"""Fast, reproducible exact-graft pilot for Qwen-style SwiGLU FFNs.

This uses a tiny synthetic module with the same projection algebra because a
Qwen checkpoint/Transformers installation is intentionally not a dependency
of the repository.  It validates conversion correctness and reports an
offline top-contribution sparsity upper bound.  The latter evaluates all
circuits and is diagnostic only, not a deployable router.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

from neural_engine.pretrained_transfer import SwiGLUCircuitBank, top_contribution_circuits


class SyntheticQwenMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class PilotChunkRouter(nn.Module):
    """Small diagnostic router; its all-address score is not the final design."""

    def __init__(self, hidden_size: int, num_circuits: int):
        super().__init__()
        width = max(32, min(128, hidden_size))
        self.network = nn.Sequential(
            nn.Linear(hidden_size, width),
            nn.SiLU(),
            nn.Linear(width, num_circuits),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)


def _route_recall(predicted: torch.Tensor, target: torch.Tensor) -> float:
    matches = (predicted.unsqueeze(-1) == target.unsqueeze(-2)).any(dim=-1)
    return float(matches.to(torch.float32).mean())


def run_router_pilot(
    bank: SwiGLUCircuitBank,
    *,
    hidden_size: int,
    batch_size: int,
    seed: int,
    active_circuits: int,
    steps: int,
) -> dict[str, float | int]:
    """Train a tiny router against offline top-contribution labels.

    The labels are deliberately produced by ``top_contribution_circuits``.
    This asks whether a cheap learned function of ``x`` can predict useful
    pretrained chunks, while making clear that the oracle itself is not a
    deployable routing algorithm.
    """
    train_inputs = torch.randn(4096, hidden_size)
    eval_inputs = torch.randn(1024, hidden_size)
    with torch.no_grad():
        train_targets = top_contribution_circuits(bank, train_inputs, active_circuits)
        eval_targets = top_contribution_circuits(bank, eval_inputs, active_circuits)
        exact = bank(eval_inputs)
        oracle = bank.forward_selected(eval_inputs, eval_targets)

    router = PilotChunkRouter(hidden_size, bank.num_circuits)
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-2)
    target_hot = torch.zeros(train_inputs.shape[0], bank.num_circuits)
    target_hot.scatter_(1, train_targets, 1.0)
    positive_weight = torch.tensor(
        (bank.num_circuits - active_circuits) / active_circuits,
        dtype=train_inputs.dtype,
    )
    generator = torch.Generator().manual_seed(seed + 91)
    for _ in range(steps):
        indices = torch.randint(
            train_inputs.shape[0], (batch_size,), generator=generator
        )
        logits = router(train_inputs[indices])
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target_hot[indices],
            pos_weight=positive_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predicted = router(eval_inputs).topk(active_circuits, dim=-1).indices
        routed = bank.forward_selected(eval_inputs, predicted)
        random_generator = torch.Generator().manual_seed(seed + 92)
        random_ids = torch.stack([
            torch.randperm(bank.num_circuits, generator=random_generator)[:active_circuits]
            for _ in range(eval_inputs.shape[0])
        ])
        random_routed = bank.forward_selected(eval_inputs, random_ids)
    return {
        "active_circuits": active_circuits,
        "router_steps": steps,
        "oracle_mse": float(F.mse_loss(oracle, exact)),
        "learned_router_mse": float(F.mse_loss(routed, exact)),
        "random_router_mse": float(F.mse_loss(random_routed, exact)),
        "learned_route_recall": _route_recall(predicted, eval_targets),
        "router_parameter_count": sum(parameter.numel() for parameter in router.parameters()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    source = SyntheticQwenMlp(args.hidden_size, args.intermediate_size)
    bank = SwiGLUCircuitBank.from_qwen_mlp(source, args.chunk_size)
    inputs = torch.randn(args.batch_size, args.hidden_size)
    exact = source(inputs)
    converted = bank(inputs)
    result: dict[str, object] = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "chunk_size": args.chunk_size,
        "num_circuits": bank.num_circuits,
        "max_abs_conversion_error": float((converted - exact).abs().max()),
        "mean_abs_conversion_error": float((converted - exact).abs().mean()),
        "parameter_report": bank.parameter_report(),
        "sparse_oracle": [],
        "learned_router_pilot": {},
    }
    dense_macs = 3 * args.batch_size * args.hidden_size * args.intermediate_size
    for active in sorted(set((bank.num_circuits, max(1, bank.num_circuits // 2),
                              max(1, bank.num_circuits // 4)))):
        ids = top_contribution_circuits(bank, inputs, active)
        sparse = bank.forward_selected(inputs, ids)
        mse = F.mse_loss(sparse, exact)
        active_intermediate = sum(int(bank.chunk_sizes[index]) for index in ids[0].tolist())
        result["sparse_oracle"].append({
            "active_circuits": active,
            "active_intermediate_for_first_sample": active_intermediate,
            "estimated_ffn_mac_fraction": float(active_intermediate / args.intermediate_size),
            "mse_to_exact": float(mse),
            "max_abs_error_to_exact": float((sparse - exact).abs().max()),
        })
    result["learned_router_pilot"] = run_router_pilot(
        bank,
        hidden_size=args.hidden_size,
        batch_size=args.router_batch_size,
        seed=args.seed,
        active_circuits=max(1, bank.num_circuits // 4),
        steps=args.router_steps,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--router-batch-size", type=int, default=256)
    parser.add_argument("--router-steps", type=int, default=800)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
