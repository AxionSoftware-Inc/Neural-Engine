"""Train and evaluate a small learned sparse router on a real Qwen checkpoint.

The Qwen weights and FFN banks stay frozen. Routers learn to predict the
highest-contribution chunks from teacher hidden states, then are evaluated on
held-out text while every Qwen MLP is replaced by a hard top-k circuit route.
The oracle and random controls use the same active count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from benchmark_qwen_sparse import SparseQwenMlp
from neural_engine.pretrained_transfer import SwiGLUCircuitBank, top_contribution_circuits


TRAIN_TEXT = "\n".join(
    f"Training example {index}: sparse computation should preserve useful "
    f"language transformations while reducing unnecessary parameter traffic."
    for index in range(160)
)
EVAL_TEXT = "\n".join(
    f"Evaluation example {index}: the model must predict the next token and "
    f"retain the meaning of this sentence under selective feed-forward execution."
    for index in range(80, 144)
)


class LearnedChunkRouter(nn.Module):
    def __init__(self, hidden_size: int, num_circuits: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.SiLU(),
            nn.Linear(128, num_circuits),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)


class LearnedSparseQwenMlp(nn.Module):
    def __init__(self, bank: SwiGLUCircuitBank, router: LearnedChunkRouter, active_circuits: int):
        super().__init__()
        self.bank = bank
        self.router = router
        self.active_circuits = active_circuits

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ids = self.router(hidden_states).topk(self.active_circuits, dim=-1).indices
        return self.bank.forward_selected(hidden_states, ids)


def _token_stream(tokenizer, text: str, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    needed = batch_size * sequence_length
    if ids.numel() < needed:
        repeats = (needed + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(repeats)
    return ids[:needed].reshape(batch_size, sequence_length).to(device)


def _capture_mlp_inputs(model, input_ids: torch.Tensor) -> list[torch.Tensor]:
    layers = model.model.layers
    captured: list[torch.Tensor | None] = [None] * len(layers)
    hooks = []
    for index, layer in enumerate(layers):
        hooks.append(layer.register_forward_pre_hook(
            lambda _module, inputs, index=index: captured.__setitem__(index, inputs[0].detach())
        ))
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    if any(value is None for value in captured):
        raise RuntimeError("failed to capture the input to every Qwen MLP")
    return [value for value in captured if value is not None]


def _ce(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    return float(F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ))


def _set_route(model, banks, routers, active: int, route: str, seed: int) -> None:
    for index, (layer, bank) in enumerate(zip(model.model.layers, banks)):
        if route == "learned":
            layer.mlp = LearnedSparseQwenMlp(bank, routers[index], active)
        else:
            layer.mlp = SparseQwenMlp(
                bank,
                active_circuits=active,
                route_mode=route,
                seed=seed + index * 1009,
            )


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_qwen_learned_sparse.py requires optional packages: "
            "pip install -r requirements-transfer.txt"
        ) from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
    )
    train_ids = _token_stream(tokenizer, TRAIN_TEXT, args.batch_size, args.sequence_length, device)
    eval_ids = _token_stream(tokenizer, EVAL_TEXT, args.batch_size, args.sequence_length, device)
    with torch.no_grad():
        full_logits = model(input_ids=eval_ids, use_cache=False).logits
    full_ce = _ce(full_logits.float(), eval_ids)

    train_hidden = _capture_mlp_inputs(model, train_ids)
    banks = [
        SwiGLUCircuitBank.from_qwen_mlp(layer.mlp, args.chunk_size).to(device=device)
        for layer in model.model.layers
    ]
    num_circuits = banks[0].num_circuits
    active = max(1, round(num_circuits * args.active_fraction))
    train_targets = [
        top_contribution_circuits(bank, hidden, active).reshape(-1, active)
        for bank, hidden in zip(banks, train_hidden)
    ]
    train_hot = []
    for targets in train_targets:
        hot = torch.zeros(targets.shape[0], num_circuits, device=device)
        hot.scatter_(1, targets, 1.0)
        train_hot.append(hot)

    routers = nn.ModuleList([
        LearnedChunkRouter(int(model.config.hidden_size), num_circuits).to(device=device)
        for _ in banks
    ])
    optimizer = torch.optim.AdamW(routers.parameters(), lr=args.router_lr)
    pos_weight = torch.tensor((num_circuits - active) / active, device=device)
    last_loss = 0.0
    for _ in range(args.router_steps):
        losses = []
        for router, hidden, hot in zip(routers, train_hidden, train_hot):
            logits = router(hidden.reshape(-1, hidden.shape[-1]))
            losses.append(F.binary_cross_entropy_with_logits(logits, hot, pos_weight=pos_weight))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss)

    eval_hidden = _capture_mlp_inputs(model, eval_ids)
    with torch.no_grad():
        learned_recall = []
        for router, bank, hidden in zip(routers, banks, eval_hidden):
            predicted = router(hidden).topk(active, dim=-1).indices
            target = top_contribution_circuits(bank, hidden, active)
            learned_recall.append((
                (predicted.unsqueeze(-1) == target.unsqueeze(-2)).any(dim=-1)
            ).to(torch.float32).mean())
    result: dict[str, object] = {
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": device,
        "dtype": args.dtype,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": int(model.config.hidden_size),
        "layer_count": len(banks),
        "chunk_size": args.chunk_size,
        "circuits_per_layer": num_circuits,
        "active_circuits": active,
        "active_fraction": float(active / num_circuits),
        "full_model_ce": full_ce,
        "router_steps": args.router_steps,
        "router_train_bce": last_loss,
        "learned_route_recall_vs_local_oracle": float(torch.stack(learned_recall).mean()),
        "levels": [],
    }

    for route in ("oracle", "learned", "random"):
        _set_route(model, banks, routers, active, route, args.seed + 5000)
        with torch.no_grad():
            logits = model(input_ids=eval_ids, use_cache=False).logits
        difference = (logits - full_logits).float()
        ce = _ce(logits.float(), eval_ids)
        result["levels"].append({
            "route": route,
            "ce": ce,
            "ce_delta": ce - full_ce,
            "logit_mse": float(F.mse_loss(logits.float(), full_logits.float())),
            "max_abs_logit_error": float(difference.abs().max()),
            "top1_agreement": float((
                logits.argmax(dim=-1) == full_logits.argmax(dim=-1)
            ).to(torch.float32).mean()),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--active-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--router-steps", type=int, default=300)
    parser.add_argument("--router-lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

