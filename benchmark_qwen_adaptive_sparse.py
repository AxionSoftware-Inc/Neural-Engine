"""Upper-bound audit for an adaptive active-circuit budget on Qwen FFNs.

For each token and layer, this diagnostic computes every converted circuit,
orders them by contribution norm, and retains the smallest prefix whose local
output error is below a relative tolerance.  It is intentionally an oracle:
the full scan is not deployable.  Its purpose is to answer a more fundamental
question before training a router: can an input-dependent active budget retain
the teacher at all?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from neural_engine.pretrained_transfer import SwiGLUCircuitBank


EVAL_TEXT = "\n".join(
    f"Adaptive evaluation example {index}: simple tokens may use fewer "
    f"circuits, while difficult context should activate more computation "
    f"to preserve the teacher output."
    for index in range(256)
)


def _token_batches(tokenizer, text, batch_size, sequence_length, count, device):
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    needed = batch_size * sequence_length * count
    if ids.numel() < needed:
        ids = ids.repeat((needed + ids.numel() - 1) // ids.numel())
    ids = ids[:needed].reshape(count, batch_size, sequence_length)
    return [batch.to(device) for batch in ids]


def _ce(logits, input_ids):
    return float(F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ))


def _metrics(logits, teacher_logits, input_ids):
    difference = (logits.float() - teacher_logits.float())
    ce = _ce(logits.float(), input_ids)
    teacher_ce = _ce(teacher_logits.float(), input_ids)
    return {
        "ce": ce,
        "ce_delta": ce - teacher_ce,
        "logit_mse": float(F.mse_loss(logits.float(), teacher_logits.float())),
        "max_abs_logit_error": float(difference.abs().max()),
        "top1_agreement": float((
            logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)
        ).to(torch.float32).mean()),
    }


class AdaptiveContributionOracle(nn.Module):
    """Non-deployable local oracle with a token-dependent circuit count."""

    def __init__(self, bank: SwiGLUCircuitBank, relative_tolerance: float):
        super().__init__()
        if relative_tolerance < 0:
            raise ValueError("relative_tolerance must be non-negative")
        self.bank = bank
        self.relative_tolerance = float(relative_tolerance)
        self.last_mean_active = 0.0
        self.last_max_active = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        contributions = self.bank.chunk_outputs(hidden_states)
        norms = contributions.norm(dim=-1)
        order = norms.argsort(dim=-1, descending=True)
        ordered = contributions.gather(
            -2, order.unsqueeze(-1).expand(*order.shape, self.bank.hidden_size)
        )
        full = contributions.sum(dim=-2)
        cumulative = ordered.cumsum(dim=-2)
        relative_error = (
            (full.unsqueeze(-2) - cumulative).norm(dim=-1)
            / full.norm(dim=-1).clamp_min(1e-6).unsqueeze(-1)
        )
        acceptable = relative_error <= self.relative_tolerance
        has_acceptable = acceptable.any(dim=-1)
        first = acceptable.to(torch.long).argmax(dim=-1) + 1
        circuit_count = torch.where(
            has_acceptable,
            first,
            torch.full_like(first, self.bank.num_circuits),
        )
        slots = torch.arange(
            self.bank.num_circuits, device=hidden_states.device
        ).view(*([1] * (circuit_count.ndim)), -1)
        weights = (slots < circuit_count.unsqueeze(-1)).to(ordered.dtype)
        result = (ordered * weights.unsqueeze(-1)).sum(dim=-2) + self.bank.down_bias
        self.last_mean_active = float(circuit_count.float().mean())
        self.last_max_active = int(circuit_count.max())
        return result


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_qwen_adaptive_sparse.py requires optional packages: "
            "pip install -r requirements-transfer.txt"
        ) from exc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
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
    batches = _token_batches(
        tokenizer, EVAL_TEXT, args.batch_size, args.sequence_length,
        args.eval_batches, device,
    )
    input_ids = torch.cat(batches, dim=0)
    with torch.no_grad():
        teacher_logits = model(input_ids=input_ids, use_cache=False).logits
    teacher_ce = _ce(teacher_logits.float(), input_ids)

    layer_count = len(model.model.layers)
    if args.layer_indices:
        layer_indices = [int(value.strip()) for value in args.layer_indices.split(",") if value.strip()]
    else:
        layer_indices = list(range(layer_count))
    if not layer_indices or any(index < 0 or index >= layer_count for index in layer_indices):
        raise ValueError("layer indices must refer to existing Qwen layers")
    banks = {
        index: SwiGLUCircuitBank.from_qwen_mlp(
            model.model.layers[index].mlp, args.chunk_size
        ).to(device=device, dtype=dtype)
        for index in layer_indices
    }
    num_circuits = banks[layer_indices[0]].num_circuits
    if any(bank.num_circuits != num_circuits for bank in banks.values()):
        raise ValueError("all selected layers must have the same circuit count")

    results = []
    for tolerance in args.tolerances:
        modules = {
            index: AdaptiveContributionOracle(bank, tolerance).to(device)
            for index, bank in banks.items()
        }
        for index, module in modules.items():
            model.model.layers[index].mlp = module
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
        metrics = _metrics(logits, teacher_logits, input_ids)
        active_counts = [module.last_mean_active for module in modules.values()]
        results.append({
            "relative_tolerance": tolerance,
            "mean_active_circuits": sum(active_counts) / len(active_counts),
            "active_fraction": sum(active_counts) / len(active_counts) / num_circuits,
            "max_active_circuits": max(module.last_max_active for module in modules.values()),
            **metrics,
        })

    report = {
        "experiment": "qwen_adaptive_contribution_oracle",
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "eval_batches": args.eval_batches,
        "layer_indices": layer_indices,
        "layer_count": layer_count,
        "hidden_size": int(model.config.hidden_size),
        "chunk_size": args.chunk_size,
        "circuits_per_layer": num_circuits,
        "teacher_ce": teacher_ce,
        "results": results,
        "interpretation": (
            "oracle computes every circuit and is an upper bound; it cannot be "
            "used as evidence of deployable routing latency"
        ),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--layer-indices", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_adaptive_oracle_chunk64.json")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
