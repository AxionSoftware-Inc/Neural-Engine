"""Per-layer FFN sparsity sensitivity audit for Qwen3-0.6B.

Uniformly sparsifying every FFN accumulates error.  This diagnostic measures
each layer independently with the same local contribution oracle, so a later
adaptive-depth design can keep sensitive layers dense and sparsify only layers
whose output has a smaller global effect.

The oracle still computes all chunks and is therefore an upper bound, not a
runtime result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark_qwen_sparse import SparseQwenMlp
from neural_engine.pretrained_transfer import SwiGLUCircuitBank


EVAL_TEXT = "\n".join(
    f"Layer sensitivity example {index}: preserve important feed-forward "
    f"transformations while allowing harmless layers to use fewer circuits."
    for index in range(256)
)
EVAL_TEXT_ALT = "\n".join(
    f"Independent layer audit {index}: a fixed sparse schedule should keep "
    f"important transformations intact while reducing feed-forward traffic "
    f"on less sensitive layers."
    for index in range(384, 640)
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
    difference = logits.float() - teacher_logits.float()
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


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_qwen_layer_sensitivity.py requires optional packages: "
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
    eval_text = EVAL_TEXT if args.eval_variant == 0 else EVAL_TEXT_ALT
    batches = _token_batches(
        tokenizer, eval_text, args.batch_size, args.sequence_length,
        args.eval_batches, device,
    )
    input_ids = torch.cat(batches, dim=0)
    with torch.no_grad():
        teacher_logits = model(input_ids=input_ids, use_cache=False).logits
    teacher_ce = _ce(teacher_logits.float(), input_ids)

    banks = {
        index: SwiGLUCircuitBank.from_qwen_mlp(
            layer.mlp, args.chunk_size
        ).to(device=device, dtype=dtype)
        for index, layer in enumerate(model.model.layers)
    }
    num_circuits = banks[0].num_circuits
    fractions = tuple(args.active_fractions)
    results = []
    for index, bank in banks.items():
        for fraction in fractions:
            active = max(1, round(num_circuits * fraction))
            if active >= num_circuits:
                continue
            model.model.layers[index].mlp = SparseQwenMlp(
                bank,
                active_circuits=active,
                route_mode="oracle",
                seed=args.seed + index * 1009,
            )
            with torch.no_grad():
                logits = model(input_ids=input_ids, use_cache=False).logits
            results.append({
                "layer_index": index,
                "active_circuits": active,
                "active_fraction": float(active / num_circuits),
                **_metrics(logits, teacher_logits, input_ids),
            })
        # The next layer remains independent; restore the original module
        # slot so model structure is unambiguous after the audit.
        model.model.layers[index].mlp = bank

    results.sort(key=lambda row: (row["active_fraction"], row["ce_delta"]))
    combination_results = []
    single_25 = [
        row for row in results
        if abs(row["active_fraction"] - 0.25) < 1e-8
    ]
    ranked_layers = [row["layer_index"] for row in sorted(
        single_25, key=lambda row: row["ce_delta"]
    )]
    # The ranking is only a cheap screening heuristic.  These combined rows
    # reveal whether individually harmless layers remain harmless together.
    for count in args.combination_counts:
        if count < 1 or count > len(ranked_layers):
            continue
        selected = set(ranked_layers[:count])
        for index, bank in banks.items():
            if index in selected:
                active = max(1, round(num_circuits * 0.25))
                model.model.layers[index].mlp = SparseQwenMlp(
                    bank,
                    active_circuits=active,
                    route_mode="oracle",
                    seed=args.seed + index * 1009,
                )
            else:
                model.model.layers[index].mlp = bank
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
        combined_fraction = 1.0 - (count / len(banks)) * 0.75
        combination_results.append({
            "sparse_layer_count": count,
            "selected_layers_by_single_layer_screen": sorted(selected),
            "effective_target_ffn_active_fraction": combined_fraction,
            **_metrics(logits, teacher_logits, input_ids),
        })
    for index, bank in banks.items():
        model.model.layers[index].mlp = bank
    report = {
        "experiment": "qwen_per_layer_ffn_sensitivity",
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "eval_batches": args.eval_batches,
        "eval_variant": args.eval_variant,
        "layer_count": len(model.model.layers),
        "hidden_size": int(model.config.hidden_size),
        "chunk_size": args.chunk_size,
        "circuits_per_layer": num_circuits,
        "teacher_ce": teacher_ce,
        "results": results,
        "combined_ranked_25pct": combination_results,
        "interpretation": (
            "each row sparsifies one layer only with a local contribution oracle; "
            "this measures layer sensitivity, not deployable route quality"
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
    parser.add_argument("--active-fractions", type=float, nargs="+", default=[0.25, 0.50])
    parser.add_argument("--combination-counts", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--eval-variant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_layer_sensitivity_chunk64.json")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
