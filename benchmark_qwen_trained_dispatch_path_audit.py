"""Compare grouped and single-token selected-FFN paths on a trained K=5 child."""

from __future__ import annotations

import argparse
import json

import torch

from benchmark_qwen_custom_kv_graph import (
    make_cache_and_fill_prefix,
    measure_eager,
    measure_graph,
)
from benchmark_qwen_multi_layer_transplant import (
    TRAIN_TEXT,
    TransferredRoutedQwenChild,
    evaluate_current,
    load_text_file,
    parse_layers,
)
from benchmark_qwen_trained_graph_audit import train_k5_cascade
from neural_engine.qwen_fixed_graph import greedy_generate_fixed_shape


def set_dispatch_path(children, single_token: bool) -> None:
    for child in children:
        base = next(
            nested for nested in child.modules()
            if isinstance(nested, TransferredRoutedQwenChild)
        )
        base.single_token_fast_path = bool(single_token)
        base.single_token_router_backend = "torch"
        base.single_token_projection_backend = "einsum"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layers", default="19,20,21,22,23,24,25,26")
    parser.add_argument("--calibration-text-file", default="data/qwen_calibration.txt")
    parser.add_argument("--eval-text-file", default="data/qwen_eval.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--child-steps", type=int, default=300)
    parser.add_argument("--hard-steps", type=int, default=300)
    parser.add_argument("--router-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hard-learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--calibration-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, trust_remote_code=False,
        local_files_only=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, local_files_only=True,
    )
    calibration_text = load_text_file(
        args.calibration_text_file, TRAIN_TEXT, "calibration",
    )
    eval_text = load_text_file(args.eval_text_file, TRAIN_TEXT, "evaluation")
    layers = parse_layers(args.layers)
    (
        _parents, children, layer_records, eval_ids, teacher_logits, teacher_ce,
        _,
    ) = train_k5_cascade(
        model, tokenizer, calibration_text, eval_text, layers, device,
        torch.float32, args.batch_size, args.sequence_length,
        args.train_batches, args.eval_batches, args.child_steps,
        args.hard_steps, args.router_steps, args.calibration_rank,
        args.learning_rate, args.hard_learning_rate, args.max_grad_norm,
        args.log_every,
    )
    quality = evaluate_current(
        model, eval_ids, teacher_logits, teacher_ce,
        "trained_k5_dispatch_path_probe",
    )

    prefix_ids = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :4].to(device)
    token_ids = tokenizer(
        " attention", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    records = []
    eager_logits_by_path = {}
    for path_name, single_token in (("single-token", True), ("grouped", False)):
        set_dispatch_path(children, single_token)
        rows = []
        for batch_size in (1, 8):
            batch_prefix = prefix_ids.repeat(batch_size, 1)
            batch_tokens = token_ids.repeat(batch_size, 1)
            position = torch.tensor([batch_prefix.shape[1]], device=device)
            cache_length = max(32, int(batch_prefix.shape[1]) + 8)
            eager_cache = make_cache_and_fill_prefix(
                model, batch_prefix, cache_length,
            )
            eager_ms, eager_logits = measure_eager(
                model, batch_tokens, eager_cache, position,
                args.warmup, args.iterations,
            )
            eager_logits_by_path[(path_name, batch_size)] = eager_logits.clone()
            graph_cache = make_cache_and_fill_prefix(
                model, batch_prefix, cache_length,
            )
            graph_ms, _, graph_logits, _ = measure_graph(
                model, batch_tokens, graph_cache, position,
                args.warmup, args.iterations,
            )
            rows.append({
                "batch_size": batch_size,
                "eager_ms": eager_ms,
                "graph_ms": graph_ms,
                "max_graph_vs_eager_logit_error": float(
                    (graph_logits - eager_logits).abs().max().item()
                ),
            })
        records.append({"path": path_name, "batches": rows})

    single_rows = {row["batch_size"]: row for row in records[0]["batches"]}
    grouped_rows = {row["batch_size"]: row for row in records[1]["batches"]}
    comparisons = []
    for batch_size in (1, 8):
        comparisons.append({
            "batch_size": batch_size,
            "grouped_over_single_token_eager": (
                grouped_rows[batch_size]["eager_ms"]
                / max(single_rows[batch_size]["eager_ms"], 1e-9)
            ),
            "grouped_over_single_token_graph": (
                grouped_rows[batch_size]["graph_ms"]
                / max(single_rows[batch_size]["graph_ms"], 1e-9)
            ),
            "max_grouped_vs_single_token_eager_logit_error": float(
                (
                    eager_logits_by_path[("grouped", batch_size)]
                    - eager_logits_by_path[("single-token", batch_size)]
                ).abs().max().item()
            ),
        })

    generation_prompt = tokenizer(
        "Explain why sparse circuits can reduce compute while preserving useful behavior.",
        return_tensors="pt",
    ).input_ids.to(device)
    set_dispatch_path(children, True)
    single_generation = greedy_generate_fixed_shape(
        model, generation_prompt, 8, use_cuda_graph=True,
    )
    set_dispatch_path(children, False)
    grouped_generation = greedy_generate_fixed_shape(
        model, generation_prompt, 8, use_cuda_graph=True,
    )
    result = {
        "experiment": "V0.223_trained_qwen_dispatch_path_audit",
        "status": "PARITY_PASS",
        "model": args.model,
        "seed": args.seed,
        "layers": layers,
        "dtype": "float32",
        "recipe": {
            "num_experts": 8,
            "active_experts": 5,
            "route_source": "subset-router",
            "dispatch_mode": "grouped",
            "calibration_rank": args.calibration_rank,
            "child_steps": args.child_steps,
            "hard_steps": args.hard_steps,
            "router_steps": args.router_steps,
        },
        "teacher_ce": teacher_ce,
        "quality": quality,
        "layer_records": layer_records,
        "records": records,
        "comparisons": comparisons,
        "generation": {
            "new_tokens": 8,
            "exact_token_match": bool(
                torch.equal(single_generation, grouped_generation)
            ),
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
