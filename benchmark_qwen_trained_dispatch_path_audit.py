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
    CrossGroupOutputMixRoutedQwenChild,
    TRAIN_TEXT,
    TransferredRoutedQwenChild,
    evaluate_current,
    load_text_file,
    parse_layers,
)
from benchmark_qwen_trained_graph_audit import train_k5_cascade
from neural_engine.qwen_fixed_graph import greedy_generate_fixed_shape


def set_dispatch_path(
    children, single_token: bool, dispatch_mode: str = "grouped",
    fused_correction: bool = False,
) -> None:
    for child in children:
        base = next(
            nested for nested in child.modules()
            if isinstance(nested, TransferredRoutedQwenChild)
        )
        base.single_token_fast_path = bool(single_token)
        base.single_token_router_backend = "torch"
        base.single_token_projection_backend = "einsum"
        base.dispatch_mode = dispatch_mode
        for nested in child.modules():
            if isinstance(nested, CrossGroupOutputMixRoutedQwenChild):
                nested.correction_dispatch_backend = (
                    "grouped-fused-correction" if fused_correction else "vectorized"
                )


def install_children(model, layers, children) -> None:
    for layer_index, child in zip(layers, children):
        model.model.layers[layer_index].mlp = child


def install_parents(model, layers, parents) -> None:
    for layer_index, parent in zip(layers, parents):
        model.model.layers[layer_index].mlp = parent


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
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--calibration-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--experiment", default="V0.224_trained_grouped_fused_audit")
    parser.add_argument(
        "--include-cached-grouped", action="store_true",
        help="include the opt-in route-independent grouped metadata cache",
    )
    parser.add_argument(
        "--include-grouped-correction-fused", action="store_true",
        help="include the opt-in grouped selected-output/correction fusion",
    )
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

    prefix_pool = tokenizer(
        "Neural Engine sparse circuits " * 256, return_tensors="pt",
    ).input_ids.to(device)
    if max(args.prefix_lengths) > prefix_pool.shape[1]:
        raise ValueError("prefix-length exceeds the generated prefix pool")
    token_ids = tokenizer(
        " attention", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    dense_records = []
    install_parents(model, layers, _parents)
    for prefix_length in args.prefix_lengths:
        prefix_ids = prefix_pool[:, :prefix_length]
        for batch_size in args.batch_sizes:
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
            graph_cache = make_cache_and_fill_prefix(
                model, batch_prefix, cache_length,
            )
            graph_ms, _, graph_logits, _ = measure_graph(
                model, batch_tokens, graph_cache, position,
                args.warmup, args.iterations,
            )
            dense_records.append({
                "prefix_length": prefix_length,
                "batch_size": batch_size,
                "eager_ms": eager_ms,
                "graph_ms": graph_ms,
                "max_graph_vs_eager_logit_error": float(
                    (graph_logits - eager_logits).abs().max().item()
                ),
            })
    install_children(model, layers, children)
    records = []
    eager_logits_by_path = {}
    path_specs = [
        ("single-token", True, "grouped"),
        ("grouped", False, "grouped"),
        ("grouped-fused", False, "grouped-fused"),
    ]
    if args.include_cached_grouped:
        path_specs.append(("grouped-cached", False, "grouped-cached"))
    if args.include_grouped_correction_fused:
        path_specs.append((
            "grouped-correction-fused", False, "grouped", True,
        ))
    for path_spec in path_specs:
        if len(path_spec) == 3:
            path_name, single_token, dispatch_mode = path_spec
            fused_correction = False
        else:
            path_name, single_token, dispatch_mode, fused_correction = path_spec
        set_dispatch_path(
            children, single_token, dispatch_mode,
            fused_correction=fused_correction,
        )
        rows = []
        for prefix_length in args.prefix_lengths:
            prefix_ids = prefix_pool[:, :prefix_length]
            for batch_size in args.batch_sizes:
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
                eager_logits_by_path[(path_name, prefix_length, batch_size)] = (
                    eager_logits.clone()
                )
                graph_cache = make_cache_and_fill_prefix(
                    model, batch_prefix, cache_length,
                )
                graph_ms, _, graph_logits, _ = measure_graph(
                    model, batch_tokens, graph_cache, position,
                    args.warmup, args.iterations,
                )
                rows.append({
                    "prefix_length": prefix_length,
                    "batch_size": batch_size,
                    "eager_ms": eager_ms,
                    "graph_ms": graph_ms,
                    "max_graph_vs_eager_logit_error": float(
                        (graph_logits - eager_logits).abs().max().item()
                    ),
                })
        records.append({"path": path_name, "batches": rows})

    rows_by_path = {
        record["path"]: {
            (row["prefix_length"], row["batch_size"]): row
            for row in record["batches"]
        }
        for record in records
    }
    comparisons = []
    for prefix_length in args.prefix_lengths:
        for batch_size in args.batch_sizes:
            single = rows_by_path["single-token"][(prefix_length, batch_size)]
            comparison = {
                "prefix_length": prefix_length,
                "batch_size": batch_size,
            }
            dense = next(
                row for row in dense_records
                if row["prefix_length"] == prefix_length
                and row["batch_size"] == batch_size
            )
            candidates = ["grouped", "grouped-fused"]
            if args.include_cached_grouped:
                candidates.append("grouped-cached")
            if args.include_grouped_correction_fused:
                candidates.append("grouped-correction-fused")
            for candidate in candidates:
                candidate_row = rows_by_path[candidate][
                    (prefix_length, batch_size)
                ]
                eager_error = (
                    eager_logits_by_path[(candidate, prefix_length, batch_size)]
                    - eager_logits_by_path[("single-token", prefix_length, batch_size)]
                ).abs().max().item()
                comparison[f"{candidate}_over_single_token_eager"] = (
                    candidate_row["eager_ms"] / max(single["eager_ms"], 1e-9)
                )
                comparison[f"{candidate}_over_single_token_graph"] = (
                    candidate_row["graph_ms"] / max(single["graph_ms"], 1e-9)
                )
                comparison[f"max_{candidate}_vs_single_token_eager_logit_error"] = (
                    float(eager_error)
                )
                comparison[f"{candidate}_over_dense_eager"] = (
                    candidate_row["eager_ms"] / max(dense["eager_ms"], 1e-9)
                )
                comparison[f"{candidate}_over_dense_graph"] = (
                    candidate_row["graph_ms"] / max(dense["graph_ms"], 1e-9)
                )
            comparisons.append(comparison)

    generation_prompt = tokenizer(
        "Explain why sparse circuits can reduce compute while preserving useful behavior.",
        return_tensors="pt",
    ).input_ids.to(device)
    set_dispatch_path(children, True, "grouped")
    single_generation = greedy_generate_fixed_shape(
        model, generation_prompt, 8, use_cuda_graph=True,
    )
    set_dispatch_path(children, False, "grouped")
    grouped_generation = greedy_generate_fixed_shape(
        model, generation_prompt, 8, use_cuda_graph=True,
    )
    set_dispatch_path(children, False, "grouped-fused")
    grouped_fused_generation = greedy_generate_fixed_shape(
        model, generation_prompt, 8, use_cuda_graph=True,
    )
    cached_grouped_generation = None
    if args.include_cached_grouped:
        set_dispatch_path(children, False, "grouped-cached")
        cached_grouped_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    correction_fused_generation = None
    if args.include_grouped_correction_fused:
        set_dispatch_path(
            children, False, "grouped", fused_correction=True,
        )
        correction_fused_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    result = {
        "experiment": args.experiment,
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
        "batch_sizes": args.batch_sizes,
        "prefix_lengths": args.prefix_lengths,
        "teacher_ce": teacher_ce,
        "quality": quality,
        "layer_records": layer_records,
        "dense_records": dense_records,
        "records": records,
        "comparisons": comparisons,
        "generation": {
            "new_tokens": 8,
            "single_vs_grouped_exact_token_match": bool(
                torch.equal(single_generation, grouped_generation)
            ),
            "grouped_vs_grouped_fused_exact_token_match": bool(
                torch.equal(grouped_generation, grouped_fused_generation)
            ),
            **({
                "grouped_vs_grouped_cached_exact_token_match": bool(
                    torch.equal(grouped_generation, cached_grouped_generation)
                ),
            } if cached_grouped_generation is not None else {}),
            **({
                "grouped_vs_grouped_correction_fused_exact_token_match": bool(
                    torch.equal(grouped_generation, correction_fused_generation)
                ),
            } if correction_fused_generation is not None else {}),
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
