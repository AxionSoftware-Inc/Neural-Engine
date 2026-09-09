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
    fused_correction: bool = False, uniform_accum: bool = False,
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
        base.grouped_uniform_accum = bool(uniform_accum)
        for nested in child.modules():
            if isinstance(nested, CrossGroupOutputMixRoutedQwenChild):
                if dispatch_mode == "fused-effective-output":
                    nested.correction_dispatch_backend = (
                        "cuda-fused-effective-output"
                    )
                elif dispatch_mode in {
                    "grouped-adaptive-effective-output",
                    "grouped-adaptive-atomic-effective-output",
                }:
                    nested.correction_dispatch_backend = "grouped-effective-output"
                else:
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
    parser.add_argument(
        "--include-grouped-uniform-fused", action="store_true",
        help="include the opt-in uniform K-subset accumulation shortcut",
    )
    parser.add_argument(
        "--include-grouped-prepacked", action="store_true",
        help="include the opt-in contiguous grouped-BMM weight layout probe",
    )
    parser.add_argument(
        "--include-grouped-prepacked-fused", action="store_true",
        help="include the fused-projection variant of the weight layout probe",
    )
    parser.add_argument(
        "--include-grouped-tiled", action="store_true",
        help="include the opt-in tiled CUDA grouped projection kernel",
    )
    parser.add_argument(
        "--include-grouped-optimized", action="store_true",
        help="include the combined cached/prepacked/fused grouped path",
    )
    parser.add_argument(
        "--include-grouped-adaptive", action="store_true",
        help="include the shape-aware grouped path (plain B1, optimized B>1)",
    )
    parser.add_argument(
        "--include-grouped-adaptive-nozero", action="store_true",
        help="include adaptive grouped dispatch without zero-filling padded rows",
    )
    parser.add_argument(
        "--include-grouped-adaptive-effective-output", action="store_true",
        help="include adaptive grouped dispatch with folded low-rank correction",
    )
    parser.add_argument(
        "--include-grouped-adaptive-atomic-pack", action="store_true",
        help="include adaptive grouped dispatch with CUDA atomic route packing",
    )
    parser.add_argument(
        "--include-grouped-adaptive-atomic-effective-output", action="store_true",
        help="include atomic route packing with folded correction output",
    )
    parser.add_argument(
        "--include-fused-effective-output", action="store_true",
        help="include direct fused dispatch with folded correction output",
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
    if args.include_grouped_uniform_fused:
        path_specs.append((
            "grouped-uniform-correction-fused", False, "grouped", True, True,
        ))
    if args.include_grouped_prepacked:
        path_specs.append(("grouped-prepacked", False, "grouped-prepacked"))
    if args.include_grouped_prepacked_fused:
        path_specs.append((
            "grouped-prepacked-fused", False, "grouped-prepacked-fused",
        ))
    if args.include_grouped_tiled:
        path_specs.append(("grouped-tiled", False, "grouped-tiled"))
    if args.include_grouped_optimized:
        path_specs.append((
            "grouped-optimized", False, "grouped-optimized", True, True,
        ))
    if args.include_grouped_adaptive:
        path_specs.append((
            "grouped-adaptive", False, "grouped-adaptive", True, True,
        ))
    if args.include_grouped_adaptive_nozero:
        path_specs.append((
            "grouped-adaptive-nozero", False, "grouped-adaptive-nozero", True, True,
        ))
    if args.include_grouped_adaptive_effective_output:
        path_specs.append((
            "grouped-adaptive-effective-output", False,
            "grouped-adaptive-effective-output", False, True,
        ))
    if args.include_grouped_adaptive_atomic_pack:
        path_specs.append((
            "grouped-adaptive-atomic-pack", False,
            "grouped-adaptive-atomic-pack", True, True,
        ))
    if args.include_grouped_adaptive_atomic_effective_output:
        path_specs.append((
            "grouped-adaptive-atomic-effective-output", False,
            "grouped-adaptive-atomic-effective-output", False, True,
        ))
    if args.include_fused_effective_output:
        path_specs.append((
            "fused-effective-output", False, "fused-effective-output",
        ))
    for path_spec in path_specs:
        if len(path_spec) == 3:
            path_name, single_token, dispatch_mode = path_spec
            fused_correction = False
            uniform_accum = False
        elif len(path_spec) == 4:
            path_name, single_token, dispatch_mode, fused_correction = path_spec
            uniform_accum = False
        else:
            (
                path_name, single_token, dispatch_mode, fused_correction,
                uniform_accum,
            ) = path_spec
        set_dispatch_path(
            children, single_token, dispatch_mode,
            fused_correction=fused_correction,
            uniform_accum=uniform_accum,
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
            if args.include_grouped_uniform_fused:
                candidates.append("grouped-uniform-correction-fused")
            if args.include_grouped_prepacked:
                candidates.append("grouped-prepacked")
            if args.include_grouped_prepacked_fused:
                candidates.append("grouped-prepacked-fused")
            if args.include_grouped_tiled:
                candidates.append("grouped-tiled")
            if args.include_grouped_optimized:
                candidates.append("grouped-optimized")
            if args.include_grouped_adaptive:
                candidates.append("grouped-adaptive")
            if args.include_grouped_adaptive_nozero:
                candidates.append("grouped-adaptive-nozero")
            if args.include_grouped_adaptive_effective_output:
                candidates.append("grouped-adaptive-effective-output")
            if args.include_grouped_adaptive_atomic_pack:
                candidates.append("grouped-adaptive-atomic-pack")
            if args.include_grouped_adaptive_atomic_effective_output:
                candidates.append("grouped-adaptive-atomic-effective-output")
            if args.include_fused_effective_output:
                candidates.append("fused-effective-output")
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
    grouped_prepacked_generation = None
    if args.include_grouped_prepacked:
        set_dispatch_path(children, False, "grouped-prepacked")
        grouped_prepacked_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    grouped_prepacked_fused_generation = None
    if args.include_grouped_prepacked_fused:
        set_dispatch_path(children, False, "grouped-prepacked-fused")
        grouped_prepacked_fused_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    grouped_tiled_generation = None
    if args.include_grouped_tiled:
        set_dispatch_path(children, False, "grouped-tiled")
        grouped_tiled_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    grouped_optimized_generation = None
    if args.include_grouped_optimized:
        set_dispatch_path(
            children, False, "grouped-optimized",
            fused_correction=True, uniform_accum=True,
        )
        grouped_optimized_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    grouped_adaptive_generation = None
    if args.include_grouped_adaptive:
        set_dispatch_path(
            children, False, "grouped-adaptive",
            fused_correction=True, uniform_accum=True,
        )
        grouped_adaptive_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    grouped_adaptive_nozero_generation = None
    if args.include_grouped_adaptive_nozero:
        set_dispatch_path(
            children, False, "grouped-adaptive-nozero",
            fused_correction=True, uniform_accum=True,
        )
        grouped_adaptive_nozero_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    grouped_adaptive_effective_output_generation = None
    if args.include_grouped_adaptive_effective_output:
        set_dispatch_path(
            children, False, "grouped-adaptive-effective-output",
            uniform_accum=True,
        )
        grouped_adaptive_effective_output_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    grouped_adaptive_atomic_pack_generation = None
    if args.include_grouped_adaptive_atomic_pack:
        set_dispatch_path(
            children, False, "grouped-adaptive-atomic-pack",
            fused_correction=True, uniform_accum=True,
        )
        grouped_adaptive_atomic_pack_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    grouped_adaptive_atomic_effective_output_generation = None
    if args.include_grouped_adaptive_atomic_effective_output:
        set_dispatch_path(
            children, False, "grouped-adaptive-atomic-effective-output",
            uniform_accum=True,
        )
        grouped_adaptive_atomic_effective_output_generation = greedy_generate_fixed_shape(
            model, generation_prompt, 8, use_cuda_graph=True,
        )
    fused_effective_output_generation = None
    if args.include_fused_effective_output:
        set_dispatch_path(children, False, "fused-effective-output")
        fused_effective_output_generation = greedy_generate_fixed_shape(
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
    uniform_correction_fused_generation = None
    if args.include_grouped_uniform_fused:
        set_dispatch_path(
            children, False, "grouped", fused_correction=True,
            uniform_accum=True,
        )
        uniform_correction_fused_generation = greedy_generate_fixed_shape(
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
            **({
                "grouped_vs_grouped_uniform_correction_fused_exact_token_match": bool(
                    torch.equal(grouped_generation, uniform_correction_fused_generation)
                ),
            } if uniform_correction_fused_generation is not None else {}),
            **({
                "grouped_vs_grouped_prepacked_exact_token_match": bool(
                    torch.equal(grouped_generation, grouped_prepacked_generation)
                ),
            } if grouped_prepacked_generation is not None else {}),
            **({
                "grouped_vs_grouped_prepacked_fused_exact_token_match": bool(
                    torch.equal(
                        grouped_generation, grouped_prepacked_fused_generation,
                    )
                ),
            } if grouped_prepacked_fused_generation is not None else {}),
            **({
                "grouped_vs_grouped_tiled_exact_token_match": bool(
                    torch.equal(grouped_generation, grouped_tiled_generation)
                ),
            } if grouped_tiled_generation is not None else {}),
            **({
                "grouped_vs_grouped_optimized_exact_token_match": bool(
                    torch.equal(
                        grouped_generation, grouped_optimized_generation,
                    )
                ),
            } if grouped_optimized_generation is not None else {}),
            **({
                "grouped_vs_grouped_adaptive_exact_token_match": bool(
                    torch.equal(
                        grouped_generation, grouped_adaptive_generation,
                    )
                ),
            } if grouped_adaptive_generation is not None else {}),
            **({
                "grouped_vs_grouped_adaptive_nozero_exact_token_match": bool(
                    torch.equal(
                        grouped_generation, grouped_adaptive_nozero_generation,
                    )
                ),
            } if grouped_adaptive_nozero_generation is not None else {}),
            **({
                "grouped_vs_grouped_adaptive_effective_output_exact_token_match": bool(
                    torch.equal(
                        grouped_generation,
                        grouped_adaptive_effective_output_generation,
                    )
                ),
            } if grouped_adaptive_effective_output_generation is not None else {}),
            **({
                "grouped_vs_grouped_adaptive_atomic_pack_exact_token_match": bool(
                    torch.equal(
                        grouped_generation,
                        grouped_adaptive_atomic_pack_generation,
                    )
                ),
            } if grouped_adaptive_atomic_pack_generation is not None else {}),
            **({
                "grouped_vs_grouped_adaptive_atomic_effective_output_exact_token_match": bool(
                    torch.equal(
                        grouped_generation,
                        grouped_adaptive_atomic_effective_output_generation,
                    )
                ),
            } if grouped_adaptive_atomic_effective_output_generation is not None else {}),
            **({
                "grouped_vs_fused_effective_output_exact_token_match": bool(
                    torch.equal(grouped_generation, fused_effective_output_generation)
                ),
            } if fused_effective_output_generation is not None else {}),
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
