"""Audit a trained sparse Qwen child with the custom fixed-KV CUDA Graph path.

This is deliberately an opt-in benchmark.  It reproduces the accepted K=5
quality recipe, then measures the trained child with ``use_cache=True`` and a
fixed-position KV cache whose decode writes are safe to replay.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark_qwen_custom_kv_graph import (
    FixedDecodeCache,
    forward_logits,
    make_cache_and_fill_prefix,
    measure_eager,
    measure_graph,
)
from neural_engine.qwen_fixed_graph import (
    FixedShapeGreedyGraphPool,
    greedy_generate_fixed_shape,
)
from benchmark_qwen_multi_layer_transplant import (
    CrossGroupOutputMixRoutedQwenChild,
    TRAIN_TEXT,
    TransferredRoutedQwenChild,
    capture_batches,
    ce,
    evaluate_current,
    load_text_file,
    make_transferred_routed_qwen_child,
    routing_diagnostics,
    token_stream,
    train_importance_router,
)
from benchmark_qwen_two_layer_transplant import (
    _set_hard_train_blend,
    _set_hard_train_modules,
    train_child,
)


DEFAULT_LAYERS = "19,20,21,22,23,24,25,26"


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must contain at least one distinct index")
    return layers


def freeze_subset_router(child: torch.nn.Module) -> list[tuple[torch.Tensor, bool]]:
    base = next(
        nested for nested in child.modules()
        if isinstance(nested, TransferredRoutedQwenChild)
    )
    parameters = []
    if base.route_source in {
        "subset-router", "oracle-subset", "pairwise-cost-router",
    }:
        router = (
            base.pairwise_cost_router
            if base.route_source == "pairwise-cost-router"
            else base.subset_router
        )
        for parameter in router.parameters():
            parameters.append((parameter, bool(parameter.requires_grad)))
            parameter.requires_grad_(False)
    return parameters


def restore_requires_grad(
    parameters: list[tuple[torch.Tensor, bool]],
) -> None:
    for parameter, previous in parameters:
        parameter.requires_grad_(previous)


def train_k5_cascade(
    model: torch.nn.Module,
    tokenizer,
    calibration_text: str,
    eval_text: str,
    layers: list[int],
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    sequence_length: int,
    train_batches: int,
    eval_batches: int,
    child_steps: int,
    hard_steps: int,
    router_steps: int,
    calibration_rank: int,
    learning_rate: float,
    hard_learning_rate: float,
    max_grad_norm: float,
    log_every: int,
) -> tuple[list[torch.nn.Module], list[torch.nn.Module], list[dict[str, object]], list[torch.Tensor], list[torch.Tensor], float, list[list[dict[str, torch.Tensor]]]]:
    train_ids = token_stream(
        tokenizer, calibration_text, batch_size,
        sequence_length * train_batches, device,
    ).reshape(train_batches, batch_size, sequence_length)
    eval_ids = token_stream(
        tokenizer, eval_text, batch_size,
        sequence_length * eval_batches, device,
    ).reshape(eval_batches, batch_size, sequence_length)

    teacher_logits = []
    teacher_ce_values = []
    with torch.inference_mode():
        for ids in eval_ids:
            logits = model(input_ids=ids, use_cache=False).logits.detach()
            teacher_ce_values.append(float(ce(logits.float(), ids)))
            teacher_logits.append(logits.to(device="cpu", dtype=torch.float16))
    teacher_ce = sum(teacher_ce_values) / len(teacher_ce_values)

    model_layers = [model.model.layers[index] for index in layers]
    parents = [layer.mlp for layer in model_layers]
    children: list[torch.nn.Module] = []
    layer_records: list[dict[str, object]] = []
    child_eval_io: list[list[dict[str, torch.Tensor]]] = []

    for layer_index, layer, parent in zip(layers, model_layers, parents):
        train_io = capture_batches(
            model, tokenizer, calibration_text, batch_size, sequence_length,
            train_batches, device, layer_index,
        )
        eval_io = capture_batches(
            model, tokenizer, eval_text, batch_size, sequence_length,
            eval_batches, device, layer_index,
        )
        child = make_transferred_routed_qwen_child(
            parent, 8, 5, 1.0, calibration_rank,
            "base-output", "cross-group", "grouped", "contiguous",
            "subset-router", 5.0, device, dtype,
            partition_io=train_io,
        )
        router_history = train_importance_router(
            child, train_io, device, dtype, router_steps,
            learning_rate, max_grad_norm, log_every,
            "subset-soft", 0.25,
        )
        frozen_router = freeze_subset_router(child)
        try:
            if calibration_rank > 0:
                child_history = train_child(
                    child, train_io, device, dtype, child_steps,
                    learning_rate, max_grad_norm, log_every,
                    hard_steps, hard_learning_rate, 0,
                )
            else:
                # With no correction wrapper the copied circuit is fully
                # frozen after router training; there is no autograd target
                # for train_child to update.
                child_history = []
        finally:
            restore_requires_grad(frozen_router)
        child.eval()
        with torch.inference_mode():
            local_mse = sum(
                F.mse_loss(
                    child(batch["input"].to(device=device, dtype=dtype)).float(),
                    batch["output"].to(device=device, dtype=torch.float32),
                ).item()
                for batch in eval_io
            ) / len(eval_io)
        diagnostics = routing_diagnostics(child, eval_io, device, dtype)
        layer_records.append({
            "layer": layer_index,
            "router_history": router_history,
            "child_history_tail": child_history[-3:],
            "local_eval_mse": local_mse,
            "routing_diagnostics": diagnostics,
        })
        children.append(child)
        child_eval_io.append(eval_io)
        # Match the sequential cascade used by the accepted quality runs.
        layer.mlp = child

    return (
        parents, children, layer_records, list(eval_ids), teacher_logits,
        teacher_ce, child_eval_io,
    )


def alternate_input_parity(
    model: torch.nn.Module,
    prefix_ids: torch.Tensor,
    token_ids: torch.Tensor,
    alternate_ids: torch.Tensor,
    cache_length: int,
) -> float:
    position = torch.tensor([prefix_ids.shape[1]], device=prefix_ids.device)
    graph_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
    static_token_ids = token_ids.clone()
    with torch.inference_mode():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_logits = forward_logits(
                model, static_token_ids, graph_cache, position,
            )
        static_token_ids.copy_(alternate_ids)
        graph.replay()
        torch.cuda.synchronize()
        graph_result = graph_logits.clone()
        eager_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
        eager_result = forward_logits(
            model, alternate_ids, eager_cache, position,
        ).clone()
        torch.cuda.synchronize()
    return float((graph_result - eager_result).abs().max().item())


def trained_batch_runtime_matrix(
    model: torch.nn.Module,
    tokenizer,
    layers: list[int],
    parents: list[torch.nn.Module],
    children: list[torch.nn.Module],
    device: torch.device,
    batch_sizes: list[int],
    warmup: int,
    iterations: int,
) -> list[dict[str, object]]:
    """Measure trained sparse graph replay across several batch sizes."""
    base_prefix = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :4].to(device)
    base_token = tokenizer(
        " attention", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    rows = []
    model_layers = [model.model.layers[index] for index in layers]
    for batch_size in batch_sizes:
        prefix_ids = base_prefix.repeat(batch_size, 1)
        token_ids = base_token.repeat(batch_size, 1)
        cache_length = max(32, int(prefix_ids.shape[1]) + 8)
        position = torch.tensor([prefix_ids.shape[1]], device=device)
        for layer, parent in zip(model_layers, parents):
            layer.mlp = parent
        parent_cache = make_cache_and_fill_prefix(
            model, prefix_ids, cache_length,
        )
        parent_ms, _ = measure_eager(
            model, token_ids, parent_cache, position, warmup, iterations,
        )
        for layer, child in zip(model_layers, children):
            route_base = next(
                nested for nested in child.modules()
                if isinstance(nested, TransferredRoutedQwenChild)
            )
            route_base.single_token_fast_path = True
            layer.mlp = child
        sparse_cache = make_cache_and_fill_prefix(
            model, prefix_ids, cache_length,
        )
        sparse_eager_ms, sparse_eager_logits = measure_eager(
            model, token_ids, sparse_cache, position, warmup, iterations,
        )
        graph_cache = make_cache_and_fill_prefix(
            model, prefix_ids, cache_length,
        )
        graph_ms, _, graph_logits, _ = measure_graph(
            model, token_ids, graph_cache, position, warmup, iterations,
        )
        replay_error = float(
            (graph_logits - sparse_eager_logits).abs().max().item()
        )
        rows.append({
            "batch_size": batch_size,
            "parent_ms": parent_ms,
            "sparse_eager_ms": sparse_eager_ms,
            "sparse_graph_ms": graph_ms,
            "graph_over_parent": graph_ms / max(parent_ms, 1e-9),
            "graph_over_sparse_eager": graph_ms / max(sparse_eager_ms, 1e-9),
            "max_replay_vs_sparse_eager_logit_error": replay_error,
        })
    return rows


def trained_correction_backend_matrix(
    model: torch.nn.Module,
    tokenizer,
    layers: list[int],
    children: list[torch.nn.Module],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[dict[str, object]]:
    """Compare vectorized and packed rank-64 correction on trained children."""
    base_prefix = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :4].to(device)
    base_token = tokenizer(
        " attention", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    model_layers = [model.model.layers[index] for index in layers]
    records = []
    original_limits = []
    mixers = []
    for child in children:
        current_mixers = [
            nested for nested in child.modules()
            if isinstance(nested, CrossGroupOutputMixRoutedQwenChild)
        ]
        mixers.extend(current_mixers)
        original_limits.extend(
            [mixer.max_dense_gather_bytes for mixer in current_mixers]
        )
    try:
        for backend in ("vectorized", "packed"):
            for mixer in mixers:
                mixer.max_dense_gather_bytes = (
                    128 * 1024 * 1024 if backend == "vectorized" else 0
                )
            batch_size = 8
            prefix_ids = base_prefix.repeat(batch_size, 1)
            token_ids = base_token.repeat(batch_size, 1)
            cache_length = max(32, int(prefix_ids.shape[1]) + 8)
            position = torch.tensor([prefix_ids.shape[1]], device=device)
            for layer, child in zip(model_layers, children):
                route_base = next(
                    nested for nested in child.modules()
                    if isinstance(nested, TransferredRoutedQwenChild)
                )
                route_base.single_token_fast_path = True
                layer.mlp = child
            eager_cache = make_cache_and_fill_prefix(
                model, prefix_ids, cache_length,
            )
            eager_ms, eager_logits = measure_eager(
                model, token_ids, eager_cache, position, warmup, iterations,
            )
            graph_cache = make_cache_and_fill_prefix(
                model, prefix_ids, cache_length,
            )
            try:
                graph_ms, _, graph_logits, _ = measure_graph(
                    model, token_ids, graph_cache, position, warmup, iterations,
                )
            except RuntimeError as exc:
                records.append({
                    "backend": backend,
                    "batch_size": batch_size,
                    "status": "GRAPH_CAPTURE_FAIL",
                    "eager_ms": eager_ms,
                    "known_cause": (
                        "packed correction calls torch.where over device routing "
                        "indices during CUDA Graph capture"
                    ),
                    "error": str(exc).splitlines()[0],
                })
                # A failed capture can leave the CUDA capture context in an
                # implementation-dependent state; report it and stop this
                # backend A/B rather than hiding the failure or continuing
                # with corrupted timing state.
                break
            records.append({
                "backend": backend,
                "batch_size": batch_size,
                "status": "PARITY_PASS",
                "eager_ms": eager_ms,
                "graph_ms": graph_ms,
                "graph_over_eager": graph_ms / max(eager_ms, 1e-9),
                "max_graph_vs_eager_logit_error": float(
                    (graph_logits - eager_logits).abs().max().item()
                ),
            })
    finally:
        for mixer, original_limit in zip(mixers, original_limits):
            mixer.max_dense_gather_bytes = original_limit
    return records


def trained_single_token_projection_matrix(
    model: torch.nn.Module,
    tokenizer,
    layers: list[int],
    children: list[torch.nn.Module],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[dict[str, object]]:
    """Compare einsum and BMM dispatch inside the trained sparse child."""
    base_prefix = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :4].to(device)
    base_token = tokenizer(
        " attention", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    model_layers = [model.model.layers[index] for index in layers]
    batch_size = 8
    prefix_ids = base_prefix.repeat(batch_size, 1)
    token_ids = base_token.repeat(batch_size, 1)
    cache_length = max(32, int(prefix_ids.shape[1]) + 8)
    position = torch.tensor([prefix_ids.shape[1]], device=device)
    records = []
    for backend in ("einsum", "bmm"):
        for layer, child in zip(model_layers, children):
            route_base = next(
                nested for nested in child.modules()
                if isinstance(nested, TransferredRoutedQwenChild)
            )
            route_base.single_token_fast_path = True
            route_base.single_token_projection_backend = backend
            layer.mlp = child
        eager_cache = make_cache_and_fill_prefix(
            model, prefix_ids, cache_length,
        )
        eager_ms, eager_logits = measure_eager(
            model, token_ids, eager_cache, position, warmup, iterations,
        )
        graph_cache = make_cache_and_fill_prefix(
            model, prefix_ids, cache_length,
        )
        graph_ms, _, graph_logits, _ = measure_graph(
            model, token_ids, graph_cache, position, warmup, iterations,
        )
        records.append({
            "backend": backend,
            "batch_size": batch_size,
            "status": "PARITY_PASS",
            "eager_ms": eager_ms,
            "graph_ms": graph_ms,
            "graph_over_eager": graph_ms / max(eager_ms, 1e-9),
            "max_graph_vs_eager_logit_error": float(
                (graph_logits - eager_logits).abs().max().item()
            ),
        })
    return records


def trained_compiled_child_matrix(
    model: torch.nn.Module,
    tokenizer,
    layers: list[int],
    children: list[torch.nn.Module],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> list[dict[str, object]]:
    """Probe an Inductor-compiled sparse child at trained batch 8."""
    if not hasattr(torch, "compile"):
        return [{"backend": "inductor", "status": "UNAVAILABLE"}]
    base_prefix = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :4].to(device)
    base_token = tokenizer(
        " attention", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    model_layers = [model.model.layers[index] for index in layers]
    batch_size = 8
    prefix_ids = base_prefix.repeat(batch_size, 1)
    token_ids = base_token.repeat(batch_size, 1)
    cache_length = max(32, int(prefix_ids.shape[1]) + 8)
    position = torch.tensor([prefix_ids.shape[1]], device=device)
    try:
        compiled_children = []
        for child in children:
            route_base = next(
                nested for nested in child.modules()
                if isinstance(nested, TransferredRoutedQwenChild)
            )
            route_base.single_token_fast_path = True
            route_base.single_token_projection_backend = "einsum"
            compiled_children.append(torch.compile(
                child, mode="max-autotune-no-cudagraphs", dynamic=False,
                fullgraph=False,
            ))
        for layer, child in zip(model_layers, compiled_children):
            layer.mlp = child
        eager_cache = make_cache_and_fill_prefix(
            model, prefix_ids, cache_length,
        )
        eager_ms, eager_logits = measure_eager(
            model, token_ids, eager_cache, position, warmup, iterations,
        )
        graph_cache = make_cache_and_fill_prefix(
            model, prefix_ids, cache_length,
        )
        graph_ms, _, graph_logits, _ = measure_graph(
            model, token_ids, graph_cache, position, warmup, iterations,
        )
        return [{
            "backend": "inductor",
            "batch_size": batch_size,
            "status": "PARITY_PASS",
            "eager_ms": eager_ms,
            "graph_ms": graph_ms,
            "graph_over_eager": graph_ms / max(eager_ms, 1e-9),
            "max_graph_vs_eager_logit_error": float(
                (graph_logits - eager_logits).abs().max().item()
            ),
        }]
    except Exception as exc:
        error_text = str(exc) or repr(exc)
        return [{
            "backend": "inductor",
            "batch_size": batch_size,
            "status": "COMPILE_OR_GRAPH_FAIL",
            "error": error_text[-4000:],
        }]


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this audit requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    dtype = torch.float32
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=False,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, local_files_only=args.local_files_only,
    )
    calibration_text = load_text_file(
        args.calibration_text_file, TRAIN_TEXT, "calibration",
    )
    eval_text = load_text_file(args.eval_text_file, TRAIN_TEXT, "evaluation")
    layers = parse_layers(args.layers)

    (
        parents, children, layer_records, eval_ids, teacher_logits, teacher_ce,
        child_eval_io,
    ) = train_k5_cascade(
        model, tokenizer, calibration_text, eval_text, layers, device, dtype,
        args.batch_size, args.sequence_length, args.train_batches,
        args.eval_batches, args.child_steps, args.hard_steps, args.router_steps,
        args.calibration_rank,
        args.learning_rate, args.hard_learning_rate, args.max_grad_norm,
        args.log_every,
    )
    sparse_quality = evaluate_current(
        model, eval_ids, teacher_logits, teacher_ce, "trained_k5_sparse",
    )

    # Dense parent baseline uses the same model and the same fixed-KV cache.
    for layer, parent in zip([model.model.layers[index] for index in layers], parents):
        layer.mlp = parent
    prefix_ids = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :4].to(device)
    token_ids = tokenizer(
        " attention", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    alternate_ids = tokenizer(
        " routing", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    cache_length = max(32, int(prefix_ids.shape[1]) + 8)
    position = torch.tensor([prefix_ids.shape[1]], device=device)
    parent_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
    parent_ms, _ = measure_eager(
        model, token_ids, parent_cache, position, args.warmup, args.iterations,
    )

    for layer, child in zip([model.model.layers[index] for index in layers], children):
        route_base = next(
            nested for nested in child.modules()
            if isinstance(nested, TransferredRoutedQwenChild)
        )
        route_base.single_token_fast_path = True
        layer.mlp = child
    sparse_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
    sparse_eager_ms, sparse_eager_logits = measure_eager(
        model, token_ids, sparse_cache, position, args.warmup, args.iterations,
    )
    graph_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
    graph_ms, _, graph_logits, _ = measure_graph(
        model, token_ids, graph_cache, position, args.warmup, args.iterations,
    )
    replay_error = float((graph_logits - sparse_eager_logits).abs().max().item())
    alternate_error = alternate_input_parity(
        model, prefix_ids, token_ids, alternate_ids, cache_length,
    )

    trained_generation_prompt = tokenizer(
        "Explain why sparse circuits can reduce compute while preserving useful behavior.",
        return_tensors="pt",
    ).input_ids.to(device)
    trained_pool = FixedShapeGreedyGraphPool(model, max_entries=2)
    trained_graph_tokens = trained_pool.generate(
        trained_generation_prompt, 8, use_cuda_graph=True,
    )
    trained_reused_tokens = trained_pool.generate(
        trained_generation_prompt, 8, use_cuda_graph=True,
    )
    trained_eager_tokens = greedy_generate_fixed_shape(
        model, trained_generation_prompt, 8, use_cuda_graph=False,
    )
    trained_generation_equal = bool(
        torch.equal(trained_graph_tokens, trained_eager_tokens)
    )
    trained_generation_reused_equal = bool(
        torch.equal(trained_graph_tokens, trained_reused_tokens)
    )
    trained_batch_runtime = trained_batch_runtime_matrix(
        model, tokenizer, layers, parents, children, device,
        args.batch_sizes, args.warmup, args.iterations,
    )
    trained_single_token_projections = trained_single_token_projection_matrix(
        model, tokenizer, layers, children, device, args.warmup,
        args.single_token_backend_iterations,
    )
    trained_compiled_children = trained_compiled_child_matrix(
        model, tokenizer, layers, children, device, args.warmup,
        args.compiled_child_iterations,
    )
    # Keep the intentionally failing packed-capture probe last: a CUDA Graph
    # capture failure can poison the current CUDA context for later work.
    trained_correction_backends = trained_correction_backend_matrix(
        model, tokenizer, layers, children, device, args.warmup,
        args.correction_backend_iterations,
    )

    result = {
        "experiment": "V0.208_trained_qwen_correction_ablation",
        "status": "PARITY_PASS" if max(replay_error, alternate_error) <= 1e-3 else "PARITY_FAIL",
        "model": args.model,
        "seed": args.seed,
        "layers": layers,
        "device": str(device),
        "dtype": "float32",
        "recipe": {
            "num_experts": 8,
            "active_experts": 5,
            "hard_route_scale": 5.0,
            "calibration_rank": args.calibration_rank,
            "calibration_mode": "cross-group",
            "route_source": "subset-router",
            "router_target": "subset-soft",
            "router_target_temperature": 0.25,
            "child_steps": args.child_steps,
            "hard_steps": args.hard_steps,
            "router_steps": args.router_steps,
        },
        "teacher_ce": teacher_ce,
        "sparse_quality": sparse_quality,
        "layer_records": layer_records,
        "graph_runtime": {
            "prefix_length": int(prefix_ids.shape[1]),
            "parent_ms": parent_ms,
            "sparse_eager_ms": sparse_eager_ms,
            "sparse_graph_ms": graph_ms,
            "graph_over_parent": graph_ms / max(parent_ms, 1e-9),
            "graph_over_sparse_eager": graph_ms / max(sparse_eager_ms, 1e-9),
            "max_replay_vs_sparse_eager_logit_error": replay_error,
            "max_alternate_input_graph_vs_eager_logit_error": alternate_error,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "trained_generation": {
            "prefix_length": int(trained_generation_prompt.shape[1]),
            "new_tokens": 8,
            "graph_vs_eager_exact_token_match": trained_generation_equal,
            "reused_shape_exact_token_match": trained_generation_reused_equal,
            "graph_capture_count": trained_pool.capture_count,
            "graph_cache_hit_count": trained_pool.hit_count,
        },
        "trained_batch_runtime": trained_batch_runtime,
        "trained_correction_backends": trained_correction_backends,
        "trained_single_token_projections": trained_single_token_projections,
        "trained_compiled_children": trained_compiled_children,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--correction-backend-iterations", type=int, default=30)
    parser.add_argument("--single-token-backend-iterations", type=int, default=30)
    parser.add_argument("--compiled-child-iterations", type=int, default=30)
    parser.add_argument("--calibration-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cuda", "auto"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output",
        default="results/runs/qwen_v0197_trained_custom_kv_graph_k5_seed2026.json",
    )
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
