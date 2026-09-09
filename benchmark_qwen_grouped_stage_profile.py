"""Profile the major grouped selected-FFN stages on a trained Qwen child."""

from __future__ import annotations

import argparse
import json

import torch
from torch.profiler import ProfilerActivity, profile

from benchmark_qwen_custom_kv_graph import forward_logits, make_cache_and_fill_prefix
from benchmark_qwen_multi_layer_transplant import (
    TRAIN_TEXT,
    load_text_file,
    parse_layers,
)
from benchmark_qwen_trained_dispatch_path_audit import (
    install_children,
    set_dispatch_path,
)
from benchmark_qwen_trained_graph_audit import train_k5_cascade


def _event_time(event: object, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layers", default="19,20,21,22,23,24,25,26")
    parser.add_argument("--calibration-text-file", default="data/qwen_calibration.txt")
    parser.add_argument("--eval-text-file", default="data/qwen_eval.txt")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[4])
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--child-steps", type=int, default=600)
    parser.add_argument("--hard-steps", type=int, default=600)
    parser.add_argument("--router-steps", type=int, default=200)
    parser.add_argument("--calibration-rank", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profile-iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this profile requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        trust_remote_code=False,
        local_files_only=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=True,
    )
    calibration_text = load_text_file(
        args.calibration_text_file, TRAIN_TEXT, "calibration",
    )
    eval_text = load_text_file(args.eval_text_file, TRAIN_TEXT, "evaluation")
    layers = parse_layers(args.layers)
    (
        _parents, children, _records, _eval_ids, _teacher_logits, _teacher_ce,
        _,
    ) = train_k5_cascade(
        model,
        tokenizer,
        calibration_text,
        eval_text,
        layers,
        device,
        torch.float32,
        max(args.batch_sizes),
        args.sequence_length,
        args.train_batches,
        args.eval_batches,
        args.child_steps,
        args.hard_steps,
        args.router_steps,
        args.calibration_rank,
        3e-3,
        3e-4,
        1.0,
        100,
    )
    install_children(model, layers, children)
    set_dispatch_path(
        children,
        False,
        "grouped-adaptive",
        fused_correction=True,
        uniform_accum=True,
    )
    prefix_pool = tokenizer(
        "Neural Engine sparse circuits " * 256,
        return_tensors="pt",
    ).input_ids.to(device)
    token_ids = tokenizer(
        " attention", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    rows = []
    for prefix_length in args.prefix_lengths:
        prefix_ids = prefix_pool[:, :prefix_length]
        for batch_size in args.batch_sizes:
            batch_prefix = prefix_ids.repeat(batch_size, 1)
            batch_tokens = token_ids.repeat(batch_size, 1)
            position = torch.tensor([prefix_length], device=device)
            cache = make_cache_and_fill_prefix(
                model, batch_prefix, max(32, prefix_length + 8),
            )
            with torch.inference_mode():
                for _ in range(args.warmup):
                    forward_logits(model, batch_tokens, cache, position)
                torch.cuda.synchronize()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as prof:
                    for _ in range(args.profile_iterations):
                        forward_logits(model, batch_tokens, cache, position)
                    torch.cuda.synchronize()
            stages = []
            for event in prof.key_averages():
                if not event.key.startswith("neural_engine.grouped."):
                    continue
                stages.append({
                    "stage": event.key.rsplit(".", 1)[-1],
                    "calls": int(event.count),
                    "self_cuda_us": _event_time(
                        event, "self_device_time_total", "self_cuda_time_total",
                    ),
                    "total_cuda_us": _event_time(
                        event, "device_time_total", "cuda_time_total",
                    ),
                    "self_cpu_us": _event_time(
                        event, "self_cpu_time_total",
                    ),
                    "total_cpu_us": _event_time(
                        event, "cpu_time_total",
                    ),
                })
            rows.append({
                "prefix_length": prefix_length,
                "batch_size": batch_size,
                "profile_iterations": args.profile_iterations,
                "stages": stages,
            })
    result = {
        "experiment": "V0.240_grouped_stage_profile",
        "model": args.model,
        "seed": args.seed,
        "layers": layers,
        "dtype": "float32",
        "dispatch_policy": "grouped-adaptive",
        "rows": rows,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
