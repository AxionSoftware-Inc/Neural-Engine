"""Measure Qwen selected-dispatch modes on one real decode-shaped token."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from benchmark_qwen_multi_layer_transplant import make_transferred_routed_qwen_child
from benchmark_qwen_parent_transplant import capture_mlp_io
from benchmark_qwen_two_layer_transplant import token_stream


def measure(module, hidden: torch.Tensor, warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            module(hidden)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            module(hidden)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32,
        trust_remote_code=False, local_files_only=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", local_files_only=True,
    )
    text = Path("data/qwen_eval.txt").read_text(encoding="utf-8")
    ids = token_stream(tokenizer, text, 1, 1, device).reshape(1, 1)
    parent = model.model.layers[args.layer].mlp
    hidden = capture_mlp_io(model, ids, args.layer)["input"]
    child = make_transferred_routed_qwen_child(
        parent, 8, 6, 1.0, 0, "base-output", "low-rank",
        "grouped", "contiguous", "router", 6.0,
        device, torch.float32,
    ).eval()
    child.single_token_fast_path = True
    child.dispatch_mode = "token-loop"
    token_loop_output = child(hidden)
    token_loop_ms = measure(child, hidden, args.warmup, args.iterations)
    child.dispatch_mode = "grouped"
    grouped_output = child(hidden)
    grouped_ms = measure(child, hidden, args.warmup, args.iterations)
    diff = (grouped_output - token_loop_output).abs()
    print({
        "shape": list(hidden.shape),
        "token_loop_ms": token_loop_ms,
        "grouped_single_token_ms": grouped_ms,
        "grouped_over_token_loop": grouped_ms / token_loop_ms,
        "max_grouped_vs_token_loop_error": diff.max().item(),
    })


if __name__ == "__main__":
    main()
