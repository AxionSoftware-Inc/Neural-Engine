"""Profile the CUDA stages of one selected-group Qwen dispatch backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmark_qwen_multi_layer_transplant import make_transferred_routed_qwen_child
from benchmark_qwen_parent_transplant import capture_mlp_io
from benchmark_qwen_two_layer_transplant import token_stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dispatch-mode",
        choices=("parent", "grouped", "packed", "packed-fused", "packed-fp16"),
        default="grouped",
    )
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this profile requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=torch.float32,
        trust_remote_code=False,
        local_files_only=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", local_files_only=True,
    )
    text = Path("data/qwen_eval.txt").read_text(encoding="utf-8")
    ids = token_stream(
        tokenizer, text, args.batch_size, args.sequence_length, device,
    ).reshape(args.batch_size, args.sequence_length)
    layer = model.model.layers[args.layer]
    parent = layer.mlp
    hidden = capture_mlp_io(model, ids, args.layer)["input"]
    if args.dispatch_mode != "parent":
        child = make_transferred_routed_qwen_child(
            parent, 8, 6, 1.0, 0, "base-output", "low-rank",
            args.dispatch_mode, "contiguous", "router", 6.0,
            device, torch.float32,
        )
        target = child.eval()
    else:
        target = parent
    with torch.inference_mode():
        for _ in range(2):
            target(hidden)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
        ) as profile:
            target(hidden)
            torch.cuda.synchronize()
    print(f"dispatch_mode={args.dispatch_mode}")
    print(profile.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=45,
    ))


if __name__ == "__main__":
    main()
