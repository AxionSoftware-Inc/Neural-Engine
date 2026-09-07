"""Profile a full Qwen one-token sparse forward after dispatch selection."""

from __future__ import annotations

import argparse

import torch

from benchmark_qwen_multi_layer_transplant import make_transferred_routed_qwen_child
from benchmark_qwen_two_layer_transplant import token_stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,26")
    parser.add_argument("--row-limit", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this profile requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32,
        trust_remote_code=False, local_files_only=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", local_files_only=True,
    )
    ids = token_stream(tokenizer, "Neural Engine", 1, 1, device).reshape(1, 1)
    layer_indices = [int(value) for value in args.layers.split(",") if value.strip()]
    for index in layer_indices:
        layer = model.model.layers[index]
        child = make_transferred_routed_qwen_child(
            layer.mlp, 8, 6, 1.0, 0, "base-output", "low-rank",
            "grouped", "contiguous", "router", 6.0,
            device, torch.float32,
        ).eval()
        child.single_token_fast_path = True
        layer.mlp = child
    with torch.inference_mode():
        for _ in range(10):
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
        ) as profile:
            model(input_ids=ids, use_cache=False)
            torch.cuda.synchronize()
    print(f"layers={layer_indices}")
    print(profile.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=args.row_limit,
    ))


if __name__ == "__main__":
    main()
