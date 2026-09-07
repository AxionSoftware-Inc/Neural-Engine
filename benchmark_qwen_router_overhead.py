"""Measure the router/controller contribution in a full Qwen one-token smoke."""

from __future__ import annotations

import argparse
import time

import torch

from benchmark_qwen_multi_layer_transplant import make_transferred_routed_qwen_child
from benchmark_qwen_two_layer_transplant import token_stream


class StaticZeroRouter(torch.nn.Module):
    """Return the same zero logits as the fresh router, without MLP launches."""

    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.num_experts = int(num_experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            *hidden_states.shape[:-1], self.num_experts,
            device=hidden_states.device, dtype=hidden_states.dtype,
        )


def forward_logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids, use_cache=False).logits


def measure(
    model: torch.nn.Module,
    ids: torch.Tensor,
    warmup: int,
    iterations: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            forward_logits(model, ids)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            forward_logits(model, ids)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def install_children(
    model: torch.nn.Module,
    parents: list[torch.nn.Module],
    layer_indices: list[int],
    device: torch.device,
    *,
    static_router: bool,
) -> list[torch.nn.Module]:
    children = []
    for index, parent in zip(layer_indices, parents):
        child = make_transferred_routed_qwen_child(
            parent, 8, 6, 1.0, 0, "base-output", "low-rank",
            "grouped", "contiguous", "router", 6.0,
            device, torch.float32,
        ).eval()
        child.single_token_fast_path = True
        if static_router:
            child.router = StaticZeroRouter(child.num_experts)
        model.model.layers[index].mlp = child
        children.append(child)
    return children


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,26")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
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
    ids = token_stream(tokenizer, "Neural Engine", 1, 1, device).reshape(1, 1)
    layer_indices = [int(value) for value in args.layers.split(",") if value.strip()]
    parents = [model.model.layers[index].mlp for index in layer_indices]

    parent_ms = measure(model, ids, args.warmup, args.iterations)
    install_children(
        model, parents, layer_indices, device, static_router=False,
    )
    regular_logits = forward_logits(model, ids)
    regular_ms = measure(model, ids, args.warmup, args.iterations)
    install_children(
        model, parents, layer_indices, device, static_router=True,
    )
    static_logits = forward_logits(model, ids)
    static_ms = measure(model, ids, args.warmup, args.iterations)
    diff = (static_logits - regular_logits).abs()
    print({
        "layers": layer_indices,
        "parent_ms": parent_ms,
        "regular_router_ms": regular_ms,
        "static_router_ms": static_ms,
        "router_mlp_ms_estimate": regular_ms - static_ms,
        "regular_over_static": regular_ms / static_ms,
        "static_over_parent": static_ms / parent_ms,
        "max_static_vs_regular_logit_error": diff.max().item(),
        "mean_static_vs_regular_logit_error": diff.mean().item(),
    })


if __name__ == "__main__":
    main()
