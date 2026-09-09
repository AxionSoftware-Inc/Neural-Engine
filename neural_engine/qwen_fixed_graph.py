"""Small fixed-shape greedy decoder for the custom Qwen KV cache."""

from __future__ import annotations

import torch

from benchmark_qwen_custom_kv_graph import (
    FixedDecodeCache,
    forward_logits,
)


@torch.inference_mode()
def greedy_generate_fixed_shape(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    use_cuda_graph: bool = True,
) -> torch.Tensor:
    """Greedily generate with fixed-shape one-token graph replay.

    The graph path is intentionally explicit: callers choose it only for a
    shape that has been validated.  ``use_cuda_graph=False`` uses the same
    custom cache and positions but executes eager forwards, which is the safe
    fallback for uncaptured or unsupported shapes.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return input_ids.clone()
    if input_ids.shape[1] < 1:
        raise ValueError("input_ids must contain at least one prefix token")
    if use_cuda_graph and input_ids.device.type != "cuda":
        raise ValueError("CUDA Graph generation requires CUDA input_ids")

    device = input_ids.device
    prefix_length = int(input_ids.shape[1])
    cache_length = prefix_length + max_new_tokens + 8
    cache = FixedDecodeCache(model.config, cache_length)
    prefix_position = torch.arange(prefix_length, device=device)
    prefix_logits = forward_logits(
        model, input_ids, cache, prefix_position,
    )
    next_token = prefix_logits[:, -1:, :].argmax(dim=-1)
    generated = torch.cat((input_ids, next_token), dim=1)
    if max_new_tokens == 1:
        return generated

    position = torch.tensor([prefix_length], device=device)
    cache.set_decode_position(position)
    static_token = next_token.clone()

    if not use_cuda_graph:
        for step in range(1, max_new_tokens):
            logits = forward_logits(model, static_token, cache, position)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat((generated, next_token), dim=1)
            if step + 1 < max_new_tokens:
                static_token.copy_(next_token)
                position.copy_(
                    torch.tensor([prefix_length + step], device=device),
                )
                cache.set_decode_position(position)
        return generated

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_logits = forward_logits(model, static_token, cache, position)
    for step in range(1, max_new_tokens):
        graph.replay()
        torch.cuda.synchronize()
        next_token = graph_logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat((generated, next_token), dim=1)
        if step + 1 < max_new_tokens:
            static_token.copy_(next_token)
            position.copy_(
                torch.tensor([prefix_length + step], device=device),
            )
    return generated
