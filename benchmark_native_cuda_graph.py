"""Smoke-test a static-shape CUDA Graph for Native Engine serving."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from data.generator import SyntheticTaskGenerator
from train import make_model, seed_everything


class StaticServingModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(inputs, adaptive=False, collect_stats=False)
        return logits


def timed(model: nn.Module, inputs: torch.Tensor, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(10):
            model(inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not hasattr(torch.cuda, "make_graphed_callables"):
        raise RuntimeError("CUDA Graph support is unavailable")

    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    seed_everything(int(config["seed"]))
    device = torch.device("cuda")
    eager = make_model(config).to(device).eval()
    eager.load_state_dict(payload["model_state"])
    wrapper = StaticServingModel(eager).eval()
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=int(config["seed"]) + 9,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    inputs = generator.task_balanced_batch(args.batch_size, device).inputs
    with torch.inference_mode():
        eager_logits = wrapper(inputs)
    eager_ms_before_capture = timed(wrapper, inputs, args.iterations)
    result = {
        "batch_size": args.batch_size,
        "adaptive": False,
        "eager_ms_before_capture": eager_ms_before_capture,
    }
    try:
        graph_model = torch.cuda.make_graphed_callables(
            wrapper, (inputs,), num_warmup_iters=5, allow_unused_input=True,
        )
        with torch.inference_mode():
            graph_logits = graph_model(inputs)
            torch.cuda.synchronize()
        max_error = (graph_logits - eager_logits).abs().max().item()
        eager_ms = timed(wrapper, inputs, args.iterations)
        graph_ms = timed(graph_model, inputs, args.iterations)
        result.update({
            "status": "ok",
            "max_logit_error": max_error,
            "eager_ms": eager_ms,
            "cuda_graph_ms": graph_ms,
            "speed_ratio_graph_over_eager": graph_ms / eager_ms,
        })
    except Exception as exc:
        result.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[-2000:],
        })
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
