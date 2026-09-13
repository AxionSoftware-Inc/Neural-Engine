"""Smoke-test torch.compile on the dynamic-register serving path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from data.dynamic_composition import DynamicCompositionGenerator
from train_dynamic_composition import make_model


class CompactServing(torch.nn.Module):
    """Return only factorized digit logits so no Cartesian class matrix is built."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        _, stats = self.model(
            inputs,
            collect_state_stats=False,
            return_full_logits=False,
        )
        return stats["digit_logits"]


def decode(digit_logits: tuple[torch.Tensor, ...], base: int) -> torch.Tensor:
    result = torch.zeros(
        digit_logits[0].shape[0], dtype=torch.long, device=digit_logits[0].device
    )
    for logits in digit_logits:
        result = result * base + logits[:, -1].argmax(dim=-1)
    return result


def max_difference(
    left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...]
) -> float:
    return max((a - b).abs().max().item() for a, b in zip(left, right))


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    output: dict[str, Any] = {
        "benchmark": "v0.340_dynamic_torch_compile_smoke",
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "circuit_mode": args.circuit_mode,
    }
    if not torch.cuda.is_available() or not hasattr(torch, "compile"):
        output.update({"status": "unavailable", "reason": "CUDA or torch.compile missing"})
        print(json.dumps(output, indent=2))
        return output

    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    if config.get("architecture") != "dynamic_register":
        raise ValueError("this smoke benchmark requires a dynamic-register checkpoint")
    model = make_model(config).cuda().eval()
    model.load_state_dict(payload["model_state"])
    model.circuit_mode = args.circuit_mode
    eager = CompactServing(model).eval()
    generator = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        split=str(config.get("eval_split", "all")),
        seed=int(config["seed"]) + 1813,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        modulus=None if config.get("modulus") is None else int(config["modulus"]),
        target_offset=int(config.get("target_offset", 0)),
    )
    batch = generator.task_balanced_batch(args.batch_size, torch.device("cuda"))
    eager_logits = eager(batch.inputs)
    eager_predictions = decode(eager_logits, int(model.output_digit_base))
    output["eager_accuracy"] = float(eager_predictions.eq(batch.targets).float().mean().cpu())

    compile_start = time.perf_counter()
    try:
        compiled = torch.compile(eager, mode="reduce-overhead", fullgraph=False)
        compiled_logits = compiled(batch.inputs)
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - depends on local compiler stack
        output.update({
            "status": "failed",
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "compile_seconds": time.perf_counter() - compile_start,
        })
        print(json.dumps(output, indent=2))
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        return output

    output["compile_seconds"] = time.perf_counter() - compile_start
    output["status"] = "ok"
    output["max_digit_logit_difference"] = max_difference(eager_logits, compiled_logits)
    output["prediction_agreement"] = float(
        eager_predictions.eq(decode(compiled_logits, int(model.output_digit_base))).float().mean().cpu()
    )
    for _ in range(args.warmup):
        eager(batch.inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        eager(batch.inputs)
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - start) * 1000.0 / args.iterations
    for _ in range(args.warmup):
        compiled(batch.inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        compiled(batch.inputs)
    torch.cuda.synchronize()
    compiled_ms = (time.perf_counter() - start) * 1000.0 / args.iterations
    output.update({
        "eager_ms": eager_ms,
        "compiled_ms": compiled_ms,
        "compiled_over_eager": compiled_ms / eager_ms,
    })
    encoded = json.dumps(output, indent=2)
    print(encoded)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic-register torch.compile smoke benchmark")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--circuit-mode", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--output", default="results/runs/v0_340_dynamic_torch_compile_smoke.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
