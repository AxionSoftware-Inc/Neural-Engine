"""Measure wall-clock cost of fixed and learned native active widths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from data.generator import SyntheticTaskGenerator
from train import make_model, seed_everything


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _timed(model: torch.nn.Module, inputs: torch.Tensor,
           warmup: int, repeats: int, device: torch.device) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(inputs, adaptive=False, collect_stats=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(inputs, adaptive=False, collect_stats=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    result: dict[str, Any] = {
        "experiment": "native_dynamic_width_runtime",
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "examples_per_task": args.examples_per_task,
        "checkpoints": {},
    }
    for checkpoint_name in args.checkpoints:
        checkpoint_path = Path(checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        base_config = dict(checkpoint["config"])
        configs = {}
        for name, width, mode in (("fixed_k8", 8, "none"), ("fixed_k16", 16, "none")):
            config = dict(base_config)
            config["active_circuits"] = width
            config["dynamic_width_mode"] = mode
            configs[name] = config
        learned_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_learned_width.pt")
        learned_checkpoint = torch.load(learned_path, map_location="cpu", weights_only=True)
        configs["learned_k8_k16"] = dict(learned_checkpoint["config"])
        if args.include_prefix_split:
            prefix_config = dict(learned_checkpoint["config"])
            prefix_config["dynamic_width_dispatch"] = "prefix_split"
            configs["learned_prefix_split"] = prefix_config
        models = {}
        for name, config in configs.items():
            model = make_model(config).to(device)
            source_checkpoint = (learned_checkpoint
                                 if name in {"learned_k8_k16", "learned_prefix_split"}
                                 else checkpoint)
            model.load_state_dict(
                source_checkpoint["model_state"])
            model.eval()
            models[name] = model
        seed = int(base_config.get("seed", 17))
        seed_everything(seed + 9000)
        generator = SyntheticTaskGenerator(seq_len=int(base_config["seq_len"]), seed=seed + 9000)
        batch = generator.balanced_batch(args.examples_per_task, device)
        checkpoint_result: dict[str, Any] = {"batch_size": int(batch.inputs.shape[0]), "variants": {}}
        for name, model in models.items():
            with torch.no_grad():
                _, stats = model(batch.inputs, adaptive=False, collect_stats=True)
            timing = _timed(model, batch.inputs, args.warmup, args.repeats, device)
            variant: dict[str, Any] = {
                "mean_ms": timing,
                "samples_per_second": float(batch.inputs.shape[0] * 1000.0 / timing),
            }
            if "active_widths" in stats:
                widths = stats["active_widths"].float()
                executed = stats["executed_mask"]
                active = widths[executed]
                variant["active_width_mean"] = float(active.mean().cpu())
                variant["active_width_fraction"] = float(active.mean().cpu() / 16.0)
            checkpoint_result["variants"][name] = variant
            del model
        result["checkpoints"][checkpoint_path.name] = checkpoint_result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--include-prefix-split", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/diagnostic_native_width_runtime_20260910.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
