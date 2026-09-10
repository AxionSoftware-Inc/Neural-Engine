"""Sweep serving batch sizes for the native fused factorized dispatch."""

from __future__ import annotations

import argparse
import gc
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


def _timed(model: torch.nn.Module, inputs: torch.Tensor, warmup: int,
           repeats: int, device: torch.device) -> float:
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


def _active_width(stats: dict[str, torch.Tensor]) -> float | None:
    if "active_widths" not in stats:
        return None
    widths = stats["active_widths"].float()
    executed = stats["executed_mask"]
    active = widths[executed]
    return float(active.mean().cpu()) if active.numel() else None


def _build_variants(checkpoint: dict[str, Any], learned_checkpoint: dict[str, Any],
                    device: torch.device):
    base_config = dict(checkpoint["config"])
    fixed_config = dict(base_config)
    fixed_config["active_circuits"] = 16
    fixed_config["dynamic_width_mode"] = "none"
    learned_config = dict(learned_checkpoint["config"])
    fused_fixed_config = dict(fixed_config)
    fused_fixed_config["circuit_dispatch_backend"] = "native_cuda_fused"
    fused_learned_config = dict(learned_config)
    fused_learned_config["circuit_dispatch_backend"] = "native_cuda_fused"
    configs = {
        "fixed_k16": (fixed_config, checkpoint),
        "native_fused_fixed_k16": (fused_fixed_config, checkpoint),
        "learned_k8_k16": (learned_config, learned_checkpoint),
        "native_fused_learned": (fused_learned_config, learned_checkpoint),
    }
    models = {}
    for name, (config, source) in configs.items():
        model = make_model(config).to(device)
        model.load_state_dict(source["model_state"])
        model.eval()
        models[name] = model
    return models, configs


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    result: dict[str, Any] = {
        "experiment": "native_fused_shape_sweep",
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "examples_per_task_values": args.examples_per_task_values,
        "checkpoints": {},
    }
    for checkpoint_name in args.checkpoints:
        checkpoint_path = Path(checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        learned_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_learned_width.pt")
        learned_checkpoint = torch.load(learned_path, map_location="cpu", weights_only=True)
        models, configs = _build_variants(checkpoint, learned_checkpoint, device)
        seed = int(checkpoint["config"].get("seed", 17))
        checkpoint_result: dict[str, Any] = {"shapes": {}}
        for examples_per_task in args.examples_per_task_values:
            seed_everything(seed + 9000 + int(examples_per_task))
            generator = SyntheticTaskGenerator(
                seq_len=int(checkpoint["config"]["seq_len"]),
                seed=seed + 9000 + int(examples_per_task),
            )
            batch = generator.balanced_batch(examples_per_task, device)
            shape_result: dict[str, Any] = {
                "examples_per_task": int(examples_per_task),
                "batch_size": int(batch.inputs.shape[0]),
                "variants": {},
            }
            reference_logits: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for name in ("fixed_k16", "learned_k8_k16"):
                    reference_logits[name], _ = models[name](
                        batch.inputs, adaptive=False, collect_stats=False)
            for name, model in models.items():
                with torch.no_grad():
                    _, stats = model(batch.inputs, adaptive=False, collect_stats=True)
                timing = _timed(model, batch.inputs, args.warmup, args.repeats, device)
                variant: dict[str, Any] = {
                    "mean_ms": timing,
                    "samples_per_second": float(batch.inputs.shape[0] * 1000.0 / timing),
                    "active_width_mean": _active_width(stats),
                }
                if name == "native_fused_fixed_k16":
                    with torch.no_grad():
                        logits, _ = model(batch.inputs, adaptive=False, collect_stats=False)
                    variant["max_logit_error_vs_torch"] = float(
                        (logits - reference_logits["fixed_k16"]).abs().max().cpu())
                elif name == "native_fused_learned":
                    with torch.no_grad():
                        logits, _ = model(batch.inputs, adaptive=False, collect_stats=False)
                    variant["max_logit_error_vs_torch"] = float(
                        (logits - reference_logits["learned_k8_k16"]).abs().max().cpu())
                shape_result["variants"][name] = variant
            checkpoint_result["shapes"][str(batch.inputs.shape[0])] = shape_result
        result["checkpoints"][checkpoint_path.name] = checkpoint_result
        del models
        gc.collect()
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
    parser.add_argument("--examples-per-task-values", nargs="+", type=int,
                        default=[1, 8, 16, 32, 64])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/diagnostic_native_fused_shape_sweep_20260910.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
