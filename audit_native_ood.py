"""Evaluate matched native checkpoints on harder distribution probes.

The native generator has a fixed value vocabulary [0, 63], so this audit does
not invent an invalid [64, 127] OOD range.  It measures combination holdout
and edge-value probes while keeping the task and model semantics unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.generator import SyntheticTaskGenerator
from train import BatchSource, evaluate, make_model, seed_everything


DEFAULT_CHECKPOINTS = (
    "results/checkpoints/ne300_shared_routekeys_ordered_s17_10000.pt",
    "results/checkpoints/ne300_shared_routekeys_ordered_s18_10000.pt",
    "results/checkpoints/ne500_shared_routekeys_ordered_s17_10000.pt",
    "results/checkpoints/ne500_shared_routekeys_ordered_s18_10000.pt",
)

HARD_TASKS = {
    "reverse_sum",
    "lookup",
    "chain3",
    "compose_add_mul",
    "compose_if",
    "state_machine",
}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    task_accuracy = metrics.get("task_accuracy", {})
    hard_values = [float(task_accuracy[name]) for name in HARD_TASKS if name in task_accuracy]
    depth_accuracy = metrics.get("depth_accuracy", {})
    deep_values = [float(depth_accuracy[str(depth)]) for depth in (2, 3) if str(depth) in depth_accuracy]
    return {
        "val_loss": float(metrics["val_loss"]),
        "exact_accuracy": float(metrics["exact_accuracy"]),
        "task_accuracy": task_accuracy,
        "depth_accuracy": depth_accuracy,
        "hard_task_mean_accuracy": sum(hard_values) / len(hard_values) if hard_values else None,
        "depth_2_3_mean_accuracy": sum(deep_values) / len(deep_values) if deep_values else None,
        "circuits_used": int(metrics.get("circuits_used", 0)),
        "dead_circuit_fraction": float(metrics.get("dead_circuit_fraction", 0.0)),
        "factor_rows_used": int(metrics.get("factor_rows_used", 0)),
        "factor_dead_fraction": float(metrics.get("factor_dead_fraction", 0.0)),
        "routing_entropy": float(metrics.get("routing_entropy", 0.0)),
        "active_width_mean": float(metrics.get("active_width_mean", 0.0)),
        "active_width_fraction": float(metrics.get("active_width_fraction", 0.0)),
        "wide_width_fraction": float(metrics.get("wide_width_fraction", 0.0)),
    }


def _evaluate_condition(model: torch.nn.Module, config: dict[str, Any], device: torch.device,
                        *, seed: int, value_min: int, value_max: int, split: str,
                        batches: int) -> dict[str, Any]:
    generator = SyntheticTaskGenerator(
        seq_len=int(config["seq_len"]), seed=seed,
        value_min=value_min, value_max=value_max, split=split,
    )
    source = BatchSource(generator, 256, device)
    return _summarize(evaluate(model, source, batches=batches))


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    checkpoints = [Path(path) for path in args.checkpoints]
    conditions = {
        "uniform_all": {"value_min": 0, "value_max": 63, "split": "all"},
        "combination_heldout": {"value_min": 0, "value_max": 63, "split": "heldout"},
        "low_edge_values": {"value_min": 0, "value_max": 7, "split": "all"},
        "high_edge_values": {"value_min": 56, "value_max": 63, "split": "all"},
    }
    output: dict[str, Any] = {
        "audit": "native_ood_distribution_probes",
        "device": str(device),
        "batches": args.batches,
        "examples_per_task_per_condition": args.batches * 32,
        "value_vocabulary": [0, 63],
        "conditions": conditions,
        "hard_tasks": sorted(HARD_TASKS),
        "checkpoints": {},
    }

    for checkpoint_path in checkpoints:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = dict(checkpoint["config"])
        if args.dynamic_width_mode is not None:
            config["dynamic_width_mode"] = args.dynamic_width_mode
        if args.dynamic_width_min is not None:
            config["dynamic_width_min"] = args.dynamic_width_min
        if args.dynamic_width_threshold is not None:
            config["dynamic_width_threshold"] = args.dynamic_width_threshold
        model = make_model(config).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        seed = int(config.get("seed", 17))
        checkpoint_result: dict[str, Any] = {
            "model_name": config["model"],
            "seed": seed,
            "total_params": sum(parameter.numel() for parameter in model.parameters()),
            "conditions": {},
        }
        for offset, (name, condition) in enumerate(conditions.items()):
            seed_everything(seed + 1000 + offset)
            checkpoint_result["conditions"][name] = _evaluate_condition(
                model, config, device, seed=seed + 3000 + offset,
                batches=args.batches, **condition,
            )
        output["checkpoints"][checkpoint_path.name] = checkpoint_result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dynamic-width-mode", choices=("none", "topk_entropy", "learned"), default=None)
    parser.add_argument("--dynamic-width-min", type=int, default=None)
    parser.add_argument("--dynamic-width-threshold", type=float, default=None)
    parser.add_argument("--output", default="results/diagnostic_native_ood_300m_500m_10000.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
