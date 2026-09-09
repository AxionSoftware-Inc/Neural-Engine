"""Inference-only task-conditioned correction-scale diagnostic for P-007.

This is deliberately an upper-bound style diagnostic, not a proposed default:
it chooses one correction scale per known synthetic task from a calibration
batch, then evaluates that policy on a different balanced batch.  It asks
whether the mixed per-example correction advantage is at least predictable at
the task level before investing in a learned gate.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data.generator import SyntheticTaskGenerator
from train import load_config, make_model, seed_everything


def _per_example_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, targets, reduction="none")


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(targets).float().mean().cpu()),
        "ce": float(_per_example_loss(logits, targets).mean().cpu()),
    }


@torch.no_grad()
def _forward_at_scale(
    model: nn.Module,
    inputs: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    original_scale = float(model.circuit_delta_scale)
    model.circuit_delta_scale = float(scale)
    try:
        logits, _stats = model(inputs, adaptive=False)
    finally:
        model.circuit_delta_scale = original_scale
    return logits


@torch.no_grad()
def _task_conditioned_eval(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    task_ids: torch.Tensor,
    selected_scales: dict[str, float],
) -> dict[str, float]:
    losses: list[torch.Tensor] = []
    correct = 0
    total = 0
    for task_id in torch.unique(task_ids, sorted=True).tolist():
        mask = task_ids.eq(task_id)
        logits = _forward_at_scale(model, inputs[mask], selected_scales[str(int(task_id))])
        losses.append(_per_example_loss(logits, targets[mask]))
        correct += int(logits.argmax(dim=-1).eq(targets[mask]).sum().cpu())
        total += int(mask.sum().cpu())
    return {
        "accuracy": correct / max(total, 1),
        "ce": float(torch.cat(losses).mean().cpu()),
    }


@torch.no_grad()
def analyze_checkpoint(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = load_config(args.config, smoke=False) if args.config else dict(payload["config"])
    if config["model"] == "baseline":
        raise ValueError("P-007 task-conditioned scale requires a Neural Engine checkpoint")
    seed_everything(int(config["seed"]))
    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device))
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload.get("model_state", payload))
    checkpoint_report = payload.get("report", {})
    if "routing_capacity" in checkpoint_report or "routing_depth" in checkpoint_report:
        model.router.set_routing_state(
            capacity=int(checkpoint_report.get("routing_capacity", model.router.routing_capacity)),
            depth=int(checkpoint_report.get("routing_depth", model.router.active_depth)),
        )

    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=args.calibration_seed,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    calibration = generator.balanced_batch(args.examples_per_task, device)
    eval_generator = SyntheticTaskGenerator(
        config["seq_len"], seed=args.eval_seed,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    evaluation = eval_generator.balanced_batch(args.examples_per_task, device)
    scales = [float(scale) for scale in args.scales]
    if not scales or any(scale < 0.0 for scale in scales):
        raise ValueError("--scales must contain non-negative values")

    calibration_choices: dict[str, dict[str, float]] = {}
    selected_scales: dict[str, float] = {}
    for task_id in torch.unique(calibration.task_ids, sorted=True).tolist():
        mask = calibration.task_ids.eq(task_id)
        by_scale: dict[str, float] = {}
        for scale in scales:
            logits = _forward_at_scale(model, calibration.inputs[mask], scale)
            by_scale[str(scale)] = _metrics(logits, calibration.targets[mask])["ce"]
        best_scale = min(scales, key=lambda scale: by_scale[str(scale)])
        calibration_choices[str(int(task_id))] = by_scale
        selected_scales[str(int(task_id))] = best_scale

    natural = _metrics(
        _forward_at_scale(model, evaluation.inputs, 1.0), evaluation.targets,
    )
    no_correction = _metrics(
        _forward_at_scale(model, evaluation.inputs, 0.0), evaluation.targets,
    )
    policy = _task_conditioned_eval(
        model, evaluation.inputs, evaluation.targets, evaluation.task_ids, selected_scales,
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "model": config["model"],
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "calibration_seed": args.calibration_seed,
        "eval_seed": args.eval_seed,
        "scales": scales,
        "calibration_choices_ce": calibration_choices,
        "selected_scales": selected_scales,
        "evaluation": {
            "natural_scale_1": natural,
            "no_correction_scale_0": no_correction,
            "task_conditioned_policy": policy,
            "policy_minus_natural_accuracy_pp": (policy["accuracy"] - natural["accuracy"]) * 100.0,
            "policy_minus_natural_ce": policy["ce"] - natural["ce"],
        },
    }
    del model, payload, calibration, evaluation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="P-007 task-conditioned scale diagnostic")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=1712)
    parser.add_argument("--eval-seed", type=int, default=2712)
    parser.add_argument("--scales", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = [analyze_checkpoint(args, Path(path)) for path in args.checkpoint]
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
