"""Inference-only correction-gain and route-sensitivity diagnostic for P-007.

The checkpoint, router, circuit bank, and data batch stay fixed.  Only the
inference-time circuit correction multiplier is changed.  For each multiplier
the natural route is replayed after a global and a within-task route shift.
This is a diagnostic, not a training recipe or an adoption benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from data.generator import SyntheticTaskGenerator
from train import load_config, make_model, seed_everything


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(targets).float().mean().cpu())


def _loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, targets).cpu())


def _shift_route(values: torch.Tensor, task_ids: torch.Tensor, *, within_task: bool) -> torch.Tensor:
    if not within_task:
        return torch.roll(values, shifts=1, dims=0)
    shifted = values.clone()
    for task_id in torch.unique(task_ids):
        indices = torch.where(task_ids.eq(task_id))[0]
        if indices.numel() > 1:
            shifted[indices] = values[indices.roll(1)]
    return shifted


@torch.no_grad()
def _evaluate_scale(model, batch, scale: float) -> dict[str, Any]:
    model.circuit_delta_scale = float(scale)
    natural_logits, natural_stats = model(batch.inputs, adaptive=False)
    global_ids = _shift_route(natural_stats["selected_ids"], batch.task_ids, within_task=False)
    global_weights = _shift_route(natural_stats["selected_weights"], batch.task_ids, within_task=False)
    global_gains = _shift_route(natural_stats["route_gains"], batch.task_ids, within_task=False)
    global_logits, _ = model(
        batch.inputs,
        adaptive=False,
        forced_selected_ids=global_ids,
        forced_selected_weights=global_weights,
        forced_route_gains=global_gains,
    )
    within_ids = _shift_route(natural_stats["selected_ids"], batch.task_ids, within_task=True)
    within_weights = _shift_route(natural_stats["selected_weights"], batch.task_ids, within_task=True)
    within_gains = _shift_route(natural_stats["route_gains"], batch.task_ids, within_task=True)
    within_logits, _ = model(
        batch.inputs,
        adaptive=False,
        forced_selected_ids=within_ids,
        forced_selected_weights=within_weights,
        forced_route_gains=within_gains,
    )
    natural_loss = _loss(natural_logits, batch.targets)
    global_loss = _loss(global_logits, batch.targets)
    within_loss = _loss(within_logits, batch.targets)
    natural_accuracy = _accuracy(natural_logits, batch.targets)
    global_accuracy = _accuracy(global_logits, batch.targets)
    within_accuracy = _accuracy(within_logits, batch.targets)
    return {
        "scale": float(scale),
        "natural": {
            "accuracy": natural_accuracy,
            "ce": natural_loss,
        },
        "global_route_replay": {
            "accuracy": global_accuracy,
            "ce": global_loss,
            "accuracy_delta_pp": (global_accuracy - natural_accuracy) * 100.0,
            "ce_increase": global_loss - natural_loss,
        },
        "within_task_route_replay": {
            "accuracy": within_accuracy,
            "ce": within_loss,
            "accuracy_delta_pp": (within_accuracy - natural_accuracy) * 100.0,
            "ce_increase": within_loss - natural_loss,
        },
    }


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = load_config(args.config, smoke=False) if args.config else dict(payload["config"])
    if config["model"] == "baseline":
        raise ValueError("P-007 correction-gain sweep requires a Neural Engine checkpoint")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = make_model(config).to(device).eval()
    load_result = model.load_state_dict(payload.get("model_state", payload), strict=False)
    optional_route_keys = {
        name for name in model.state_dict()
        if name == "route_context_scale" or name.startswith("route_value_encoder.")
    }
    unexpected = set(load_result.unexpected_keys)
    missing = set(load_result.missing_keys) - optional_route_keys
    if unexpected or missing:
        raise RuntimeError(
            f"checkpoint/model mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]),
        seed=args.seed,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    batch = generator.balanced_batch(args.examples_per_task, device)
    original_scale = float(model.circuit_delta_scale)
    rows = [_evaluate_scale(model, batch, scale) for scale in args.scales]
    model.circuit_delta_scale = original_scale
    return {
        "checkpoint": str(checkpoint_path),
        "seed": int(config["seed"]),
        "device": str(device),
        "examples_per_task": int(args.examples_per_task),
        "batch_examples": int(batch.targets.numel()),
        "task_count": int(torch.unique(batch.task_ids).numel()),
        "checkpoint_circuit_delta_scale": original_scale,
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="P-007 inference-only correction-gain sweep")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1712)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = [evaluate_checkpoint(args, Path(path)) for path in args.checkpoint]
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
