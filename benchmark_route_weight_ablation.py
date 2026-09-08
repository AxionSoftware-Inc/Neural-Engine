"""Diagnose whether Native route quality is limited by selected-route weights.

The checkpoint, router decisions, selected circuit IDs, and route gains stay
fixed.  Only the mixture weights assigned to the already selected K circuits
are changed and replayed through the frozen model.  This separates a selector
problem from a weighting problem without training a new model or increasing
the active circuit budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_candidate_pool_screen import _load_checkpoint, _make_batches
from neural_engine.model import NeuralEngineV0


def _weight_variants(natural: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return fixed alternatives with the same shape and unit row mass."""
    if natural.ndim < 1 or natural.shape[-1] < 1:
        raise ValueError("natural route weights must have a non-empty last dimension")
    uniform = torch.full_like(natural, 1.0 / natural.shape[-1])
    # Natural weights are a softmax over the selected key scores.  Power
    # transforms probe whether the router is too flat or too concentrated,
    # while keeping the selected IDs and active K exactly unchanged.
    flattened = natural.clamp_min(1e-8)
    sharp = flattened.square()
    sharp = sharp / sharp.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    flat = flattened.sqrt()
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    top1 = torch.zeros_like(natural)
    top1.scatter_(-1, natural.argmax(dim=-1, keepdim=True), 1.0)
    return {
        "uniform": uniform,
        "softmax_power2": sharp,
        "softmax_power_half": flat,
        "top1": top1,
    }


@torch.inference_mode()
def _evaluate(model: NeuralEngineV0,
              batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    totals = {
        "examples": 0,
        "natural_ce": 0.0,
        "natural_correct": 0,
        "variants": {},
    }
    for inputs, targets in batches:
        natural_logits, stats = model(inputs, adaptive=False, collect_stats=True)
        natural_weights = stats["selected_weights"]
        variants = _weight_variants(natural_weights)
        totals["examples"] += int(targets.numel())
        totals["natural_ce"] += float(F.cross_entropy(
            natural_logits, targets, reduction="sum",
        ).cpu())
        totals["natural_correct"] += int(
            natural_logits.argmax(dim=-1).eq(targets).sum().cpu()
        )
        for name, weights in variants.items():
            logits, _ = model(
                inputs,
                adaptive=False,
                forced_selected_ids=stats["selected_ids"],
                forced_selected_weights=weights,
                forced_route_gains=stats["route_gains"],
                collect_stats=False,
            )
            slot = totals["variants"].setdefault(
                name, {"ce": 0.0, "correct": 0},
            )
            slot["ce"] += float(F.cross_entropy(
                logits, targets, reduction="sum",
            ).cpu())
            slot["correct"] += int(logits.argmax(dim=-1).eq(targets).sum().cpu())

    examples = totals["examples"]
    natural_ce = totals["natural_ce"] / examples
    natural_accuracy = totals["natural_correct"] / examples
    rows = []
    for name, values in totals["variants"].items():
        ce = values["ce"] / examples
        accuracy = values["correct"] / examples
        rows.append({
            "variant": name,
            "mean_ce": ce,
            "accuracy": accuracy,
            "delta_ce": ce - natural_ce,
            "delta_accuracy_pp": (accuracy - natural_accuracy) * 100.0,
        })
    return {
        "examples": examples,
        "natural_ce": natural_ce,
        "natural_accuracy": natural_accuracy,
        "variants": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Native route weight alternatives")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    result = {
        "device": str(device),
        "batches": int(args.batches),
        "examples_per_task": int(args.examples_per_task),
        "checkpoints": [],
    }
    for checkpoint_name in args.checkpoint:
        path = Path(checkpoint_name)
        model, config = _load_checkpoint(path, device)
        batches = _make_batches(config, device, args.batches, args.examples_per_task)
        evaluation = _evaluate(model, batches)
        result["checkpoints"].append({
            "checkpoint": str(path),
            "model": config.get("model"),
            "seed": int(config.get("seed", -1)),
            "active_circuits": int(model.active_circuits),
            "internal_steps": int(model.internal_steps),
            "evaluation": evaluation,
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
