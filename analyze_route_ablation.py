from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data.generator import SyntheticTaskGenerator
from train import load_config, make_model, seed_everything


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(targets).float().mean().cpu())


def route_source_indices(task_ids: torch.Tensor, mode: str) -> torch.Tensor:
    source = torch.arange(task_ids.numel(), device=task_ids.device)
    if mode == "global":
        return torch.roll(source, shifts=1)
    if mode != "within_task":
        raise ValueError("route swap mode must be 'global' or 'within_task'")
    for task_id in torch.unique(task_ids, sorted=True):
        indices = (task_ids == task_id).nonzero(as_tuple=False).squeeze(-1)
        source[indices] = torch.roll(indices, shifts=1)
    return source


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = load_config(args.config, smoke=False) if args.config else dict(payload["config"])
    if config["model"] == "baseline":
        raise ValueError("route ablation requires a Neural Engine checkpoint")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload.get("model_state", payload))
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=args.seed,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    batch = generator.balanced_batch(args.examples_per_task, device)
    # Fixed execution is used so every sample has a complete route tensor. The
    # model is trained to execute this path even when adaptive inference is on.
    natural_logits, natural_stats = model(batch.inputs, adaptive=False)
    natural_loss = nn.functional.cross_entropy(natural_logits, batch.targets)
    fractions = (0.0, 0.25, 0.5, 1.0)
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model": config["model"],
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "batch_size": int(batch.inputs.shape[0]),
        "fixed_execution": True,
        "natural_accuracy": accuracy(natural_logits, batch.targets),
        "natural_loss": float(natural_loss.cpu()),
        "ablation": {},
    }
    for mode in ("global", "within_task"):
        source = route_source_indices(batch.task_ids, mode)
        conditions = {}
        for fraction in fractions:
            count = int(round(fraction * batch.inputs.shape[0]))
            swap_mask = torch.zeros(batch.inputs.shape[0], dtype=torch.bool, device=device)
            if count:
                # A cyclic source is a derangement, so a swapped item never
                # receives its own route in either ablation mode.
                spread_indices = torch.arange(count, device=device) * batch.inputs.shape[0] // count
                swap_mask[spread_indices] = True
            forced_ids = natural_stats["selected_ids"].clone()
            forced_weights = natural_stats["selected_weights"].clone()
            forced_gains = natural_stats["route_gains"].clone()
            forced_ids[swap_mask] = natural_stats["selected_ids"][source[swap_mask]]
            forced_weights[swap_mask] = natural_stats["selected_weights"][source[swap_mask]]
            forced_gains[swap_mask] = natural_stats["route_gains"][source[swap_mask]]
            swapped_logits, _ = model(
                batch.inputs, adaptive=False,
                forced_selected_ids=forced_ids,
                forced_selected_weights=forced_weights,
                forced_route_gains=forced_gains,
            )
            swapped_loss = nn.functional.cross_entropy(swapped_logits, batch.targets)
            conditions[f"{fraction:.2f}"] = {
                "swapped_examples": count,
                "accuracy": accuracy(swapped_logits, batch.targets),
                "accuracy_drop": result["natural_accuracy"] - accuracy(swapped_logits, batch.targets),
                "loss": float(swapped_loss.cpu()),
                "loss_increase": float((swapped_loss - natural_loss).cpu()),
            }
        result["ablation"][mode] = conditions
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure quality dependence on selected circuit routes")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1711)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rendered = json.dumps(analyze(args), indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
