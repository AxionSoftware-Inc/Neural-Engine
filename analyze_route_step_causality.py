"""Measure how a one-step route change propagates through Native Engine state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.generator import SyntheticTaskGenerator
from train import make_model


def _route_source_indices(task_ids: torch.Tensor, mode: str) -> torch.Tensor:
    source = torch.arange(task_ids.numel(), device=task_ids.device)
    if mode == "global":
        return torch.roll(source, shifts=1)
    if mode != "within_task":
        raise ValueError("route swap mode must be global or within_task")
    for task_id in torch.unique(task_ids, sorted=True):
        indices = (task_ids == task_id).nonzero(as_tuple=False).squeeze(-1)
        source[indices] = torch.roll(indices, shifts=1)
    return source


def _checkpoint(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    return model, config


def _heldout_batch(config: dict[str, Any], examples_per_task: int,
                   device: torch.device):
    value_min = int(config.get("heldout_value_min", config.get("eval_value_min", 0)))
    value_max = int(config.get("heldout_value_max", config.get("eval_value_max", 63)))
    split = str(config.get("heldout_split", "all"))
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config["seed"]) + 3,
        value_min=value_min, value_max=value_max, split=split,
    )
    return generator.balanced_batch(examples_per_task, device)


@torch.inference_mode()
def analyze(path: Path, device: torch.device, examples_per_task: int) -> dict[str, Any]:
    model, config = _checkpoint(path, device)
    batch = _heldout_batch(config, examples_per_task, device)
    natural_logits, natural_stats = model(batch.inputs, adaptive=False)
    natural_loss = F.cross_entropy(natural_logits, batch.targets)
    natural_accuracy = natural_logits.argmax(dim=-1).eq(batch.targets).float().mean()
    selected = natural_stats["selected_ids"]
    weights = natural_stats["selected_weights"]
    gains = natural_stats["route_gains"]
    batch_size, steps, active = selected.shape
    result: dict[str, Any] = {
        "checkpoint": str(path),
        "seed": int(config["seed"]),
        "examples": int(batch_size),
        "internal_steps": int(steps),
        "active_circuits": int(active),
        "natural_loss": float(natural_loss),
        "natural_accuracy": float(natural_accuracy),
        "modes": {},
    }
    for mode in ("global", "within_task"):
        source = _route_source_indices(batch.task_ids, mode)
        mode_rows = []
        for changed_step in range(steps):
            forced_ids = torch.full_like(selected, -1)
            forced_weights = torch.zeros_like(weights)
            forced_gains = torch.ones_like(gains)
            forced_ids[:, changed_step] = selected[source, changed_step]
            forced_weights[:, changed_step] = weights[source, changed_step]
            forced_gains[:, changed_step] = gains[source, changed_step]
            swapped_logits, swapped_stats = model(
                batch.inputs, adaptive=False,
                forced_selected_ids=forced_ids,
                forced_selected_weights=forced_weights,
                forced_route_gains=forced_gains,
            )
            swapped_loss = F.cross_entropy(swapped_logits, batch.targets)
            swapped_accuracy = swapped_logits.argmax(dim=-1).eq(batch.targets).float().mean()
            step_logit_delta = (
                swapped_stats["step_logits"] - natural_stats["step_logits"]
            ).float().norm(dim=-1).mean(dim=0)
            query_delta = (
                swapped_stats["query_states"] - natural_stats["query_states"]
            ).float().norm(dim=-1).mean(dim=0)
            route_delta_delta = (
                swapped_stats["route_deltas"] - natural_stats["route_deltas"]
            ).float().norm(dim=-1).mean(dim=0)
            final_logit_delta = (swapped_logits - natural_logits).float().norm(dim=-1).mean()
            mode_rows.append({
                "changed_step": changed_step,
                "loss_change": float(swapped_loss - natural_loss),
                "accuracy_change": float(swapped_accuracy - natural_accuracy),
                "final_logit_l2": float(final_logit_delta),
                "step_logit_l2_by_stage": [float(value) for value in step_logit_delta],
                "query_l2_by_stage": [float(value) for value in query_delta],
                "route_delta_l2_by_stage": [float(value) for value in route_delta_delta],
            })
        result["modes"][mode] = mode_rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one-step route causality")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [analyze(Path(path), device, args.examples_per_task)
               for path in args.checkpoint]
    rendered = json.dumps({"experiment": "route_step_causality", "results": results}, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
