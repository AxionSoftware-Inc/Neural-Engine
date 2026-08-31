from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from data.generator import SyntheticTaskGenerator
from data.tasks import TASKS
from neural_engine.model import NeuralEngineV0
from train import load_config, make_model, seed_everything


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def build_task_batch(generator: SyntheticTaskGenerator, examples_per_task: int,
                     device: torch.device):
    rows = []
    for task in TASKS:
        for _ in range(examples_per_task):
            tokens, target, stage_targets, stage_mask = generator._one(task)
            rows.append((tokens, target, task.task_id, task.depth, stage_targets, stage_mask))
    return generator._make_batch(rows, device)


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    if args.config:
        config = load_config(args.config, smoke=False)
    else:
        config = dict(payload["config"])
    if config["model"] == "baseline":
        raise ValueError("route analysis requires a Neural Engine checkpoint")
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
    batch = build_task_batch(generator, args.examples_per_task, device)
    _, stats = model(batch.inputs, adaptive=model.adaptive_inference)
    selected = stats["selected_ids"].detach().cpu()
    task_ids = batch.task_ids.detach().cpu()
    route_sets: list[set[int]] = []
    for sample in selected:
        route_sets.append(set(int(value) for value in sample.flatten().tolist() if int(value) >= 0))

    per_task: dict[int, list[set[int]]] = {task.task_id: [] for task in TASKS}
    for task_id, route_set in zip(task_ids.tolist(), route_sets):
        per_task[int(task_id)].append(route_set)
    task_unions = {task_id: set().union(*sets) for task_id, sets in per_task.items()}
    within_values = []
    task_rows = []
    for task in TASKS:
        sets = per_task[task.task_id]
        pair_values = [jaccard(left, right) for left, right in combinations(sets, 2)]
        within = sum(pair_values) / len(pair_values) if pair_values else 1.0
        task_rows.append({
            "task": task.name,
            "depth": task.depth,
            "samples": len(sets),
            "mean_unique_active_circuits": sum(map(len, sets)) / max(len(sets), 1),
            "union_circuits": len(task_unions[task.task_id]),
            "within_task_jaccard": within,
        })
        within_values.extend(pair_values)
    between_values = [jaccard(task_unions[left], task_unions[right])
                      for left, right in combinations(task_unions, 2)]
    union_pair_values = [
        (jaccard(task_unions[left], task_unions[right]), left, right)
        for left, right in combinations(task_unions, 2)
    ]
    most_overlapping = max(union_pair_values) if union_pair_values else (0.0, 0, 0)
    least_overlapping = min(union_pair_values) if union_pair_values else (0.0, 0, 0)

    flat_selected = selected[selected.ge(0)]
    counts = torch.bincount(flat_selected, minlength=model.router.num_circuits).float()
    probabilities = counts / counts.sum().clamp_min(1)
    example_hot = torch.zeros(model.router.num_circuits)
    for route_set in route_sets:
        if route_set:
            example_hot[list(route_set)] += 1
    example_hot /= len(route_sets)
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model": config["model"],
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "split": str(config.get("eval_split", "all")),
        "avg_executed_steps": float(stats["executed_steps"].float().mean().cpu()),
        "active_circuits_per_step": model.active_circuits,
        "circuit_bank_size": model.router.num_circuits,
        "circuits_used": int((counts > 0).sum()),
        "dead_circuit_fraction": float((counts == 0).float().mean()),
        "always_hot_fraction_ge_50pct_examples": float((example_hot >= 0.5).float().mean()),
        "routing_entropy": float(-(probabilities[probabilities > 0]
                                    * probabilities[probabilities > 0].log()).sum()),
        "routing_max_load_fraction": float(probabilities.max()),
        "within_task_jaccard_mean": sum(within_values) / max(len(within_values), 1),
        "between_task_union_jaccard_mean": sum(between_values) / max(len(between_values), 1),
        "most_overlapping_task_pair": [TASKS[most_overlapping[1]].name, TASKS[most_overlapping[2]].name]
        if union_pair_values else [],
        "most_overlapping_task_pair_jaccard": most_overlapping[0],
        "least_overlapping_task_pair": [TASKS[least_overlapping[1]].name, TASKS[least_overlapping[2]].name]
        if union_pair_values else [],
        "least_overlapping_task_pair_jaccard": least_overlapping[0],
        "per_task": task_rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Neural Engine route stability and task overlap")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = analyze(args)
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
