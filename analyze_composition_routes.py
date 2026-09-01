from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from data.composition import OPERATIONS
from evaluate_composition import grid_rows
from train import make_model, seed_everything


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def summarize_group(route_sets: list[set[int]]) -> dict[str, float | int]:
    pair_values = [jaccard(left, right) for left, right in combinations(route_sets, 2)]
    union = set().union(*route_sets) if route_sets else set()
    return {
        "examples": len(route_sets),
        "mean_active_circuits": sum(map(len, route_sets)) / max(len(route_sets), 1),
        "union_circuits": len(union),
        "within_route_jaccard": sum(pair_values) / max(len(pair_values), 1),
    }


@torch.no_grad()
def analyze(checkpoint: str, grid_size: int, batch_size: int,
            device_name: str) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    seed_everything(int(config["seed"]))
    device = torch.device(device_name)
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    inputs, _, labels = grid_rows(config, grid_size)
    selected_parts: list[torch.Tensor] = []
    for start in range(0, inputs.shape[0], batch_size):
        _, stats = model(inputs[start:start + batch_size].to(device), adaptive=False)
        selected_parts.append(stats["selected_ids"].cpu())
    selected = torch.cat(selected_parts, dim=0)

    # Stage 1 is typed by op1, stage 2 by op2, and stage 3 is the shared
    # readout/refinement stage.  The label list preserves pair context for
    # cross-context reuse measurements.
    op_names = list(OPERATIONS)
    stage_groups: dict[str, list[set[int]]] = {}
    stage_union: dict[str, set[int]] = {}
    for index in range(inputs.shape[0]):
        op_ids = (inputs[index, 1:3] - 2).tolist()
        for stage, op_id in enumerate((*op_ids, -1)):
            group = f"stage_{stage + 1}:{op_names[op_id] if op_id >= 0 else 'readout'}"
            route = set(int(value) for value in selected[index, stage].flatten().tolist())
            stage_groups.setdefault(group, []).append(route)
            stage_union.setdefault(group, set()).update(route)

    per_stage: dict[str, Any] = {}
    for stage in range(3):
        stage_name = f"stage_{stage + 1}"
        groups = {
            group: summarize_group(routes)
            for group, routes in stage_groups.items()
            if group.startswith(stage_name + ":")
        }
        group_names = sorted(groups)
        union_pairs = [
            jaccard(stage_union[left], stage_union[right])
            for left, right in combinations(group_names, 2)
        ]
        per_stage[stage_name] = {
            "groups": groups,
            "between_group_union_jaccard": sum(union_pairs) / max(len(union_pairs), 1),
        }

    flat_selected = selected.reshape(-1)
    counts = torch.bincount(flat_selected, minlength=model.router.num_circuits).float()
    probabilities = counts / counts.sum().clamp_min(1)
    result: dict[str, Any] = {
        "checkpoint": checkpoint,
        "model": config["model"],
        "device": str(device),
        "grid_size": grid_size,
        "examples": int(inputs.shape[0]),
        "examples_per_pair": grid_size ** 3,
        "active_circuits_per_stage": int(model.router.active_circuits),
        "circuit_bank_size": model.router.num_circuits,
        "circuits_used": int((counts > 0).sum()),
        "dead_circuit_fraction": float((counts == 0).float().mean()),
        "routing_entropy": float(-(probabilities[probabilities > 0]
                                    * probabilities[probabilities > 0].log()).sum()),
        "routing_max_load_fraction": float(probabilities.max()),
        "per_stage": per_stage,
        "interpretation": {
            "same_operator_within_route_jaccard":
                "higher means stronger reuse within a typed operator group",
            "between_group_union_jaccard":
                "lower means more separated circuit families between operators",
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure typed operator route sharing on the composition grid")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = analyze(args.checkpoint, args.grid_size, args.batch_size, args.device)
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
