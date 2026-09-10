"""Measure selected-circuit redundancy by task depth and value regime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from data.generator import SyntheticTaskGenerator, accuracy_by_depth
from train import make_model, seed_everything


DEFAULT_CHECKPOINTS = (
    "results/checkpoints/ne300_shared_routekeys_ordered_s17_10000.pt",
    "results/checkpoints/ne300_shared_routekeys_ordered_s18_10000.pt",
    "results/checkpoints/ne500_shared_routekeys_ordered_s17_10000.pt",
    "results/checkpoints/ne500_shared_routekeys_ordered_s18_10000.pt",
)

CONDITIONS = {
    "uniform_all": (0, 63, "all"),
    "low_edge_values": (0, 7, "all"),
    "high_edge_values": (56, 63, "all"),
}


def mean_pair_cosine(values: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(values, dim=-1)
    summed = normalized.sum(dim=1)
    pair_sum = summed.square().sum(dim=-1) - values.shape[1]
    denominator = max(1, values.shape[1] * (values.shape[1] - 1))
    return pair_sum / denominator


def device_for(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


@torch.no_grad()
def analyze_condition(model: torch.nn.Module, config: dict[str, Any], device: torch.device,
                      *, seed: int, value_min: int, value_max: int, split: str,
                      examples_per_task: int) -> dict[str, Any]:
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), seed=seed, value_min=value_min,
        value_max=value_max, split=split,
    )
    batch = generator.balanced_batch(examples_per_task, device)
    logits, stats = model(batch.inputs, adaptive=False)
    predictions = logits.argmax(dim=-1)
    selected_ids = stats["selected_ids"]
    selected_weights = stats["selected_weights"]
    queries = stats["query_states"]
    result: dict[str, Any] = {
        "value_range": [value_min, value_max],
        "split": split,
        "examples_per_task": examples_per_task,
        "accuracy": float(predictions.eq(batch.targets).float().mean().cpu()),
        "accuracy_by_depth": accuracy_by_depth(predictions, batch),
        "depths": {},
    }

    for depth in (1, 2, 3):
        depth_mask = batch.depths.eq(depth)
        if not depth_mask.any():
            continue
        pair_cosines = []
        selected_norms = []
        query_norms = []
        route_norms = []
        for step in range(selected_ids.shape[1]):
            ids = selected_ids[depth_mask, step]
            query = queries[depth_mask, step]
            weights = selected_weights[depth_mask, step]
            expanded_query = query[:, None, :].expand(-1, ids.shape[-1], -1)
            unit_weights = torch.ones_like(weights)
            outputs = model.circuits(
                expanded_query.reshape(-1, expanded_query.shape[-1]),
                ids.reshape(-1, 1), unit_weights.reshape(-1, 1),
            ).reshape(ids.shape[0], ids.shape[1], -1)
            pair_cosines.append(mean_pair_cosine(outputs))
            selected_norms.append(outputs.norm(dim=-1).mean(dim=-1))
            query_norms.append(query.norm(dim=-1))
            route_norms.append((outputs * weights.unsqueeze(-1)).sum(dim=1).norm(dim=-1))
        pair = torch.cat(pair_cosines)
        selected = torch.cat(selected_norms)
        query_values = torch.cat(query_norms)
        route = torch.cat(route_norms)
        result["depths"][str(depth)] = {
            "rows": int(depth_mask.sum().item()),
            "selected_pair_cosine_mean": float(pair.mean().cpu()),
            "selected_pair_cosine_p95": float(pair.quantile(0.95).cpu()),
            "selected_output_norm_mean": float(selected.mean().cpu()),
            "query_norm_mean": float(query_values.mean().cpu()),
            "route_output_norm_mean": float(route.mean().cpu()),
            "route_to_query_norm_ratio": float((route / query_values.clamp_min(1e-8)).mean().cpu()),
        }

    routed = selected_ids.reshape(-1).cpu()
    counts = torch.bincount(routed, minlength=model.circuits.num_circuits).float()
    result.update({
        "selected_unique": int(counts.gt(0).sum()),
        "dead_circuit_fraction": float(counts.eq(0).float().mean()),
    })
    if getattr(model, "circuit_bank_mode", None) == "factorized":
        factor_count = int(model.circuits.factor_count)
        factor_ids = torch.cat((routed.remainder(factor_count), routed.div(factor_count, rounding_mode="floor")))
        factor_counts = torch.bincount(factor_ids, minlength=factor_count).float()
        result["factor_dead_fraction"] = float(factor_counts.eq(0).float().mean())
    return result


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = device_for(args.device)
    output: dict[str, Any] = {
        "audit": "native_depth_specialization",
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "conditions": {name: {"value_min": values[0], "value_max": values[1], "split": values[2]}
                       for name, values in CONDITIONS.items()},
        "checkpoints": {},
    }
    for checkpoint_name in args.checkpoints:
        checkpoint_path = Path(checkpoint_name)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config = dict(payload["config"])
        model = make_model(config).to(device).eval()
        model.load_state_dict(payload["model_state"])
        seed = int(config.get("seed", 17))
        checkpoint_result: dict[str, Any] = {"model_name": config["model"], "seed": seed, "conditions": {}}
        for offset, (name, (value_min, value_max, split)) in enumerate(CONDITIONS.items()):
            seed_everything(seed + 4100 + offset)
            checkpoint_result["conditions"][name] = analyze_condition(
                model, config, device, seed=seed + 5100 + offset,
                value_min=value_min, value_max=value_max, split=split,
                examples_per_task=args.examples_per_task,
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
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/diagnostic_native_depth_specialization_300m_500m_10000.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
