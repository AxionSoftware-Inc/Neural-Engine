"""Measure whether routed circuits are functionally distinct.

This is a no-training diagnostic for P-002/P-007.  It evaluates the same
query through each circuit in the model's candidate pool and reports output
diversity, route composition, and task concentration.  It intentionally does
not change checkpoints or model parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.generator import SyntheticTaskGenerator
from train import load_config, make_model, seed_everything


def _mean_pair_cosine(vectors: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(vectors, dim=-1)
    summed = normalized.sum(dim=1)
    pair_sum = summed.square().sum(dim=-1) - vectors.shape[1]
    denominator = max(1, vectors.shape[1] * (vectors.shape[1] - 1))
    return pair_sum / denominator


def _entropy(counts: torch.Tensor) -> torch.Tensor:
    probabilities = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


@torch.no_grad()
def analyze_checkpoint(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = load_config(args.config, smoke=False) if args.config else dict(payload["config"])
    if config["model"] == "baseline":
        raise ValueError("circuit specialization diagnostic requires a Neural Engine checkpoint")
    seed_everything(int(config["seed"]))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload.get("model_state", payload))
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=args.seed,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    batch = generator.balanced_batch(args.examples_per_task, device)
    logits, stats = model(batch.inputs, adaptive=False)
    candidate_ids = stats["candidate_ids"]
    selected_ids = stats["selected_ids"]
    selected_weights = stats["selected_weights"]
    query_states = stats["query_states"]
    batch_size, step_count, pool_width = candidate_ids.shape
    state_dim = query_states.shape[-1]
    candidate_pair_cosines = []
    candidate_norms = []
    route_norms = []
    route_pair_cosines = []
    selected_counts = torch.zeros(model.circuits.num_circuits, dtype=torch.long)
    task_counts = torch.zeros(
        model.circuits.num_circuits,
        int(config.get("num_classes", 64)),
        dtype=torch.long,
    )
    selected_not_in_candidates = 0
    row_count = batch_size * step_count
    for step in range(step_count):
        step_candidates = candidate_ids[:, step]
        step_selected = selected_ids[:, step]
        step_weights = selected_weights[:, step]
        step_query = query_states[:, step]
        flat_ids = step_candidates.reshape(-1)
        flat_query = step_query[:, None, :].expand(-1, pool_width, -1).reshape(-1, state_dim)
        outputs = []
        for start in range(0, flat_ids.shape[0], args.chunk_size):
            end = min(flat_ids.shape[0], start + args.chunk_size)
            ids = flat_ids[start:end].unsqueeze(1)
            weights = torch.ones(ids.shape[0], 1, device=device)
            outputs.append(model.circuits(flat_query[start:end], ids, weights))
        candidate_outputs = torch.cat(outputs, dim=0).reshape(batch_size, pool_width, state_dim)
        candidate_pair_cosines.append(_mean_pair_cosine(candidate_outputs).cpu())
        candidate_norms.append(candidate_outputs.norm(dim=-1).mean(dim=-1).cpu())

        selected_output = torch.zeros(batch_size, state_dim, device=device)
        selected_found = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for slot in range(step_selected.shape[1]):
            matches = step_candidates.eq(step_selected[:, slot, None])
            found = matches.any(dim=-1)
            positions = matches.float().argmax(dim=-1)
            selected_output = selected_output + step_weights[:, slot, None] * candidate_outputs[
                torch.arange(batch_size, device=device), positions
            ]
            selected_found |= found
        selected_not_in_candidates += int((~selected_found).sum().cpu())
        route_norms.append(selected_output.norm(dim=-1).cpu())
        route_pair_cosines.append(
            _mean_pair_cosine(
                torch.stack([
                    candidate_outputs[
                        torch.arange(batch_size, device=device),
                        step_candidates.eq(step_selected[:, slot, None]).float().argmax(dim=-1),
                    ] for slot in range(step_selected.shape[1])
                ], dim=1)
            ).cpu()
        )

        selected_counts.scatter_add_(0, step_selected.reshape(-1).cpu(), torch.ones(step_selected.numel(), dtype=torch.long))
        task_ids = batch.task_ids.cpu().remainder(task_counts.shape[1])
        for slot in range(step_selected.shape[1]):
            # The explicit loop is small (tasks x selected rows) and avoids a
            # dense bank-sized one-hot allocation on large checkpoints.
            for circuit_id, task_id in zip(step_selected[:, slot].cpu().tolist(), task_ids.tolist()):
                task_counts[circuit_id, task_id] += 1

    used = selected_counts.gt(0)
    used_task_counts = task_counts[used]
    task_entropy = _entropy(used_task_counts) if used_task_counts.numel() else torch.empty(0)
    route_pair = torch.cat(route_pair_cosines)
    mean_candidate_norm = torch.cat(candidate_norms)
    mean_route_norm = torch.cat(route_norms)
    return {
        "checkpoint": str(checkpoint_path),
        "model": config["model"],
        "num_circuits": int(model.circuits.num_circuits),
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "rows": row_count,
        "natural_accuracy": float(logits.argmax(dim=-1).eq(batch.targets).float().mean().cpu()),
        "candidate_pool": pool_width,
        "selected_width": int(selected_ids.shape[-1]),
        "candidate_mean_pair_cosine": float(torch.cat(candidate_pair_cosines).mean()),
        "candidate_p95_pair_cosine": float(torch.cat(candidate_pair_cosines).quantile(0.95)),
        "selected_mean_pair_cosine": float(route_pair.mean()),
        "selected_p95_pair_cosine": float(route_pair.quantile(0.95)),
        "candidate_mean_individual_norm": float(mean_candidate_norm.mean()),
        "route_mean_norm": float(mean_route_norm.mean()),
        "route_to_candidate_norm_ratio": float((mean_route_norm / mean_candidate_norm.clamp_min(1e-8)).mean()),
        "selected_not_in_candidate_rows": selected_not_in_candidates,
        "selected_unique": int(used.sum()),
        "selected_fraction_of_bank": float(used.float().mean()),
        "selected_usage_entropy": float(_entropy(selected_counts[used].float().unsqueeze(0)).item())
        if used.any() else 0.0,
        "task_entropy_mean_used_circuits": float(task_entropy.mean()) if task_entropy.numel() else 0.0,
        "task_entropy_p95_used_circuits": float(task_entropy.quantile(0.95)) if task_entropy.numel() else 0.0,
        "task_count_classes": task_counts.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure functional diversity of routed circuits")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1712)
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
