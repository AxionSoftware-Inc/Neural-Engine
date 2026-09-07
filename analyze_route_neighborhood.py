"""Measure local selection regret versus a limited full-bank retrieval probe.

For each natural route, this diagnostic replaces one selected circuit at a
time and evaluates the real final CE of the frozen model.  The first search
uses every circuit in the hierarchical candidate pool.  The second search
uses the full-bank key top-k circuits.  It is a one-swap neighborhood, not an
exhaustive K-subset oracle, so the numbers are conservative and comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from neural_engine.router import HierarchicalRouter
from train import make_model


def _load_checkpoint(path: Path, device: torch.device) -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("route neighborhood requires a NeuralEngineV0 checkpoint")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    if not isinstance(model.router, HierarchicalRouter):
        raise ValueError("route neighborhood currently targets HierarchicalRouter")
    return model, config


def _batches(config: dict, device: torch.device, count: int,
             examples_per_task: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config["seed"]) + 3,
        value_min=int(config.get("heldout_value_min", config.get("eval_value_min", 0))),
        value_max=int(config.get("heldout_value_max", config.get("eval_value_max", 63))),
        split=str(config.get("heldout_split", "all")),
    )
    return [
        (batch.inputs, batch.targets)
        for batch in (
            generator.balanced_batch(examples_per_task, device)
            for _ in range(count)
        )
    ]


def _alternative_route(base: torch.Tensor, alternative: torch.Tensor,
                       replace_slot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the least-weighted selected member, avoiding duplicate routes."""
    selected = base.clone()
    replaced = selected.scatter(1, replace_slot.unsqueeze(-1), alternative.unsqueeze(-1))
    already_selected = selected.eq(alternative.unsqueeze(-1)).any(dim=-1)
    selected = torch.where(already_selected.unsqueeze(-1), base, replaced)
    return selected, already_selected


@torch.inference_mode()
def _forced_losses(model: NeuralEngineV0, inputs: torch.Tensor, natural_ids: torch.Tensor,
                   natural_weights: torch.Tensor, natural_gains: torch.Tensor,
                   query: torch.Tensor, step: int, alternative_ids: torch.Tensor,
                   replace_slot: torch.Tensor) -> torch.Tensor:
    route_ids, already_selected = _alternative_route(
        natural_ids[:, step], alternative_ids, replace_slot,
    )
    route_scores = torch.einsum(
        "bd,bkd->bk", query, model.router.keys[route_ids],
    ) / (query.shape[-1] ** 0.5)
    route_weights = F.softmax(route_scores, dim=-1)
    route_weights = torch.where(
        already_selected.unsqueeze(-1), natural_weights[:, step], route_weights,
    )
    forced_ids = torch.full_like(natural_ids, -1)
    forced_ids[:, step] = route_ids
    forced_weights = natural_weights.clone()
    forced_weights[:, step] = route_weights
    logits, _ = model(
        inputs,
        adaptive=False,
        forced_selected_ids=forced_ids,
        forced_selected_weights=forced_weights,
        forced_route_gains=natural_gains,
        collect_stats=False,
    )
    return logits


@torch.inference_mode()
def _audit(model: NeuralEngineV0, batches: list[tuple[torch.Tensor, torch.Tensor]],
           global_topk: int) -> dict:
    step_rows = []
    overall = {
        "examples": 0,
        "natural_loss": 0.0,
        "best_local_loss": 0.0,
        "best_global_loss": 0.0,
        "best_external_loss": 0.0,
        "external_examples": 0,
        "local_improvement_examples": 0,
        "global_improvement_examples": 0,
        "global_topk_members": 0,
        "global_topk_members_in_candidate": 0,
    }
    for step in range(model.internal_steps):
        totals = {key: 0.0 for key in (
            "examples", "natural_loss", "best_local_loss", "best_global_loss",
            "best_external_loss", "external_examples", "local_improvement_examples",
            "global_improvement_examples", "global_topk_members",
            "global_topk_members_in_candidate",
        )}
        for inputs, targets in batches:
            natural_logits, stats = model(inputs, adaptive=False, collect_stats=True)
            natural_loss = F.cross_entropy(natural_logits, targets, reduction="none")
            natural_ids = stats["selected_ids"]
            natural_weights = stats["selected_weights"]
            natural_gains = stats["route_gains"]
            candidate_ids = stats["candidate_ids"][:, step]
            query = stats["query_states"][:, step]
            full_scores = torch.einsum(
                "bd,cd->bc", query, model.router.keys[:model.router.routing_capacity],
            ) / (query.shape[-1] ** 0.5)
            topk = min(global_topk, full_scores.shape[-1])
            global_ids = full_scores.topk(topk, dim=-1).indices
            in_candidate = candidate_ids.unsqueeze(1).eq(global_ids.unsqueeze(-1)).any(dim=-1)
            replace_slot = natural_weights[:, step].argmin(dim=-1)
            best_local = natural_loss.clone()
            best_global = natural_loss.clone()
            best_external = torch.full_like(natural_loss, float("inf"))
            for rank in range(candidate_ids.shape[-1]):
                alternative = candidate_ids[:, rank]
                logits = _forced_losses(
                    model, inputs, natural_ids, natural_weights, natural_gains,
                    query, step, alternative, replace_slot,
                )
                losses = F.cross_entropy(logits, targets, reduction="none")
                best_local = torch.minimum(best_local, losses)
                best_global = torch.minimum(best_global, losses)
            for rank in range(global_ids.shape[-1]):
                alternative = global_ids[:, rank]
                logits = _forced_losses(
                    model, inputs, natural_ids, natural_weights, natural_gains,
                    query, step, alternative, replace_slot,
                )
                losses = F.cross_entropy(logits, targets, reduction="none")
                best_global = torch.minimum(best_global, losses)
                external = ~in_candidate[:, rank]
                best_external = torch.where(
                    external, torch.minimum(best_external, losses), best_external,
                )
            n = targets.numel()
            totals["examples"] += n
            totals["natural_loss"] += float(natural_loss.sum().cpu())
            totals["best_local_loss"] += float(best_local.sum().cpu())
            totals["best_global_loss"] += float(best_global.sum().cpu())
            finite_external = torch.isfinite(best_external)
            totals["best_external_loss"] += float(best_external[finite_external].sum().cpu())
            totals["external_examples"] += int(finite_external.sum().cpu())
            totals["local_improvement_examples"] += int((best_local < natural_loss - 1e-6).sum().cpu())
            totals["global_improvement_examples"] += int((best_global < best_local - 1e-6).sum().cpu())
            totals["global_topk_members"] += int(in_candidate.numel())
            totals["global_topk_members_in_candidate"] += int(in_candidate.sum().cpu())
        overall = {
            key: overall[key] + totals[key]
            for key in overall
        }
        step_rows.append({
            "step": step,
            "examples": int(totals["examples"]),
            "natural_ce": totals["natural_loss"] / totals["examples"],
            "best_local_ce": totals["best_local_loss"] / totals["examples"],
            "best_global_probe_ce": totals["best_global_loss"] / totals["examples"],
            "local_selection_gain_ce": (totals["natural_loss"] - totals["best_local_loss"]) / totals["examples"],
            "global_retrieval_gain_ce": (totals["best_local_loss"] - totals["best_global_loss"]) / totals["examples"],
            "local_improvement_fraction": totals["local_improvement_examples"] / totals["examples"],
            "global_improvement_fraction": totals["global_improvement_examples"] / totals["examples"],
            "full_key_topk_candidate_recall": totals["global_topk_members_in_candidate"] / totals["global_topk_members"],
            "external_probe_examples": int(totals["external_examples"]),
            "external_probe_ce": (
                totals["best_external_loss"] / totals["external_examples"]
                if totals["external_examples"] else None
            ),
        })
    return {
        "steps": step_rows,
        "overall": {
            "examples": int(overall["examples"]),
            "natural_ce": overall["natural_loss"] / overall["examples"],
            "best_local_ce": overall["best_local_loss"] / overall["examples"],
            "best_global_probe_ce": overall["best_global_loss"] / overall["examples"],
            "local_selection_gain_ce": (overall["natural_loss"] - overall["best_local_loss"]) / overall["examples"],
            "global_retrieval_gain_ce": (overall["best_local_loss"] - overall["best_global_loss"]) / overall["examples"],
            "local_improvement_fraction": overall["local_improvement_examples"] / overall["examples"],
            "global_improvement_fraction": overall["global_improvement_examples"] / overall["examples"],
            "full_key_topk_candidate_recall": overall["global_topk_members_in_candidate"] / overall["global_topk_members"],
            "external_probe_examples": int(overall["external_examples"]),
            "external_probe_ce": (
                overall["best_external_loss"] / overall["external_examples"]
                if overall["external_examples"] else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit route neighborhood regret")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--global-topk", type=int, default=8)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    result = {
        "device": str(device),
        "global_topk": int(args.global_topk),
        "batches": int(args.batches),
        "examples_per_task": int(args.examples_per_task),
        "checkpoints": [],
    }
    for checkpoint_name in args.checkpoint:
        path = Path(checkpoint_name)
        model, config = _load_checkpoint(path, device)
        result["checkpoints"].append({
            "checkpoint": str(path),
            "model": config.get("model"),
            "seed": int(config.get("seed", -1)),
            "candidate_pool": int(model.router.candidate_pool),
            "active_circuits": int(model.active_circuits),
            "audit": _audit(model, _batches(config, device, args.batches, args.examples_per_task), args.global_topk),
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
