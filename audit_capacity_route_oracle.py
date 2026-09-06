"""Separate route retrieval from route selection on frozen Neural Engine banks.

The audit keeps circuit weights frozen and changes one recurrent decision at a
time.  At the changed step every compared pair uses uniform weights and unit
gain; this makes the learned-route, candidate-pool oracle, and full-bank oracle
comparable without turning a weight/gain change into a route claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from train import BatchSource, make_model


def _checkpoint(path: Path, device: torch.device) -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("route oracle audit requires a NeuralEngineV0 checkpoint")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, config


def _heldout_source(config: dict, device: torch.device) -> BatchSource:
    heldout_min = int(config.get("heldout_value_min", config.get("eval_value_min", 0)))
    heldout_max = int(config.get("heldout_value_max", config.get("eval_value_max", 63)))
    heldout_split = str(config.get("heldout_split", "all"))
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config["seed"]) + 3,
        value_min=heldout_min, value_max=heldout_max, split=heldout_split,
    )
    return BatchSource(generator, 256, device)


def _step_plan(base_ids: torch.Tensor, step: int, pair: torch.Tensor) -> torch.Tensor:
    plan = torch.full_like(base_ids, -1)
    plan[:, step] = pair
    return plan


@torch.inference_mode()
def audit_checkpoint(path: Path, device: torch.device, batches: int,
                     examples_per_task: int) -> dict:
    model, config = _checkpoint(path, device)
    source = _heldout_source(config, device)
    active = model.active_circuits
    if active != 2:
        raise ValueError("this audit currently expects active_circuits=2")
    bank = model.router.num_circuits
    all_pairs = list(itertools.combinations(range(bank), active))
    totals = {
        "examples": 0,
        "native_loss": 0.0,
        "fixed_learned_loss": 0.0,
        "candidate_oracle_loss": 0.0,
        "full_oracle_loss": 0.0,
        "candidate_regret": [],
        "full_regret": [],
        "native_correct": 0,
        "fixed_learned_correct": 0,
        "candidate_oracle_correct": 0,
        "full_oracle_correct": 0,
        "candidate_pairs": 0,
        "full_pairs": len(all_pairs),
    }

    for _ in range(batches):
        batch = source.balanced(examples_per_task)
        native_logits, native_stats = model(batch.inputs, adaptive=False)
        native_loss = F.cross_entropy(native_logits, batch.targets, reduction="none")
        base_ids = native_stats["selected_ids"]
        candidate_ids = native_stats["candidate_ids"]
        if candidate_ids.shape[-1] != model.router.candidate_pool:
            raise ValueError("checkpoint did not expose a fixed candidate pool")
        totals["examples"] += batch.targets.numel()
        totals["native_loss"] += float(native_loss.sum())
        totals["native_correct"] += int(native_logits.argmax(-1).eq(batch.targets).sum())

        for step in range(model.internal_steps):
            learned_pair = base_ids[:, step]
            learned_plan = _step_plan(base_ids, step, learned_pair)
            learned_logits, _ = model(
                batch.inputs, adaptive=False, forced_selected_ids=learned_plan,
            )
            learned_loss = F.cross_entropy(learned_logits, batch.targets, reduction="none")
            candidate_best = torch.full_like(learned_loss, float("inf"))
            full_best = torch.full_like(learned_loss, float("inf"))
            candidate_best_logits = torch.zeros_like(learned_logits)
            full_best_logits = torch.zeros_like(learned_logits)
            seen_candidate_pairs = set()
            for pair in all_pairs:
                pair_tensor = torch.tensor(pair, device=device, dtype=torch.long).view(1, -1)
                pair_tensor = pair_tensor.expand(batch.targets.shape[0], -1)
                plan = _step_plan(base_ids, step, pair_tensor)
                logits, _ = model(batch.inputs, adaptive=False, forced_selected_ids=plan)
                losses = F.cross_entropy(logits, batch.targets, reduction="none")
                better = losses < full_best
                full_best = torch.where(better, losses, full_best)
                full_best_logits = torch.where(better.unsqueeze(-1), logits, full_best_logits)
                valid = (
                    candidate_ids[:, step].eq(pair[0]).any(dim=-1)
                    & candidate_ids[:, step].eq(pair[1]).any(dim=-1)
                )
                if bool(valid.any()):
                    seen_candidate_pairs.add(pair)
                    candidate_better = valid & (losses < candidate_best)
                    candidate_best = torch.where(candidate_better, losses, candidate_best)
                    candidate_best_logits = torch.where(
                        candidate_better.unsqueeze(-1), logits, candidate_best_logits,
                    )
            if not torch.isfinite(candidate_best).all():
                raise RuntimeError("candidate pool did not contain a valid pair for every example")
            totals["candidate_pairs"] = max(totals["candidate_pairs"], len(seen_candidate_pairs))
            totals["fixed_learned_loss"] += float(learned_loss.sum())
            totals["candidate_oracle_loss"] += float(candidate_best.sum())
            totals["full_oracle_loss"] += float(full_best.sum())
            totals["candidate_regret"].append((learned_loss - candidate_best).cpu())
            totals["full_regret"].append((learned_loss - full_best).cpu())
            totals["fixed_learned_correct"] += int(learned_logits.argmax(-1).eq(batch.targets).sum())
            totals["candidate_oracle_correct"] += int(
                candidate_best_logits.argmax(-1).eq(batch.targets).sum()
            )
            totals["full_oracle_correct"] += int(full_best_logits.argmax(-1).eq(batch.targets).sum())

    count = totals["examples"]
    decision_count = count * model.internal_steps
    candidate_regret = torch.cat(totals.pop("candidate_regret"))
    full_regret = torch.cat(totals.pop("full_regret"))
    result = {
        "checkpoint": str(path),
        "seed": int(config["seed"]),
        "bank": bank,
        "active_circuits": active,
        "internal_steps": model.internal_steps,
        "batches": batches,
        "examples": count,
        "candidate_pool": model.router.candidate_pool,
        "candidate_pair_count": totals["candidate_pairs"],
        "full_pair_count": totals["full_pairs"],
        "native_loss": totals["native_loss"] / count,
        "fixed_learned_loss": totals["fixed_learned_loss"] / decision_count,
        "candidate_oracle_loss": totals["candidate_oracle_loss"] / decision_count,
        "full_oracle_loss": totals["full_oracle_loss"] / decision_count,
        "candidate_gain_vs_fixed_learned": (
            totals["fixed_learned_loss"] - totals["candidate_oracle_loss"]
        ) / decision_count,
        "full_gain_vs_fixed_learned": (
            totals["fixed_learned_loss"] - totals["full_oracle_loss"]
        ) / decision_count,
        "candidate_regret_mean": float(candidate_regret.mean()),
        "candidate_regret_p95": float(torch.quantile(candidate_regret, 0.95)),
        "full_regret_mean": float(full_regret.mean()),
        "full_regret_p95": float(torch.quantile(full_regret, 0.95)),
        "native_accuracy": totals["native_correct"] / count,
        "fixed_learned_accuracy": totals["fixed_learned_correct"] / decision_count,
        "candidate_oracle_accuracy": totals["candidate_oracle_correct"] / decision_count,
        "full_oracle_accuracy": totals["full_oracle_correct"] / decision_count,
        "comparison_note": (
            "At the replaced step all compared pairs use uniform weights and unit gain; "
            "other steps re-route normally. This is a one-decision counterfactual audit."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen-bank route retrieval and selection")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--examples-per-task", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    results = [audit_checkpoint(Path(path), device, args.batches, args.examples_per_task)
               for path in args.checkpoint]
    payload = {"experiment": "capacity_route_oracle_audit", "results": results}
    text = json.dumps(payload, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
