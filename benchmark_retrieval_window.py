"""Counterfactual audit for hierarchical candidate-window width.

The audit keeps the frozen model and its native route states fixed.  At one
recurrent decision it exhaustively evaluates every pair in the 32-circuit
bank, then asks what would happen if the native contiguous candidate window
were widened from M=8 to M=16/24/32.  This separates pair retrieval loss from
pair selection loss without training a new router.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from audit_capacity_route_oracle import _checkpoint, _heldout_source, _step_plan


def _canonical_pair_index(bank: int, active: int) -> dict[tuple[int, int], int]:
    return {
        pair: index
        for index, pair in enumerate(itertools.combinations(range(bank), active))
    }


def _window(base: torch.Tensor, width: int, capacity: int) -> torch.Tensor:
    offsets = torch.arange(width, device=base.device, dtype=torch.long)
    return (base.unsqueeze(-1) + offsets).remainder(capacity)


def _new_bucket() -> dict[str, Any]:
    return {
        "decisions": 0,
        "candidate_loss": 0.0,
        "selector_loss": 0.0,
        "full_loss": 0.0,
        "candidate_regret": [],
        "selection_regret": [],
        "retrieval_regret": [],
        "candidate_correct": 0,
        "selector_correct": 0,
        "full_correct": 0,
        "recall": 0,
    }


def _summarize(bucket: dict[str, Any], width: int) -> dict[str, Any]:
    candidate_regret = torch.tensor(bucket.pop("candidate_regret"), dtype=torch.float32)
    selection_regret = torch.tensor(bucket.pop("selection_regret"), dtype=torch.float32)
    retrieval_regret = torch.tensor(bucket.pop("retrieval_regret"), dtype=torch.float32)
    decisions = bucket.pop("decisions")
    return {
        "candidate_pool": width,
        "decisions": decisions,
        "candidate_oracle_loss": bucket.pop("candidate_loss") / decisions,
        "selector_loss": bucket.pop("selector_loss") / decisions,
        "full_oracle_loss": bucket.pop("full_loss") / decisions,
        "candidate_regret_mean": float(candidate_regret.mean()),
        "candidate_regret_p95": float(torch.quantile(candidate_regret, 0.95)),
        "selection_regret_mean": float(selection_regret.mean()),
        "selection_regret_p95": float(torch.quantile(selection_regret, 0.95)),
        "retrieval_regret_mean": float(retrieval_regret.mean()),
        "retrieval_regret_p95": float(torch.quantile(retrieval_regret, 0.95)),
        "candidate_oracle_accuracy": bucket.pop("candidate_correct") / decisions,
        "selector_accuracy": bucket.pop("selector_correct") / decisions,
        "full_oracle_accuracy": bucket.pop("full_correct") / decisions,
        "candidate_recall": bucket.pop("recall") / decisions,
    }


@torch.inference_mode()
def audit_checkpoint(path: Path, device: torch.device, batches: int,
                     examples_per_task: int, widths: list[int]) -> dict[str, Any]:
    model, config = _checkpoint(path, device)
    source = _heldout_source(config, device)
    bank = model.router.num_circuits
    active = model.active_circuits
    if active != 2:
        raise ValueError("retrieval-window audit currently expects active_circuits=2")
    if model.router.candidate_pool != 8:
        raise ValueError("retrieval-window audit expects the native M=8 checkpoint")
    if any(width < active or width > bank for width in widths):
        raise ValueError("every candidate width must fit active circuits and the bank")
    all_pairs = list(itertools.combinations(range(bank), active))
    pair_index = _canonical_pair_index(bank, active)
    totals = {width: _new_bucket() for width in widths}
    per_step = {
        width: {str(step): _new_bucket() for step in range(model.internal_steps)}
        for width in widths
    }
    examples = 0

    for _ in range(batches):
        batch = source.balanced(examples_per_task)
        native_logits, native_stats = model(batch.inputs, adaptive=False)
        base_ids = native_stats["selected_ids"]
        native_candidates = native_stats["candidate_ids"]
        query_states = native_stats["query_states"]
        examples += batch.targets.numel()

        for step in range(model.internal_steps):
            base = native_candidates[:, step, 0]
            query = query_states[:, step]
            pools = {width: _window(base, width, bank) for width in widths}
            key_logits = query @ model.router.keys[:bank].T
            key_logits = key_logits / (query.shape[-1] ** 0.5)
            selector_indices: dict[int, torch.Tensor] = {}
            for width, pool in pools.items():
                values, positions = key_logits.gather(1, pool).topk(active, dim=-1)
                selected = pool.gather(1, positions)
                canonical = selected.sort(dim=-1).values
                selector_indices[width] = torch.tensor(
                    [pair_index[tuple(row.tolist())] for row in canonical.cpu()],
                    device=device, dtype=torch.long,
                )

            full_best = torch.full((batch.targets.shape[0],), float("inf"), device=device)
            full_best_logits = torch.zeros(
                batch.targets.shape[0], native_logits.shape[-1], device=device,
            )
            candidate_best = {
                width: torch.full_like(full_best, float("inf")) for width in widths
            }
            candidate_best_logits = {
                width: torch.zeros_like(full_best_logits) for width in widths
            }
            selector_loss = {width: torch.zeros_like(full_best) for width in widths}
            selector_logits = {
                width: torch.zeros_like(full_best_logits) for width in widths
            }
            pair_losses = []

            for pair_number, pair in enumerate(all_pairs):
                pair_tensor = torch.tensor(pair, device=device, dtype=torch.long).view(1, -1)
                pair_tensor = pair_tensor.expand(batch.targets.shape[0], -1)
                plan = _step_plan(base_ids, step, pair_tensor)
                logits, _ = model(
                    batch.inputs, adaptive=False, forced_selected_ids=plan,
                )
                losses = F.cross_entropy(logits, batch.targets, reduction="none")
                pair_losses.append(losses)
                better = losses < full_best
                full_best = torch.where(better, losses, full_best)
                full_best_logits = torch.where(
                    better.unsqueeze(-1), logits, full_best_logits,
                )
                for width in widths:
                    pool = pools[width]
                    valid = pool.eq(pair[0]).any(dim=-1) & pool.eq(pair[1]).any(dim=-1)
                    candidate_better = valid & (losses < candidate_best[width])
                    candidate_best[width] = torch.where(
                        candidate_better, losses, candidate_best[width],
                    )
                    candidate_best_logits[width] = torch.where(
                        candidate_better.unsqueeze(-1), logits,
                        candidate_best_logits[width],
                    )
                    chosen = selector_indices[width].eq(pair_number)
                    selector_loss[width] = torch.where(chosen, losses, selector_loss[width])
                    selector_logits[width] = torch.where(
                        chosen.unsqueeze(-1), logits, selector_logits[width],
                    )

            pair_losses = torch.stack(pair_losses, dim=1)
            full_ties = pair_losses.le(full_best.unsqueeze(1) + 1e-6)
            pair_left = torch.tensor(
                [pair[0] for pair in all_pairs], device=device, dtype=torch.long,
            ).view(1, -1, 1)
            pair_right = torch.tensor(
                [pair[1] for pair in all_pairs], device=device, dtype=torch.long,
            ).view(1, -1, 1)
            for width in widths:
                pool = pools[width].unsqueeze(1)
                valid_pairs = pool.eq(pair_left).any(dim=-1) & pool.eq(pair_right).any(dim=-1)
                recall = (full_ties & valid_pairs).any(dim=1)
                selection_regret = selector_loss[width] - candidate_best[width]
                retrieval_regret = candidate_best[width] - full_best
                for bucket in (totals[width], per_step[width][str(step)]):
                    bucket["decisions"] += batch.targets.numel()
                    bucket["candidate_loss"] += float(candidate_best[width].sum())
                    bucket["selector_loss"] += float(selector_loss[width].sum())
                    bucket["full_loss"] += float(full_best.sum())
                    bucket["candidate_regret"].extend(
                        (candidate_best[width] - full_best).cpu().tolist()
                    )
                    bucket["selection_regret"].extend(selection_regret.cpu().tolist())
                    bucket["retrieval_regret"].extend(retrieval_regret.cpu().tolist())
                    bucket["candidate_correct"] += int(
                        candidate_best_logits[width].argmax(-1).eq(batch.targets).sum()
                    )
                    bucket["selector_correct"] += int(
                        selector_logits[width].argmax(-1).eq(batch.targets).sum()
                    )
                    bucket["full_correct"] += int(
                        full_best_logits.argmax(-1).eq(batch.targets).sum()
                    )
                    bucket["recall"] += int(recall.sum())

    return {
        "checkpoint": str(path),
        "seed": int(config["seed"]),
        "bank": bank,
        "active_circuits": active,
        "internal_steps": model.internal_steps,
        "batches": batches,
        "examples": examples,
        "widths": [_summarize(totals[width], width) for width in widths],
        "by_step": {
            str(step): [_summarize(per_step[width][str(step)], width) for width in widths]
            for step in range(model.internal_steps)
        },
        "comparison_note": (
            "Native M=8 route states and frozen bank are held fixed. At one step, "
            "the contiguous native window is widened counterfactually; every bank "
            "pair is evaluated with uniform weights and unit gain."
        ),
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P-001 retrieval-window diagnostic",
        "",
        "This is a frozen-bank counterfactual audit, not a trained-router result.",
        "The native M=8 route state is held fixed while its contiguous candidate",
        "window is widened. Pair quality is evaluated exhaustively at one changed",
        "recurrent decision, with uniform weights and unit gain.",
        "",
        "## Aggregate results",
        "",
        "| Seed | M | Candidate oracle CE | Same-key selector CE | Full oracle CE | Retrieval regret | Selection regret | p95 retrieval | Recall | Candidate acc | Selector acc | Full acc |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        for item in result["widths"]:
            lines.append(
                f"| {result['seed']} | {item['candidate_pool']} | "
                f"{item['candidate_oracle_loss']:.4f} | {item['selector_loss']:.4f} | "
                f"{item['full_oracle_loss']:.4f} | {item['retrieval_regret_mean']:.4f} | "
                f"{item['selection_regret_mean']:.4f} | {item['retrieval_regret_p95']:.4f} | "
                f"{item['candidate_recall']:.1%} | {item['candidate_oracle_accuracy']:.1%} | "
                f"{item['selector_accuracy']:.1%} | {item['full_oracle_accuracy']:.1%} |"
            )
    lines += [
        "",
        "## Interpretation rule",
        "",
        "If widening M sharply reduces retrieval regret/raises recall while the",
        "same-key selector remains poor, retrieval is a real bottleneck but the",
        "selector/objective still needs separate work. If M=16/24 barely changes",
        "the result, candidate width is not the main explanation for scaling failure.",
        "",
        "Exact JSON: `results/runs/p001_retrieval_window_s17_s18.json`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit candidate-window retrieval headroom")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--examples-per-task", type=int, default=17)
    parser.add_argument("--width", action="append", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/runs/p001_retrieval_window_s17_s18.json")
    parser.add_argument("--markdown", default="results/P001_RETRIEVAL_WINDOW_S17_S18.md")
    args = parser.parse_args()
    widths = args.width or [8, 16, 24, 32]
    device = torch.device(args.device)
    payload = {
        "experiment": "p001_retrieval_window_diagnostic",
        "results": [
            audit_checkpoint(Path(path), device, args.batches, args.examples_per_task, widths)
            for path in args.checkpoint
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
