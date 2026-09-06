"""Paired seed17/18 benchmark for the opt-in P-001 strided M=8 schedule.

This experiment changes only candidate retrieval geometry.  It keeps the
frozen/trained circuit bank, M=8 candidate count, active=2 execution, existing
key-score selector, route weights/gain, circuit body, recurrent update, and
model defaults unchanged.

Example (paths are local because checkpoints are intentionally not committed):

    python benchmark_p001_strided_retrieval.py \
      --checkpoint /path/to/seed17.pt \
      --checkpoint /path/to/seed18.pt \
      --device cuda \
      --output results/runs/p001_strided_retrieval_s17_s18.json \
      --markdown results/P001_STRIDED_RETRIEVAL_S17_S18.md

The script performs two paired evaluations per checkpoint:

1. native contiguous M=8 retrieval;
2. opt-in four-window M=8 retrieval.

It separately reports final held-out quality and one-decision frozen-bank
retrieval/selection regret.  The final decision is PASS only when every
pre-registered P-001 acceptance gate in problems.md is evidenced; otherwise it
is REJECTED.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from contextlib import nullcontext
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from audit_capacity_route_oracle import _checkpoint, _heldout_source, _step_plan
from neural_engine.p001_retrieval_patch import strided_candidate_schedule
from neural_engine.router import HierarchicalRouter


def _schedule(model, patched: bool, windows: int):
    if not patched:
        return nullcontext()
    if not isinstance(model.router, HierarchicalRouter):
        raise TypeError("P-001 strided benchmark requires HierarchicalRouter")
    return strided_candidate_schedule(model.router, windows=windows)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _checkpoint_steps(path: Path) -> int | None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    report = payload.get("report", {})
    value = report.get("steps")
    return None if value is None else int(value)


def _active_cost(model) -> dict[str, Any]:
    router = model.router
    state_dim = int(model.state_dim)
    candidate_pool = int(router.candidate_pool)
    active = int(model.active_circuits)
    steps = int(model.internal_steps)
    tree_muladds = int(
        router.active_depth * router.num_addresses * state_dim * router.branch
    )
    candidate_muladds = int(candidate_pool * state_dim)
    result: dict[str, Any] = {
        "candidate_key_rows_per_decision": candidate_pool,
        "candidate_key_scalar_reads_per_decision": candidate_pool * state_dim,
        "candidate_score_muladds_per_decision": candidate_muladds,
        "tree_score_muladds_per_decision": tree_muladds,
        "executed_circuit_rows_per_decision": active,
        "candidate_key_scalar_reads_per_example": candidate_pool * state_dim * steps,
        "candidate_score_muladds_per_example": candidate_muladds * steps,
        "tree_score_muladds_per_example": tree_muladds * steps,
        "executed_circuit_rows_per_example": active * steps,
        "training_probe_extra_forwards": 0,
        "training_probe_extra_parameters": 0,
        "schedule_trainable_parameters": 0,
    }
    if hasattr(model, "parameter_report"):
        report = model.parameter_report()
        for key in (
            "total_params",
            "active_params_estimate",
            "active_fraction",
            "active_circuit_params",
        ):
            if key in report:
                result[key] = report[key]
    if "total_params" not in result:
        result["total_params"] = sum(parameter.numel() for parameter in model.parameters())
    return result


@torch.inference_mode()
def _quality(
    model,
    config: dict[str, Any],
    device: torch.device,
    *,
    patched: bool,
    windows: int,
    batches: int,
    examples_per_task: int,
) -> dict[str, Any]:
    source = _heldout_source(config, device)
    loss_sum = 0.0
    correct = 0
    examples = 0
    usage = torch.zeros(model.router.num_circuits, dtype=torch.long, device=device)

    # Warm-up uses an independent source so paired held-out examples remain identical.
    warmup = _heldout_source(config, device).balanced(min(4, examples_per_task))
    with _schedule(model, patched, windows):
        model(warmup.inputs, adaptive=False)
        _sync(device)
        start = time.perf_counter()
        for _ in range(batches):
            batch = source.balanced(examples_per_task)
            logits, stats = model(batch.inputs, adaptive=False)
            losses = F.cross_entropy(logits, batch.targets, reduction="sum")
            loss_sum += float(losses)
            correct += int(logits.argmax(dim=-1).eq(batch.targets).sum())
            examples += batch.targets.numel()
            selected = stats["selected_ids"].reshape(-1)
            selected = selected[selected.ge(0)]
            usage += torch.bincount(selected, minlength=model.router.num_circuits)
        _sync(device)
        elapsed = time.perf_counter() - start

    dead = int((usage == 0).sum())
    return {
        "heldout_ce": loss_sum / examples,
        "hard_accuracy": correct / examples,
        "examples": examples,
        "dead_circuits": dead,
        "circuits_used": int((usage > 0).sum()),
        "seconds": elapsed,
        "seconds_per_example": elapsed / examples,
        "active_cost": _active_cost(model),
    }


def _bucket() -> dict[str, Any]:
    return {
        "decisions": 0,
        "candidate_loss": 0.0,
        "selector_loss": 0.0,
        "full_loss": 0.0,
        "retrieval_regret": [],
        "selection_regret": [],
        "candidate_correct": 0,
        "selector_correct": 0,
        "full_correct": 0,
        "recall": 0,
    }


def _summary(bucket: dict[str, Any]) -> dict[str, Any]:
    decisions = int(bucket["decisions"])
    retrieval = torch.tensor(bucket["retrieval_regret"], dtype=torch.float32)
    selection = torch.tensor(bucket["selection_regret"], dtype=torch.float32)
    return {
        "decisions": decisions,
        "candidate_oracle_ce": bucket["candidate_loss"] / decisions,
        "selector_ce": bucket["selector_loss"] / decisions,
        "full_oracle_ce": bucket["full_loss"] / decisions,
        "retrieval_regret_mean": float(retrieval.mean()),
        "retrieval_regret_p95": float(torch.quantile(retrieval, 0.95)),
        "selection_regret_mean": float(selection.mean()),
        "selection_regret_p95": float(torch.quantile(selection, 0.95)),
        "candidate_recall": bucket["recall"] / decisions,
        "candidate_oracle_accuracy": bucket["candidate_correct"] / decisions,
        "selector_accuracy": bucket["selector_correct"] / decisions,
        "full_oracle_accuracy": bucket["full_correct"] / decisions,
    }


@torch.inference_mode()
def _route_diagnostic(
    model,
    config: dict[str, Any],
    device: torch.device,
    *,
    patched: bool,
    windows: int,
    batches: int,
    examples_per_task: int,
) -> dict[str, Any]:
    bank = int(model.router.num_circuits)
    active = int(model.active_circuits)
    if bank != 32 or active != 2 or model.router.candidate_pool != 8:
        raise ValueError("P-001 diagnostic requires E=32, active=2, M=8")
    if model.internal_steps != 3:
        raise ValueError("P-001 diagnostic requires T=3")

    pairs = list(itertools.combinations(range(bank), active))
    pair_to_index = {pair: index for index, pair in enumerate(pairs)}
    source = _heldout_source(config, device)
    total = _bucket()
    by_step = {str(step): _bucket() for step in range(model.internal_steps)}

    with _schedule(model, patched, windows):
        for _ in range(batches):
            batch = source.balanced(examples_per_task)
            native_logits, native_stats = model(batch.inputs, adaptive=False)
            base_ids = native_stats["selected_ids"]
            candidate_steps = native_stats["candidate_ids"]
            query_steps = native_stats["query_states"]

            for step in range(model.internal_steps):
                candidates = candidate_steps[:, step]
                query = query_steps[:, step]
                key_logits = query @ model.router.keys[:bank].T
                key_logits = key_logits / math.sqrt(query.shape[-1])
                _, key_positions = key_logits.gather(1, candidates).topk(active, dim=-1)
                key_selected = candidates.gather(1, key_positions).sort(dim=-1).values
                selector_pair_index = torch.tensor(
                    [pair_to_index[tuple(row.tolist())] for row in key_selected.cpu()],
                    device=device,
                    dtype=torch.long,
                )

                batch_size = batch.targets.shape[0]
                full_best = torch.full((batch_size,), float("inf"), device=device)
                candidate_best = torch.full_like(full_best, float("inf"))
                selector_loss = torch.zeros_like(full_best)
                full_logits = torch.zeros(batch_size, native_logits.shape[-1], device=device)
                candidate_logits = torch.zeros_like(full_logits)
                selector_logits = torch.zeros_like(full_logits)
                pair_losses: list[torch.Tensor] = []

                for pair_number, pair in enumerate(pairs):
                    pair_tensor = torch.tensor(pair, device=device, dtype=torch.long).view(1, 2)
                    pair_tensor = pair_tensor.expand(batch_size, -1)
                    plan = _step_plan(base_ids, step, pair_tensor)
                    logits, _ = model(batch.inputs, adaptive=False, forced_selected_ids=plan)
                    losses = F.cross_entropy(logits, batch.targets, reduction="none")
                    pair_losses.append(losses)

                    better = losses < full_best
                    full_best = torch.where(better, losses, full_best)
                    full_logits = torch.where(better.unsqueeze(-1), logits, full_logits)

                    valid = candidates.eq(pair[0]).any(dim=-1) & candidates.eq(pair[1]).any(dim=-1)
                    candidate_better = valid & (losses < candidate_best)
                    candidate_best = torch.where(candidate_better, losses, candidate_best)
                    candidate_logits = torch.where(
                        candidate_better.unsqueeze(-1), logits, candidate_logits
                    )

                    chosen = selector_pair_index.eq(pair_number)
                    selector_loss = torch.where(chosen, losses, selector_loss)
                    selector_logits = torch.where(chosen.unsqueeze(-1), logits, selector_logits)

                if not torch.isfinite(candidate_best).all():
                    raise RuntimeError("candidate pool failed to expose a valid pair")

                all_losses = torch.stack(pair_losses, dim=1)
                full_ties = all_losses.le(full_best.unsqueeze(1) + 1e-6)
                left = torch.tensor([pair[0] for pair in pairs], device=device).view(1, -1, 1)
                right = torch.tensor([pair[1] for pair in pairs], device=device).view(1, -1, 1)
                pool = candidates.unsqueeze(1)
                valid_pairs = pool.eq(left).any(dim=-1) & pool.eq(right).any(dim=-1)
                recall = (full_ties & valid_pairs).any(dim=1)
                retrieval = candidate_best - full_best
                selection = selector_loss - candidate_best

                for target in (total, by_step[str(step)]):
                    target["decisions"] += batch_size
                    target["candidate_loss"] += float(candidate_best.sum())
                    target["selector_loss"] += float(selector_loss.sum())
                    target["full_loss"] += float(full_best.sum())
                    target["retrieval_regret"].extend(retrieval.cpu().tolist())
                    target["selection_regret"].extend(selection.cpu().tolist())
                    target["candidate_correct"] += int(
                        candidate_logits.argmax(-1).eq(batch.targets).sum()
                    )
                    target["selector_correct"] += int(
                        selector_logits.argmax(-1).eq(batch.targets).sum()
                    )
                    target["full_correct"] += int(
                        full_logits.argmax(-1).eq(batch.targets).sum()
                    )
                    target["recall"] += int(recall.sum())

    return {
        "aggregate": _summary(total),
        "by_step": {step: _summary(bucket) for step, bucket in by_step.items()},
        "note": (
            "At the replaced decision every pair uses uniform weights and unit gain; "
            "retrieval regret = candidate oracle CE - full-bank oracle CE; "
            "selection regret = same-key selector CE - candidate oracle CE."
        ),
    }


def _reduction(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return 0.0 if abs(after) < 1e-12 else -math.inf
    return (before - after) / abs(before)


def _paired_checkpoint(
    path: Path,
    device: torch.device,
    *,
    windows: int,
    quality_batches: int,
    quality_examples_per_task: int,
    diagnostic_batches: int,
    diagnostic_examples_per_task: int,
) -> dict[str, Any]:
    model, config = _checkpoint(path, device)
    model.eval()
    seed = int(config["seed"])
    baseline_quality = _quality(
        model, config, device,
        patched=False, windows=windows,
        batches=quality_batches, examples_per_task=quality_examples_per_task,
    )
    patched_quality = _quality(
        model, config, device,
        patched=True, windows=windows,
        batches=quality_batches, examples_per_task=quality_examples_per_task,
    )
    baseline_route = _route_diagnostic(
        model, config, device,
        patched=False, windows=windows,
        batches=diagnostic_batches, examples_per_task=diagnostic_examples_per_task,
    )
    patched_route = _route_diagnostic(
        model, config, device,
        patched=True, windows=windows,
        batches=diagnostic_batches, examples_per_task=diagnostic_examples_per_task,
    )
    latency_ratio = patched_quality["seconds_per_example"] / max(
        baseline_quality["seconds_per_example"], 1e-12
    )
    return {
        "checkpoint": str(path),
        "checkpoint_training_steps": _checkpoint_steps(path),
        "seed": seed,
        "protocol": {
            "bank": int(model.router.num_circuits),
            "active": int(model.active_circuits),
            "candidate_pool": int(model.router.candidate_pool),
            "internal_steps": int(model.internal_steps),
            "strided_windows": windows,
        },
        "baseline": {"quality": baseline_quality, "route": baseline_route},
        "patched": {"quality": patched_quality, "route": patched_route},
        "delta": {
            "heldout_ce": patched_quality["heldout_ce"] - baseline_quality["heldout_ce"],
            "hard_accuracy_pp": 100.0 * (
                patched_quality["hard_accuracy"] - baseline_quality["hard_accuracy"]
            ),
            "candidate_recall_pp": 100.0 * (
                patched_route["aggregate"]["candidate_recall"]
                - baseline_route["aggregate"]["candidate_recall"]
            ),
            "retrieval_regret_p95_reduction": _reduction(
                baseline_route["aggregate"]["retrieval_regret_p95"],
                patched_route["aggregate"]["retrieval_regret_p95"],
            ),
            "selection_regret_p95_reduction": _reduction(
                baseline_route["aggregate"]["selection_regret_p95"],
                patched_route["aggregate"]["selection_regret_p95"],
            ),
            "latency_ratio": latency_ratio,
        },
    }


def _gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted(item["seed"] for item in results)
    protocol_ok = seeds == [17, 18] and all(
        item["protocol"]["bank"] == 32
        and item["protocol"]["active"] == 2
        and item["protocol"]["candidate_pool"] == 8
        and item["protocol"]["internal_steps"] == 3
        for item in results
    )
    full_run = all(
        item["checkpoint_training_steps"] is not None
        and item["checkpoint_training_steps"] >= 5000
        for item in results
    )
    recall_ok = all(
        item["patched"]["route"]["aggregate"]["candidate_recall"]
        >= item["baseline"]["route"]["aggregate"]["candidate_recall"]
        for item in results
    )
    mean_retrieval_reduction = sum(
        item["delta"]["retrieval_regret_p95_reduction"] for item in results
    ) / len(results)
    mean_selection_reduction = sum(
        item["delta"]["selection_regret_p95_reduction"] for item in results
    ) / len(results)
    regret_ok = mean_retrieval_reduction >= 0.10 and mean_selection_reduction >= 0.10
    mean_accuracy_pp = sum(item["delta"]["hard_accuracy_pp"] for item in results) / len(results)
    accuracy_ok = mean_accuracy_pp >= 2.0
    dead_ok = all(item["patched"]["quality"]["dead_circuits"] <= 3 for item in results)
    latency_ok = all(item["delta"]["latency_ratio"] <= 1.25 for item in results)
    active_cost_same = all(
        item["patched"]["quality"]["active_cost"]
        == item["baseline"]["quality"]["active_cost"]
        for item in results
    )
    checks = {
        "seed17_18_protocol": protocol_ok,
        "full_5000_step_checkpoint": full_run,
        "candidate_recall_non_decreasing": recall_ok,
        "mean_p95_retrieval_regret_reduction_gte_10pct": mean_retrieval_reduction >= 0.10,
        "mean_p95_selection_regret_reduction_gte_10pct": mean_selection_reduction >= 0.10,
        "mean_hard_accuracy_delta_gte_2pp": accuracy_ok,
        "dead_circuits_lte_3": dead_ok,
        "latency_lte_1_25x": latency_ok,
        "active_cost_unchanged": active_cost_same,
    }
    passed = all(checks.values())
    return {
        "decision": "PASS" if passed else "REJECTED",
        "checks": checks,
        "mean_hard_accuracy_delta_pp": mean_accuracy_pp,
        "mean_p95_retrieval_regret_reduction": mean_retrieval_reduction,
        "mean_p95_selection_regret_reduction": mean_selection_reduction,
        "note": "CE is reported but is never sufficient by itself for adoption.",
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P-001 strided M=8 retrieval schedule — seed17/18 paired audit",
        "",
        f"**Decision: {payload['gate']['decision']}**",
        "",
        "The patch is opt-in and changes candidate retrieval schedule only. M=8,",
        "active=2, the existing key-score selector, circuit body, recurrent state",
        "update, correction path, and default model are unchanged.",
        "",
        "## Quality",
        "",
        "| Seed | Base CE | Patch CE | Δ CE | Base acc | Patch acc | Δ acc | Dead | Latency |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["results"]:
        base = item["baseline"]["quality"]
        patch = item["patched"]["quality"]
        delta = item["delta"]
        lines.append(
            f"| {item['seed']} | {base['heldout_ce']:.4f} | {patch['heldout_ce']:.4f} | "
            f"{delta['heldout_ce']:+.4f} | {base['hard_accuracy']:.2%} | "
            f"{patch['hard_accuracy']:.2%} | {delta['hard_accuracy_pp']:+.2f} pp | "
            f"{patch['dead_circuits']}/32 | {delta['latency_ratio']:.3f}x |"
        )
    lines += [
        "",
        "## Retrieval vs selection",
        "",
        "| Seed | Variant | Recall | Retrieval mean | Retrieval p95 | Selection mean | Selection p95 | Selector CE | Selector acc |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["results"]:
        for label in ("baseline", "patched"):
            route = item[label]["route"]["aggregate"]
            lines.append(
                f"| {item['seed']} | {label} | {route['candidate_recall']:.1%} | "
                f"{route['retrieval_regret_mean']:.4f} | {route['retrieval_regret_p95']:.4f} | "
                f"{route['selection_regret_mean']:.4f} | {route['selection_regret_p95']:.4f} | "
                f"{route['selector_ce']:.4f} | {route['selector_accuracy']:.1%} |"
            )
    lines += [
        "",
        "## Active cost",
        "",
        "The schedule adds no trainable parameter and no probe forward. Candidate key",
        "rows remain M=8 and executed circuit rows remain active=2. Measured latency",
        "is still gated separately because Python hook overhead is real experiment cost.",
        "",
        "## Acceptance gate",
        "",
    ]
    for name, value in payload["gate"]["checks"].items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — `{name}`")
    lines += [
        "",
        "A failed gate means REJECTED for adoption. CE-only improvement is explicitly",
        "insufficient. See the JSON for by-step regret and complete active-cost fields.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python benchmark_p001_strided_retrieval.py \\",
        "  --checkpoint /path/to/seed17.pt \\",
        "  --checkpoint /path/to/seed18.pt \\",
        "  --device cuda \\",
        "  --output results/runs/p001_strided_retrieval_s17_s18.json \\",
        "  --markdown results/P001_STRIDED_RETRIEVAL_S17_S18.md",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark P-001 opt-in strided M=8 retrieval")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--quality-batches", type=int, default=8)
    parser.add_argument("--quality-examples-per-task", type=int, default=32)
    parser.add_argument("--diagnostic-batches", type=int, default=2)
    parser.add_argument("--diagnostic-examples-per-task", type=int, default=17)
    parser.add_argument(
        "--output", default="results/runs/p001_strided_retrieval_s17_s18.json"
    )
    parser.add_argument(
        "--markdown", default="results/P001_STRIDED_RETRIEVAL_S17_S18.md"
    )
    args = parser.parse_args()

    if len(args.checkpoint) != 2:
        raise ValueError("P-001 acceptance benchmark requires exactly seed17 and seed18 checkpoints")
    device = torch.device(args.device)
    results = [
        _paired_checkpoint(
            Path(path), device,
            windows=args.windows,
            quality_batches=args.quality_batches,
            quality_examples_per_task=args.quality_examples_per_task,
            diagnostic_batches=args.diagnostic_batches,
            diagnostic_examples_per_task=args.diagnostic_examples_per_task,
        )
        for path in args.checkpoint
    ]
    payload = {
        "experiment": "p001_strided_candidate_schedule",
        "patch": {
            "candidate_pool": 8,
            "active_circuits": 2,
            "windows": args.windows,
            "trainable_parameters_added": 0,
            "selector_changed": False,
            "circuit_body_changed": False,
            "recurrent_update_changed": False,
            "default_model_changed": False,
        },
        "results": results,
    }
    payload["gate"] = _gate(results)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown = Path(args.markdown)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
