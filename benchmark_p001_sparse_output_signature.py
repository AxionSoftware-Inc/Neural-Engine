"""Handoff D: distill the local output-aware proxy into sparse inference.

The frozen Neural Engine retriever/circuit bank stay unchanged. Training may use
candidate-only real circuit outputs as a teacher, but inference computes only a
small learned signature for the M retrieved candidates and executes two real
circuits. Dense full-bank circuit execution is never used by treatment inference.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.nn import functional as F

from audit_capacity_route_oracle import _checkpoint, _heldout_source, _step_plan
from benchmark_retrieval_window import audit_checkpoint as retrieval_window_audit
from data.generator import SyntheticTaskGenerator
from neural_engine.p001_sparse_output_selector import (
    SparseOutputSignatureSelector,
    SparseStepContext,
    candidate_local_teacher,
    distillation_loss,
    selector_cost_report,
    sparse_selector_rollout,
    validate_plain_sparse_contract,
)
from train import BatchSource


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _checkpoint_steps(path: Path) -> int | None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    report = payload.get("report", {})
    value = report.get("steps")
    return None if value is None else int(value)


def _train_source(config: dict[str, Any], device: torch.device, batch_size: int) -> BatchSource:
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]),
        int(config["seed"]) + 1,
        value_min=int(config.get("train_value_min", 0)),
        value_max=int(config.get("train_value_max", 63)),
        split=str(config.get("train_split", "train")),
    )
    return BatchSource(generator, batch_size, device, task_balanced=True)


def _heldout_batches(
    config: dict[str, Any],
    device: torch.device,
    count: int,
    examples_per_task: int,
) -> list[Any]:
    source = _heldout_source(config, device)
    return [source.balanced(examples_per_task) for _ in range(count)]


@torch.no_grad()
def _baseline_contexts(model, batch) -> list[SparseStepContext]:
    validate_plain_sparse_contract(model)
    encoded = model.encode(batch.inputs)
    _, stats = model(batch.inputs, adaptive=False)
    contexts = []
    for step in range(model.internal_steps):
        query = stats["query_states"][:, step].detach()
        contexts.append(
            SparseStepContext(
                step=step,
                query=query,
                state_before=(query - model.step_embedding[step]).detach(),
                encoded=encoded.detach(),
                candidate_ids=stats["candidate_ids"][:, step].detach(),
                route_gain=stats["route_gains"][:, step].detach(),
            )
        )
    return contexts


def _make_selector(model, args: argparse.Namespace, seed: int, device: torch.device):
    if args.freeze_signature and not args.init_from_bank:
        raise ValueError("--freeze-signature requires --init-from-bank")
    random.seed(seed + 4107)
    torch.manual_seed(seed + 4107)
    selector = SparseOutputSignatureSelector(
        state_dim=int(model.state_dim),
        num_circuits=int(model.router.num_circuits),
        candidate_pool=int(model.router.candidate_pool),
        active_circuits=int(model.active_circuits),
        signature_rank=args.signature_rank,
        signature_dim=args.signature_dim,
        key_prior_weight=args.key_prior_weight,
    )
    if args.key_prior_weight:
        selector.set_reference_keys(model.router.keys)
    if args.init_from_bank:
        if args.signature_rank != int(model.circuits.down.shape[-1]):
            raise ValueError(
                "--init-from-bank requires signature-rank to match the source circuit rank"
            )
        projection_source = torch.randn(
            int(model.state_dim), args.signature_dim, device=device
        )
        projection, _ = torch.linalg.qr(projection_source, mode="reduced")
        selector.initialize_from_circuit_bank(
            model.circuits.down,
            model.circuits.up,
            model.circuits.bias,
            projection,
        )
        if args.freeze_signature:
            for name in ("signature_down", "signature_up", "signature_bias"):
                getattr(selector, name).requires_grad_(False)
    return selector.to(device)


def _freeze_model(model) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def train_selector(
    model,
    config: dict[str, Any],
    selector: SparseOutputSignatureSelector,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _freeze_model(model)
    source = _train_source(config, device, args.train_batch_size)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    losses: list[float] = []
    teacher_matches: list[float] = []
    teacher_regrets: list[float] = []
    teacher_candidate_rows = 0
    teacher_pair_state_evals = 0
    peak_vram = 0
    start = time.perf_counter()
    selector.train()

    for step_number in range(1, args.steps + 1):
        batch = source.batch()
        with torch.no_grad():
            if step_number <= args.warmup_steps:
                contexts = _baseline_contexts(model, batch)
            else:
                _, _, contexts = sparse_selector_rollout(
                    model, selector, batch.inputs, return_contexts=True
                )
        context = contexts[(step_number - 1) % model.internal_steps]
        teacher = candidate_local_teacher(model, context, batch.targets)
        student_scores, _ = selector(context.query, context.candidate_ids)
        teacher_target = teacher[args.teacher_target]
        loss, diagnostics = distillation_loss(
            student_scores,
            teacher_target,
            temperature=args.distill_temperature,
            hard_weight=args.hard_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.grad_clip)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        teacher_matches.append(diagnostics["teacher_pair_match"])
        teacher_regrets.append(diagnostics["teacher_local_regret"])
        teacher_candidate_rows += int(batch.targets.numel() * model.router.candidate_pool)
        teacher_pair_state_evals += int(batch.targets.numel() * selector.pair_count)
        if device.type == "cuda":
            peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated(device) // 2**20))
        if args.log_every and (
            step_number == 1
            or step_number % args.log_every == 0
            or step_number == args.steps
        ):
            print(
                f"[signature] seed={config['seed']} step={step_number}/{args.steps} "
                f"loss={losses[-1]:.4f} match={teacher_matches[-1]:.3f} "
                f"local_regret={teacher_regrets[-1]:.4f}"
            )

    _sync(device)
    seconds = time.perf_counter() - start
    return {
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "seconds": seconds,
        "peak_vram_mb": peak_vram,
        "mean_last_100_loss": sum(losses[-100:]) / min(100, len(losses)),
        "mean_last_100_teacher_pair_match": sum(teacher_matches[-100:]) / min(100, len(teacher_matches)),
        "mean_last_100_teacher_local_regret": sum(teacher_regrets[-100:]) / min(100, len(teacher_regrets)),
        "teacher_candidate_real_circuit_rows_evaluated": teacher_candidate_rows,
        "teacher_pair_gru_head_evaluations": teacher_pair_state_evals,
        "teacher_full_bank_real_circuit_rows_evaluated": 0,
        "teacher_scope": "candidate-only M real outputs; training only",
        "teacher_target": args.teacher_target,
    }


def _usage_update(usage: torch.Tensor, selected: torch.Tensor) -> None:
    ids = selected.reshape(-1)
    ids = ids[ids.ge(0)]
    if ids.numel():
        usage += torch.bincount(ids, minlength=usage.numel()).to(usage.device)


@torch.inference_mode()
def evaluate_quality(
    model,
    selector: SparseOutputSignatureSelector,
    batches: list[Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    selector.eval()
    n = int(model.router.num_circuits)
    control_usage = torch.zeros(n, dtype=torch.long, device=device)
    sparse_usage = torch.zeros(n, dtype=torch.long, device=device)
    control_loss = sparse_loss = 0.0
    control_correct = sparse_correct = 0
    examples = 0
    candidate_same = candidate_total = 0
    route_same = route_total = 0

    warmup = batches[0]
    model(warmup.inputs, adaptive=False)
    sparse_selector_rollout(model, selector, warmup.inputs)
    _sync(device)

    start = time.perf_counter()
    control_cache: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for batch in batches:
        logits, stats = model(batch.inputs, adaptive=False)
        control_cache.append((logits, stats))
        control_loss += float(F.cross_entropy(logits, batch.targets, reduction="sum").cpu())
        control_correct += int(logits.argmax(-1).eq(batch.targets).sum())
        _usage_update(control_usage, stats["selected_ids"])
        examples += int(batch.targets.numel())
    _sync(device)
    control_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for batch, (control_logits, control_stats) in zip(batches, control_cache):
        logits, stats, _ = sparse_selector_rollout(model, selector, batch.inputs)
        sparse_loss += float(F.cross_entropy(logits, batch.targets, reduction="sum").cpu())
        sparse_correct += int(logits.argmax(-1).eq(batch.targets).sum())
        _usage_update(sparse_usage, stats["selected_ids"])
        candidate_same += int(stats["candidate_ids"].eq(control_stats["candidate_ids"]).sum())
        candidate_total += int(stats["candidate_ids"].numel())
        route_same += int(stats["selected_ids"].eq(control_stats["selected_ids"]).sum())
        route_total += int(stats["selected_ids"].numel())
    _sync(device)
    sparse_seconds = time.perf_counter() - start

    return {
        "control": {
            "heldout_ce": control_loss / examples,
            "hard_accuracy": control_correct / examples,
            "seconds": control_seconds,
            "seconds_per_example": control_seconds / examples,
            "circuit_usage": control_usage.cpu().tolist(),
            "used_circuits": int(control_usage.gt(0).sum().cpu()),
            "dead_circuits": int(control_usage.eq(0).sum().cpu()),
        },
        "sparse_signature": {
            "heldout_ce": sparse_loss / examples,
            "hard_accuracy": sparse_correct / examples,
            "seconds": sparse_seconds,
            "seconds_per_example": sparse_seconds / examples,
            "circuit_usage": sparse_usage.cpu().tolist(),
            "used_circuits": int(sparse_usage.gt(0).sum().cpu()),
            "dead_circuits": int(sparse_usage.eq(0).sum().cpu()),
        },
        "examples": examples,
        "cascade_candidate_slot_stability": candidate_same / max(candidate_total, 1),
        "selected_route_slot_stability": route_same / max(route_total, 1),
        "latency_ratio": sparse_seconds / max(control_seconds, 1e-12),
        "delta": {
            "heldout_ce": sparse_loss / examples - control_loss / examples,
            "hard_accuracy_pp": 100.0 * (sparse_correct - control_correct) / examples,
        },
    }


def _pair_index_from_positions(positions: torch.Tensor, pair_map: dict[tuple[int, int], int]) -> torch.Tensor:
    canonical = positions.sort(dim=-1).values.cpu()
    return torch.tensor(
        [pair_map[tuple(row.tolist())] for row in canonical],
        device=positions.device,
        dtype=torch.long,
    )


def _regret_summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(tensor.mean()),
        "p95": float(torch.quantile(tensor, 0.95)),
    }


@torch.inference_mode()
def selector_ablation_diagnostic(
    model,
    selector: SparseOutputSignatureSelector,
    batches: list[Any],
) -> dict[str, Any]:
    """Isolate output, pair, GRU and output-head contributions on frozen states."""
    selector.eval()
    m = int(model.router.candidate_pool)
    if m != selector.candidate_pool:
        raise ValueError("selector/candidate-pool mismatch")
    pair_positions = list(itertools.combinations(range(m), 2))
    pair_map = {pair: index for index, pair in enumerate(pair_positions)}
    methods = (
        "key_score",
        "individual_output_additive",
        "joint_output_no_gru",
        "joint_gru_state_norm",
        "full_local_gru_head",
        "sparse_signature",
    )
    regrets = {name: [] for name in methods}
    correct = {name: 0 for name in methods}
    candidate_correct = 0
    decisions = 0

    for batch in batches:
        encoded = model.encode(batch.inputs)
        _, stats = model(batch.inputs, adaptive=False)
        base_ids = stats["selected_ids"]
        for step in range(model.internal_steps):
            query = stats["query_states"][:, step]
            context = SparseStepContext(
                step=step,
                query=query,
                state_before=query - model.step_embedding[step],
                encoded=encoded,
                candidate_ids=stats["candidate_ids"][:, step],
                route_gain=stats["route_gains"][:, step],
            )
            teacher = candidate_local_teacher(model, context, batch.targets)
            candidates = context.candidate_ids
            batch_size = batch.targets.numel()
            student_scores, _ = selector(query, candidates)

            key_logits = torch.einsum(
                "bd,bmd->bm", query, model.router.keys[candidates]
            ) / math.sqrt(query.shape[-1])
            key_positions = key_logits.topk(2, dim=-1).indices
            chosen = {
                "key_score": _pair_index_from_positions(key_positions, pair_map),
                "individual_output_additive": teacher["individual_additive_losses"].argmin(dim=-1),
                "joint_output_no_gru": teacher["no_gru_losses"].argmin(dim=-1),
                "joint_gru_state_norm": teacher["gru_state_change"].argmax(dim=-1),
                "full_local_gru_head": teacher["full_local_losses"].argmin(dim=-1),
                "sparse_signature": student_scores.argmax(dim=-1),
            }

            pair_final_losses = []
            pair_final_logits = []
            for left, right in pair_positions:
                pair_ids = torch.stack([candidates[:, left], candidates[:, right]], dim=-1)
                plan = _step_plan(base_ids, step, pair_ids)
                logits, _ = model(batch.inputs, adaptive=False, forced_selected_ids=plan)
                pair_final_logits.append(logits)
                pair_final_losses.append(
                    F.cross_entropy(logits, batch.targets, reduction="none")
                )
            losses = torch.stack(pair_final_losses, dim=1)
            logits = torch.stack(pair_final_logits, dim=1)
            best_loss, best_index = losses.min(dim=1)
            rows = torch.arange(batch_size, device=query.device)
            candidate_correct += int(
                logits[rows, best_index].argmax(-1).eq(batch.targets).sum()
            )
            decisions += batch_size
            for name, index in chosen.items():
                selected_loss = losses[rows, index]
                regrets[name].extend((selected_loss - best_loss).clamp_min(0).cpu().tolist())
                correct[name] += int(
                    logits[rows, index].argmax(-1).eq(batch.targets).sum()
                )

    return {
        "decisions": decisions,
        "candidate_oracle_accuracy": candidate_correct / decisions,
        "methods": {
            name: {
                "final_accuracy": correct[name] / decisions,
                "selection_regret": _regret_summary(regrets[name]),
            }
            for name in methods
        },
        "interpretation": {
            "key_to_individual": "gain from real circuit-output awareness without joint pair cost",
            "individual_to_full_local": "additional gain from joint pair interaction plus recurrent/head processing",
            "no_gru_to_full_local": "increment attributable to the learned immediate GRU transition",
            "state_norm_to_full_local": "importance of output-head/label alignment beyond state-change magnitude",
            "full_local_to_sparse_signature": "distillation gap paid to obtain sparse inference",
        },
    }


def _reduction(before: float, after: float) -> float:
    if before <= 1e-12:
        return 0.0 if after <= 1e-12 else -math.inf
    return (before - after) / before


def run_checkpoint(path: Path, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    model, config = _checkpoint(path, device)
    validate_plain_sparse_contract(model)
    if int(model.router.num_circuits) != 32 or int(model.router.candidate_pool) != 8:
        raise ValueError("Handoff D full protocol requires E=32 and M=8")
    if int(model.active_circuits) != 2 or int(model.internal_steps) != 3:
        raise ValueError("Handoff D full protocol requires active=2 and T=3")
    seed = int(config["seed"])
    selector = _make_selector(model, args, seed, device)
    training = train_selector(model, config, selector, device, args)
    batches = _heldout_batches(config, device, args.eval_batches, args.examples_per_task)
    quality = evaluate_quality(model, selector, batches, device)
    diagnostic_batches = batches[: min(args.diagnostic_batches, len(batches))]
    ablation = selector_ablation_diagnostic(model, selector, diagnostic_batches)

    retrieval = retrieval_window_audit(
        path,
        device,
        args.retrieval_batches,
        args.retrieval_examples_per_task,
        [8],
    )["widths"][0]
    baseline_regret = ablation["methods"]["key_score"]["selection_regret"]
    sparse_regret = ablation["methods"]["sparse_signature"]["selection_regret"]
    return {
        "checkpoint": str(path),
        "checkpoint_training_steps": _checkpoint_steps(path),
        "seed": seed,
        "protocol": {
            "bank": 32,
            "candidate_pool": 8,
            "active": 2,
            "internal_steps": 3,
            "retriever_changed": False,
            "circuit_bank_changed": False,
            "dense_full_bank_inference": False,
        },
        "training": training,
        "quality": quality,
        "ablation": ablation,
        "retrieval_frozen_state": {
            "candidate_recall": retrieval["candidate_recall"],
            "candidate_recall_delta_pp": 0.0,
            "retrieval_regret_mean": retrieval["retrieval_regret_mean"],
            "retrieval_regret_p95": retrieval["retrieval_regret_p95"],
            "note": "retrieval mechanism/candidate IDs on the isolated frozen-state audit are unchanged",
        },
        "cost": {
            "inference": selector_cost_report(model, selector),
            "training_teacher": {
                "candidate_real_rows_per_teacher_decision": 8,
                "pair_gru_head_evals_per_teacher_decision": selector.pair_count,
                "full_bank_real_rows_per_teacher_decision": 0,
            },
        },
        "delta": {
            "hard_accuracy_pp": quality["delta"]["hard_accuracy_pp"],
            "heldout_ce": quality["delta"]["heldout_ce"],
            "latency_ratio": quality["latency_ratio"],
            "selection_regret_mean_reduction": _reduction(
                baseline_regret["mean"], sparse_regret["mean"]
            ),
            "selection_regret_p95_reduction": _reduction(
                baseline_regret["p95"], sparse_regret["p95"]
            ),
        },
        "selector_state": {key: value.detach().cpu() for key, value in selector.state_dict().items()},
    }


def gate(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    seeds = sorted(item["seed"] for item in results)
    full_steps = all(
        item["checkpoint_training_steps"] is None
        or int(item["checkpoint_training_steps"]) >= 5000
        for item in results
    ) and args.steps >= 5000
    acc = [item["delta"]["hard_accuracy_pp"] for item in results]
    mean_regret = [item["delta"]["selection_regret_mean_reduction"] for item in results]
    p95_regret = [item["delta"]["selection_regret_p95_reduction"] for item in results]
    latency = [item["delta"]["latency_ratio"] for item in results]
    recall_delta = [item["retrieval_frozen_state"]["candidate_recall_delta_pp"] for item in results]
    sparse_cost_ok = all(
        item["cost"]["inference"]["full_bank_real_circuit_rows_scored_per_decision"] == 0
        and item["cost"]["inference"]["candidate_real_circuit_rows_scored_at_inference"] == 0
        and item["cost"]["inference"]["real_circuit_rows_executed_per_decision"] == 2
        for item in results
    )
    accuracy_gate = mean(acc) >= 2.0 and min(acc) >= 0.0
    regret_gate = mean(mean_regret) >= 0.10 and mean(p95_regret) >= 0.10
    recall_guard = min(recall_delta) >= 0.0
    latency_guard = max(latency) <= 1.25
    protocol_guard = seeds == [17, 18] and full_steps and sparse_cost_ok
    accepted = accuracy_gate and regret_gate and recall_guard and latency_guard and protocol_guard
    return {
        "accepted": accepted,
        "decision": "PASS" if accepted else "REJECTED",
        "accuracy_gate": accuracy_gate,
        "regret_gate": regret_gate,
        "recall_guard": recall_guard,
        "latency_guard": latency_guard,
        "protocol_guard": protocol_guard,
        "mean_hard_accuracy_delta_pp": mean(acc),
        "mean_selection_regret_reduction": mean(mean_regret),
        "mean_p95_selection_regret_reduction": mean(p95_regret),
        "mean_latency_ratio": mean(latency),
        "max_latency_ratio": max(latency),
        "criteria": {
            "accuracy": "seed17/18 mean hard accuracy >= +2 pp and neither seed negative",
            "regret": "mean and p95 candidate-selection regret each improve >= 10%",
            "recall": "isolated frozen-state candidate recall cannot decrease",
            "latency": "each seed sparse inference latency <= 1.25x control",
            "protocol": "seed17/18, 5000-step selector, E=32/M=8/active=2/T=3, no dense full-bank inference",
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Handoff D — Sparse output-signature selector",
        "",
        f"**Decision:** `{report['decision']['decision']}`",
        "",
        "Dense full-bank inference is forbidden. The treatment keeps the existing retriever and circuit bank, scores only M=8 miniature signatures, then executes two real circuits.",
        "",
        "## Quality / cost",
        "",
        "| Seed | Control acc | Sparse acc | Δ acc | Δ CE | Mean regret red. | P95 regret red. | Recall Δ | Latency | Selector touched/decision |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["results"]:
        q = item["quality"]
        d = item["delta"]
        cost = item["cost"]["inference"]
        lines.append(
            f"| {item['seed']} | {q['control']['hard_accuracy']*100:.2f}% | "
            f"{q['sparse_signature']['hard_accuracy']*100:.2f}% | {d['hard_accuracy_pp']:+.2f} pp | "
            f"{d['heldout_ce']:+.4f} | {d['selection_regret_mean_reduction']*100:+.1f}% | "
            f"{d['selection_regret_p95_reduction']*100:+.1f}% | "
            f"{item['retrieval_frozen_state']['candidate_recall_delta_pp']:+.1f} pp | "
            f"{d['latency_ratio']:.3f}x | {cost['selector_total_touched_params_per_decision']:,} |"
        )
    lines += ["", "## Proxy component ablation", ""]
    for item in report["results"]:
        lines += [
            f"### Seed {item['seed']}",
            "",
            "| Method | Final acc | Mean regret | P95 regret |",
            "|---|---:|---:|---:|",
        ]
        for name, values in item["ablation"]["methods"].items():
            r = values["selection_regret"]
            lines.append(
                f"| {name} | {values['final_accuracy']*100:.2f}% | {r['mean']:.4f} | {r['p95']:.4f} |"
            )
        lines.append("")
    decision = report["decision"]
    lines += ["## Gate", ""]
    for key in ("accuracy", "regret", "recall", "latency", "protocol"):
        flag_name = f"{key}_gate" if f"{key}_gate" in decision else f"{key}_guard"
        lines.append(f"- {key}: `{decision[flag_name]}` — {decision['criteria'][key]}.")
    lines += [
        "",
        "## Reproduction",
        "",
        "```bash",
        report["command"],
        "```",
        "",
    ]
    return "\n".join(lines)


def append_rejection(report: dict[str, Any], path: Path) -> None:
    if report["decision"]["accepted"]:
        return
    marker = "### C-P001-OUTPUT-SIGNATURE-001 — Sparse output-aware selector"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    d = report["decision"]
    block = (
        f"\n\n{marker}\n\n**Status:** `REJECTED`  \n**Muammo:** P-001 / Handoff D  \n"
        f"**Natija:** seed17/18 mean hard accuracy delta `{d['mean_hard_accuracy_delta_pp']:+.3f} pp`, "
        f"mean selection-regret reduction `{d['mean_selection_regret_reduction']*100:.2f}%`, "
        f"p95 reduction `{d['mean_p95_selection_regret_reduction']*100:.2f}%`, "
        f"max latency `{d['max_latency_ratio']:.3f}x`. Gate bajarilmadi. "
        "Dense full-bank inference ishlatilmadi; P-001 `ACTIVE` qoladi.\n"
    )
    path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Handoff D sparse output-signature benchmark")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--signature-rank", type=int, default=2)
    parser.add_argument("--signature-dim", type=int, default=8)
    parser.add_argument("--init-from-bank", action="store_true")
    parser.add_argument("--freeze-signature", action="store_true")
    parser.add_argument(
        "--teacher-target",
        choices=("full_local_losses", "individual_additive_losses"),
        default="full_local_losses",
    )
    parser.add_argument("--key-prior-weight", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=0.10)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--diagnostic-batches", type=int, default=2)
    parser.add_argument("--retrieval-batches", type=int, default=2)
    parser.add_argument("--retrieval-examples-per-task", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--output", default="results/runs/p001_sparse_output_signature_s17_s18.json")
    parser.add_argument("--markdown", default="results/P001_SPARSE_OUTPUT_SIGNATURE_S17_S18.md")
    parser.add_argument("--selector-dir", default="results/checkpoints/p001_sparse_output_signature")
    parser.add_argument("--update-problems-on-reject", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.steps = min(args.steps, 6)
        args.warmup_steps = min(args.warmup_steps, 2)
        args.train_batch_size = min(args.train_batch_size, 16)
        args.eval_batches = 1
        args.examples_per_task = 2
        args.diagnostic_batches = 1
        args.retrieval_batches = 1
        args.retrieval_examples_per_task = 1
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    paths = [Path(value) for value in args.checkpoint]
    results = [run_checkpoint(path, device, args) for path in paths]
    decision = gate(results, args)
    command = (
        "python benchmark_p001_sparse_output_signature.py "
        + " ".join(f"--checkpoint {path}" for path in paths)
        + f" --device {args.device} --steps {args.steps}"
        + f" --signature-rank {args.signature_rank} --signature-dim {args.signature_dim}"
        + (" --init-from-bank" if args.init_from_bank else "")
        + (" --freeze-signature" if args.freeze_signature else "")
        + f" --teacher-target {args.teacher_target}"
        + f" --key-prior-weight {args.key_prior_weight}"
        + " --update-problems-on-reject"
    )

    serializable_results = copy.deepcopy(results)
    selector_states = []
    for item in serializable_results:
        selector_states.append((item["seed"], item.pop("selector_state")))
    report = {
        "experiment": "Handoff D sparse output-signature selector",
        "hypothesis": "candidate-only distilled output signatures retain local proxy selection quality without dense bank inference",
        "results": serializable_results,
        "decision": decision,
        "command": command,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown(report), encoding="utf-8")

    selector_dir = Path(args.selector_dir)
    selector_dir.mkdir(parents=True, exist_ok=True)
    for seed, state in selector_states:
        torch.save(state, selector_dir / f"seed{seed}_selector.pt")
    if args.update_problems_on_reject:
        append_rejection(report, Path("problems.md"))
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
