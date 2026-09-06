"""P001--P006 bounded diagnostic and bank-row credit control experiment.

No router architecture/weights, recurrent controller or output-head changes.
Only existing circuit rows can train. Use CPU by default to avoid GPU jobs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_capacity_route_oracle import _checkpoint
from benchmark_probe_router import common_oracle, oracle_metrics, source, latency
from neural_engine.sparse_audit import (CircuitLedger, final_margin_loss,
                                        sparse_cost_report, sparse_row_credit,
                                        training_phase_budget)


def replay_args(stats, rows, step, replacements):
    selected = stats["selected_ids"][rows]
    plan = torch.full_like(selected, -1)
    plan[:, :step] = selected[:, :step]
    plan[:, step] = replacements
    return plan, stats["selected_weights"][rows].detach(), stats["route_gains"][rows].detach()


@torch.inference_mode()
def quality(model, batches):
    count, ce, correct = 0, 0.0, 0
    usage = torch.zeros(model.circuits.num_circuits, dtype=torch.long)
    task_correct, task_count = torch.zeros(15), torch.zeros(15)
    for batch in batches:
        logits, stats = model(batch.inputs, adaptive=False)
        ce += float(F.cross_entropy(logits, batch.targets, reduction="sum"))
        hit = logits.argmax(-1).eq(batch.targets)
        correct += int(hit.sum())
        count += batch.targets.numel()
        usage += torch.bincount(stats["selected_ids"].flatten().cpu(), minlength=usage.numel())
        task_correct.scatter_add_(0, batch.task_ids.cpu(), hit.float().cpu())
        task_count.scatter_add_(0, batch.task_ids.cpu(), torch.ones_like(hit, dtype=torch.float).cpu())
    return {"ce": ce / count, "hard_accuracy": correct / count, "examples": count,
            "usage": usage.tolist(), "used_circuits": int(usage.gt(0).sum()),
            "dead_circuits": int(usage.eq(0).sum()),
            "task_accuracy": (task_correct / task_count.clamp_min(1)).tolist()}


@torch.inference_mode()
def functional_ablation(model, batch):
    """Conditional contribution, not proof of independent learned algorithms.

    Remove one routed row's weighted contribution, without renormalization,
    with the entire native route/weights/gains otherwise fixed. No row IDs are
    matched between independently trained models of different bank sizes.
    """
    baseline, stats = model(batch.inputs, adaptive=False)
    base_loss = F.cross_entropy(baseline, batch.targets, reduction="none")
    ids = stats["selected_ids"]
    E = model.circuits.num_circuits
    sums, counts = torch.zeros(E, 15), torch.zeros(E, 15, dtype=torch.long)
    for circuit in ids.unique().tolist():
        rows = ids.eq(circuit).any(-1).any(-1)
        weights = stats["selected_weights"][rows].clone()
        weights[ids[rows].eq(circuit)] = 0
        logits, _ = model(batch.inputs[rows], adaptive=False,
                           forced_selected_ids=ids[rows], forced_selected_weights=weights,
                           forced_route_gains=stats["route_gains"][rows])
        delta = F.cross_entropy(logits, batch.targets[rows], reduction="none") - base_loss[rows]
        tasks = batch.task_ids[rows].cpu()
        sums[circuit].scatter_add_(0, tasks, delta.cpu())
        counts[circuit].scatter_add_(0, tasks, torch.ones_like(tasks))
    means = sums / counts.clamp_min(1)
    observed = counts.ge(2)
    return {"task_conditional_ablation_ce": means.tolist(), "task_observations": counts.tolist(),
            "observed_useful_circuit_task_cells": int((means.gt(0.01) & observed).sum()),
            "definition": "CE without row minus CE with row; fixed complete route; absent cells unobserved, not zero effect",
            "specialization_proven": False,
            "limitation": "Conditional ablation is not identification of each parameter's causal contribution or cross-model circuit alignment."}


@torch.inference_mode()
def cascade_diagnostic(model, batch):
    logits, stats = model(batch.inputs, adaptive=False)
    base_ce = F.cross_entropy(logits, batch.targets, reduction="none")
    result = []
    for t in range(model.internal_steps):
        current = stats["selected_ids"][:, t]
        alternative = current.clone()
        for b, candidates in enumerate(stats["candidate_ids"][:, t]):
            options = [x for x in candidates.tolist() if x not in current[b].tolist()]
            alternative[b, -1] = options[0]
        plan, weights, gains = replay_args(stats, slice(None), t, alternative)
        free_logits, free_stats = model(batch.inputs, adaptive=False, forced_selected_ids=plan,
                                        forced_selected_weights=weights, forced_route_gains=gains)
        fixed_plan = stats["selected_ids"].clone()
        fixed_plan[:, t] = alternative
        fixed_logits, _ = model(batch.inputs, adaptive=False, forced_selected_ids=fixed_plan,
                                 forced_selected_weights=weights, forced_route_gains=gains)
        free_ce = F.cross_entropy(free_logits, batch.targets, reduction="none")
        fixed_ce = F.cross_entropy(fixed_logits, batch.targets, reduction="none")
        free_suffix = free_stats["selected_ids"][:, t + 1:]
        native_suffix = stats["selected_ids"][:, t + 1:]
        result.append({"changed_step": t,
                       "prefix_query_max_error": float((free_stats["query_states"][:, :t + 1] - stats["query_states"][:, :t + 1]).abs().max()),
                       "final_ce_gain_natural_suffix": float((base_ce - free_ce).mean()),
                       "final_ce_gain_fixed_suffix": float((base_ce - fixed_ce).mean()),
                       "suffix_rerouting_ce_effect": float((free_ce - fixed_ce).mean()),
                       "suffix_route_change_fraction": float(free_suffix.ne(native_suffix).float().mean()) if free_suffix.numel() else 0.0,
                       "per_example_final_ce_gain": (base_ce - free_ce).tolist()})
    return {"steps": result,
            "note": "One candidate swap with identical original coefficients. Fixed suffix is diagnostic only; not a trajectory oracle."}


@torch.inference_mode()
def calibration_drift(initial, current_model, batch):
    """Same input and SAME alternative IDs, re-labelled on current states."""
    _, reference = initial(batch.inputs, adaptive=False)
    old_values, new_values = [], []
    for t in range(initial.internal_steps):
        original = reference["selected_ids"][:, t]
        alternate = original.clone()
        for b, candidates in enumerate(reference["candidate_ids"][:, t]):
            alternate[b, -1] = next(x for x in candidates.tolist() if x not in original[b].tolist())
        for model, values in ((initial, old_values), (current_model, new_values)):
            _, stats = model(batch.inputs, adaptive=False)
            losses = []
            for pair in (original, alternate):
                plan, weights, gains = replay_args(stats, slice(None), t, pair)
                weights, gains = weights.clone(), gains.clone()
                weights[:, t], gains[:, t] = 0.5, 1.0
                logits, _ = model(batch.inputs, adaptive=False, forced_selected_ids=plan,
                                   forced_selected_weights=weights, forced_route_gains=gains)
                losses.append(F.cross_entropy(logits, batch.targets, reduction="none"))
            values.append(losses[0] - losses[1])
    old, new = torch.cat(old_values), torch.cat(new_values)
    decisive = old.abs().gt(0.02)
    return {"mean_absolute_target_drift": float((old - new).abs().mean()),
            "sign_agreement_on_old_decisive_targets": float(old[decisive].sign().eq(new[decisive].sign()).float().mean()) if decisive.any() else None,
            "decisive_targets": int(decisive.sum()),
            "definition": "Same input, same two route IDs, uniform changed-step coefficients; new natural prefix/suffix and bank. Diagnostic, not a proof that DAgger helps."}


def calibration_refit(base, config, mode, args):
    """Static calibration versus fresh on-policy re-labelling, frozen bank.

    Same eight input batches, schedule, update count and original key head.
    Only existing router keys train. Target is final output hinge improvement.
    Static arm also computes a fresh diagnostic rollout, but uses cached labels;
    this charges equal replay work rather than comparing unequal compute.
    """
    model = copy.deepcopy(base)
    for p in model.parameters():
        p.requires_grad_(False)
    model.router.keys.requires_grad_(True)
    device = next(model.parameters()).device
    data = source(config, device, int(config["seed"]) + 31001, "train")
    batches = [data.balanced(args.train_examples_per_task) for _ in range(8)]
    optimizer = torch.optim.AdamW([model.router.keys], lr=args.lr, weight_decay=0.0)

    @torch.no_grad()
    def collect(batch, t):
        _, stats = model(batch.inputs, adaptive=False)
        pair = stats["selected_ids"][:, t]
        alt = pair.clone()
        for b, pool in enumerate(stats["candidate_ids"][:, t]):
            alt[b, -1] = next(i for i in pool.tolist() if i not in pair[b].tolist())
        losses = []
        for ids in (pair, alt):
            plan, weights, gains = replay_args(stats, slice(None), t, ids)
            weights, gains = weights.clone(), gains.clone()
            weights[:, t], gains[:, t] = 0.5, 1.0
            logits, _ = model(batch.inputs, adaptive=False, forced_selected_ids=plan,
                               forced_selected_weights=weights, forced_route_gains=gains)
            losses.append(final_margin_loss(logits, batch.targets))
        return stats["query_states"][:, t].detach(), pair, alt, (losses[0] - losses[1]).detach()

    cached = {(b, t): collect(batch, t) for b, batch in enumerate(batches) for t in range(3)}
    for step in range(args.calibration_steps):
        b, t = step % 8, step % 3
        fresh = collect(batches[b], t)
        query, pair, alt, delta = cached[b, t] if mode == "static" else fresh
        def score(ids):
            return (query[:, None] * model.router.keys[ids]).sum(-1).mean(-1) / query.shape[-1] ** 0.5
        loss = F.smooth_l1_loss(score(alt) - score(pair), delta, beta=0.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.router.keys], 1.0)
        optimizer.step()
    for n, p in model.named_parameters():
        if n != "router.keys" and not torch.equal(p, dict(base.named_parameters())[n]):
            raise AssertionError(f"calibration changed {n}")
    return model


def train_credit(base, config, arm, args):
    model = copy.deepcopy(base)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("circuits."))
    model.eval()  # freeze stochastic policy; existing bank rows still receive gradients
    unchanged = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("circuits.")}
    device = next(model.parameters()).device
    seed = int(config["seed"])
    rng = torch.Generator().manual_seed(seed + 17001)
    data = source(config, device, seed + 18001, "train")
    ledger = CircuitLedger(model)
    optimizer = torch.optim.AdamW(model.circuits.parameters(), lr=args.lr, weight_decay=0.0)
    sampled = torch.zeros(model.circuits.num_circuits, dtype=torch.long)
    try:
        for step in range(args.steps):
            batch = data.balanced(args.train_examples_per_task)
            optimizer.zero_grad(set_to_none=True)
            with ledger.phase("main", batch.task_ids):
                logits, stats = model(batch.inputs, adaptive=False)
                F.cross_entropy(logits, batch.targets).backward()
            # Identical number of probe forward/backward trajectories in every arm.
            rows = torch.randperm(len(batch.targets), generator=rng)[:max(1, len(batch.targets) // 4)].to(device)
            t = step % model.internal_steps
            current = stats["selected_ids"][rows, t].detach()
            alternate = current.clone()
            picked = []
            for b in range(len(rows)):
                weights = torch.ones(model.circuits.num_circuits)
                if arm != "uniform_margin":
                    weights = 0.5 / model.circuits.num_circuits + 0.5 * (
                        (ledger.grad_steps.float() + 1).rsqrt() / (ledger.grad_steps.float() + 1).rsqrt().sum())
                weights[current[b].cpu()] = 0
                p = int(torch.multinomial(weights, 1, generator=rng))
                alternate[b, -1] = p
                picked.append(p)
                sampled[p] += 1
            plan, weights, gains = replay_args(stats, rows, t, alternate)
            objective = "ce" if arm == "underused_ce" else "margin"
            with ledger.phase("training_probe", batch.task_ids[rows]):
                _, credit = sparse_row_credit(model, batch.inputs[rows], batch.targets[rows],
                                                plan, weights, gains,
                                                torch.tensor(picked, device=device), objective)
            if arm != "control":
                for p, index, gradient in credit:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.index_add_(0, index, args.credit_weight * gradient)
            torch.nn.utils.clip_grad_norm_(model.circuits.parameters(), 1.0)
            ledger.before_step()
            optimizer.step()
            ledger.after_step()
            if args.log_every and (step + 1) % args.log_every == 0:
                print(f"seed={seed} {arm} {step + 1}/{args.steps}", flush=True)
        for n, p in model.named_parameters():
            if n in unchanged and not torch.equal(p, unchanged[n]):
                raise AssertionError(f"non-bank weight changed: {n}")
        return model, {**ledger.report(), "sampled_credit_count": sampled.tolist(),
                       "non_bank_weights_unchanged": True,
                       "probe_objective": objective,
                       "auxiliary_weight": 0 if arm == "control" else args.credit_weight}
    finally:
        ledger.close()


def evaluate_arm(model, tests, audit, args):
    q = quality(model, tests)
    pairs, costs, queries = common_oracle(model, audit, 32)
    metric = oracle_metrics(model.router, pairs, costs, queries)
    metric["reference"] = "This arm's own states/bank/suffix; exhaustive ONE-decision pair oracle, not global trajectory oracle. Cross-arm gap shrinkage alone is not improvement."
    with torch.no_grad():
        _, stats = model(tests[0].inputs, adaptive=False)
    return {**q, **metric, "sparse_audit_v2": sparse_cost_report(model, tests[0].inputs, stats),
            "latency": latency(model, tests[0].inputs, 30),
            "functional_ablation": functional_ablation(model, audit),
            "cascade": cascade_diagnostic(model, audit)}


def scale_audit(directory, args):
    """Matched evaluation of available checkpoints, explicitly NOT matched training."""
    names = ("ne20_v12_full.pt", "ne50_v12_coverage_full.pt", "ne100_v12_coverage_full.pt")
    results = []
    for name in names:
        path = directory / name
        if not path.exists():
            results.append({"checkpoint": name, "status": "missing"})
            continue
        print(f"capacity read-only audit: {name}", flush=True)
        model, config = _checkpoint(path, torch.device(args.device))
        data = source(config, torch.device(args.device), 27001, "heldout")
        batches = [data.balanced(8) for _ in range(args.capacity_batches)]
        with torch.no_grad():
            _, stats = model(batches[0].inputs, adaptive=False)
        r = model.router
        theoretical = min(r.num_circuits, r.branch ** r.active_depth + r.candidates_per_address - 1)
        natural = quality(model, batches)
        # Paired depth-shortening diagnostic on the SAME weights/data.
        depth_control = None
        if r.active_depth > 4:
            r.set_routing_state(depth=4)
            depth_control = quality(model, batches)
            r.set_routing_state(depth=config["router_depth"])
        results.append({"checkpoint": name, "config": config, "quality": natural,
                        "sparse_audit_v2": sparse_cost_report(model, batches[0].inputs, stats),
                        "theoretically_reachable_circuits": theoretical,
                        "depth4_same_weight_control": depth_control,
                        "historical_gradient_update_counts": None,
                        "protocol_warning": "Available v12 checkpoints are not the original 53.62/54.77/54.43% runs. NE20 lacks the NE50/100 coverage regularizer; all-split historical training means this is not a clean unseen-training split. No causal per-new-parameter alignment."})
    return {"results": results, "matched_evaluation": True, "matched_training": False,
            "unresolved": ["5000-step sufficiency requires matched learning curves/continuations",
                           "active-budget causality requires matched K ablation, not inference-only K changes",
                           "optimization stability requires repeated training seeds",
                           "new scalar parameter attribution is not identified across independent checkpoints"]}


def run(args):
    torch.set_num_threads(2)
    results = []
    directory = Path(args.checkpoint_dir)
    for seed in (17, 18):
        base, config = _checkpoint(directory / f"capacity_audit_c32_global_s{seed}.pt", torch.device(args.device))
        if (base.circuits.num_circuits, base.active_circuits, base.router.candidate_pool, base.internal_steps) != (32, 2, 8, 3):
            raise ValueError("credit screen requires E32/K2/M8/T3")
        data = source(config, torch.device(args.device), seed + 28001, "heldout")
        tests = [data.balanced(8) for _ in range(args.eval_batches)]
        audit = source(config, torch.device(args.device), seed + 29001, "heldout").balanced(2)
        print(f"seed={seed} frozen baseline audit", flush=True)
        arms = {"frozen": evaluate_arm(base, tests, audit, args)}
        for arm in ("control", "uniform_margin", "underused_ce", "underused_margin"):
            model, training = train_credit(base, config, arm, args)
            print(f"seed={seed} audit {arm}", flush=True)
            arms[arm] = evaluate_arm(model, tests, audit, args)
            arms[arm]["training"] = training
            # Directly remeasured stale-calibration target sign drift; never used to fit.
            arms[arm]["calibration_target_drift"] = calibration_drift(base, model, audit)
        for mode in ("static", "on_policy"):
            print(f"seed={seed} frozen-bank calibration {mode}", flush=True)
            calibrated = calibration_refit(base, config, mode, args)
            arms[f"calibration_{mode}"] = evaluate_arm(calibrated, tests, audit, args)
        results.append({"seed": seed, "arms": arms})
    comparisons = []
    for row in results:
        c, p = row["arms"]["control"], row["arms"]["underused_margin"]
        comparisons.append({"seed": row["seed"], "accuracy_gain": p["hard_accuracy"] - c["hard_accuracy"],
                            "ce_gain": c["ce"] - p["ce"],
                            "mean_regret_reduction": 1 - p["mean_regret_ce"] / max(c["mean_regret_ce"], 1e-9),
                            "p95_regret_reduction": 1 - p["p95_regret_ce"] / max(c["p95_regret_ce"], 1e-9),
                            "candidate_recall_gain": p["candidate_recall"] - c["candidate_recall"]})
    passed = (sum(r["accuracy_gain"] for r in comparisons) / 2 >= 0.02 and
              all(r["accuracy_gain"] >= 0 and r["ce_gain"] >= 0 and
                  r["mean_regret_reduction"] >= 0.1 and r["p95_regret_reduction"] >= 0.1 and
                  r["candidate_recall_gain"] >= 0 for r in comparisons))
    return {"experiment": "sparse_credit_and_cost_audit", "config": vars(args), "results": results,
            "comparisons": comparisons, "decision": "screen_pass_needs_confirmation" if passed else "reject_for_adoption",
            "gate": "mean accuracy +2pp, neither seed negative; CE noninferior; each seed mean/p95 one-decision regret -10%; recall noninferior",
            "capacity": scale_audit(directory, args) if args.capacity else {"status": "not_run"},
            "scope": "No router/body architecture changes, no dense-bank inference, no load-balancing loss. Bank-only sparse credit experiment; does not claim joint training/DAgger superiority or 20M-100M training-budget causality."}


def add_phase_budgets(result):
    """Add accounting only; never change recorded experiment metrics/gates."""
    config = result["config"]
    for seed in result["results"]:
        for name, arm in seed["arms"].items():
            if "training" in arm:
                arm["training_cost_v2"] = training_phase_budget(arm["training"], arm["sparse_audit_v2"])
            elif name.startswith("calibration_"):
                batch_size = 15 * config["train_examples_per_task"]
                initial = 8 * 3 * 3 * batch_size
                relabel = config["calibration_steps"] * 3 * batch_size
                macs = arm["sparse_audit_v2"]["inference_macs_per_example"][0]
                arm["training_cost_v2"] = {
                    "initial_calibration_reference_example_forwards": initial,
                    "fresh_diagnostic_relabel_example_forwards": relabel,
                    "forward_macs": (initial + relabel) * macs,
                    "router_only_backward_calls": config["calibration_steps"],
                    "body_backward_calls": 0,
                    "note": "Both static/on-policy arms pay the same fresh relabel forward cost. Static ignores fresh labels."
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--credit-weight", type=float, default=0.1)
    parser.add_argument("--calibration-steps", type=int, default=200)
    parser.add_argument("--train-examples-per-task", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--capacity-batches", type=int, default=16)
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--from-result", help="Accounting-only enrichment of a completed result; no training/evaluation")
    args = parser.parse_args()
    if min(args.steps, args.calibration_steps, args.train_examples_per_task, args.eval_batches, args.capacity_batches) < 1:
        parser.error("counts must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("choose a new output path; existing results are never overwritten")
    if args.from_result:
        result = json.loads(Path(args.from_result).read_text(encoding="utf-8"))
        result["accounting_source_result"] = args.from_result
    else:
        result = run(args)
    result = add_phase_budgets(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "comparisons": result["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
