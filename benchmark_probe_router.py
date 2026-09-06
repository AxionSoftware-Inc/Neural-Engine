"""Small paired frozen-bank screen. Defaults to CPU; never changes a checkpoint.

Example:
  python benchmark_probe_router.py --steps 2000 --output results/runs/coupled_probe_screen.json

old = original ProbeRoute objective; old-signed = architecture control with the
new objective; coupled = compact shared-utility router with the new objective.
All arms see the same fresh data, sample counts, rollout rule and update budget.
Exact oracle costs are EVALUATION ONLY on a separate, common reference policy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from audit_capacity_route_oracle import _checkpoint
from data.generator import SyntheticTaskGenerator
from neural_engine.coupled_probe import CoupledProbeRouter, signed_probe_loss
from neural_engine.router import ProbeRouteRouter
from probe_route_frozen import (_imitation_loss, _retrieval_inclusion_loss,
                                _sample_probe_pairs, _selection_loss)
from train import BatchSource


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def body_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith("router."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source(config, device, seed, split):
    prefix = "train" if split == "train" else "heldout"
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), seed,
        value_min=int(config.get(prefix + "_value_min", 0)),
        value_max=int(config.get(prefix + "_value_max", 63)), split=split)
    return BatchSource(generator, 128, device)


def subset_stats(stats, indices):
    return {name: stats[name][indices] for name in
            ("selected_ids", "selected_weights", "route_gains")}


@torch.no_grad()
def rollout(model, inputs, stats, step, pairs):
    """Replay the identical prefix, override one decision, re-route the suffix.

Uses only the public model API. No reconstruction of the recurrent body.
The prefix is recomputed (and charged), not falsely reported as cached.
"""
    plan = torch.full_like(stats["selected_ids"], -1)
    plan[:, :step] = stats["selected_ids"][:, :step]
    plan[:, step] = pairs
    weights = stats["selected_weights"].clone()
    gains = stats["route_gains"].clone()
    weights[:, step] = 0.5
    gains[:, step] = 1.0
    return model(inputs, adaptive=False, forced_selected_ids=plan,
                 forced_selected_weights=weights, forced_route_gains=gains)[0]


def new_model(teacher, seed, variant):
    model = copy.deepcopy(teacher)
    torch.manual_seed(seed + 1907)
    cls = CoupledProbeRouter if variant == "coupled" else ProbeRouteRouter
    model.router = cls(teacher.state_dim, 32, candidate_pool=8, active_circuits=2)
    # Match V0.180's old-router initialization from the hierarchical keys.
    if variant != "coupled":
        with torch.no_grad():
            model.router.keys.copy_(teacher.router.keys.cpu())
    model.to(next(teacher.parameters()).device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.router.parameters():
        parameter.requires_grad_(True)
    return model


def train_router(teacher, model, config, variant, args):
    device = next(model.parameters()).device
    seed = int(config["seed"])
    batches = source(config, device, seed + 1001, "train")
    rng = torch.Generator(device=device).manual_seed(seed + 991)
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    kind_counts = {"inside": 0, "outside": 0, "double": 0}
    loss_totals = {"selection": 0.0, "retrieval": 0.0}
    updates = 0
    forwarded_examples = 0
    start_digest = body_digest(model)
    synchronize(device)
    started = time.perf_counter()
    for step in range(args.steps):
        batch = batches.balanced(args.examples_per_task)
        optimizer.zero_grad(set_to_none=True)
        t = int(torch.randint(3, (), generator=rng, device=device))
        with torch.no_grad():
            current_logits, stats = (teacher if step < args.imitation_steps else model)(
                batch.inputs, adaptive=False)
            forwarded_examples += batch.targets.numel()
        query = stats["query_states"][:, t].detach()
        current = stats["selected_ids"][:, t]
        candidates = stats["candidate_ids"][:, t]
        if step < args.imitation_steps:
            loss = _imitation_loss(model.router, query, candidates, current, "full")
        else:
            alternative, kinds, mask = _sample_probe_pairs(model.router, current, candidates, rng)
            for kind, name in ((1, "inside"), (2, "outside"), (3, "double")):
                kind_counts[name] += int(kinds.eq(kind).sum())
            if not bool(mask.any()):
                continue
            # Reuse the ordinary CE; replay ONLY probed examples, not the full batch.
            with torch.no_grad():
                alt_logits = rollout(model, batch.inputs[mask], subset_stats(stats, mask),
                                     t, alternative[mask])
                forwarded_examples += int(mask.sum())
                delta = (F.cross_entropy(current_logits[mask], batch.targets[mask], reduction="none")
                         - F.cross_entropy(alt_logits, batch.targets[mask], reduction="none"))
            if not bool(torch.isfinite(delta).all()):
                raise RuntimeError("non-finite probe target")
            if variant == "old":
                full_delta = torch.zeros_like(kinds, dtype=query.dtype)
                full_delta[mask] = delta
                selection = _selection_loss(model.router, query, current, alternative, full_delta, mask)
                positive = mask & kinds.ge(2) & full_delta.gt(0.02)
                retrieval = _retrieval_inclusion_loss(
                    model.router, query, alternative,
                    (full_delta.abs() / 0.1).clamp_max(2) * positive)
                loss = selection + 0.1 * retrieval
                metrics = {"selection": selection.detach(), "retrieval": retrieval.detach()}
            else:
                loss, metrics = signed_probe_loss(model.router, query[mask], current[mask],
                                                  alternative[mask], delta)
            for name in loss_totals:
                loss_totals[name] += float(metrics[name])
            updates += 1
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite router loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.router.parameters(), 1.0)
        optimizer.step()
        if args.log_every and (step + 1) % args.log_every == 0:
            print(f"seed={seed} {variant} step={step + 1}/{args.steps}", flush=True)
    synchronize(device)
    seconds = time.perf_counter() - started
    if body_digest(model) != start_digest:
        raise AssertionError("frozen body changed")
    return {"training_seconds": seconds, "probe_counts": kind_counts,
            "forwarded_examples_including_replay": forwarded_examples,
            "circuit_row_forwards": forwarded_examples * 3 * 2,
            "mean_losses": {k: v / max(1, updates) for k, v in loss_totals.items()},
            "body_unchanged": True}


@torch.inference_mode()
def evaluate(model, batches):
    model.eval()
    ce, correct, count = 0.0, 0, 0
    usage = torch.zeros(32, device=next(model.parameters()).device, dtype=torch.long)
    union_counts = []
    for batch in batches:
        logits, stats = model(batch.inputs, adaptive=False)
        ce += float(F.cross_entropy(logits, batch.targets, reduction="sum"))
        correct += int(logits.argmax(-1).eq(batch.targets).sum())
        count += batch.targets.numel()
        ids = stats["selected_ids"]
        usage += torch.bincount(ids.reshape(-1), minlength=32)
        union_counts.extend(float(row.unique().numel()) for row in ids.reshape(-1, 6))
    return {"ce": ce / count, "hard_accuracy": correct / count, "examples": count,
            "dead_circuits": int(usage.eq(0).sum()), "circuit_usage": usage.cpu().tolist(),
            "mean_distinct_circuits_per_example": sum(union_counts) / count}


def parameter_metrics(model):
    router = model.router
    total = sum(p.numel() for p in model.parameters())
    bank = sum(p.numel() for p in model.circuits.parameters())
    router_total = sum(p.numel() for p in router.parameters())
    shared = total - bank - router_total
    # Both probe routers score all utility keys. Only candidate pair embeddings
    # are gathered; model.parameter_report() omits these router projections.
    touched = router_total
    if hasattr(router, "pair_embeddings"):
        touched -= (32 - 8) * router.pair_embeddings.shape[-1]
    elif hasattr(router, "level_projections"):
        touched -= (32 - 8) * router.keys.shape[-1]
    return {"total_parameters": total, "router_parameters": router_total,
            "router_parameters_touched_per_decision": touched,
            "active_circuit_parameters_per_decision": bank // 32 * 2,
            "active_parameters_per_decision_upper_bound": shared + touched + bank // 32 * 2,
            "active_parameters_per_example_upper_bound": shared + router_total + bank // 32 * 6}


@torch.inference_mode()
def latency(model, inputs, repeats):
    device = inputs.device
    model.eval()
    for _ in range(10):
        model(inputs, adaptive=False)
    values = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        model(inputs, adaptive=False)
        synchronize(device)
        values.append((time.perf_counter() - start) * 1000)
    times = torch.tensor(values)
    return {"batch_size": inputs.shape[0], "median_ms": float(times.median()),
            "p95_ms": float(torch.quantile(times, 0.95)), "repeats": repeats}


@torch.inference_mode()
def common_oracle(teacher, batch, chunk_size):
    """Costs at common teacher states, with teacher re-routing the suffix.

Every compared replacement uses uniform weights/unit gain. Costs are shared
across arms; no arm gets a different oracle because its states became worse.
"""
    pairs = torch.tensor(list(itertools.combinations(range(32), 2)), device=batch.inputs.device)
    _, stats = teacher(batch.inputs, adaptive=False)
    batch_size = batch.inputs.shape[0]
    costs = []
    for t in range(3):
        chunks = []
        for start in range(0, pairs.shape[0], chunk_size):
            choices = pairs[start:start + chunk_size]
            n = choices.shape[0]
            rows = torch.arange(batch_size, device=batch.inputs.device).repeat_interleave(n)
            logits = rollout(teacher, batch.inputs[rows], subset_stats(stats, rows), t,
                             choices.repeat(batch_size, 1))
            loss = F.cross_entropy(logits, batch.targets[rows], reduction="none")
            chunks.append(loss.reshape(batch_size, n))
        costs.append(torch.cat(chunks, 1))
    return pairs, torch.stack(costs, 1), stats["query_states"]


@torch.inference_mode()
def oracle_metrics(router, pairs, costs, queries):
    lookup = torch.full((32, 32), -1, dtype=torch.long, device=pairs.device)
    lookup[pairs[:, 0], pairs[:, 1]] = torch.arange(len(pairs), device=pairs.device)
    lookup[pairs[:, 1], pairs[:, 0]] = torch.arange(len(pairs), device=pairs.device)
    selection, retrieval, regrets, recalls = [], [], [], []
    for t in range(3):
        selected, _, stats = router(queries[:, t])
        candidates = stats["candidate_ids"]
        contained = ((candidates[:, :, None] == pairs[None, :, 0]).any(1)
                     & (candidates[:, :, None] == pairs[None, :, 1]).any(1))
        loss = costs[:, t]
        global_best = loss.min(-1).values
        candidate_best = loss.masked_fill(~contained, float("inf")).min(-1).values
        chosen = loss.gather(1, lookup[selected[:, 0], selected[:, 1]][:, None]).squeeze(1)
        selection.append(chosen - candidate_best)
        retrieval.append(candidate_best - global_best)
        regrets.append(chosen - global_best)
        # Tie-aware: any CE-optimal pair present counts as a successful retrieval.
        recalls.append((contained & (loss <= global_best[:, None] + 1e-6)).any(-1).float())
    regret = torch.cat(regrets)
    return {"candidate_recall": float(torch.cat(recalls).mean()),
            "selection_regret_ce": float(torch.cat(selection).mean()),
            "retrieval_regret_ce": float(torch.cat(retrieval).mean()),
            "mean_regret_ce": float(regret.mean()),
            "p95_regret_ce": float(torch.quantile(regret, 0.95)),
            "audit_examples": costs.shape[0], "audit_decisions": costs.shape[0] * 3,
            "reference": "same frozen hierarchical prefix and suffix; one-decision CE oracle"}


def acceptance(rows):
    """Predeclared paired screen; a small/noisy gain is NOT acceptance."""
    comparisons = []
    for row in rows:
        old, new = row["models"]["old"], row["models"]["coupled"]
        comparisons.append({
            "seed": row["seed"],
            "accuracy_gain": new["hard_accuracy"] - old["hard_accuracy"],
            "ce_reduction": old["ce"] - new["ce"],
            "p95_regret_ratio": new["p95_regret_ce"] / max(old["p95_regret_ce"], 1e-8),
            "recall_gain": new["candidate_recall"] - old["candidate_recall"],
            "latency_ratio": new["latency_batch"]["median_ms"] / old["latency_batch"]["median_ms"],
            "dead_circuits": new["dead_circuits"],
            "vs_hierarchical_accuracy": new["hard_accuracy"] - row["models"]["hierarchical"]["hard_accuracy"],
        })
    passes = (len(comparisons) >= 2
              and sum(c["accuracy_gain"] for c in comparisons) / len(comparisons) >= 0.02
              and all(c["accuracy_gain"] > 0 and c["ce_reduction"] >= 0.05
                      and c["p95_regret_ratio"] <= 0.9 and c["recall_gain"] >= 0
                      and c["latency_ratio"] <= 1.25 and c["dead_circuits"] <= 3
                      and c["vs_hierarchical_accuracy"] >= 0 for c in comparisons))
    return {"decision": "screen_pass_not_scaling_proof" if passes else "reject_for_adoption",
            "comparisons": comparisons,
            "gate": "mean +2pp accuracy; each seed positive accuracy, CE -0.05, p95 regret -10%, recall noninferior, latency <=1.25x, dead<=3, accuracy>=hierarchical",
            "architecture_control": "old-signed separates the new objective from the compact architecture",
            "limitations": "Two seeds and a small oracle audit are a screen, not statistical proof or a 300M gate."}


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    results = []
    for seed in (17, 18):
        path = Path(args.checkpoint_dir) / f"capacity_audit_c32_global_s{seed}.pt"
        teacher, config = _checkpoint(path, device)
        if (teacher.router.num_circuits, teacher.active_circuits, teacher.router.candidate_pool,
                teacher.internal_steps, teacher.circuit_mode) != (32, 2, 8, 3, "parallel"):
            raise ValueError("benchmark requires E32/K2/M8/T3 and parallel circuits")
        for p in teacher.parameters():
            p.requires_grad_(False)
        original_digest = body_digest(teacher)
        test_source = source(config, device, seed + 10003, "heldout")
        test_data = [test_source.balanced(args.examples_per_task) for _ in range(args.eval_batches)]
        audit_batch = source(config, device, seed + 20003, "heldout").balanced(args.audit_examples_per_task)
        print(f"seed={seed} common oracle ({len(audit_batch.targets)} examples)", flush=True)
        pairs, costs, queries = common_oracle(teacher, audit_batch, args.oracle_chunk)
        models = {"hierarchical": teacher}
        training = {}
        # Reverse arm order on the second seed to reduce fixed timing-order bias.
        variants = ("old", "old-signed", "coupled") if seed == 17 else ("coupled", "old-signed", "old")
        for variant in variants:
            print(f"seed={seed} training {variant}", flush=True)
            model = new_model(teacher, seed, variant)
            training[variant] = train_router(teacher, model, config, variant, args)
            assert body_digest(model) == original_digest
            models[variant] = model.eval()
        measured = {}
        for name, model in models.items():
            measured[name] = {
                **evaluate(model, test_data), **parameter_metrics(model),
                **oracle_metrics(model.router, pairs, costs, queries),
                "latency_one": latency(model, test_data[0].inputs[:1], args.latency_repeats),
                "latency_batch": latency(model, test_data[0].inputs, args.latency_repeats),
            }
            if name in training:
                measured[name]["training"] = training[name]
            print(f"seed={seed} {name}: CE={measured[name]['ce']:.5f} accuracy={measured[name]['hard_accuracy']:.4f}", flush=True)
        results.append({"seed": seed, "checkpoint": str(path), "body_sha256": original_digest,
                        "models": measured})
    return {"experiment": "coupled_probe_paired_frozen_screen", "config": vars(args),
            "torch_version": str(torch.__version__), "device": str(device),
            "results": results, "acceptance": acceptance(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--imitation-steps", type=int, default=200)
    parser.add_argument("--examples-per-task", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--audit-examples-per-task", type=int, default=2)
    parser.add_argument("--oracle-chunk", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--latency-repeats", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.imitation_steps < args.steps:
        parser.error("require 0 <= imitation-steps < steps")
    for name in ("examples_per_task", "eval_batches", "audit_examples_per_task",
                 "oracle_chunk", "threads", "latency_repeats"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new path (no silent overwrite)")
    result = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["acceptance"], indent=2))


if __name__ == "__main__":
    main()
