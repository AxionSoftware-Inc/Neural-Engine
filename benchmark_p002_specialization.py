from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from neural_engine.p002_credit import CounterfactualCircuitCredit
from train import BatchSource, load_config, make_model, make_optimizer, seed_everything


def make_source(
    config: dict[str, Any], device: torch.device, split: str
) -> BatchSource:
    if split not in {"train", "eval", "heldout"}:
        raise ValueError("split must be train, eval or heldout")
    if split == "train":
        prefix, seed_offset, batch_size, task_balanced = (
            "train", 1, int(config["batch_size"]), True
        )
    elif split == "eval":
        prefix, seed_offset, batch_size, task_balanced = "eval", 2, 256, False
    else:
        prefix, seed_offset, batch_size, task_balanced = "heldout", 3, 256, False
    fallback_prefix = "eval" if split == "heldout" else prefix
    generator = SyntheticTaskGenerator(
        config["seq_len"],
        int(config["seed"]) + seed_offset,
        value_min=int(
            config.get(
                f"{prefix}_value_min",
                config.get(f"{fallback_prefix}_value_min", 0),
            )
        ),
        value_max=int(
            config.get(
                f"{prefix}_value_max",
                config.get(f"{fallback_prefix}_value_max", 63),
            )
        ),
        split=str(
            config.get(
                f"{prefix}_split",
                config.get(f"{fallback_prefix}_split", "all"),
            )
        ),
    )
    return BatchSource(
        generator,
        batch_size,
        device,
        task_balanced=task_balanced,
    )


def sampled_grad_norm(model: NeuralEngineV0, circuit_id: int) -> float:
    squared = None
    for name in ("down", "up", "bias"):
        parameter = getattr(model.circuits, name)
        if parameter.grad is None:
            continue
        value = parameter.grad[circuit_id].detach().float().pow(2).sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt().cpu())


def specialization_metrics(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.double()
    if matrix.numel() == 0 or float(matrix.sum()) <= 0:
        return {
            "task_circuit_nmi": 0.0,
            "usage_weighted_specialization": 0.0,
            "usage_weighted_task_purity": 0.0,
        }
    circuit_load = matrix.sum(0)
    used = circuit_load.gt(0)
    probabilities = torch.zeros_like(matrix)
    probabilities[:, used] = matrix[:, used] / circuit_load[used].unsqueeze(0)
    entropy = torch.zeros(matrix.shape[1], dtype=torch.float64)
    purity = torch.zeros(matrix.shape[1], dtype=torch.float64)
    if used.any():
        p = probabilities[:, used]
        entropy[used] = -(p * p.clamp_min(1e-30).log()).sum(0) / math.log(matrix.shape[0])
        purity[used] = p.max(0).values
    weights = circuit_load / circuit_load.sum().clamp_min(1)
    specialization = float((weights * (1.0 - entropy)).sum())
    weighted_purity = float((weights * purity).sum())

    joint = matrix / matrix.sum().clamp_min(1)
    pt = joint.sum(1, keepdim=True)
    pc = joint.sum(0, keepdim=True)
    nz = joint.gt(0)
    ratio = joint / (pt * pc).clamp_min(1e-30)
    mi = float((joint[nz] * ratio[nz].log()).sum())
    ht = float(-(pt[pt.gt(0)] * pt[pt.gt(0)].log()).sum())
    hc = float(-(pc[pc.gt(0)] * pc[pc.gt(0)].log()).sum())
    nmi = mi / math.sqrt(max(ht * hc, 1e-30))
    return {
        "task_circuit_nmi": nmi,
        "usage_weighted_specialization": specialization,
        "usage_weighted_task_purity": weighted_purity,
    }


@torch.no_grad()
def evaluate_specialization(
    model: NeuralEngineV0,
    source: BatchSource,
    batches: int,
    examples_per_task: int,
) -> dict[str, Any]:
    model.eval()
    num_circuits = model.router.num_circuits
    matrix = torch.zeros(15, num_circuits, dtype=torch.float64)
    correct = total = 0
    losses: list[float] = []
    for _ in range(batches):
        batch = source.balanced(examples_per_task)
        logits, stats = model(batch.inputs, adaptive=False)
        losses.append(float(F.cross_entropy(logits, batch.targets).cpu()))
        correct += int(logits.argmax(-1).eq(batch.targets).sum())
        total += int(batch.targets.numel())
        selected = stats["selected_ids"].detach().cpu()
        tasks = batch.task_ids.detach().cpu().long()
        for task_id in range(15):
            routed = selected[tasks.eq(task_id)].reshape(-1)
            routed = routed[routed.ge(0)]
            if routed.numel():
                matrix[task_id] += torch.bincount(
                    routed, minlength=num_circuits
                ).double()

    circuit_load = matrix.sum(0)
    used = circuit_load.gt(0)
    metrics = specialization_metrics(matrix)
    return {
        "accuracy": correct / max(total, 1),
        "loss": sum(losses) / max(len(losses), 1),
        "circuits_used": int(used.sum()),
        "dead_circuit_fraction": float((~used).double().mean()),
        **metrics,
        "circuit_load": circuit_load.long().tolist(),
        "task_circuit_counts": matrix.long().tolist(),
    }


@torch.no_grad()
def evaluate_counterfactual_bank(
    model: NeuralEngineV0,
    source: BatchSource,
    batches: int,
    examples_per_task: int,
    min_advantage: float,
) -> dict[str, Any]:
    """Measure held-out functional specialization independent of routing.

    Each batch audits one (recurrent-step, active-slot) role. Every circuit that
    is absent from an example's on-policy trajectory is substituted into that
    role while all route IDs/weights/gains otherwise stay fixed. The best
    positive final-loss substitution owns responsibility for that example.

    Rotating roles prevents the metric from rewarding only one recurrent slot.
    No gradients or route changes occur in this audit.
    """
    model.eval()
    n = model.router.num_circuits
    roles = [
        (step, slot)
        for step in range(model.internal_steps)
        for slot in range(model.active_circuits)
    ]
    counts = torch.zeros(15, n, dtype=torch.long)
    positive_examples = 0
    audited_examples = 0
    positive_advantage_sum = 0.0
    positive_advantages: list[torch.Tensor] = []
    responsible_circuits = torch.zeros(n, dtype=torch.long)

    for batch_index in range(batches):
        batch = source.balanced(examples_per_task)
        logits, stats = model(batch.inputs, adaptive=False)
        baseline_loss = F.cross_entropy(logits, batch.targets, reduction="none")
        selected = stats["selected_ids"].detach()
        weights = stats["selected_weights"].detach()
        gains = stats["route_gains"].detach()
        target_step, target_slot = roles[batch_index % len(roles)]
        executed = selected[:, target_step, target_slot].ge(0)
        advantages = torch.full(
            (n, batch.inputs.shape[0]),
            -torch.inf,
            device=batch.inputs.device,
            dtype=baseline_loss.dtype,
        )

        for circuit_id in range(n):
            already_used = selected.eq(circuit_id).reshape(selected.shape[0], -1).any(dim=1)
            eligible = executed & (~already_used)
            if not eligible.any():
                continue
            forced = selected[eligible].clone()
            forced[:, target_step, target_slot] = circuit_id
            candidate_logits, _ = model(
                batch.inputs[eligible],
                adaptive=False,
                forced_selected_ids=forced,
                forced_selected_weights=weights[eligible],
                forced_route_gains=gains[eligible],
            )
            candidate_loss = F.cross_entropy(
                candidate_logits, batch.targets[eligible], reduction="none"
            )
            advantages[circuit_id, eligible] = (
                baseline_loss[eligible] - candidate_loss
            )

        best_advantage, best_circuit = advantages.max(dim=0)
        responsible = (
            executed
            & torch.isfinite(best_advantage)
            & best_advantage.ge(min_advantage)
        )
        audited_examples += int(executed.sum())
        count = int(responsible.sum())
        if not count:
            continue
        positive_examples += count
        positive_advantage_sum += float(best_advantage[responsible].sum().cpu())
        positive_advantages.append(best_advantage[responsible].detach().cpu())
        tasks = batch.task_ids[responsible].detach().cpu().long()
        winners = best_circuit[responsible].detach().cpu().long()
        for task_id in range(15):
            task_winners = winners[tasks.eq(task_id)]
            if task_winners.numel():
                counts[task_id] += torch.bincount(
                    task_winners, minlength=n
                )
        responsible_circuits += torch.bincount(winners, minlength=n)

    metrics = specialization_metrics(counts)
    joined_advantages = (
        torch.cat(positive_advantages)
        if positive_advantages
        else torch.empty(0)
    )
    return {
        **metrics,
        "positive_responsibility_fraction": (
            positive_examples / max(audited_examples, 1)
        ),
        "mean_positive_advantage": (
            positive_advantage_sum / max(positive_examples, 1)
        ),
        "median_positive_advantage": (
            float(joined_advantages.median()) if joined_advantages.numel() else 0.0
        ),
        "responsible_circuits": int(responsible_circuits.gt(0).sum()),
        "positive_examples": positive_examples,
        "audited_examples": audited_examples,
        "responsibility_task_circuit_counts": counts.tolist(),
        "responsibility_per_circuit": responsible_circuits.tolist(),
        "roles_audited": [
            list(roles[index % len(roles)]) for index in range(batches)
        ],
        "min_advantage": min_advantage,
    }


def train_arm(
    config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    treatment: bool,
) -> tuple[NeuralEngineV0, dict[str, Any]]:
    seed_everything(int(config["seed"]))
    model = make_model(config).to(device)
    if not isinstance(model, NeuralEngineV0):
        raise TypeError("P-002 requires NeuralEngineV0")
    model.load_state_dict(initial_state, strict=True)
    model.routing_mode = "learned"
    if config.get("routing_mode", "learned") != "learned":
        raise ValueError("router control must remain learned")
    if getattr(model.router, "soft_routing_temperature", 0.0) != 0.0:
        raise ValueError("P-002 requires hard routing")
    if config.get("route_exploration_prob", 0.0) or config.get("routing_coverage_weight", 0.0):
        raise ValueError("route exploration/coverage are forbidden in this P-002 experiment")
    if config.get("adaptive_halting", False):
        raise ValueError("adaptive halting is disabled in the controlled P-002 protocol")

    optimizer = make_optimizer(model, config)
    if optimizer.__class__.__name__.lower().startswith("lazy"):
        raise ValueError("row-local auxiliary credit requires standard AdamW")
    source = make_source(config, device, "train")
    credit = (
        CounterfactualCircuitCredit(
            model.router.num_circuits,
            model.internal_steps,
            model.active_circuits,
            interval=args.credit_interval,
            candidates=args.credit_candidates,
            weight=args.credit_weight,
            min_eligible=args.credit_min_eligible,
            min_responsible=args.credit_min_responsible,
            min_advantage=args.credit_min_advantage,
            advantage_clip=args.credit_advantage_clip,
            seed=int(config["seed"]) + 2002,
        )
        if treatment
        else None
    )

    n = model.router.num_circuits
    route_usage = torch.zeros(n, dtype=torch.long)
    grad_samples = torch.zeros(n, dtype=torch.long)
    grad_nonzero = torch.zeros(n, dtype=torch.long)
    grad_norm_sum = torch.zeros(n, dtype=torch.float64)
    events: list[dict[str, Any]] = []
    losses: list[float] = []
    peak_vram = 0
    start = time.perf_counter()
    model.train()

    for step in range(1, args.steps + 1):
        batch = source.batch()
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(batch.inputs, adaptive=False)
        selected = stats["selected_ids"].detach()
        routed = selected.reshape(-1)
        routed = routed[routed.ge(0)].cpu()
        route_usage += torch.bincount(routed, minlength=n)
        if credit is not None:
            credit.observe_on_policy(selected)
        loss = F.cross_entropy(logits, batch.targets)
        pending = (
            credit.prepare_update(
                model,
                batch.inputs,
                batch.targets,
                logits,
                stats,
                step,
                task_ids=batch.task_ids,
            )
            if credit is not None
            else None
        )
        loss.backward()

        sampled_id = (step - 1) % n
        norm = sampled_grad_norm(model, sampled_id)
        grad_samples[sampled_id] += 1
        grad_norm_sum[sampled_id] += norm
        if norm > 1e-12:
            grad_nonzero[sampled_id] += 1
        if credit is not None:
            credit.observe_main_grad(model.circuits, sampled_id)
            credit.apply_update(model.circuits, pending)
            if pending is not None:
                events.append({
                    "step": step,
                    "target_step": pending.target_step,
                    "target_slot": pending.target_slot,
                    "responsible_examples": pending.responsible_examples,
                    "mean_advantage": pending.mean_advantage,
                    "rows": [
                        {
                            "circuit_id": row.circuit_id,
                            "examples": row.examples,
                            "mean_advantage": row.mean_advantage,
                            "auxiliary_loss": row.auxiliary_loss,
                        }
                        for row in pending.rows
                    ],
                })

        nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if device.type == "cuda":
            peak_vram = max(
                peak_vram,
                int(torch.cuda.max_memory_allocated(device) // 2**20),
            )
        if args.log_every and (
            step == 1 or step % args.log_every == 0 or step == args.steps
        ):
            print(
                f"[{'crca' if treatment else 'control'}] seed={config['seed']} "
                f"step={step}/{args.steps} loss={losses[-1]:.4f}"
            )

    seconds = time.perf_counter() - start
    credit_report = credit.report() if credit is not None else None
    if credit_report is not None:
        normal_examples = args.steps * int(config["batch_size"])
        credit_report["actual_probe_forward_example_ratio"] = (
            credit_report["actual_probe_forward_examples"] / max(normal_examples, 1)
        )
        credit_report["actual_gradient_forward_example_ratio"] = (
            credit_report["actual_gradient_forward_examples"] / max(normal_examples, 1)
        )
    return model, {
        "seconds": seconds,
        "samples_per_second": (
            args.steps * int(config["batch_size"]) / max(seconds, 1e-9)
        ),
        "peak_vram_mb": peak_vram,
        "final_loss": losses[-1],
        "mean_last_100_loss": sum(losses[-100:]) / len(losses[-100:]),
        "on_policy_route_usage": route_usage.tolist(),
        "dead_training_usage_circuits": int(route_usage.eq(0).sum()),
        "main_grad_sample_count": grad_samples.tolist(),
        "main_grad_nonzero_samples": grad_nonzero.tolist(),
        "main_grad_norm_sum": grad_norm_sum.tolist(),
        "zero_main_gradient_sample_circuits": int(
            ((grad_samples > 0) & (grad_nonzero == 0)).sum()
        ),
        "credit_events": events,
        "credit": credit_report,
    }


def run_seed(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["seed"] = seed
    seed_everything(seed)
    initial_model = make_model(config).to(device)
    if not isinstance(initial_model, NeuralEngineV0):
        raise TypeError("P-002 requires NeuralEngineV0")
    parameter_report = initial_model.parameter_report()
    if (
        not args.smoke
        and not 18_000_000 <= int(parameter_report["total_params"]) <= 22_000_000
    ):
        raise RuntimeError(
            f"expected ~20M params, got {parameter_report['total_params']:,}"
        )
    if int(config["num_circuits"]) != 32:
        raise RuntimeError("acceptance protocol requires exactly 32 circuits")
    router_class = initial_model.router.__class__.__name__
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in initial_model.state_dict().items()
    }
    del initial_model

    result: dict[str, Any] = {
        "seed": seed,
        "parameter_report": parameter_report,
        "router_contract": {
            "router_class": router_class,
            "candidate_pool": int(config["candidate_pool"]),
            "active_circuits": int(config["active_circuits"]),
            "internal_steps": int(config["internal_steps"]),
        },
    }

    for treatment in (False, True):
        name = "crca" if treatment else "control"
        model, training = train_arm(
            config, initial_state, device, args, treatment
        )
        evaluation = evaluate_specialization(
            model,
            make_source(config, device, "eval"),
            args.eval_batches,
            args.examples_per_task,
        )
        heldout = evaluate_specialization(
            model,
            make_source(config, device, "heldout"),
            args.eval_batches,
            args.examples_per_task,
        )
        functional = evaluate_counterfactual_bank(
            model,
            make_source(config, device, "heldout"),
            args.cf_batches,
            args.cf_examples_per_task,
            args.credit_min_advantage,
        )
        result[name] = {
            "training": training,
            "evaluation": evaluation,
            "heldout": heldout,
            "counterfactual_bank": functional,
        }
        if args.checkpoint_dir:
            path = Path(args.checkpoint_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "evaluation": evaluation,
                    "heldout": heldout,
                    "counterfactual_bank": functional,
                },
                path / f"p002_crca_seed{seed}_{name}.pt",
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    c = result["control"]["evaluation"]
    t = result["crca"]["evaluation"]
    h_c = result["control"]["heldout"]
    h_t = result["crca"]["heldout"]
    cf_c = result["control"]["counterfactual_bank"]
    cf_t = result["crca"]["counterfactual_bank"]
    result["delta"] = {
        "accuracy_pp": 100.0 * (h_t["accuracy"] - h_c["accuracy"]),
        "heldout_loss": h_t["loss"] - h_c["loss"],
        "in_domain_accuracy_pp": 100.0 * (t["accuracy"] - c["accuracy"]),
        "in_domain_loss": t["loss"] - c["loss"],
        "route_task_circuit_nmi": (
            h_t["task_circuit_nmi"] - h_c["task_circuit_nmi"]
        ),
        "route_usage_weighted_specialization": (
            h_t["usage_weighted_specialization"]
            - h_c["usage_weighted_specialization"]
        ),
        "dead_circuit_fraction": (
            h_t["dead_circuit_fraction"] - h_c["dead_circuit_fraction"]
        ),
        "cf_task_circuit_nmi": (
            cf_t["task_circuit_nmi"] - cf_c["task_circuit_nmi"]
        ),
        "cf_usage_weighted_specialization": (
            cf_t["usage_weighted_specialization"]
            - cf_c["usage_weighted_specialization"]
        ),
        "cf_mean_positive_advantage": (
            cf_t["mean_positive_advantage"]
            - cf_c["mean_positive_advantage"]
        ),
        "cf_positive_responsibility_fraction": (
            cf_t["positive_responsibility_fraction"]
            - cf_c["positive_responsibility_fraction"]
        ),
        "cf_responsible_circuits": (
            cf_t["responsible_circuits"] - cf_c["responsible_circuits"]
        ),
    }
    return result


def decide(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [seed["delta"]["accuracy_pp"] for seed in seeds]
    cf_nmi = [seed["delta"]["cf_task_circuit_nmi"] for seed in seeds]
    cf_spec = [
        seed["delta"]["cf_usage_weighted_specialization"]
        for seed in seeds
    ]
    cf_adv = [
        seed["delta"]["cf_mean_positive_advantage"]
        for seed in seeds
    ]
    dead = [seed["delta"]["dead_circuit_fraction"] for seed in seeds]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    accuracy_gate = mean(acc) >= 2.0 and min(acc) >= -1.0
    functional_specialization_gate = (
        mean(cf_nmi) >= 0.05
        and mean(cf_spec) >= 0.05
        and mean(cf_adv) >= 0.01
        and min(cf_adv) >= -0.005
        and mean(dead) <= 1 / 32
    )
    accepted = accuracy_gate or functional_specialization_gate
    return {
        "accepted": accepted,
        "decision": "ACCEPT" if accepted else "REJECTED",
        "accuracy_gate": accuracy_gate,
        "functional_specialization_gate": functional_specialization_gate,
        "mean_accuracy_delta_pp": mean(acc),
        "mean_cf_task_circuit_nmi_delta": mean(cf_nmi),
        "mean_cf_specialization_delta": mean(cf_spec),
        "mean_cf_positive_advantage_delta": mean(cf_adv),
        "mean_dead_circuit_fraction_delta": mean(dead),
        "per_seed_accuracy_delta_pp": acc,
        "per_seed_cf_positive_advantage_delta": cf_adv,
        "criteria": {
            "accuracy": "held-out mean >= +2.0 pp and no seed worse than -1.0 pp",
            "functional_specialization": (
                "held-out counterfactual NMI >= +0.05, specialization >= +0.05, "
                "mean positive final-CE advantage >= +0.01, no seed advantage "
                "delta below -0.005, dead delta <= +1/32"
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# P-002 — Causal Responsibility Credit Assignment",
        "",
        f"**Decision:** `{report['decision']['decision']}`",
        "",
        "| Seed | Arm | Held-out acc | In-domain acc | Held-out route NMI | CF NMI | CF specialization | CF +adv | Dead | Train s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in report["seeds"]:
        for arm in ("control", "crca"):
            evaluation = seed[arm]["evaluation"]
            heldout = seed[arm]["heldout"]
            functional = seed[arm]["counterfactual_bank"]
            training = seed[arm]["training"]
            lines.append(
                f"| {seed['seed']} | {arm} | {heldout['accuracy']*100:.2f}% | "
                f"{evaluation['accuracy']*100:.2f}% | "
                f"{heldout['task_circuit_nmi']:.4f} | "
                f"{functional['task_circuit_nmi']:.4f} | "
                f"{functional['usage_weighted_specialization']:.4f} | "
                f"{functional['mean_positive_advantage']:.4f} | "
                f"{heldout['dead_circuit_fraction']*32:.0f}/32 | "
                f"{training['seconds']:.1f} |"
            )
        delta = seed["delta"]
        lines.append(
            f"| {seed['seed']} | **delta** | **{delta['accuracy_pp']:+.2f} pp** | "
            f"**{delta['in_domain_accuracy_pp']:+.2f} pp** | "
            f"**{delta['route_task_circuit_nmi']:+.4f}** | "
            f"**{delta['cf_task_circuit_nmi']:+.4f}** | "
            f"**{delta['cf_usage_weighted_specialization']:+.4f}** | "
            f"**{delta['cf_mean_positive_advantage']:+.4f}** | "
            f"**{delta['dead_circuit_fraction']*32:+.1f}/32** | — |"
        )
    decision = report["decision"]
    lines += [
        "",
        "## Gate",
        "",
        f"- Accuracy: `{decision['accuracy_gate']}` — {decision['criteria']['accuracy']}.",
        (
            f"- Held-out functional specialization: "
            f"`{decision['functional_specialization_gate']}` — "
            f"{decision['criteria']['functional_specialization']}."
        ),
        "",
        "The router class, candidate pool and hard inference path are identical in both arms.",
        "CRCA changes training credit only. Actual probe/gradient example overhead is recorded",
        "in each treatment training report; inference active parameters remain the control value.",
        "",
        "## Reproduction",
        "",
        "```bash",
        report["command"],
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_rejection(report: dict[str, Any], path: Path) -> None:
    if report["decision"]["accepted"]:
        return
    marker = "### C-P002-CRCA-001 — Causal responsibility sparse credit"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    decision = report["decision"]
    block = (
        f"\n\n{marker}\n\n"
        "**Status:** `REJECTED`  \n"
        "**Muammo:** P-002  \n"
        f"**Natija:** seed17/18 held-out mean accuracy delta "
        f"`{decision['mean_accuracy_delta_pp']:+.3f} pp`, held-out "
        f"counterfactual NMI delta "
        f"`{decision['mean_cf_task_circuit_nmi_delta']:+.5f}`, specialization "
        f"delta `{decision['mean_cf_specialization_delta']:+.5f}`, positive "
        f"final-CE advantage delta "
        f"`{decision['mean_cf_positive_advantage_delta']:+.5f}`. "
        "Pre-registered P-002 gate bajarilmadi; CRCA default uchun qabul qilinmadi.\n"
    )
    path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P-002 CRCA paired benchmark")
    parser.add_argument("--config", default="configs/ne_p002_20m_32.yaml")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--cf-batches", type=int, default=6)
    parser.add_argument("--cf-examples-per-task", type=int, default=8)
    parser.add_argument("--credit-interval", type=int, default=8)
    parser.add_argument("--credit-candidates", type=int, default=4)
    parser.add_argument("--credit-weight", type=float, default=0.25)
    parser.add_argument("--credit-min-eligible", type=int, default=16)
    parser.add_argument("--credit-min-responsible", type=int, default=4)
    parser.add_argument("--credit-min-advantage", type=float, default=0.02)
    parser.add_argument("--credit-advantage-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--output", default="results/P002_CRCA_SEED17_18.json")
    parser.add_argument("--markdown", default="results/P002_CRCA_SEED17_18.md")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/p002_crca")
    parser.add_argument("--update-problems-on-reject", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, smoke=False)
    if args.smoke:
        args.steps = min(args.steps, 4)
        args.eval_batches = 1
        args.examples_per_task = 2
        args.cf_batches = 1
        args.cf_examples_per_task = 1
        args.credit_interval = 2
        args.credit_candidates = 2
        args.credit_min_eligible = 1
        args.credit_min_responsible = 1
        args.credit_min_advantage = 0.0
        args.seeds = args.seeds[:1]
        config = copy.deepcopy(config)
        config.update(
            d_model=64,
            state_dim=64,
            circuit_rank=4,
            batch_size=32,
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    seeds = [
        run_seed(config, seed, device, args)
        for seed in args.seeds
    ]
    command = (
        f"python benchmark_p002_specialization.py --config {args.config} "
        f"--steps {args.steps} --device {args.device} "
        f"--seeds {' '.join(map(str, args.seeds))} "
        "--update-problems-on-reject"
    )
    report = {
        "experiment": "P002 causal responsibility credit assignment",
        "hypothesis": (
            "per-example positive final-loss responsibility can specialize "
            "starved circuit rows without changing the router"
        ),
        "config": config,
        "credit": {
            "interval": args.credit_interval,
            "candidates": args.credit_candidates,
            "weight": args.credit_weight,
            "min_eligible": args.credit_min_eligible,
            "min_responsible": args.credit_min_responsible,
            "min_advantage": args.credit_min_advantage,
            "advantage_clip": args.credit_advantage_clip,
            "planned_probe_forward_equivalents_per_step": (
                args.credit_candidates / args.credit_interval
            ),
            "planned_aux_backward_upper_bound_per_step": (
                args.credit_candidates / args.credit_interval
            ),
        },
        "counterfactual_audit": {
            "split": "heldout",
            "seed_offset": 3,
            "batches": args.cf_batches,
            "examples_per_task": args.cf_examples_per_task,
            "roles_rotate_over_all_step_slot_pairs": True,
        },
        "seeds": seeds,
        "decision": decide(seeds),
        "command": command,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_markdown(report, Path(args.markdown))
    if args.update_problems_on_reject:
        append_rejection(report, Path("problems.md"))
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
