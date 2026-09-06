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
from neural_engine.p004_cascade_credit import CascadeCredit
from train import BatchSource, load_config, make_model, make_optimizer, seed_everything


ARMS = ("control", "cascade", "frozen_bank")


def make_source(
    config: dict[str, Any],
    device: torch.device,
    split: str,
    *,
    seed_offset: int | None = None,
    balanced: bool = False,
) -> BatchSource:
    if split not in {"train", "heldout"}:
        raise ValueError("split must be train or heldout")
    if split == "train":
        prefix = "train"
        offset = 1 if seed_offset is None else seed_offset
        batch_size = int(config["batch_size"])
    else:
        prefix = "heldout"
        offset = 3 if seed_offset is None else seed_offset
        batch_size = 256
    fallback = "eval" if split == "heldout" else "train"
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]),
        int(config["seed"]) + offset,
        value_min=int(config.get(f"{prefix}_value_min", config.get(f"{fallback}_value_min", 0))),
        value_max=int(config.get(f"{prefix}_value_max", config.get(f"{fallback}_value_max", 63))),
        split=str(config.get(f"{prefix}_split", config.get(f"{fallback}_split", split))),
    )
    return BatchSource(generator, batch_size, device, task_balanced=balanced)


class GradientCoverage:
    def __init__(self, model: NeuralEngineV0) -> None:
        n = model.router.num_circuits
        self.circuit_samples = torch.zeros(n, dtype=torch.long)
        self.circuit_nonzero = torch.zeros(n, dtype=torch.long)
        self.router_key_samples = torch.zeros(n, dtype=torch.long)
        self.router_key_nonzero = torch.zeros(n, dtype=torch.long)

    @staticmethod
    def _circuit_row_norms(model: NeuralEngineV0) -> torch.Tensor | None:
        squared = None
        for name in ("down", "up", "bias"):
            parameter = getattr(model.circuits, name, None)
            if parameter is None or parameter.grad is None:
                continue
            grad = parameter.grad.detach().float().reshape(parameter.shape[0], -1)
            value = grad.pow(2).sum(dim=1)
            squared = value if squared is None else squared + value
        return None if squared is None else squared.sqrt().cpu()

    def observe(self, model: NeuralEngineV0) -> None:
        circuit_norms = self._circuit_row_norms(model)
        if circuit_norms is not None:
            self.circuit_samples += 1
            self.circuit_nonzero += circuit_norms.gt(1e-12)
        keys = getattr(model.router, "keys", None)
        if keys is not None and keys.grad is not None:
            norms = keys.grad.detach().float().reshape(keys.shape[0], -1).norm(dim=1).cpu()
            self.router_key_samples += 1
            self.router_key_nonzero += norms.gt(1e-12)

    @staticmethod
    def _report(samples: torch.Tensor, nonzero: torch.Tensor) -> dict[str, Any]:
        observed = samples.gt(0)
        rates = nonzero.float() / samples.clamp_min(1).float()
        return {
            "covered_rows": int((nonzero > 0).sum()),
            "rows": int(samples.numel()),
            "coverage_fraction": float((nonzero > 0).float().mean()),
            "mean_nonzero_rate": float(rates[observed].mean()) if bool(observed.any()) else 0.0,
            "per_row_nonzero_rate": rates.tolist(),
        }

    def report(self) -> dict[str, Any]:
        return {
            "circuit": self._report(self.circuit_samples, self.circuit_nonzero),
            "router_keys": self._report(self.router_key_samples, self.router_key_nonzero),
        }


def route_usage(selected: torch.Tensor, counts: torch.Tensor) -> None:
    routed = selected.detach().reshape(-1)
    routed = routed[routed.ge(0)].cpu()
    if routed.numel():
        counts += torch.bincount(routed, minlength=counts.numel())


def _canonical_alternative(selected: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alternative = selected.clone()
    valid = torch.zeros(selected.shape[0], dtype=torch.bool, device=selected.device)
    for row in range(selected.shape[0]):
        current = set(int(x) for x in selected[row].tolist() if int(x) >= 0)
        replacement = next((int(x) for x in candidates[row].tolist() if int(x) >= 0 and int(x) not in current), None)
        if replacement is not None:
            alternative[row, 0] = replacement
            valid[row] = True
    return alternative, valid


def _route_overlap(a: torch.Tensor, b: torch.Tensor, valid_examples: torch.Tensor) -> tuple[float, int, float, int]:
    if not bool(valid_examples.any()) or a.numel() == 0:
        return 0.0, 0, 0.0, 0
    left = a[valid_examples]
    right = b[valid_examples]
    valid = left.ge(0) & right.ge(0)
    slot_same = left.eq(right) & valid
    slot_sum = float(slot_same.sum().cpu())
    slot_count = int(valid.sum().cpu())
    exact = left.eq(right).all(dim=-1).all(dim=-1)
    return slot_sum, slot_count, float(exact.float().sum().cpu()), int(exact.numel())


@torch.inference_mode()
def evaluate_natural(
    model: NeuralEngineV0,
    source: BatchSource,
    batches: int,
    examples_per_task: int,
) -> dict[str, Any]:
    model.eval()
    n = model.router.num_circuits
    usage = torch.zeros(n, dtype=torch.long)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for _ in range(batches):
        batch = source.balanced(examples_per_task)
        logits, stats = model(batch.inputs, adaptive=False)
        losses = F.cross_entropy(logits, batch.targets, reduction="sum")
        total_loss += float(losses.cpu())
        total_correct += int(logits.argmax(-1).eq(batch.targets).sum())
        total_examples += int(batch.targets.numel())
        route_usage(stats["selected_ids"], usage)
    return {
        "accuracy": total_correct / max(total_examples, 1),
        "ce": total_loss / max(total_examples, 1),
        "examples": total_examples,
        "circuit_usage": usage.tolist(),
        "used_circuits": int(usage.gt(0).sum()),
        "dead_circuits": int(usage.eq(0).sum()),
    }


@torch.inference_mode()
def cascade_regret_audit(
    model: NeuralEngineV0,
    source: BatchSource,
    batches: int,
    examples_per_task: int,
) -> dict[str, Any]:
    """One-step within-candidate oracle with suffix rerouted on changed state."""
    model.eval()
    regrets: list[torch.Tensor] = []
    prefix_same = prefix_count = 0.0
    suffix_same = suffix_count = 0.0
    suffix_exact = suffix_exact_count = 0.0
    cascade_shift_sum = 0.0
    cascade_shift_examples = 0
    alternatives_evaluated = 0

    for _ in range(batches):
        batch = source.balanced(examples_per_task)
        natural_logits, stats = model(batch.inputs, adaptive=False)
        natural_loss = F.cross_entropy(natural_logits, batch.targets, reduction="none")
        best_loss = natural_loss.clone()
        selected_all = stats["selected_ids"].detach()
        candidates_all = stats["candidate_ids"].detach()
        weights = stats["selected_weights"].detach()
        gains = stats["route_gains"].detach()

        for step in range(model.internal_steps):
            selected = selected_all[:, step]
            candidates = candidates_all[:, step]
            for side in range(model.active_circuits):
                for position in range(model.router.candidate_pool):
                    replacement = candidates[:, position]
                    duplicate = selected.eq(replacement.unsqueeze(1)).any(dim=1)
                    valid = replacement.ge(0) & ~duplicate
                    if not bool(valid.any()):
                        continue
                    alternative = selected.clone()
                    alternative[valid, side] = replacement[valid]
                    plan = torch.full_like(selected_all, -1)
                    plan[valid, step] = alternative[valid]
                    alt_logits, alt_stats = model(
                        batch.inputs,
                        adaptive=False,
                        forced_selected_ids=plan,
                        forced_selected_weights=weights,
                        forced_route_gains=gains,
                    )
                    alt_loss = F.cross_entropy(alt_logits, batch.targets, reduction="none")
                    best_loss = torch.where(valid, torch.minimum(best_loss, alt_loss), best_loss)
                    alternatives_evaluated += int(valid.sum())

                    if step > 0:
                        p_same, p_count, _, _ = _route_overlap(
                            selected_all[:, :step], alt_stats["selected_ids"][:, :step], valid
                        )
                        prefix_same += p_same
                        prefix_count += p_count
                    if step < model.internal_steps - 1:
                        s_same, s_count, s_exact, s_exact_count = _route_overlap(
                            selected_all[:, step + 1 :], alt_stats["selected_ids"][:, step + 1 :], valid
                        )
                        suffix_same += s_same
                        suffix_count += s_count
                        suffix_exact += s_exact
                        suffix_exact_count += s_exact_count

            canonical, valid = _canonical_alternative(selected, candidates)
            if bool(valid.any()):
                cascade_plan = torch.full_like(selected_all, -1)
                cascade_plan[valid, step] = canonical[valid]
                fixed_plan = selected_all.clone()
                fixed_plan[valid, step] = canonical[valid]
                cascade_logits, _ = model(
                    batch.inputs,
                    adaptive=False,
                    forced_selected_ids=cascade_plan,
                    forced_selected_weights=weights,
                    forced_route_gains=gains,
                )
                fixed_logits, _ = model(
                    batch.inputs,
                    adaptive=False,
                    forced_selected_ids=fixed_plan,
                    forced_selected_weights=weights,
                    forced_route_gains=gains,
                )
                cascade_loss = F.cross_entropy(cascade_logits, batch.targets, reduction="none")
                fixed_loss = F.cross_entropy(fixed_logits, batch.targets, reduction="none")
                cascade_shift_sum += float((cascade_loss - fixed_loss)[valid].sum().cpu())
                cascade_shift_examples += int(valid.sum())

        regrets.append((natural_loss - best_loss).clamp_min(0).cpu())

    regret = torch.cat(regrets) if regrets else torch.zeros(1)
    return {
        "mean_regret": float(regret.mean()),
        "p95_regret": float(torch.quantile(regret, 0.95)),
        "zero_regret_fraction": float(regret.le(1e-8).float().mean()),
        "prefix_route_stability": prefix_same / prefix_count if prefix_count else 1.0,
        "suffix_slot_stability": suffix_same / suffix_count if suffix_count else 1.0,
        "suffix_exact_stability": suffix_exact / suffix_exact_count if suffix_exact_count else 1.0,
        "mean_cascade_shift_ce": cascade_shift_sum / max(cascade_shift_examples, 1),
        "alternative_example_evaluations": alternatives_evaluated,
    }


def train_arm(
    config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    arm: str,
) -> tuple[NeuralEngineV0, dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    seed_everything(int(config["seed"]))
    model = make_model(config).to(device)
    if not isinstance(model, NeuralEngineV0):
        raise TypeError("P-004 requires NeuralEngineV0")
    model.load_state_dict(initial_state, strict=True)
    model.routing_mode = "learned"
    if model.router.__class__.__name__ != "HierarchicalRouter":
        raise RuntimeError("P-004 must keep the existing HierarchicalRouter")
    if getattr(model.router, "soft_routing_temperature", 0.0) != 0.0:
        raise RuntimeError("P-004 requires hard on-policy routing")
    if int(config.get("route_exploration_prob", 0.0)) != 0:
        raise RuntimeError("P-004 does not use route exploration")
    if arm == "frozen_bank":
        for parameter in model.circuits.parameters():
            parameter.requires_grad_(False)

    optimizer = make_optimizer(model, config)
    source = make_source(config, device, "train", balanced=True)
    credit = None if arm == "control" else CascadeCredit(
        internal_steps=model.internal_steps,
        active_circuits=model.active_circuits,
        candidate_pool=model.router.candidate_pool,
        interval=args.credit_interval,
        diagnostic_interval=args.diagnostic_interval,
        margin=args.credit_margin,
        temperature=args.credit_temperature,
        weight=args.credit_weight,
        seed=int(config["seed"]) + 4004,
    )
    tracker = GradientCoverage(model)
    usage = torch.zeros(model.router.num_circuits, dtype=torch.long)
    main_losses: list[float] = []
    aux_losses: list[float] = []
    peak_vram = 0
    start = time.perf_counter()
    model.train()

    for step in range(1, args.steps + 1):
        batch = source.batch()
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(batch.inputs, adaptive=False)
        route_usage(stats["selected_ids"], usage)
        main_loss = F.cross_entropy(logits, batch.targets)
        aux_loss = model.router.keys.sum() * 0.0

        if credit is not None:
            probe = credit.build_probe(stats, step)
            if probe is not None:
                current_per_example = F.cross_entropy(logits.detach(), batch.targets, reduction="none")
                with torch.no_grad():
                    cascade_logits, cascade_stats = model(
                        batch.inputs,
                        adaptive=False,
                        forced_selected_ids=probe.cascade_plan,
                        forced_selected_weights=stats["selected_weights"].detach(),
                        forced_route_gains=stats["route_gains"].detach(),
                    )
                    cascade_per_example = F.cross_entropy(
                        cascade_logits, batch.targets, reduction="none"
                    )
                    if credit.should_diagnose(step):
                        fixed_logits, _ = model(
                            batch.inputs,
                            adaptive=False,
                            forced_selected_ids=probe.fixed_suffix_plan,
                            forced_selected_weights=stats["selected_weights"].detach(),
                            forced_route_gains=stats["route_gains"].detach(),
                        )
                        fixed_per_example = F.cross_entropy(
                            fixed_logits, batch.targets, reduction="none"
                        )
                        credit.observe_diagnostic(
                            probe,
                            current_per_example,
                            cascade_per_example,
                            fixed_per_example,
                            stats,
                            cascade_stats,
                        )
                aux_loss, _ = credit.auxiliary_loss(
                    model.router,
                    stats["query_states"][:, probe.step],
                    stats["selected_ids"][:, probe.step].detach(),
                    probe.alternative_ids,
                    current_per_example,
                    cascade_per_example,
                    probe.probed_mask,
                )

        total_loss = main_loss + aux_loss
        total_loss.backward()
        tracker.observe(model)
        nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        main_losses.append(float(main_loss.detach().cpu()))
        aux_losses.append(float(aux_loss.detach().cpu()))
        if device.type == "cuda":
            peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated(device) // 2**20))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == args.steps):
            print(
                f"[{arm}] seed={config['seed']} step={step}/{args.steps} "
                f"main={main_losses[-1]:.4f} aux={aux_losses[-1]:.4f}"
            )

    seconds = time.perf_counter() - start
    return model, {
        "seconds": seconds,
        "samples_per_second": args.steps * int(config["batch_size"]) / max(seconds, 1e-9),
        "peak_vram_mb": peak_vram,
        "mean_last_100_main_loss": sum(main_losses[-100:]) / min(len(main_losses), 100),
        "mean_aux_loss": sum(aux_losses) / max(len(aux_losses), 1),
        "circuit_usage": usage.tolist(),
        "used_circuits": int(usage.gt(0).sum()),
        "dead_circuits": int(usage.eq(0).sum()),
        "gradient_coverage": tracker.report(),
        "cascade_credit": credit.report() if credit is not None else None,
    }


def run_seed(config: dict[str, Any], seed: int, device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["seed"] = int(seed)
    seed_everything(seed)
    initial = make_model(config).to(device)
    if not isinstance(initial, NeuralEngineV0):
        raise TypeError("P-004 requires NeuralEngineV0")
    if initial.router.__class__.__name__ != "HierarchicalRouter":
        raise RuntimeError("P-004 refuses router variants")
    contract = {
        "router_class": initial.router.__class__.__name__,
        "num_circuits": initial.router.num_circuits,
        "candidate_pool": initial.router.candidate_pool,
        "active_circuits": initial.active_circuits,
        "internal_steps": initial.internal_steps,
        "circuit_class": initial.circuits.__class__.__name__,
    }
    state = {key: value.detach().cpu().clone() for key, value in initial.state_dict().items()}
    del initial

    result: dict[str, Any] = {"seed": seed, "contract": contract}
    for arm in ARMS:
        model, training = train_arm(config, state, device, args, arm)
        on_policy = evaluate_natural(
            model,
            make_source(config, device, "train", seed_offset=4, balanced=False),
            args.eval_batches,
            args.examples_per_task,
        )
        heldout = evaluate_natural(
            model,
            make_source(config, device, "heldout", balanced=False),
            args.eval_batches,
            args.examples_per_task,
        )
        regret = cascade_regret_audit(
            model,
            make_source(config, device, "heldout", seed_offset=5, balanced=False),
            args.regret_batches,
            args.regret_examples_per_task,
        )
        result[arm] = {
            "training": training,
            "on_policy": on_policy,
            "heldout": heldout,
            "regret": regret,
        }
        if args.checkpoint_dir:
            path = Path(args.checkpoint_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "config": config, "arm": arm},
                path / f"p004_seed{seed}_{arm}.pt",
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    control = result["control"]
    treatment = result["cascade"]
    frozen = result["frozen_bank"]
    result["delta"] = {
        "heldout_accuracy_pp": 100.0 * (treatment["heldout"]["accuracy"] - control["heldout"]["accuracy"]),
        "heldout_ce": treatment["heldout"]["ce"] - control["heldout"]["ce"],
        "on_policy_accuracy_pp": 100.0 * (treatment["on_policy"]["accuracy"] - control["on_policy"]["accuracy"]),
        "on_policy_ce": treatment["on_policy"]["ce"] - control["on_policy"]["ce"],
        "mean_regret": treatment["regret"]["mean_regret"] - control["regret"]["mean_regret"],
        "p95_regret": treatment["regret"]["p95_regret"] - control["regret"]["p95_regret"],
        "circuit_grad_coverage": (
            treatment["training"]["gradient_coverage"]["circuit"]["coverage_fraction"]
            - control["training"]["gradient_coverage"]["circuit"]["coverage_fraction"]
        ),
        "router_key_grad_coverage": (
            treatment["training"]["gradient_coverage"]["router_keys"]["coverage_fraction"]
            - control["training"]["gradient_coverage"]["router_keys"]["coverage_fraction"]
        ),
        "training_overhead_ratio": treatment["training"]["seconds"] / max(control["training"]["seconds"], 1e-9),
        "frozen_bank_heldout_accuracy_pp_vs_control": 100.0 * (
            frozen["heldout"]["accuracy"] - control["heldout"]["accuracy"]
        ),
    }
    return result


def decide(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    held_acc = [seed["delta"]["heldout_accuracy_pp"] for seed in seeds]
    on_acc = [seed["delta"]["on_policy_accuracy_pp"] for seed in seeds]
    ce = [seed["delta"]["heldout_ce"] for seed in seeds]
    circuit_cov = [seed["delta"]["circuit_grad_coverage"] for seed in seeds]
    router_cov = [seed["delta"]["router_key_grad_coverage"] for seed in seeds]
    overhead = [seed["delta"]["training_overhead_ratio"] for seed in seeds]

    regret_reduction = []
    p95_reduction = []
    for seed in seeds:
        control = seed["control"]["regret"]
        treatment = seed["cascade"]["regret"]
        regret_reduction.append(
            (control["mean_regret"] - treatment["mean_regret"]) / max(control["mean_regret"], 1e-8)
        )
        p95_reduction.append(
            (control["p95_regret"] - treatment["p95_regret"]) / max(control["p95_regret"], 1e-8)
        )

    accuracy_gate = (
        mean(held_acc) >= 2.0 and min(held_acc) >= 0.0
        and mean(on_acc) >= 1.0 and min(on_acc) >= 0.0
    )
    ce_gate = mean(ce) <= -0.02 and max(ce) <= 0.02
    regret_gate = mean(regret_reduction) >= 0.10 and mean(p95_reduction) >= 0.10
    coverage_guard = mean(circuit_cov) >= -(1 / 32) and mean(router_cov) >= -(1 / 32)
    overhead_guard = mean(overhead) <= 1.50
    prefix_guard = min(
        seed["cascade"]["regret"]["prefix_route_stability"] for seed in seeds
    ) >= 0.999
    accepted = accuracy_gate and ce_gate and regret_gate and coverage_guard and overhead_guard and prefix_guard
    return {
        "accepted": accepted,
        "decision": "ACCEPT" if accepted else "REJECTED",
        "accuracy_gate": accuracy_gate,
        "ce_gate": ce_gate,
        "regret_gate": regret_gate,
        "coverage_guard": coverage_guard,
        "overhead_guard": overhead_guard,
        "prefix_guard": prefix_guard,
        "mean_heldout_accuracy_delta_pp": mean(held_acc),
        "mean_on_policy_accuracy_delta_pp": mean(on_acc),
        "mean_heldout_ce_delta": mean(ce),
        "mean_regret_reduction_fraction": mean(regret_reduction),
        "mean_p95_regret_reduction_fraction": mean(p95_reduction),
        "mean_training_overhead_ratio": mean(overhead),
        "criteria": {
            "accuracy": "held-out mean >= +2 pp, on-policy mean >= +1 pp, no seed negative",
            "ce": "mean held-out CE delta <= -0.02 and no seed worse than +0.02",
            "regret": "mean and p95 one-step cascade regret each improve >= 10%",
            "coverage": "circuit and router-key gradient coverage do not fall by > 1/32",
            "overhead": "paired treatment/control training wall-clock <= 1.50x",
            "prefix": "counterfactual replay preserves natural prefix >= 99.9%",
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# P-004 — Cascade-consistent on-policy credit",
        "",
        f"**Decision:** `{report['decision']['decision']}`",
        "",
        "Router architecture and circuit bank are unchanged. Credit target is final corrected-output CE after a one-step route change and natural suffix rerouting.",
        "",
        "| Seed | Arm | Held-out acc | Held-out CE | On-policy acc | Mean regret | P95 regret | Suffix stability | Used circuits | Circuit grad cov | Router-key grad cov | Train s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in report["seeds"]:
        for arm in ARMS:
            item = seed[arm]
            training = item["training"]
            lines.append(
                f"| {seed['seed']} | {arm} | {item['heldout']['accuracy']*100:.2f}% | "
                f"{item['heldout']['ce']:.4f} | {item['on_policy']['accuracy']*100:.2f}% | "
                f"{item['regret']['mean_regret']:.4f} | {item['regret']['p95_regret']:.4f} | "
                f"{item['regret']['suffix_slot_stability']:.3f} | {item['heldout']['used_circuits']}/32 | "
                f"{training['gradient_coverage']['circuit']['coverage_fraction']:.3f} | "
                f"{training['gradient_coverage']['router_keys']['coverage_fraction']:.3f} | "
                f"{training['seconds']:.1f} |"
            )
        d = seed["delta"]
        lines.append(
            f"| {seed['seed']} | **cascade-control** | **{d['heldout_accuracy_pp']:+.2f} pp** | "
            f"**{d['heldout_ce']:+.4f}** | **{d['on_policy_accuracy_pp']:+.2f} pp** | "
            f"**{d['mean_regret']:+.4f}** | **{d['p95_regret']:+.4f}** | — | — | "
            f"**{d['circuit_grad_coverage']:+.3f}** | **{d['router_key_grad_coverage']:+.3f}** | "
            f"**{d['training_overhead_ratio']:.2f}x** |"
        )
    decision = report["decision"]
    lines += ["", "## Gate", ""]
    for key in ("accuracy", "ce", "regret", "coverage", "overhead", "prefix"):
        flag = decision[f"{key}_gate"] if f"{key}_gate" in decision else decision[f"{key}_guard"]
        lines.append(f"- {key}: `{flag}` — {decision['criteria'][key]}.")
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- Fixed-suffix replay is diagnostic only; it never supplies the treatment target.",
        "- Cascade replay forces only the changed step. Every later recurrent step receives the changed state and reroutes naturally.",
        "- `frozen_bank` is diagnostic and is not part of the adoption gate.",
        "- A CE-only gain is insufficient; hard accuracy and regret must pass together.",
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


def update_problems(report: dict[str, Any], path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    decision = report["decision"]
    if decision["accepted"]:
        start = text.find("### P-004 — Sparse training credit assignment va cascade shift")
        end = text.find("\n---", start)
        if start >= 0 and end > start:
            section = text[start:end]
            section = section.replace("**Status:** `ACTIVE`", "**Status:** `SOLVED`", 1)
            text = text[:start] + section + text[end:]
            path.write_text(text, encoding="utf-8")
        return

    marker = "### C-P004-CASCADE-001 — Cascade-consistent on-policy credit"
    if marker in text:
        return
    block = (
        f"\n\n{marker}\n\n**Status:** `REJECTED`  \n**Muammo:** P-004  \n"
        f"**Natija:** seed17/18 held-out accuracy mean delta "
        f"`{decision['mean_heldout_accuracy_delta_pp']:+.3f} pp`, held-out CE delta "
        f"`{decision['mean_heldout_ce_delta']:+.5f}`, mean regret reduction "
        f"`{decision['mean_regret_reduction_fraction']*100:.2f}%`, p95 regret reduction "
        f"`{decision['mean_p95_regret_reduction_fraction']*100:.2f}%`, training overhead "
        f"`{decision['mean_training_overhead_ratio']:.3f}x`. Pre-registered gate bajarilmadi; "
        "P-004 `ACTIVE` qoladi.\n"
    )
    path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P-004 cascade-consistent credit benchmark")
    parser.add_argument("--config", default="configs/ne_capacity_signal.yaml")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--credit-interval", type=int, default=4)
    parser.add_argument("--diagnostic-interval", type=int, default=16)
    parser.add_argument("--credit-margin", type=float, default=0.01)
    parser.add_argument("--credit-temperature", type=float, default=0.10)
    parser.add_argument("--credit-weight", type=float, default=0.20)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--regret-batches", type=int, default=2)
    parser.add_argument("--regret-examples-per-task", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--output", default="results/P004_CASCADE_CREDIT_SEED17_18.json")
    parser.add_argument("--markdown", default="results/P004_CASCADE_CREDIT_SEED17_18.md")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/p004_cascade")
    parser.add_argument("--update-problems", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, smoke=False)
    if args.smoke:
        args.steps = min(args.steps, 6)
        args.seeds = args.seeds[:1]
        args.eval_batches = 1
        args.examples_per_task = 2
        args.regret_batches = 1
        args.regret_examples_per_task = 1
        args.credit_interval = 2
        args.diagnostic_interval = 4
        config = copy.deepcopy(config)
        config.update(d_model=64, state_dim=64, circuit_rank=4, batch_size=32)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    seeds = [run_seed(config, seed, device, args) for seed in args.seeds]
    command = (
        f"python benchmark_p004_cascade.py --config {args.config} --steps {args.steps} "
        f"--device {args.device} --seeds {' '.join(map(str, args.seeds))} --update-problems"
    )
    report = {
        "experiment": "P004 cascade-consistent on-policy route credit",
        "hypothesis": (
            "final corrected-output credit with natural suffix rerouting reduces cascade regret "
            "without changing router architecture or circuit bank"
        ),
        "config": config,
        "seeds": seeds,
        "decision": decide(seeds),
        "command": command,
        "protocol": {
            "control_and_treatment_same_initialization": True,
            "control_and_treatment_same_data_stream": True,
            "cascade_target": "final CE after one changed step and natural suffix rerouting",
            "fixed_suffix": "diagnostic only",
            "router_architecture_changed": False,
            "circuit_bank_changed": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.markdown))
    if args.update_problems:
        update_problems(report, Path("problems.md"))
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
