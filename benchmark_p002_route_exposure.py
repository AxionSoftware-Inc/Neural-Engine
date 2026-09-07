from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from benchmark_p002_specialization import (
    evaluate_counterfactual_bank,
    evaluate_specialization,
    make_source,
    sampled_grad_norm,
)
from neural_engine.model import NeuralEngineV0
from train import controlled_task_route_ids, load_config, make_model, make_optimizer, seed_everything


def train_arm(
    config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    exposure_warmup: bool,
) -> tuple[NeuralEngineV0, dict[str, Any]]:
    """Train one paired arm with identical model and data initialization.

    The treatment uses task-stable routes only during the first warmup steps.
    After that point it is exactly the ordinary learned hard router.  No
    routing loss, exploration, coverage term, or circuit-body change is used.
    """
    seed = int(config["seed"])
    seed_everything(seed)
    model = make_model(config).to(device)
    if not isinstance(model, NeuralEngineV0):
        raise TypeError("P-002 route exposure requires NeuralEngineV0")
    model.load_state_dict(initial_state, strict=True)
    if config.get("routing_mode", "learned") != "learned":
        raise ValueError("route exposure control requires learned routing config")
    if getattr(model.router, "soft_routing_temperature", 0.0) != 0.0:
        raise ValueError("route exposure requires hard routing")
    if config.get("route_exploration_prob", 0.0) or config.get("routing_coverage_weight", 0.0):
        raise ValueError("route exposure forbids exploration and coverage regularizers")
    if config.get("adaptive_halting", False):
        raise ValueError("route exposure requires fixed internal steps")

    optimizer = make_optimizer(model, config)
    source = make_source(config, device, "train")
    n = model.router.num_circuits
    route_usage = torch.zeros(n, dtype=torch.long)
    warmup_route_usage = torch.zeros(n, dtype=torch.long)
    learned_route_usage = torch.zeros(n, dtype=torch.long)
    grad_samples = torch.zeros(n, dtype=torch.long)
    grad_nonzero = torch.zeros(n, dtype=torch.long)
    grad_norm_sum = torch.zeros(n, dtype=torch.float64)
    losses: list[float] = []
    start = time.perf_counter()
    peak_vram = 0
    model.train()

    for step in range(1, args.steps + 1):
        batch = source.batch()
        use_exposure = exposure_warmup and step <= args.warmup_steps
        model.routing_mode = "controlled_task" if use_exposure else "learned"
        forced_ids = None
        if use_exposure:
            forced_ids = controlled_task_route_ids(
                batch.task_ids,
                model.internal_steps,
                model.active_circuits,
                model.router.num_circuits,
            )

        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(
            batch.inputs,
            adaptive=False,
            forced_selected_ids=forced_ids,
        )
        selected = stats["selected_ids"].detach().reshape(-1)
        selected = selected[selected.ge(0)].cpu()
        counts = torch.bincount(selected, minlength=n)
        route_usage += counts
        if use_exposure:
            warmup_route_usage += counts
        else:
            learned_route_usage += counts

        loss = F.cross_entropy(logits, batch.targets)
        loss.backward()

        sampled_id = (step - 1) % n
        norm = sampled_grad_norm(model, sampled_id)
        grad_samples[sampled_id] += 1
        grad_norm_sum[sampled_id] += norm
        if norm > 1e-12:
            grad_nonzero[sampled_id] += 1

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
            arm = "exposure" if exposure_warmup else "control"
            phase = "warmup" if use_exposure else "learned"
            print(
                f"[{arm}/{phase}] seed={seed} step={step}/{args.steps} "
                f"loss={losses[-1]:.4f}",
                flush=True,
            )

    model.routing_mode = "learned"
    seconds = time.perf_counter() - start
    return model, {
        "seconds": seconds,
        "samples_per_second": args.steps * int(config["batch_size"]) / max(seconds, 1e-9),
        "peak_vram_mb": peak_vram,
        "final_loss": losses[-1],
        "mean_last_100_loss": sum(losses[-100:]) / len(losses[-100:]),
        "warmup_steps": int(args.warmup_steps if exposure_warmup else 0),
        "warmup_route_usage": warmup_route_usage.tolist(),
        "post_warmup_route_usage": learned_route_usage.tolist(),
        "on_policy_route_usage": route_usage.tolist(),
        "dead_training_usage_circuits": int(route_usage.eq(0).sum()),
        "dead_warmup_usage_circuits": int(warmup_route_usage.eq(0).sum()),
        "dead_post_warmup_usage_circuits": int(learned_route_usage.eq(0).sum()),
        "main_grad_sample_count": grad_samples.tolist(),
        "main_grad_nonzero_samples": grad_nonzero.tolist(),
        "main_grad_norm_sum": grad_norm_sum.tolist(),
        "zero_main_gradient_sample_circuits": int(
            ((grad_samples > 0) & (grad_nonzero == 0)).sum()
        ),
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
        raise TypeError("P-002 route exposure requires NeuralEngineV0")
    parameter_report = initial_model.parameter_report()
    if not args.smoke and not 18_000_000 <= int(parameter_report["total_params"]) <= 22_000_000:
        raise RuntimeError(f"expected ~20M params, got {parameter_report['total_params']:,}")
    if int(config["num_circuits"]) != 32:
        raise RuntimeError("route exposure protocol requires exactly 32 circuits")
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in initial_model.state_dict().items()
    }
    router_class = initial_model.router.__class__.__name__
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
    for exposure_warmup in (False, True):
        name = "exposure" if exposure_warmup else "control"
        model, training = train_arm(
            config, initial_state, device, args, exposure_warmup
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
            args.min_advantage,
        )
        result[name] = {
            "training": training,
            "evaluation": evaluation,
            "heldout": heldout,
            "counterfactual_bank": functional,
        }
        if args.checkpoint_dir:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "arm": name,
                    "warmup_steps": args.warmup_steps,
                    "evaluation": evaluation,
                    "heldout": heldout,
                    "counterfactual_bank": functional,
                },
                checkpoint_dir / f"p002_route_exposure_seed{seed}_{name}.pt",
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    control = result["control"]
    exposure = result["exposure"]
    c_eval, e_eval = control["evaluation"], exposure["evaluation"]
    c_hold, e_hold = control["heldout"], exposure["heldout"]
    c_cf, e_cf = control["counterfactual_bank"], exposure["counterfactual_bank"]
    result["delta"] = {
        "accuracy_pp": 100.0 * (e_hold["accuracy"] - c_hold["accuracy"]),
        "heldout_loss": e_hold["loss"] - c_hold["loss"],
        "in_domain_accuracy_pp": 100.0 * (e_eval["accuracy"] - c_eval["accuracy"]),
        "in_domain_loss": e_eval["loss"] - c_eval["loss"],
        "route_task_circuit_nmi": e_hold["task_circuit_nmi"] - c_hold["task_circuit_nmi"],
        "route_usage_weighted_specialization": (
            e_hold["usage_weighted_specialization"]
            - c_hold["usage_weighted_specialization"]
        ),
        "route_usage_weighted_purity": e_hold["usage_weighted_task_purity"] - c_hold["usage_weighted_task_purity"],
        "dead_circuit_fraction": e_hold["dead_circuit_fraction"] - c_hold["dead_circuit_fraction"],
        "cf_task_circuit_nmi": e_cf["task_circuit_nmi"] - c_cf["task_circuit_nmi"],
        "cf_usage_weighted_specialization": e_cf["usage_weighted_specialization"] - c_cf["usage_weighted_specialization"],
        "cf_mean_positive_advantage": e_cf["mean_positive_advantage"] - c_cf["mean_positive_advantage"],
        "cf_positive_responsibility_fraction": e_cf["positive_responsibility_fraction"] - c_cf["positive_responsibility_fraction"],
        "cf_responsible_circuits": e_cf["responsible_circuits"] - c_cf["responsible_circuits"],
    }
    return result


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def decide(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    accuracy = [item["delta"]["accuracy_pp"] for item in seeds]
    nmi = [item["delta"]["route_task_circuit_nmi"] for item in seeds]
    spec = [item["delta"]["route_usage_weighted_specialization"] for item in seeds]
    cf_nmi = [item["delta"]["cf_task_circuit_nmi"] for item in seeds]
    cf_spec = [item["delta"]["cf_usage_weighted_specialization"] for item in seeds]
    dead = [item["delta"]["dead_circuit_fraction"] for item in seeds]
    accuracy_gate = mean(accuracy) >= 2.0 and min(accuracy) >= -1.0
    specialization_gate = (
        mean(nmi) >= 0.05
        and mean(spec) >= 0.05
        and mean(cf_nmi) >= 0.05
        and mean(cf_spec) >= 0.05
        and mean(dead) <= 0.0
        and min(accuracy) >= -1.0
    )
    accepted = accuracy_gate or specialization_gate
    return {
        "accepted_for_followup": accepted,
        "decision": "PROMISING" if accepted else "REJECTED",
        "accuracy_gate": accuracy_gate,
        "specialization_gate": specialization_gate,
        "mean_accuracy_delta_pp": mean(accuracy),
        "mean_route_nmi_delta": mean(nmi),
        "mean_route_specialization_delta": mean(spec),
        "mean_cf_nmi_delta": mean(cf_nmi),
        "mean_cf_specialization_delta": mean(cf_spec),
        "mean_dead_fraction_delta": mean(dead),
        "per_seed_accuracy_delta_pp": accuracy,
        "criteria": {
            "accuracy": "held-out mean >= +2.0 pp and no seed worse than -1.0 pp",
            "specialization": (
                "route and counterfactual NMI/specialization each mean >= +0.05, "
                "no held-out accuracy seed below -1.0 pp, and no dead-fraction increase"
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# P-002 — Route-exposure warmup audit",
        "",
        f"**Decision:** `{report['decision']['decision']}`",
        "",
        "The treatment uses fixed task-stable routes only for the first "
        f"`{report['warmup_steps']}` training steps, then switches to the "
        "unchanged learned hard router. The control learns routes from step 1.",
        "",
        "| Seed | Arm | Held-out acc | In-domain acc | Held-out route NMI | Route specialization | CF NMI | CF specialization | Dead | Train s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in report["seeds"]:
        for arm in ("control", "exposure"):
            evaluation = seed[arm]["evaluation"]
            heldout = seed[arm]["heldout"]
            functional = seed[arm]["counterfactual_bank"]
            training = seed[arm]["training"]
            lines.append(
                f"| {seed['seed']} | {arm} | {heldout['accuracy']*100:.2f}% | "
                f"{evaluation['accuracy']*100:.2f}% | {heldout['task_circuit_nmi']:.4f} | "
                f"{heldout['usage_weighted_specialization']:.4f} | "
                f"{functional['task_circuit_nmi']:.4f} | "
                f"{functional['usage_weighted_specialization']:.4f} | "
                f"{heldout['dead_circuit_fraction']*32:.0f}/32 | "
                f"{training['seconds']:.1f} |"
            )
        delta = seed["delta"]
        lines.append(
            f"| {seed['seed']} | **delta** | **{delta['accuracy_pp']:+.2f} pp** | "
            f"**{delta['in_domain_accuracy_pp']:+.2f} pp** | "
            f"**{delta['route_task_circuit_nmi']:+.4f}** | "
            f"**{delta['route_usage_weighted_specialization']:+.4f}** | "
            f"**{delta['cf_task_circuit_nmi']:+.4f}** | "
            f"**{delta['cf_usage_weighted_specialization']:+.4f}** | "
            f"**{delta['dead_circuit_fraction']*32:+.1f}/32** | — |"
        )
    decision = report["decision"]
    lines += [
        "",
        "## Gate",
        "",
        f"- Accuracy: `{decision['accuracy_gate']}` — {decision['criteria']['accuracy']}.",
        f"- Specialization: `{decision['specialization_gate']}` — {decision['criteria']['specialization']}.",
        "",
        "The router class, candidate pool, circuit body, recurrent state update, "
        "active circuit count, optimizer, data stream, and total steps are paired. "
        "The only treatment difference is the initial task-stable route exposure.",
        "Training gradient exposure is recorded per sampled circuit; inference active "
        "parameters remain exactly the same in both arms.",
        "",
        "## Interpretation",
        "",
        "A `PROMISING` result is evidence for a follow-up schedule study, not an "
        "automatic default change. A `REJECTED` result closes this simple warmup "
        "recipe while leaving more structured circuit initialization or credit "
        "assignment as separate hypotheses.",
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


def append_registry(report: dict[str, Any], path: Path) -> None:
    marker = "### C-P002-EXPOSURE-WARMUP-001"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    decision = report["decision"]
    status = "PROMISING" if decision["accepted_for_followup"] else "REJECTED"
    block = (
        f"\n\n{marker} — Initial task-stable route exposure\n\n"
        f"**Status:** `{status}`  \n"
        "**Muammo:** P-002 / P-003  \n"
        f"**Natija:** seed17/18 held-out mean accuracy delta "
        f"`{decision['mean_accuracy_delta_pp']:+.3f} pp`; route NMI delta "
        f"`{decision['mean_route_nmi_delta']:+.5f}`; route specialization delta "
        f"`{decision['mean_route_specialization_delta']:+.5f}`; counterfactual "
        f"NMI/specialization deltas `{decision['mean_cf_nmi_delta']:+.5f}`/"
        f"`{decision['mean_cf_specialization_delta']:+.5f}`. "
        f"Birinchi `{report['warmup_steps']}` qadamda task-stable route, undan "
        "keyin oddiy learned hard router ishladi; router va circuit body o‘zgarmadi.\n"
    )
    path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P-002 route exposure warmup benchmark")
    parser.add_argument("--config", default="configs/ne_p002_20m_32.yaml")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--cf-batches", type=int, default=6)
    parser.add_argument("--cf-examples-per-task", type=int, default=8)
    parser.add_argument("--min-advantage", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--output", default="results/runs/p002_route_exposure_seed17_18.json")
    parser.add_argument("--markdown", default="results/P002_ROUTE_EXPOSURE_WARMUP_AUDIT.md")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/p002_route_exposure")
    parser.add_argument("--update-problems", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, smoke=False)
    if args.warmup_steps < 0 or args.warmup_steps > args.steps:
        raise ValueError("warmup-steps must be between 0 and steps")
    if args.smoke:
        args.steps = min(args.steps, 4)
        args.warmup_steps = min(args.warmup_steps, args.steps // 2)
        args.eval_batches = 1
        args.examples_per_task = 2
        args.cf_batches = 1
        args.cf_examples_per_task = 1
        args.min_advantage = 0.0
        args.seeds = args.seeds[:1]
        config = copy.deepcopy(config)
        config.update(d_model=64, state_dim=64, circuit_rank=4, batch_size=32)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    seeds = [run_seed(config, seed, device, args) for seed in args.seeds]
    command = (
        f"python benchmark_p002_route_exposure.py --config {args.config} "
        f"--steps {args.steps} --warmup-steps {args.warmup_steps} "
        f"--device {args.device} --seeds {' '.join(map(str, args.seeds))} "
        "--update-problems"
    )
    report = {
        "experiment": "P-002 initial task-stable route exposure warmup",
        "hypothesis": (
            "giving each task a stable circuit group during early training will "
            "increase circuit gradient exposure and reduce route fragmentation "
            "before learned routing takes over"
        ),
        "config": config,
        "warmup_steps": args.warmup_steps,
        "seeds": seeds,
        "decision": decide(seeds),
        "command": command,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.markdown))
    if args.update_problems:
        append_registry(report, Path("problems.md"))
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
