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
from train import load_config, make_model, make_optimizer, seed_everything


def paired_initial_states(
    independent_config: dict[str, Any],
    shared_config: dict[str, Any],
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Keep every pre-existing parameter identical across the two bank modes."""
    seed_everything(seed)
    independent = make_model(independent_config)
    if not isinstance(independent, NeuralEngineV0):
        raise TypeError("P-002 requires NeuralEngineV0")
    independent_state = {
        key: value.detach().cpu().clone()
        for key, value in independent.state_dict().items()
    }
    seed_everything(seed)
    shared = make_model(shared_config)
    if not isinstance(shared, NeuralEngineV0):
        raise TypeError("P-002 requires NeuralEngineV0")
    shared_state = {
        key: value.detach().cpu().clone()
        for key, value in shared.state_dict().items()
    }
    for key, value in independent_state.items():
        if key not in shared_state or shared_state[key].shape != value.shape:
            raise RuntimeError(f"paired state mismatch for {key}")
        shared_state[key] = value.clone()
    del independent, shared
    return {"independent": independent_state, "shared_residual": shared_state}


def train_arm(
    config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    arm: str,
) -> tuple[NeuralEngineV0, dict[str, Any]]:
    seed = int(config["seed"])
    seed_everything(seed)
    model = make_model(config).to(device)
    if not isinstance(model, NeuralEngineV0):
        raise TypeError("P-002 requires NeuralEngineV0")
    model.load_state_dict(initial_state, strict=True)
    model.routing_mode = "learned"
    if getattr(model.router, "soft_routing_temperature", 0.0) != 0.0:
        raise ValueError("shared-residual experiment requires hard routing")
    if config.get("route_exploration_prob", 0.0) or config.get("routing_coverage_weight", 0.0):
        raise ValueError("shared-residual experiment forbids exploration and coverage")
    if config.get("adaptive_halting", False):
        raise ValueError("shared-residual experiment requires fixed internal steps")

    optimizer = make_optimizer(model, config)
    source = make_source(config, device, "train")
    n = model.router.num_circuits
    route_usage = torch.zeros(n, dtype=torch.long)
    grad_samples = torch.zeros(n, dtype=torch.long)
    grad_nonzero = torch.zeros(n, dtype=torch.long)
    grad_norm_sum = torch.zeros(n, dtype=torch.float64)
    losses: list[float] = []
    start = time.perf_counter()
    peak_vram = 0
    model.train()
    for step in range(1, args.steps + 1):
        batch = source.batch()
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(batch.inputs, adaptive=False)
        selected = stats["selected_ids"].detach().reshape(-1)
        selected = selected[selected.ge(0)].cpu()
        route_usage += torch.bincount(selected, minlength=n)
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
            peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated(device) // 2**20))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == args.steps):
            print(
                f"[{arm}] seed={seed} step={step}/{args.steps} loss={losses[-1]:.4f}",
                flush=True,
            )
    model.routing_mode = "learned"
    seconds = time.perf_counter() - start
    return model, {
        "arm": arm,
        "parameter_report": model.parameter_report(),
        "seconds": seconds,
        "samples_per_second": args.steps * int(config["batch_size"]) / max(seconds, 1e-9),
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
    }


def run_seed(
    independent_config: dict[str, Any],
    shared_config: dict[str, Any],
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    independent_config = copy.deepcopy(independent_config)
    shared_config = copy.deepcopy(shared_config)
    independent_config["seed"] = seed
    shared_config["seed"] = seed
    states = paired_initial_states(independent_config, shared_config, seed)
    result: dict[str, Any] = {"seed": seed}
    for arm, config in (
        ("independent", independent_config),
        ("shared_residual", shared_config),
    ):
        model, training = train_arm(config, states[arm], device, args, arm)
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
        result[arm] = {
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
                    "arm": arm,
                    "training": training,
                    "evaluation": evaluation,
                    "heldout": heldout,
                    "counterfactual_bank": functional,
                },
                checkpoint_dir / f"p002_shared_residual_seed{seed}_{arm}.pt",
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    independent = result["independent"]
    shared = result["shared_residual"]
    i_eval, s_eval = independent["evaluation"], shared["evaluation"]
    i_hold, s_hold = independent["heldout"], shared["heldout"]
    i_cf, s_cf = independent["counterfactual_bank"], shared["counterfactual_bank"]
    result["delta"] = {
        "heldout_accuracy_pp": 100.0 * (s_hold["accuracy"] - i_hold["accuracy"]),
        "in_domain_accuracy_pp": 100.0 * (s_eval["accuracy"] - i_eval["accuracy"]),
        "heldout_loss": s_hold["loss"] - i_hold["loss"],
        "route_task_circuit_nmi": s_hold["task_circuit_nmi"] - i_hold["task_circuit_nmi"],
        "route_specialization": s_hold["usage_weighted_specialization"] - i_hold["usage_weighted_specialization"],
        "dead_circuit_fraction": s_hold["dead_circuit_fraction"] - i_hold["dead_circuit_fraction"],
        "cf_task_circuit_nmi": s_cf["task_circuit_nmi"] - i_cf["task_circuit_nmi"],
        "cf_specialization": s_cf["usage_weighted_specialization"] - i_cf["usage_weighted_specialization"],
        "cf_mean_positive_advantage": s_cf["mean_positive_advantage"] - i_cf["mean_positive_advantage"],
        "zero_main_gradient_sample_circuits": (
            shared["training"]["zero_main_gradient_sample_circuits"]
            - independent["training"]["zero_main_gradient_sample_circuits"]
        ),
    }
    return result


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def decide(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    quality = [item["delta"]["heldout_accuracy_pp"] for item in seeds]
    cf_nmi = [item["delta"]["cf_task_circuit_nmi"] for item in seeds]
    cf_spec = [item["delta"]["cf_specialization"] for item in seeds]
    cf_adv = [item["delta"]["cf_mean_positive_advantage"] for item in seeds]
    dead = [item["delta"]["dead_circuit_fraction"] for item in seeds]
    quality_gate = mean(quality) >= 2.0 and min(quality) >= -1.0
    specialization_gate = (
        mean(cf_nmi) >= 0.05
        and mean(cf_spec) >= 0.05
        and mean(cf_adv) >= 0.01
        and min(cf_adv) >= -0.005
        and mean(dead) <= 1 / 32
    )
    accepted = quality_gate or specialization_gate
    return {
        "accepted_for_followup": accepted,
        "decision": "PROMISING" if accepted else "REJECTED",
        "quality_gate": quality_gate,
        "specialization_gate": specialization_gate,
        "mean_heldout_accuracy_delta_pp": mean(quality),
        "mean_cf_nmi_delta": mean(cf_nmi),
        "mean_cf_specialization_delta": mean(cf_spec),
        "mean_cf_positive_advantage_delta": mean(cf_adv),
        "mean_dead_fraction_delta": mean(dead),
        "per_seed_heldout_accuracy_delta_pp": quality,
        "criteria": {
            "quality": "held-out mean >= +2.0 pp and no seed below -1.0 pp",
            "specialization": (
                "counterfactual NMI >= +0.05, specialization >= +0.05, "
                "positive advantage >= +0.01, no seed below -0.005, dead delta <= 1/32"
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# P-002 — Shared reusable residual bank audit",
        "",
        f"**Decision:** `{report['decision']['decision']}`",
        "",
        "The independent control and shared-residual treatment start with the "
        "same router, recurrent body, output head, and per-circuit rows. The "
        "treatment adds one always-available rank-8 shared nonlinear primitive; "
        "the sparse per-circuit residual path and active circuit budget remain.",
        "",
        "| Seed | Arm | Held-out acc | In-domain acc | Route NMI | Route specialization | CF NMI | CF specialization | CF +adv | Dead | Train s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in report["seeds"]:
        for arm in ("independent", "shared_residual"):
            x = seed[arm]
            lines.append(
                f"| {seed['seed']} | {arm} | {x['heldout']['accuracy']*100:.2f}% | "
                f"{x['evaluation']['accuracy']*100:.2f}% | {x['heldout']['task_circuit_nmi']:.4f} | "
                f"{x['heldout']['usage_weighted_specialization']:.4f} | "
                f"{x['counterfactual_bank']['task_circuit_nmi']:.4f} | "
                f"{x['counterfactual_bank']['usage_weighted_specialization']:.4f} | "
                f"{x['counterfactual_bank']['mean_positive_advantage']:.4f} | "
                f"{x['heldout']['dead_circuit_fraction']*32:.0f}/32 | "
                f"{x['training']['seconds']:.1f} |"
            )
        d = seed["delta"]
        lines.append(
            f"| {seed['seed']} | **delta shared-independent** | **{d['heldout_accuracy_pp']:+.2f} pp** | "
            f"**{d['in_domain_accuracy_pp']:+.2f} pp** | **{d['route_task_circuit_nmi']:+.4f}** | "
            f"**{d['route_specialization']:+.4f}** | **{d['cf_task_circuit_nmi']:+.4f}** | "
            f"**{d['cf_specialization']:+.4f}** | **{d['cf_mean_positive_advantage']:+.4f}** | "
            f"**{d['dead_circuit_fraction']*32:+.1f}/32** | — |"
        )
    decision = report["decision"]
    lines += [
        "",
        "## Gate",
        "",
        f"- Quality: `{decision['quality_gate']}` — {decision['criteria']['quality']}.",
        f"- Functional specialization: `{decision['specialization_gate']}` — {decision['criteria']['specialization']}.",
        "",
        "The shared path is a small additional common primitive; it does not "
        "force an active circuit ID and does not change the router API. Active "
        "parameter accounting includes the shared rank-8 path.",
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
    marker = "### C-P002-SHARED-RESIDUAL-001"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    d = report["decision"]
    status = "PROMISING" if d["accepted_for_followup"] else "REJECTED"
    block = (
        f"\n\n{marker} — Shared reusable residual primitive\n\n"
        f"**Status:** `{status}`\n"
        "**Muammo:** P-002 / P-003 / P-007\n"
        f"**Natija:** seed17/18 shared-residual held-out mean accuracy delta "
        f"`{d['mean_heldout_accuracy_delta_pp']:+.3f} pp`; counterfactual "
        f"NMI/specialization/positive-advantage deltas `"
        f"{d['mean_cf_nmi_delta']:+.5f}`/`{d['mean_cf_specialization_delta']:+.5f}`/"
        f"`{d['mean_cf_positive_advantage_delta']:+.5f}`; dead fraction delta "
        f"`{d['mean_dead_fraction_delta']:+.5f}`. Shared rank-8 primitive V0’ga "
        "opt-in sifatida qo‘shildi, mustaqil per-circuit residual va router saqlandi.\n"
    )
    path.write_text(text.rstrip() + block, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P-002 shared residual bank benchmark")
    parser.add_argument("--independent-config", default="configs/ne_p002_20m_32.yaml")
    parser.add_argument("--shared-config", default="configs/ne_p002_20m_32_shared_residual.yaml")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--cf-batches", type=int, default=6)
    parser.add_argument("--cf-examples-per-task", type=int, default=8)
    parser.add_argument("--min-advantage", type=float, default=0.02)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--output", default="results/runs/p002_shared_residual_seed17_18.json")
    parser.add_argument("--markdown", default="results/P002_SHARED_RESIDUAL_AUDIT.md")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/p002_shared_residual")
    parser.add_argument("--update-problems", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    independent_config = load_config(args.independent_config, smoke=False)
    shared_config = load_config(args.shared_config, smoke=False)
    if args.smoke:
        args.steps = min(args.steps, 4)
        args.eval_batches = 1
        args.examples_per_task = 2
        args.cf_batches = 1
        args.cf_examples_per_task = 1
        args.min_advantage = 0.0
        args.seeds = args.seeds[:1]
        args.checkpoint_dir = ""
        independent_config = copy.deepcopy(independent_config)
        shared_config = copy.deepcopy(shared_config)
        independent_config.update(d_model=64, state_dim=64, circuit_rank=4, batch_size=32)
        shared_config.update(d_model=64, state_dim=64, circuit_rank=4, batch_size=32)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seeds = [
        run_seed(independent_config, shared_config, seed, device, args)
        for seed in args.seeds
    ]
    command = (
        f"python benchmark_p002_shared_residual.py --independent-config {args.independent_config} "
        f"--shared-config {args.shared_config} --steps {args.steps} --device {args.device} "
        f"--seeds {' '.join(map(str, args.seeds))} --update-problems"
    )
    report = {
        "experiment": "P-002 shared reusable residual primitive",
        "hypothesis": (
            "a small shared nonlinear circuit receiving gradient on every route "
            "will preserve reusable computation while independent rows specialize"
        ),
        "independent_config": independent_config,
        "shared_config": shared_config,
        "steps": args.steps,
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
