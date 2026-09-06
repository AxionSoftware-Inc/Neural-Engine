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


def make_source(config: dict[str, Any], device: torch.device, train: bool) -> BatchSource:
    prefix = "train" if train else "eval"
    generator = SyntheticTaskGenerator(
        config["seq_len"], int(config["seed"]) + (1 if train else 2),
        value_min=int(config.get(f"{prefix}_value_min", 0)),
        value_max=int(config.get(f"{prefix}_value_max", 63)),
        split=str(config.get(f"{prefix}_split", "all")),
    )
    return BatchSource(generator, int(config["batch_size"] if train else 256),
                       device, task_balanced=train)


def sampled_grad_norm(model: NeuralEngineV0, circuit_id: int) -> float:
    squared = None
    for name in ("down", "up", "bias"):
        parameter = getattr(model.circuits, name)
        if parameter.grad is None:
            continue
        value = parameter.grad[circuit_id].detach().float().pow(2).sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt().cpu())


@torch.no_grad()
def evaluate_specialization(model: NeuralEngineV0, source: BatchSource,
                            batches: int, examples_per_task: int) -> dict[str, Any]:
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
                matrix[task_id] += torch.bincount(routed, minlength=num_circuits).double()

    circuit_load = matrix.sum(0)
    used = circuit_load.gt(0)
    probabilities = torch.zeros_like(matrix)
    probabilities[:, used] = matrix[:, used] / circuit_load[used].unsqueeze(0)
    entropy = torch.zeros(num_circuits, dtype=torch.float64)
    purity = torch.zeros(num_circuits, dtype=torch.float64)
    if used.any():
        p = probabilities[:, used]
        entropy[used] = -(p * p.clamp_min(1e-30).log()).sum(0) / math.log(15.0)
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
        "accuracy": correct / max(total, 1),
        "loss": sum(losses) / max(len(losses), 1),
        "circuits_used": int(used.sum()),
        "dead_circuit_fraction": float((~used).double().mean()),
        "task_circuit_nmi": nmi,
        "usage_weighted_specialization": specialization,
        "usage_weighted_task_purity": weighted_purity,
        "circuit_load": circuit_load.long().tolist(),
        "per_circuit_task_entropy": entropy.tolist(),
        "per_circuit_task_purity": purity.tolist(),
        "task_circuit_counts": matrix.long().tolist(),
    }


def train_arm(config: dict[str, Any], initial_state: dict[str, torch.Tensor],
              device: torch.device, args: argparse.Namespace,
              treatment: bool) -> tuple[NeuralEngineV0, dict[str, Any]]:
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
    source = make_source(config, device, True)
    credit = CounterfactualCircuitCredit(
        model.router.num_circuits, model.internal_steps, model.active_circuits,
        interval=args.credit_interval, candidates=args.credit_candidates,
        weight=args.credit_weight, min_eligible=args.credit_min_eligible,
        seed=int(config["seed"]) + 2002,
    ) if treatment else None

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
        pending = (credit.prepare_update(model, batch.inputs, batch.targets, logits, stats, step)
                   if credit is not None else None)
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
                    "step": step, "circuit_id": pending.circuit_id,
                    "target_step": pending.target_step, "target_slot": pending.target_slot,
                    "advantage": pending.advantage,
                    "eligible_examples": pending.eligible_examples,
                    "auxiliary_loss": pending.auxiliary_loss,
                })

        nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if device.type == "cuda":
            peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated() // 2**20))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == args.steps):
            print(f"[{'ctca' if treatment else 'control'}] seed={config['seed']} "
                  f"step={step}/{args.steps} loss={losses[-1]:.4f}")

    seconds = time.perf_counter() - start
    return model, {
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
            ((grad_samples > 0) & (grad_nonzero == 0)).sum()),
        "credit_events": events,
        "credit": credit.report() if credit is not None else None,
    }


def run_seed(config: dict[str, Any], seed: int, device: torch.device,
             args: argparse.Namespace) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["seed"] = seed
    seed_everything(seed)
    initial_model = make_model(config).to(device)
    if not isinstance(initial_model, NeuralEngineV0):
        raise TypeError("P-002 requires NeuralEngineV0")
    parameter_report = initial_model.parameter_report()
    if not args.smoke and not 18_000_000 <= int(parameter_report["total_params"]) <= 22_000_000:
        raise RuntimeError(f"expected ~20M params, got {parameter_report['total_params']:,}")
    if int(config["num_circuits"]) != 32:
        raise RuntimeError("acceptance protocol requires exactly 32 circuits")
    router_class = initial_model.router.__class__.__name__
    initial_state = {k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()}
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
        name = "ctca" if treatment else "control"
        model, training = train_arm(config, initial_state, device, args, treatment)
        evaluation = evaluate_specialization(
            model, make_source(config, device, False), args.eval_batches, args.examples_per_task)
        result[name] = {"training": training, "evaluation": evaluation}
        if args.checkpoint_dir:
            path = Path(args.checkpoint_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "config": config,
                        "evaluation": evaluation}, path / f"p002_ctca_seed{seed}_{name}.pt")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    c, t = result["control"]["evaluation"], result["ctca"]["evaluation"]
    result["delta"] = {
        "accuracy_pp": 100.0 * (t["accuracy"] - c["accuracy"]),
        "loss": t["loss"] - c["loss"],
        "task_circuit_nmi": t["task_circuit_nmi"] - c["task_circuit_nmi"],
        "usage_weighted_specialization": (
            t["usage_weighted_specialization"] - c["usage_weighted_specialization"]),
        "usage_weighted_task_purity": (
            t["usage_weighted_task_purity"] - c["usage_weighted_task_purity"]),
        "dead_circuit_fraction": t["dead_circuit_fraction"] - c["dead_circuit_fraction"],
    }
    return result


def decide(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [s["delta"]["accuracy_pp"] for s in seeds]
    nmi = [s["delta"]["task_circuit_nmi"] for s in seeds]
    spec = [s["delta"]["usage_weighted_specialization"] for s in seeds]
    dead = [s["delta"]["dead_circuit_fraction"] for s in seeds]
    mean = lambda values: sum(values) / len(values)
    accuracy_gate = mean(acc) >= 2.0 and min(acc) >= -1.0
    specialization_gate = mean(nmi) >= 0.05 and mean(spec) >= 0.05 and mean(dead) <= 1/32
    accepted = accuracy_gate or specialization_gate
    return {
        "accepted": accepted,
        "decision": "ACCEPT" if accepted else "REJECTED",
        "accuracy_gate": accuracy_gate,
        "specialization_gate": specialization_gate,
        "mean_accuracy_delta_pp": mean(acc),
        "mean_task_circuit_nmi_delta": mean(nmi),
        "mean_usage_weighted_specialization_delta": mean(spec),
        "mean_dead_circuit_fraction_delta": mean(dead),
        "per_seed_accuracy_delta_pp": acc,
        "criteria": {
            "accuracy": "mean >= +2.0 pp and no seed worse than -1.0 pp",
            "specialization": "NMI >= +0.05, specialization >= +0.05, dead delta <= +1/32",
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# P-002 — Counterfactual Tournament Credit Assignment", "",
             f"**Decision:** `{report['decision']['decision']}`", "",
             "| Seed | Arm | Accuracy | NMI | Specialization | Dead | Train s |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for seed in report["seeds"]:
        for arm in ("control", "ctca"):
            e, tr = seed[arm]["evaluation"], seed[arm]["training"]
            lines.append(f"| {seed['seed']} | {arm} | {e['accuracy']*100:.2f}% | "
                         f"{e['task_circuit_nmi']:.4f} | {e['usage_weighted_specialization']:.4f} | "
                         f"{e['dead_circuit_fraction']*32:.0f}/32 | {tr['seconds']:.1f} |")
        d = seed["delta"]
        lines.append(f"| {seed['seed']} | **delta** | **{d['accuracy_pp']:+.2f} pp** | "
                     f"**{d['task_circuit_nmi']:+.4f}** | "
                     f"**{d['usage_weighted_specialization']:+.4f}** | "
                     f"**{d['dead_circuit_fraction']*32:+.1f}/32** | — |")
    decision = report["decision"]
    lines += ["", "## Gate", "",
              f"- Accuracy: `{decision['accuracy_gate']}` — {decision['criteria']['accuracy']}.",
              f"- Specialization: `{decision['specialization_gate']}` — {decision['criteria']['specialization']}.",
              "", "Inference path and active parameters are identical to control. Training-only CTCA cost "
              "is recorded in the JSON report.", "", "## Reproduction", "", "```bash",
              report["command"], "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_rejection(report: dict[str, Any], path: Path) -> None:
    if report["decision"]["accepted"]:
        return
    marker = "### C-P002-CTCA-001 — Counterfactual tournament sparse credit"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    d = report["decision"]
    block = (f"\n\n{marker}\n\n**Status:** `REJECTED`  \n**Muammo:** P-002  \n"
             f"**Natija:** seed17/18 mean accuracy delta `{d['mean_accuracy_delta_pp']:+.3f} pp`, "
             f"NMI delta `{d['mean_task_circuit_nmi_delta']:+.5f}`, specialization delta "
             f"`{d['mean_usage_weighted_specialization_delta']:+.5f}`. Pre-registered gate "
             "bajarilmadi; CTCA default uchun qabul qilinmadi.\n")
    path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P-002 CTCA paired benchmark")
    p.add_argument("--config", default="configs/ne_p002_20m_32.yaml")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--examples-per-task", type=int, default=32)
    p.add_argument("--credit-interval", type=int, default=8)
    p.add_argument("--credit-candidates", type=int, default=4)
    p.add_argument("--credit-weight", type=float, default=0.25)
    p.add_argument("--credit-min-eligible", type=int, default=16)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--output", default="results/P002_CTCA_SEED17_18.json")
    p.add_argument("--markdown", default="results/P002_CTCA_SEED17_18.md")
    p.add_argument("--checkpoint-dir", default="results/checkpoints/p002_ctca")
    p.add_argument("--update-problems-on-reject", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, smoke=False)
    if args.smoke:
        args.steps, args.eval_batches, args.examples_per_task = min(args.steps, 4), 1, 2
        args.credit_interval, args.credit_candidates, args.credit_min_eligible = 2, 2, 1
        args.seeds = args.seeds[:1]
        config = copy.deepcopy(config)
        config.update(d_model=64, state_dim=64, circuit_rank=4, batch_size=32)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seeds = [run_seed(config, seed, device, args) for seed in args.seeds]
    command = (f"python benchmark_p002_specialization.py --config {args.config} --steps {args.steps} "
               f"--device {args.device} --seeds {' '.join(map(str, args.seeds))} "
               "--update-problems-on-reject")
    report = {
        "experiment": "P002 counterfactual tournament credit assignment",
        "hypothesis": "row-local counterfactual task credit can break sparse circuit starvation",
        "config": config,
        "credit": {
            "interval": args.credit_interval, "candidates": args.credit_candidates,
            "weight": args.credit_weight, "min_eligible": args.credit_min_eligible,
            "planned_extra_forward_equivalents_per_step": (args.credit_candidates + 1) / args.credit_interval,
            "planned_extra_circuit_backward_equivalents_per_step": 1 / args.credit_interval,
        },
        "seeds": seeds, "decision": decide(seeds), "command": command,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, Path(args.markdown))
    if args.update_problems_on_reject:
        append_rejection(report, Path("problems.md"))
    print(json.dumps(report["decision"], indent=2))


if __name__ == "__main__":
    main()
