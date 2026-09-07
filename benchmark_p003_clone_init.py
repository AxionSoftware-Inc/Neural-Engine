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

from benchmark_p002_specialization import evaluate_specialization
from data.generator import SyntheticTaskGenerator
from expand_checkpoint import expand_state
from grow_capacity import rank_parent_circuits
from neural_engine.model import NeuralEngineV0
from train import BatchSource, load_config, make_model, make_optimizer, seed_everything


def clone_expanded_state(
    parent_state: dict[str, torch.Tensor],
    target_model: NeuralEngineV0,
    parent_ids: torch.Tensor,
    clone_noise: float,
) -> dict[str, torch.Tensor]:
    """Expand a parent bank, then initialize only new rows from parent rows."""
    expanded = expand_state(parent_state, target_model.state_dict())
    parent_circuits = int(parent_state["circuits.bias"].shape[0])
    target_circuits = int(expanded["circuits.bias"].shape[0])
    extra = target_circuits - parent_circuits
    if extra <= 0:
        raise ValueError("target bank must be larger than parent bank")
    repeated = parent_ids.repeat((extra + parent_ids.numel() - 1) // parent_ids.numel())[:extra]
    for name in ("circuits.down", "circuits.up", "circuits.bias", "router.keys"):
        rows = parent_state[name].index_select(0, repeated)
        if clone_noise:
            rows = rows + torch.randn_like(rows) * parent_state[name].std() * clone_noise
        expanded[name][parent_circuits:] = rows
    return expanded


def make_heldout_source(config: dict[str, Any], device: torch.device, offset: int) -> BatchSource:
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]),
        int(config["seed"]) + offset,
        value_min=int(config.get("heldout_value_min", 0)),
        value_max=int(config.get("heldout_value_max", 63)),
        split="heldout",
    )
    return BatchSource(generator, 256, device)


def train_arm(
    config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    arm: str,
) -> tuple[NeuralEngineV0, dict[str, Any]]:
    seed = int(config["seed"])
    seed_everything(seed + 20000)
    model = make_model(config).to(device)
    if not isinstance(model, NeuralEngineV0):
        raise TypeError("P-003 clone initialization requires NeuralEngineV0")
    model.load_state_dict(initial_state, strict=True)
    model.routing_mode = "learned"
    optimizer = make_optimizer(model, config)
    source = BatchSource(
        SyntheticTaskGenerator(
            int(config["seq_len"]), seed + 1,
            value_min=int(config.get("train_value_min", 0)),
            value_max=int(config.get("train_value_max", 63)),
            split=str(config.get("train_split", "all")),
        ),
        int(config["batch_size"]), device, task_balanced=True,
    )
    n = model.router.num_circuits
    route_usage = torch.zeros(n, dtype=torch.long)
    losses: list[float] = []
    start = time.perf_counter()
    peak_vram = 0
    coverage_weight = float(config.get("routing_coverage_weight", 0.0))
    model.train()
    for step in range(1, args.steps + 1):
        batch = source.batch()
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(
            batch.inputs,
            adaptive=False,
            coverage=coverage_weight > 0.0,
        )
        selected = stats["selected_ids"].detach().reshape(-1)
        route_usage += torch.bincount(selected[selected.ge(0)].cpu(), minlength=n)
        loss = F.cross_entropy(logits, batch.targets)
        if config.get("adaptive_halting", False):
            halt_targets = (
                torch.arange(model.internal_steps, device=device).unsqueeze(0)
                >= (batch.depths.unsqueeze(1) - 1)
            ).float()
            loss = loss + float(config.get("halt_loss_weight", 0.1)) * F.binary_cross_entropy_with_logits(
                stats["halt_logits"], halt_targets
            )
            exit_weight = float(config.get("exit_loss_weight", 0.0))
            if exit_weight:
                exit_steps = (batch.depths - 1).clamp(0, model.internal_steps - 1)
                rows = torch.arange(batch.inputs.shape[0], device=device)
                exit_logits = stats["step_logits"][rows, exit_steps]
                loss = loss + exit_weight * F.cross_entropy(exit_logits, batch.targets)
        if "router_entropy" in stats:
            loss = loss - 0.0001 * stats["router_entropy"]
        if coverage_weight and "routing_coverage_loss" in stats:
            loss = loss + coverage_weight * stats["routing_coverage_loss"]
        loss.backward()
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
        "circuits_seen_during_training": int(route_usage.gt(0).sum()),
    }


def run_seed(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["seed"] = seed
    parent_path = Path(args.parent_template.format(seed=seed))
    if not parent_path.exists():
        if seed == 17:
            parent_path = Path("results/checkpoints/ne20_v12_coverage_matched_5000.pt")
        else:
            parent_path = Path(f"results/checkpoints/ne20_v12_coverage_matched_seed{seed}_5000.pt")
    if not parent_path.exists():
        raise FileNotFoundError(parent_path)
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    parent_state = parent_payload["model_state"]
    parent_config = dict(parent_payload["config"])
    if int(parent_config["num_circuits"]) >= int(config["num_circuits"]):
        raise ValueError("parent bank must be smaller than target bank")
    seed_everything(seed + 10000)
    census_ranked, census_counts = rank_parent_circuits(
        str(parent_path), config, device, args.census_batches, args.census_batch_size, "synthetic"
    )
    if census_ranked.numel() == 0:
        raise RuntimeError("parent route census found no used circuits")

    states: dict[str, dict[str, torch.Tensor]] = {}
    for arm in ("random", "clone"):
        seed_everything(seed + 10000)
        target_model = make_model(config)
        if not isinstance(target_model, NeuralEngineV0):
            raise TypeError("target must be NeuralEngineV0")
        if arm == "random":
            states[arm] = expand_state(parent_state, target_model.state_dict())
        else:
            states[arm] = clone_expanded_state(
                parent_state, target_model, census_ranked[: args.clone_source_count], args.clone_noise
            )
        del target_model

    result: dict[str, Any] = {
        "seed": seed,
        "parent_checkpoint": str(parent_path),
        "parent_config": parent_config,
        "target_config": config,
        "census": {
            "batches": args.census_batches,
            "batch_size": args.census_batch_size,
            "parent_circuits_seen": int((census_counts > 0).sum()),
            "top_parent_ids": census_ranked[: args.clone_source_count].tolist(),
        },
    }
    for arm in ("random", "clone"):
        checkpoint_path = (
            Path(args.checkpoint_dir) / f"p003_clone_init_seed{seed}_{arm}.pt"
            if args.checkpoint_dir else None
        )
        if args.reuse_checkpoints and checkpoint_path and checkpoint_path.exists():
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model = make_model(config).to(device)
            model.load_state_dict(payload["model_state"], strict=True)
            training = dict(payload.get("training", {}))
            training["parameter_report"] = model.parameter_report()
            training["reused_checkpoint"] = str(checkpoint_path)
            print(f"[reuse] seed={seed} arm={arm} checkpoint={checkpoint_path}", flush=True)
        else:
            model, training = train_arm(config, states[arm], device, args, arm)
        in_domain = evaluate_specialization(
            model,
            BatchSource(
                SyntheticTaskGenerator(
                    int(config["seq_len"]), seed + 2,
                    value_min=int(config.get("eval_value_min", 0)),
                    value_max=int(config.get("eval_value_max", 63)),
                    split=str(config.get("eval_split", "all")),
                ),
                256, device,
            ),
            args.eval_batches,
            args.examples_per_task,
        )
        heldout = evaluate_specialization(
            model,
            make_heldout_source(config, device, 3),
            args.eval_batches,
            args.examples_per_task,
        )
        result[arm] = {
            "training": training,
            "in_domain": in_domain,
            "heldout": heldout,
        }
        if args.checkpoint_dir:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "config": config, "arm": arm, "training": training},
                checkpoint_dir / f"p003_clone_init_seed{seed}_{arm}.pt",
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    random_arm, clone_arm = result["random"], result["clone"]
    result["delta"] = {
        "heldout_accuracy_pp": 100.0 * (clone_arm["heldout"]["accuracy"] - random_arm["heldout"]["accuracy"]),
        "in_domain_accuracy_pp": 100.0 * (clone_arm["in_domain"]["accuracy"] - random_arm["in_domain"]["accuracy"]),
        "heldout_loss": clone_arm["heldout"]["loss"] - random_arm["heldout"]["loss"],
        "route_task_circuit_nmi": clone_arm["heldout"]["task_circuit_nmi"] - random_arm["heldout"]["task_circuit_nmi"],
        "route_specialization": clone_arm["heldout"]["usage_weighted_specialization"] - random_arm["heldout"]["usage_weighted_specialization"],
        "dead_fraction": clone_arm["heldout"]["dead_circuit_fraction"] - random_arm["heldout"]["dead_circuit_fraction"],
        "training_seen_circuits": clone_arm["training"]["circuits_seen_during_training"] - random_arm["training"]["circuits_seen_during_training"],
    }
    return result


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def decide(seeds: list[dict[str, Any]]) -> dict[str, Any]:
    acc = [item["delta"]["heldout_accuracy_pp"] for item in seeds]
    nmi = [item["delta"]["route_task_circuit_nmi"] for item in seeds]
    spec = [item["delta"]["route_specialization"] for item in seeds]
    dead = [item["delta"]["dead_fraction"] for item in seeds]
    accepted = mean(acc) >= 2.0 and min(acc) >= -1.0
    return {
        "accepted_for_followup": accepted,
        "decision": "PROMISING" if accepted else "REJECTED",
        "mean_heldout_accuracy_delta_pp": mean(acc),
        "mean_route_nmi_delta": mean(nmi),
        "mean_route_specialization_delta": mean(spec),
        "mean_dead_fraction_delta": mean(dead),
        "per_seed_heldout_accuracy_delta_pp": acc,
        "criteria": "mean held-out accuracy >= +2.0 pp and no seed below -1.0 pp",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# P-003 — New-bank initialization: random vs parent-cloned rows",
        "",
        f"**Decision:** `{report['decision']['decision']}`",
        "",
        "A 20M parent checkpoint is expanded to the same 100M target. The random "
        "arm leaves new circuit/key rows at target initialization; the clone arm "
        f"copies the top `{report['clone_source_count']}` parent rows with "
        f"Gaussian noise `{report['clone_noise']}`. Both arms then receive the "
        f"same `{report['steps']}` full-bank training steps.",
        "",
        "| Seed | Arm | Held-out acc | In-domain acc | Held-out route NMI | Route specialization | Dead | Seen in training | Train s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in report["seeds"]:
        for arm in ("random", "clone"):
            x = seed[arm]
            lines.append(
                f"| {seed['seed']} | {arm} | {x['heldout']['accuracy']*100:.2f}% | "
                f"{x['in_domain']['accuracy']*100:.2f}% | {x['heldout']['task_circuit_nmi']:.4f} | "
                f"{x['heldout']['usage_weighted_specialization']:.4f} | "
                f"{x['heldout']['dead_circuit_fraction']*100:.2f}% | "
                f"{x['training']['circuits_seen_during_training']} | "
                f"{x['training']['seconds']:.1f} |"
            )
        d = seed["delta"]
        lines.append(
            f"| {seed['seed']} | **delta clone-random** | **{d['heldout_accuracy_pp']:+.2f} pp** | "
            f"**{d['in_domain_accuracy_pp']:+.2f} pp** | **{d['route_task_circuit_nmi']:+.4f}** | "
            f"**{d['route_specialization']:+.4f}** | **{d['dead_fraction']*100:+.2f} pp** | "
            f"**{d['training_seen_circuits']:+d}** | — |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The router, active budget, optimizer, training stream, target model and "
        "number of steps are unchanged. Only initialization of the newly added "
        "bank rows differs. This is an initialization test, not a claim that "
        "cloning alone solves large-bank routing.",
        "",
        "## Gate",
        "",
        f"- `{report['decision']['decision']}` — {report['decision']['criteria']}.",
        "- All route coverage, active parameter and training-time diagnostics are retained in the JSON report.",
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
    marker = "### C-P003-CLONE-INIT-001"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    d = report["decision"]
    status = "PROMISING" if d["accepted_for_followup"] else "REJECTED"
    block = (
        f"\n\n{marker} — Parent-cloned initialization of new bank rows\n\n"
        f"**Status:** `{status}`\n"
        "**Muammo:** P-002 / P-003\n"
        f"**Natija:** seed17/18 clone minus random held-out mean accuracy "
        f"`{d['mean_heldout_accuracy_delta_pp']:+.3f} pp`; route NMI delta "
        f"`{d['mean_route_nmi_delta']:+.5f}`; route specialization delta "
        f"`{d['mean_route_specialization_delta']:+.5f}`; dead fraction delta "
        f"`{d['mean_dead_fraction_delta']:+.5f}`. Yangi 100M bank qatorlari "
        f"top `{report['clone_source_count']}` parent circuitdan noise bilan "
        "initsializatsiya qilindi; qolgan protocol random arm bilan bir xil.\n"
    )
    path.write_text(text.rstrip() + block, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P-003 new-bank initialization benchmark")
    parser.add_argument("--config", default="configs/ne_100_v12_coverage.yaml")
    parser.add_argument("--parent-template", default="results/checkpoints/ne20_v12_coverage_matched_{seed}_5000.pt")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--census-batches", type=int, default=64)
    parser.add_argument("--census-batch-size", type=int, default=128)
    parser.add_argument("--clone-source-count", type=int, default=64)
    parser.add_argument("--clone-noise", type=float, default=0.05)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--output", default="results/runs/p003_clone_init_seed17_18.json")
    parser.add_argument("--markdown", default="results/P003_CLONE_INIT_AUDIT.md")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints/p003_clone_init")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--update-problems", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, smoke=False)
    if args.smoke:
        args.steps = min(args.steps, 4)
        args.census_batches = 1
        args.census_batch_size = 8
        args.clone_source_count = 2
        args.eval_batches = 1
        args.examples_per_task = 2
        args.seeds = args.seeds[:1]
        args.checkpoint_dir = ""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seeds = [run_seed(config, seed, device, args) for seed in args.seeds]
    command = (
        f"python benchmark_p003_clone_init.py --config {args.config} --steps {args.steps} "
        f"--device {args.device} --seeds {' '.join(map(str, args.seeds))} "
        f"--clone-source-count {args.clone_source_count} --clone-noise {args.clone_noise} "
        "--update-problems"
    )
    report = {
        "experiment": "P-003 new-bank initialization random versus parent-cloned rows",
        "hypothesis": "new circuit rows become useful faster when initialized from reusable parent primitives",
        "config": config,
        "steps": args.steps,
        "clone_source_count": args.clone_source_count,
        "clone_noise": args.clone_noise,
        "census_batches": args.census_batches,
        "census_batch_size": args.census_batch_size,
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
