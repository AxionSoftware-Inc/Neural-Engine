"""Train/evaluate an opt-in nonlinear candidate scorer from a Native checkpoint.

The treatment adds a zero-initialized residual to the existing query-key
score.  Both control and treatment use the same training batches and a
training-only soft candidate mixture; evaluation returns to hard top-k.  This
keeps the experiment focused on score expressiveness while preserving sparse
inference (`K` circuit bodies execute).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_route_cost_surrogate import _load_checkpoint, _make_batches
from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from train import seed_everything


def _loss(model: NeuralEngineV0, logits: torch.Tensor, stats: dict,
          targets: torch.Tensor, depths: torch.Tensor, config: dict) -> torch.Tensor:
    loss = F.cross_entropy(logits, targets)
    if config.get("adaptive_halting", False):
        steps = torch.arange(model.internal_steps, device=targets.device).unsqueeze(0)
        halt_targets = (steps >= (depths.unsqueeze(1) - 1)).float()
        loss = loss + float(config.get("halt_loss_weight", 0.1)) * F.binary_cross_entropy_with_logits(
            stats["halt_logits"], halt_targets,
        )
        exit_weight = float(config.get("exit_loss_weight", 0.0))
        if exit_weight:
            exit_steps = (depths - 1).clamp(0, model.internal_steps - 1)
            rows = torch.arange(targets.shape[0], device=targets.device)
            exit_logits = stats["step_logits"][rows, exit_steps]
            loss = loss + exit_weight * F.cross_entropy(exit_logits, targets)
    coverage_weight = float(config.get("routing_coverage_weight", 0.0))
    if coverage_weight and "routing_coverage_loss" in stats:
        loss = loss + coverage_weight * stats["routing_coverage_loss"]
    if "router_entropy" in stats:
        loss = loss - 0.0001 * stats["router_entropy"]
    return loss


def _train_pair(control: NeuralEngineV0, treatment: NeuralEngineV0,
                config: dict, device: torch.device, steps: int,
                soft_temperature: float, seed: int) -> dict:
    control.train()
    treatment.train()
    control.router.soft_routing_temperature = soft_temperature
    treatment.router.soft_routing_temperature = soft_temperature
    learning_rate = float(config.get("learning_rate", 3e-4))
    weight_decay = float(config.get("weight_decay", 0.01))
    control_optimizer = torch.optim.AdamW(
        control.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    treatment_optimizer = torch.optim.AdamW(
        treatment.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), seed + 1,
        value_min=int(config.get("train_value_min", 0)),
        value_max=int(config.get("train_value_max", 63)),
        split=str(config.get("train_split", "all")),
    )
    batch_size = int(config.get("batch_size", 128))
    started = time.perf_counter()
    losses = {"control": [], "treatment": []}
    peak_vram = 0
    for step in range(1, steps + 1):
        batch = generator.task_balanced_batch(batch_size, device)
        for name, model, optimizer in (
            ("control", control, control_optimizer),
            ("treatment", treatment, treatment_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            logits, stats = model(
                batch.inputs, adaptive=False,
                coverage=float(config.get("routing_coverage_weight", 0.0)) > 0.0,
            )
            loss = _loss(model, logits, stats, batch.targets, batch.depths, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            optimizer.step()
            losses[name].append(float(loss.detach().cpu()))
            if device.type == "cuda":
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated(device) // (1024 * 1024)))
        if step == 1 or step == steps or (step % max(1, steps // 4) == 0):
            print(
                f"step={step:04d}/{steps} control={losses['control'][-1]:.4f} "
                f"treatment={losses['treatment'][-1]:.4f}", flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "steps": int(steps),
        "seconds": time.perf_counter() - started,
        "mean_loss": {name: sum(values) / len(values) for name, values in losses.items()},
        "peak_vram_mb": peak_vram,
        "soft_routing_temperature": float(soft_temperature),
    }


@torch.inference_mode()
def _evaluate(model: NeuralEngineV0,
              batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    model.eval()
    model.router.soft_routing_temperature = 0.0
    ce = 0.0
    correct = 0
    count = 0
    unique = set()
    for inputs, targets in batches:
        logits, stats = model(inputs, adaptive=False, collect_stats=True)
        ce += float(F.cross_entropy(logits, targets, reduction="sum").cpu())
        correct += int(logits.argmax(dim=-1).eq(targets).sum().cpu())
        count += int(targets.numel())
        unique.update(int(value) for value in stats["selected_ids"].detach().cpu().reshape(-1).tolist())
    return {
        "examples": count,
        "mean_ce": ce / count,
        "accuracy": correct / count,
        "unique_selected_circuits": len(unique),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit nonlinear Native Engine route scorer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--soft-temperature", type=float, default=0.5)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-examples-per-task", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    seed_everything(17)
    path = Path(args.checkpoint)
    base, config = _load_checkpoint(path, device)
    control = copy.deepcopy(base)
    treatment = copy.deepcopy(base)
    if not hasattr(treatment.router, "enable_candidate_score_residual"):
        raise RuntimeError("candidate score residual support is missing")
    treatment.router.enable_candidate_score_residual(hidden_dim=32)
    treatment.to(device)
    eval_batches = _make_batches(
        config, device, "heldout", args.eval_batches, args.eval_examples_per_task,
    )
    training = _train_pair(
        control, treatment, config, device, args.steps,
        args.soft_temperature, int(config.get("seed", 17)),
    )
    control_eval = _evaluate(control, eval_batches)
    treatment_eval = _evaluate(treatment, eval_batches)
    result = {
        "checkpoint": str(path),
        "model": config.get("model"),
        "seed": int(config.get("seed", -1)),
        "device": str(device),
        "active_circuits": int(treatment.active_circuits),
        "candidate_pool": int(treatment.router.candidate_pool),
        "residual_scorer_parameters": sum(
            parameter.numel() for parameter in treatment.router.candidate_score_residual.parameters()
        ),
        "training": training,
        "control": control_eval,
        "treatment": treatment_eval,
        "delta": {
            "mean_ce": treatment_eval["mean_ce"] - control_eval["mean_ce"],
            "accuracy_pp": (treatment_eval["accuracy"] - control_eval["accuracy"]) * 100,
        },
    }
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"config": config, "model_state": control.state_dict()},
                   save_dir / "control.pt")
        treatment_config = dict(config)
        treatment_config["candidate_score_residual"] = True
        torch.save({"config": treatment_config, "model_state": treatment.state_dict()},
                   save_dir / "treatment.pt")
        result["saved_checkpoints"] = str(save_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
