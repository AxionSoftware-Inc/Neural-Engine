"""Test composition-only intermediate-stage supervision on Native checkpoints.

The treatment adds the existing deterministic stage targets only for depth-2
and depth-3 tasks.  Depth-1 examples still train on the final target but do
not receive the auxiliary stage loss.  This isolates whether the previous
overall stage-loss regression came from spending auxiliary capacity on tasks
that were already solved.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data.generator import SyntheticTaskGenerator
from diagnose_composition_stage import _evaluate as _evaluate_stages
from neural_engine.model import NeuralEngineV0
from train import make_model, seed_everything


def _load_checkpoint(path: Path, device: torch.device,
                     state_history_mode: str = "none",
                     state_history_scale: float = 1.0) -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    config["state_history_mode"] = state_history_mode
    config["state_history_scale"] = state_history_scale
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("composition stage continuation requires NeuralEngineV0")
    missing, unexpected = model.load_state_dict(
        payload["model_state"], strict=False,
    )
    expected_missing = (
        ["state_history_task_scales"]
        if state_history_mode == "task_scaled" else []
    )
    if sorted(missing) != sorted(expected_missing) or unexpected:
        raise ValueError(
            f"unexpected checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.to(device)
    return model, config


def _make_eval_batches(config: dict, device: torch.device, count: int,
                       examples_per_task: int, split: str) -> list:
    prefix = "heldout" if split == "heldout" else "eval"
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config.get("seed", 17)) + 3,
        value_min=int(config.get(prefix + "_value_min", 0)),
        value_max=int(config.get(prefix + "_value_max", 63)),
        split=split,
    )
    return [generator.balanced_batch(examples_per_task, device) for _ in range(count)]


def _loss(model: NeuralEngineV0, logits: torch.Tensor, stats: dict,
          batch, stage_weight: float, config: dict) -> torch.Tensor:
    loss = F.cross_entropy(logits, batch.targets)
    if stage_weight:
        # The stage target is auxiliary only for genuinely compositional tasks.
        valid = batch.stage_mask & batch.depths.unsqueeze(1).ge(2)
        stage_losses = []
        for stage in range(min(stats["step_logits"].shape[1], batch.stage_targets.shape[1])):
            mask = valid[:, stage]
            if mask.any():
                stage_losses.append(F.cross_entropy(
                    stats["step_logits"][mask, stage],
                    batch.stage_targets[mask, stage],
                ))
        if stage_losses:
            loss = loss + stage_weight * torch.stack(stage_losses).mean()
    if model.adaptive_halting:
        halt_targets = (torch.arange(model.internal_steps, device=batch.targets.device).unsqueeze(0)
                        >= (batch.depths.unsqueeze(1) - 1)).float()
        loss = loss + float(config.get("halt_loss_weight", 0.1)) * F.binary_cross_entropy_with_logits(
            stats["halt_logits"], halt_targets,
        )
        exit_weight = float(config.get("exit_loss_weight", 0.0))
        if exit_weight:
            exit_steps = (batch.depths - 1).clamp(0, model.internal_steps - 1)
            rows = torch.arange(batch.targets.shape[0], device=batch.targets.device)
            loss = loss + exit_weight * F.cross_entropy(
                stats["step_logits"][rows, exit_steps], batch.targets,
            )
    coverage_weight = float(config.get("routing_coverage_weight", 0.0))
    if coverage_weight and "routing_coverage_loss" in stats:
        loss = loss + coverage_weight * stats["routing_coverage_loss"]
    if "router_entropy" in stats:
        loss = loss - 0.0001 * stats["router_entropy"]
    return loss


@torch.inference_mode()
def _evaluate(model: NeuralEngineV0, batches: list) -> dict:
    model.eval()
    ce = 0.0
    correct = 0
    examples = 0
    for batch in batches:
        logits, _ = model(batch.inputs, adaptive=False, collect_stats=False)
        ce += float(F.cross_entropy(logits, batch.targets, reduction="sum").cpu())
        correct += int(logits.argmax(dim=-1).eq(batch.targets).sum().cpu())
        examples += int(batch.targets.numel())
    stage = _evaluate_stages(model, batches)
    return {
        "examples": examples,
        "mean_ce": ce / examples,
        "accuracy": correct / examples,
        "stage_rows": stage["stage_rows"],
        "task_rows": stage["task_rows"],
    }


def _train_pair(control: NeuralEngineV0, treatment: NeuralEngineV0,
                config: dict, device: torch.device, steps: int,
                stage_weight: float) -> dict:
    control.train()
    treatment.train()
    optimizers = [
        torch.optim.AdamW(
            control.parameters(),
            lr=float(config.get("learning_rate", 3e-4)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        ),
        torch.optim.AdamW(
            treatment.parameters(),
            lr=float(config.get("learning_rate", 3e-4)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        ),
    ]
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config.get("seed", 17)) + 1,
        value_min=int(config.get("train_value_min", 0)),
        value_max=int(config.get("train_value_max", 63)),
        split=str(config.get("train_split", "all")),
    )
    batch_size = int(config.get("batch_size", 128))
    started = time.perf_counter()
    losses = [[], []]
    peak_vram = 0
    for step in range(1, steps + 1):
        batch = generator.task_balanced_batch(batch_size, device)
        for index, (model, optimizer, weight) in enumerate((
            (control, optimizers[0], 0.0),
            (treatment, optimizers[1], stage_weight),
        )):
            optimizer.zero_grad(set_to_none=True)
            logits, stats = model(
                batch.inputs,
                adaptive=False,
                coverage=float(config.get("routing_coverage_weight", 0.0)) > 0.0,
            )
            loss = _loss(model, logits, stats, batch, weight, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.get("grad_clip", 1.0)),
            )
            optimizer.step()
            losses[index].append(float(loss.detach().cpu()))
            if device.type == "cuda":
                peak_vram = max(
                    peak_vram,
                    int(torch.cuda.max_memory_allocated(device) // (1024 * 1024)),
                )
        if step == 1 or step == steps or step % max(1, steps // 4) == 0:
            print(
                f"step={step:04d}/{steps} control={losses[0][-1]:.4f} "
                f"treatment={losses[1][-1]:.4f}", flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "steps": int(steps),
        "seconds": time.perf_counter() - started,
        "mean_loss": {
            "control": sum(losses[0]) / len(losses[0]),
            "treatment": sum(losses[1]) / len(losses[1]),
        },
        "peak_vram_mb": peak_vram,
        "stage_loss_weight": float(stage_weight),
        "stage_loss_depth_min": 2,
    }


def _run(path: Path, args: argparse.Namespace, device: torch.device) -> dict:
    base, config = _load_checkpoint(
        path, device, args.state_history_mode, args.state_history_scale,
    )
    control = copy.deepcopy(base)
    treatment = copy.deepcopy(base)
    eval_batches = _make_eval_batches(
        config, device, args.eval_batches, args.eval_examples_per_task, "heldout",
    )
    training = _train_pair(
        control, treatment, config, device, args.steps, args.stage_loss_weight,
    )
    control_eval = _evaluate(control, eval_batches)
    treatment_eval = _evaluate(treatment, eval_batches)
    result = {
        "checkpoint": str(path),
        "model": config.get("model"),
        "seed": int(config.get("seed", -1)),
        "device": str(device),
        "training": training,
        "control": control_eval,
        "treatment": treatment_eval,
        "delta": {
            "mean_ce": treatment_eval["mean_ce"] - control_eval["mean_ce"],
            "accuracy_pp": (treatment_eval["accuracy"] - control_eval["accuracy"]) * 100.0,
            "stage_accuracy_pp": {},
        },
    }
    for control_row, treatment_row in zip(
        control_eval["stage_rows"], treatment_eval["stage_rows"],
    ):
        stage = str(control_row["stage"])
        if control_row["accuracy"] is not None and treatment_row["accuracy"] is not None:
            result["delta"]["stage_accuracy_pp"][stage] = (
                treatment_row["accuracy"] - control_row["accuracy"]
            ) * 100.0
    del base, control, treatment
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit composition-only stage supervision")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--stage-loss-weight", type=float, default=0.1)
    parser.add_argument("--state-history-mode", choices=("none", "sum", "task_scaled"),
                        default="none")
    parser.add_argument("--state-history-scale", type=float, default=1.0)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-examples-per-task", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(17)
    device = torch.device(args.device)
    result = {
        "device": str(device),
        "steps": int(args.steps),
        "stage_loss_weight": float(args.stage_loss_weight),
        "state_history_mode": args.state_history_mode,
        "state_history_scale": float(args.state_history_scale),
        "checkpoints": [],
    }
    for checkpoint_name in args.checkpoint:
        result["checkpoints"].append(_run(Path(checkpoint_name), args, device))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
