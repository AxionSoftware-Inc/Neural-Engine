"""Audit auxiliary supervision on recurrent state instead of final output head.

The control receives only the final task loss.  The treatment adds a small
state-only class head used for depth-2/3 intermediate targets.  The head is
not part of the stats-free serving path and does not feed values back into the
model, isolating whether the recurrent state itself can learn a useful
composition representation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_composition_stage_only import (
    _evaluate,
    _make_eval_batches,
    _loss as base_loss,
)
from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from train import make_model, seed_everything


def _load_checkpoint(path: Path, device: torch.device,
                     state_stage_head: bool) -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    config["state_stage_head"] = state_stage_head
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("state stage head requires NeuralEngineV0")
    missing, unexpected = model.load_state_dict(
        payload["model_state"], strict=False,
    )
    expected_missing = []
    if state_stage_head:
        expected_missing = [
            "state_stage_head.0.weight", "state_stage_head.0.bias",
            "state_stage_head.1.weight", "state_stage_head.1.bias",
        ]
    if sorted(missing) != sorted(expected_missing) or unexpected:
        raise ValueError(
            f"unexpected checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.to(device)
    return model, config


def _state_stage_loss(stats: dict, batch, weight: float) -> torch.Tensor:
    if not weight:
        return torch.zeros((), device=batch.targets.device)
    logits = stats.get("state_stage_logits")
    if logits is None:
        raise ValueError("state stage loss requires state_stage_head=True")
    valid = batch.stage_mask & batch.depths.unsqueeze(1).ge(2)
    losses = []
    for stage in range(min(logits.shape[1], batch.stage_targets.shape[1])):
        mask = valid[:, stage]
        if mask.any():
            losses.append(F.cross_entropy(
                logits[mask, stage], batch.stage_targets[mask, stage],
            ))
    return weight * torch.stack(losses).mean() if losses else logits.sum() * 0.0


def _train_pair(control: NeuralEngineV0, treatment: NeuralEngineV0,
                config: dict, device: torch.device, steps: int,
                state_stage_weight: float) -> dict:
    control.train()
    treatment.train()
    models = (control, treatment)
    optimizers = tuple(
        torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("learning_rate", 3e-4)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
        for model in models
    )
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config.get("seed", 17)) + 1,
        value_min=int(config.get("train_value_min", 0)),
        value_max=int(config.get("train_value_max", 63)),
        split=str(config.get("train_split", "all")),
    )
    batch_size = int(config.get("batch_size", 128))
    losses = [[], []]
    started = time.perf_counter()
    peak_vram = 0
    for step in range(1, steps + 1):
        batch = generator.task_balanced_batch(batch_size, device)
        for index, (model, optimizer) in enumerate(zip(models, optimizers)):
            optimizer.zero_grad(set_to_none=True)
            logits, stats = model(
                batch.inputs,
                adaptive=False,
                coverage=float(config.get("routing_coverage_weight", 0.0)) > 0.0,
            )
            loss = base_loss(model, logits, stats, batch, 0.0, config)
            if index == 1:
                loss = loss + _state_stage_loss(
                    stats, batch, state_stage_weight,
                )
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
                f"state_stage={losses[1][-1]:.4f}", flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "steps": int(steps),
        "seconds": time.perf_counter() - started,
        "mean_loss": {
            "control": sum(losses[0]) / len(losses[0]),
            "state_stage": sum(losses[1]) / len(losses[1]),
        },
        "peak_vram_mb": peak_vram,
        "state_stage_loss_weight": float(state_stage_weight),
        "state_stage_depth_min": 2,
    }


def _run(path: Path, args, device: torch.device) -> dict:
    control, config = _load_checkpoint(path, device, False)
    treatment, _ = _load_checkpoint(path, device, True)
    eval_batches = _make_eval_batches(
        config, device, args.eval_batches, args.eval_examples_per_task, "heldout",
    )
    training = _train_pair(
        control, treatment, config, device, args.steps,
        args.state_stage_loss_weight,
    )
    control_eval = _evaluate(control, eval_batches)
    treatment_eval = _evaluate(treatment, eval_batches)
    return {
        "checkpoint": str(path),
        "model": config.get("model"),
        "seed": int(config.get("seed", -1)),
        "device": str(device),
        "training": training,
        "parameter_report": {
            "control": control.parameter_report(),
            "state_stage": treatment.parameter_report(),
        },
        "control": control_eval,
        "state_stage": treatment_eval,
        "delta": {
            "mean_ce": treatment_eval["mean_ce"] - control_eval["mean_ce"],
            "accuracy_pp": (
                treatment_eval["accuracy"] - control_eval["accuracy"]
            ) * 100.0,
        },
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Audit state-only stage supervision")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--state-stage-loss-weight", type=float, default=0.1)
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
        "state_stage_loss_weight": float(args.state_stage_loss_weight),
        "checkpoints": [
            _run(Path(checkpoint), args, device)
            for checkpoint in args.checkpoint
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
