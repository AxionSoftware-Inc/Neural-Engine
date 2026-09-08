"""Audit an operation-conditioned low-rank state-transition adapter.

The adapter is a neutral, opt-in transformation of the recurrent update:
the task/operator ID selects a low-rank matrix applied to the current update
before the GRU state write.  This isolates state-write mathematics from the
typed output bridge and from stage-loss weighting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benchmark_composition_stage_only import _evaluate, _make_eval_batches, _loss
from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from train import make_model, seed_everything


def _load_checkpoint(path: Path, device: torch.device, rank: int) -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    config["operation_transition_rank"] = rank
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("operation transition requires NeuralEngineV0")
    missing, unexpected = model.load_state_dict(
        payload["model_state"], strict=False,
    )
    expected_missing = (
        ["operation_transition_down", "operation_transition_up"]
        if rank else []
    )
    if sorted(missing) != sorted(expected_missing) or unexpected:
        raise ValueError(
            f"unexpected checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.to(device)
    return model, config


def _train_pair(control: NeuralEngineV0, treatment: NeuralEngineV0,
                config: dict, device: torch.device, steps: int) -> dict:
    optimizers = {
        "control": torch.optim.AdamW(
            control.parameters(),
            lr=float(config.get("learning_rate", 3e-4)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        ),
        "transition": torch.optim.AdamW(
            treatment.parameters(),
            lr=float(config.get("learning_rate", 3e-4)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        ),
    }
    control.train()
    treatment.train()
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config.get("seed", 17)) + 1,
        value_min=int(config.get("train_value_min", 0)),
        value_max=int(config.get("train_value_max", 63)),
        split=str(config.get("train_split", "all")),
    )
    batch_size = int(config.get("batch_size", 128))
    losses = {"control": [], "transition": []}
    started = time.perf_counter()
    peak_vram = 0
    for step in range(1, steps + 1):
        batch = generator.task_balanced_batch(batch_size, device)
        for name, model in (("control", control), ("transition", treatment)):
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            logits, stats = model(
                batch.inputs,
                adaptive=False,
                coverage=float(config.get("routing_coverage_weight", 0.0)) > 0.0,
            )
            loss = _loss(model, logits, stats, batch, 0.0, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config.get("grad_clip", 1.0)),
            )
            optimizer.step()
            losses[name].append(float(loss.detach().cpu()))
            if device.type == "cuda":
                peak_vram = max(
                    peak_vram,
                    int(torch.cuda.max_memory_allocated(device) // (1024 * 1024)),
                )
        if step == 1 or step == steps or step % max(1, steps // 4) == 0:
            print(
                f"step={step:04d}/{steps} control={losses['control'][-1]:.4f} "
                f"transition={losses['transition'][-1]:.4f}", flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "steps": int(steps),
        "seconds": time.perf_counter() - started,
        "mean_loss": {
            name: sum(values) / len(values) for name, values in losses.items()
        },
        "peak_vram_mb": peak_vram,
    }


def _run(path: Path, args: argparse.Namespace, device: torch.device) -> dict:
    control, config = _load_checkpoint(path, device, 0)
    transition, _ = _load_checkpoint(path, device, args.rank)
    eval_batches = _make_eval_batches(
        config, device, args.eval_batches, args.eval_examples_per_task, "heldout",
    )
    training = _train_pair(control, transition, config, device, args.steps)
    control_eval = _evaluate(control, eval_batches)
    transition_eval = _evaluate(transition, eval_batches)
    return {
        "checkpoint": str(path),
        "model": config.get("model"),
        "seed": int(config.get("seed", -1)),
        "device": str(device),
        "transition_rank": int(args.rank),
        "training": training,
        "parameter_report": {
            "control": control.parameter_report(),
            "transition": transition.parameter_report(),
        },
        "control": control_eval,
        "transition": transition_eval,
        "delta": {
            "mean_ce": transition_eval["mean_ce"] - control_eval["mean_ce"],
            "accuracy_pp": (
                transition_eval["accuracy"] - control_eval["accuracy"]
            ) * 100.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit operation-conditioned state transition")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
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
        "transition_rank": int(args.rank),
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
