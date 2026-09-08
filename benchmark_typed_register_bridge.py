"""2x2 audit for a differentiable typed intermediate-value register.

The bridge is intentionally opt-in.  This benchmark separates the architecture
change from the auxiliary stage objective on the same frozen Native checkpoint:

  old model / final loss
  old model / final + composition stage loss
  typed bridge / final loss
  typed bridge / final + composition stage loss

All four arms see the same generated batch at every step and use the same
held-out evaluator.  The bridge's value basis is zero-initialized, so its
initial forward pass matches the old model exactly.
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


def _load_checkpoint(path: Path, device: torch.device,
                     typed_register_bridge: bool,
                     bridge_mode: str = "soft") -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    config["typed_register_bridge"] = typed_register_bridge
    config["register_bridge_mode"] = bridge_mode
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("typed register bridge requires NeuralEngineV0")
    missing, unexpected = model.load_state_dict(
        payload["model_state"], strict=False,
    )
    expected_missing = (
        ["register_value_embedding.weight"] if typed_register_bridge else []
    )
    if sorted(missing) != sorted(expected_missing) or unexpected:
        raise ValueError(
            f"unexpected checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.to(device)
    return model, config


def _train_arms(arms: dict[str, NeuralEngineV0], config: dict,
                device: torch.device, steps: int, stage_weight: float) -> dict:
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("learning_rate", 3e-4)),
            weight_decay=float(config.get("weight_decay", 0.01)),
        )
        for name, model in arms.items()
    }
    for model in arms.values():
        model.train()
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config.get("seed", 17)) + 1,
        value_min=int(config.get("train_value_min", 0)),
        value_max=int(config.get("train_value_max", 63)),
        split=str(config.get("train_split", "all")),
    )
    batch_size = int(config.get("batch_size", 128))
    losses = {name: [] for name in arms}
    started = time.perf_counter()
    peak_vram = 0
    for step in range(1, steps + 1):
        batch = generator.task_balanced_batch(batch_size, device)
        for name, model in arms.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            logits, stats = model(
                batch.inputs,
                adaptive=False,
                coverage=float(config.get("routing_coverage_weight", 0.0)) > 0.0,
            )
            weight = stage_weight if name in {"stage_only", "bridge_stage"} else 0.0
            loss = _loss(model, logits, stats, batch, weight, config)
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
            summary = " ".join(
                f"{name}={losses[name][-1]:.4f}" for name in arms
            )
            print(f"step={step:04d}/{steps} {summary}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "steps": int(steps),
        "seconds": time.perf_counter() - started,
        "mean_loss": {
            name: sum(values) / len(values) for name, values in losses.items()
        },
        "peak_vram_mb": peak_vram,
        "stage_loss_weight": float(stage_weight),
        "stage_loss_depth_min": 2,
    }


def _run(path: Path, args: argparse.Namespace,
         device: torch.device) -> dict:
    control, config = _load_checkpoint(path, device, False, args.bridge_mode)
    stage_only, _ = _load_checkpoint(path, device, False, args.bridge_mode)
    bridge_only, _ = _load_checkpoint(path, device, True, args.bridge_mode)
    bridge_stage, _ = _load_checkpoint(path, device, True, args.bridge_mode)
    arms = {
        "control": control,
        "stage_only": stage_only,
        "bridge_only": bridge_only,
        "bridge_stage": bridge_stage,
    }
    eval_batches = _make_eval_batches(
        config, device, args.eval_batches, args.eval_examples_per_task, "heldout",
    )
    training = _train_arms(
        arms, config, device, args.steps, args.stage_loss_weight,
    )
    evaluations = {name: _evaluate(model, eval_batches) for name, model in arms.items()}
    baseline = evaluations["control"]
    deltas = {}
    for name, result in evaluations.items():
        deltas[name] = {
            "mean_ce": result["mean_ce"] - baseline["mean_ce"],
            "accuracy_pp": (result["accuracy"] - baseline["accuracy"]) * 100.0,
        }
    result = {
        "checkpoint": str(path),
        "model": config.get("model"),
        "seed": int(config.get("seed", -1)),
        "device": str(device),
        "training": training,
        "parameter_report": {
            name: model.parameter_report() for name, model in arms.items()
        },
        "evaluations": evaluations,
        "deltas_vs_control": deltas,
    }
    del arms
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit typed register bridge")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--stage-loss-weight", type=float, default=0.1)
    parser.add_argument("--bridge-mode", choices=("soft", "straight_through"),
                        default="soft")
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
        "bridge_mode": args.bridge_mode,
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
