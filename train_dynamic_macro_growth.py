from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from data.dynamic_composition import DynamicCompositionGenerator
from neural_engine.dynamic_register import DynamicRegisterNeuralEngine
from neural_engine.macro_growth import expand_macro_model
from neural_engine.instrumentation import count_parameters
from train_dynamic_composition import (
    make_model,
    make_optimizer,
    seed_everything,
    set_lazy_active_rows,
    evaluate,
)


def train_steps(
    model: DynamicRegisterNeuralEngine,
    optimizer: Any,
    generator: DynamicCompositionGenerator,
    config: dict[str, Any],
    *,
    steps: int,
    batch_size: int,
    device: torch.device,
    log_every: int,
    label: str,
) -> dict[str, Any]:
    model.train()
    losses: list[float] = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        batch = generator.task_balanced_batch(batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(batch.inputs)
        loss = nn.functional.cross_entropy(logits, batch.targets)
        stage_weight = float(config.get("stage_loss_weight", 0.0))
        if stage_weight and batch.stage_targets is not None and batch.stage_mask is not None:
            stage_losses = []
            for stage in range(model.max_ops):
                mask = batch.stage_mask[:, stage]
                if mask.any():
                    stage_losses.append(nn.functional.cross_entropy(
                        stats["step_logits"][mask, stage], batch.stage_targets[mask, stage]
                    ))
            if stage_losses:
                loss = loss + stage_weight * torch.stack(stage_losses).mean()
        loss = loss - 0.0001 * stats["router_entropy"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {label} step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        set_lazy_active_rows(model, optimizer, stats["selected_ids"])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if log_every and (step == 1 or step % log_every == 0 or step == steps):
            print(f"{label} step={step:05d} loss={losses[-1]:.5f}", flush=True)
    return {
        "steps": steps,
        "training_seconds": time.perf_counter() - start,
        "loss_first": losses[0],
        "loss_last": losses[-1],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.parent_config, "r", encoding="utf-8") as handle:
        parent_config = yaml.safe_load(handle)
    with open(args.target_config, "r", encoding="utf-8") as handle:
        target_config = yaml.safe_load(handle)
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    parent = make_model(parent_config).to(device)
    if not parent_config.get("macro_cell_count", 0):
        raise ValueError("parent config must enable macro cells")
    if int(target_config.get("macro_cell_count", 0)) <= int(parent_config["macro_cell_count"]):
        raise ValueError("target config must have a larger macro bank")

    train_generator = DynamicCompositionGenerator(
        max_ops=int(parent_config["max_ops"]),
        train_max_ops=int(parent_config.get("train_max_ops", parent_config["max_ops"])),
        seed=args.seed + 1,
        value_min=args.train_value_min,
        value_max=args.train_value_max,
        split="train" if args.heldout_depths else "all",
    )
    eval_generator = DynamicCompositionGenerator(
        max_ops=int(parent_config["max_ops"]),
        train_max_ops=int(parent_config.get("train_max_ops", parent_config["max_ops"])),
        seed=args.seed + 2,
        value_min=args.eval_value_min,
        value_max=args.eval_value_max,
        split="heldout" if args.heldout_depths else "all",
    )
    parent_train_eval_generator = DynamicCompositionGenerator(
        max_ops=int(parent_config["max_ops"]),
        train_max_ops=int(parent_config.get("train_max_ops", parent_config["max_ops"])),
        seed=args.seed + 3,
        value_min=args.train_value_min,
        value_max=args.train_value_max,
        split="train" if args.heldout_depths else "all",
    )
    batch_size = args.batch_size or int(parent_config["batch_size"])
    parent_optimizer = make_optimizer(parent, parent_config)
    parent_train = train_steps(
        parent, parent_optimizer, train_generator, parent_config,
        steps=args.parent_steps, batch_size=batch_size, device=device,
        log_every=args.log_every, label="parent16",
    )
    parent_eval = evaluate(parent, eval_generator, device, args.examples_per_depth)
    parent_train_eval = evaluate(
        parent, parent_train_eval_generator, device, args.examples_per_depth
    )

    parent_checkpoint = Path(args.checkpoint_dir) / f"{args.run_id}_parent.pt"
    parent_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": parent.state_dict(), "config": parent_config}, parent_checkpoint)

    grown = make_model(target_config).to(device)
    expand_macro_model(parent, grown)
    grown_optimizer = make_optimizer(grown, target_config)
    grown_train = train_steps(
        grown, grown_optimizer, train_generator, target_config,
        steps=args.growth_steps, batch_size=batch_size, device=device,
        log_every=args.log_every, label="grown256",
    )
    grown_train_eval_generator = DynamicCompositionGenerator(
        max_ops=int(parent_config["max_ops"]),
        train_max_ops=int(parent_config.get("train_max_ops", parent_config["max_ops"])),
        seed=args.seed + 3,
        value_min=args.train_value_min,
        value_max=args.train_value_max,
        split="train" if args.heldout_depths else "all",
    )
    grown_train_eval = evaluate(
        grown, grown_train_eval_generator, device, args.examples_per_depth
    )
    grown_eval_generator = DynamicCompositionGenerator(
        max_ops=int(parent_config["max_ops"]),
        train_max_ops=int(parent_config.get("train_max_ops", parent_config["max_ops"])),
        seed=args.seed + 2,
        value_min=args.eval_value_min,
        value_max=args.eval_value_max,
        split="heldout" if args.heldout_depths else "all",
    )
    grown_eval = evaluate(
        grown, grown_eval_generator, device, args.examples_per_depth
    )

    report = {
        "run_id": args.run_id,
        "device": str(device),
        "seed": args.seed,
        "parent_model": parent_config["model"],
        "grown_model": target_config["model"],
        "batch_size": batch_size,
        "parent": {
            "params": count_parameters(parent),
            "training": parent_train,
            "train": parent_train_eval,
            "evaluation": parent_eval,
        },
        "grown": {
            "params": count_parameters(grown),
            "training": grown_train,
            "train": grown_train_eval,
            "evaluation": grown_eval,
            "parameter_report": grown.parameter_report(),
        },
        "parent_checkpoint": str(parent_checkpoint),
    }
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start a larger Macro-Cell bank")
    parser.add_argument("--parent-config", default="configs/ne_dynamic_20m_macro_v0.yaml")
    parser.add_argument("--target-config", default="configs/ne_dynamic_20m_macro_v0_256.yaml")
    parser.add_argument("--parent-steps", type=int, default=3000)
    parser.add_argument("--growth-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="ne_dynamic_20m_macro_parent16_grow256")
    parser.add_argument("--output", default="results/runs")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--examples-per-depth", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--heldout-depths", action="store_true")
    parser.add_argument("--train-value-min", type=int, default=0)
    parser.add_argument("--train-value-max", type=int, default=63)
    parser.add_argument("--eval-value-min", type=int, default=0)
    parser.add_argument("--eval-value-max", type=int, default=63)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
