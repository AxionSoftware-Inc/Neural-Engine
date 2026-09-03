from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from data.composition import COMPOSITION_SPECS, CompositionalProgramGenerator
from neural_engine.dynamic_register import DynamicRegisterNeuralEngine
from neural_engine.instrumentation import count_parameters
from train import make_model, make_optimizer, seed_everything
from train_dynamic_composition import (
    make_model as make_dynamic_model,
    make_optimizer as make_dynamic_optimizer,
)


def evaluate(model: nn.Module, generator: CompositionalProgramGenerator,
             examples_per_task: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    batch = generator.balanced_batch(examples_per_task, device)
    logit_parts = []
    prediction_parts = []
    executed_parts = []
    eval_batch_size = 64
    with torch.no_grad():
        for start in range(0, batch.inputs.shape[0], eval_batch_size):
            inputs = batch.inputs[start:start + eval_batch_size]
            if isinstance(model, DynamicRegisterNeuralEngine):
                logits, stats = model(inputs)
            elif hasattr(model, "adaptive_inference"):
                logits, stats = model(inputs, adaptive=model.adaptive_inference)
            else:
                logits, stats = model(inputs)
            logit_parts.append(logits.cpu())
            prediction_parts.append(logits.argmax(dim=-1).cpu())
            if "executed_steps" in stats:
                executed_parts.append(stats["executed_steps"].float().cpu())
    logits = torch.cat(logit_parts)
    predictions = torch.cat(prediction_parts)
    targets = batch.targets.cpu()
    task_ids = batch.task_ids.cpu()
    correct = predictions.eq(targets)
    per_pair = {}
    for spec in generator.allowed_specs:
        mask = task_ids.eq(spec.task_id)
        per_pair[spec.name] = float(correct[mask].float().mean().cpu())
    result = {
        "accuracy": float(correct.float().mean().cpu()),
        "loss": float(nn.functional.cross_entropy(logits, targets).cpu()),
        "depth_accuracy": {"3": float(correct.float().mean().cpu())},
        "per_pair_accuracy": per_pair,
    }
    if executed_parts:
        executed = torch.cat(executed_parts)
        result.update({
            "avg_executed_steps": float(executed.mean().cpu()),
            "active_step_fraction": float((executed / float(model.internal_steps)).mean().cpu()),
        })
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_seed = int(config["seed"]) if args.seed is None else int(args.seed)
    seed_everything(run_seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    heldout_pairs = tuple(tuple(pair) for pair in config.get("heldout_pairs", []))
    train_value_min = int(config.get("train_value_min", 0))
    train_value_max = int(config.get("train_value_max", 63))
    eval_value_min = int(config.get("eval_value_min", 0))
    eval_value_max = int(config.get("eval_value_max", 63))
    train_combination_split = str(config.get("train_combination_split", "all"))
    eval_combination_split = str(config.get("eval_combination_split", "all"))
    train_generator = CompositionalProgramGenerator(
        seq_len=int(config["seq_len"]), seed=run_seed + 1,
        value_min=train_value_min, value_max=train_value_max,
        split="train", heldout_pairs=heldout_pairs,
        combination_split=train_combination_split)
    evaluation_split = "heldout" if heldout_pairs else "all"
    heldout_generator = CompositionalProgramGenerator(
        seq_len=int(config["seq_len"]), seed=run_seed + 2,
        value_min=eval_value_min, value_max=eval_value_max,
        split=evaluation_split, heldout_pairs=heldout_pairs,
        combination_split=eval_combination_split)
    if config.get("architecture") == "dynamic_register":
        model = make_dynamic_model(config).to(device)
        optimizer_factory = make_dynamic_optimizer
    else:
        model = make_model(config).to(device)
        optimizer_factory = make_optimizer
    if args.init_checkpoint:
        initialization = torch.load(Path(args.init_checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(initialization.get("model_state", initialization))
    optimizer = optimizer_factory(model, config)
    steps = args.steps if args.steps is not None else 1000
    coverage_weight = float(config.get("routing_coverage_weight", 0.0))
    coverage_enabled = coverage_weight > 0.0
    routing_warmup_steps = int(config.get("routing_warmup_steps", 0))
    losses = []
    start = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        if (hasattr(model, "router") and routing_warmup_steps
                and step == routing_warmup_steps + 1):
            model.router.set_routing_state(
                capacity=model.router.num_circuits,
                depth=model.router.depth,
            )
        batch = train_generator.task_balanced_batch(int(config["batch_size"]), device)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, DynamicRegisterNeuralEngine):
            logits, route_stats = model(batch.inputs)
        elif hasattr(model, "adaptive_inference"):
            logits, route_stats = model(batch.inputs, adaptive=False, coverage=coverage_enabled)
        else:
            logits, route_stats = model(batch.inputs)
        if hasattr(optimizer, "set_active_rows") and hasattr(model, "circuits"):
            selected = route_stats.get("selected_ids")
            if selected is not None:
                selected = selected.detach().reshape(-1)
                selected = selected[selected.ge(0)].unique()
                if selected.numel():
                    router_rows = selected
                    if hasattr(model.router, "factor_candidate_pool"):
                        first, second = model.router._factor_ids(selected)
                        router_rows = torch.cat([first, second]).unique()
                    active_rows = {model.router.keys: router_rows}
                    if hasattr(model.circuits, "down"):
                        active_rows.update({
                            model.circuits.down: selected,
                            model.circuits.up: selected,
                            model.circuits.bias: selected,
                        })
                    elif hasattr(model.circuits, "down_factors"):
                        first, second = model.circuits._factor_ids(selected)
                        factor_rows = torch.cat([first, second]).unique()
                        active_rows.update({
                            model.circuits.down_factors: factor_rows,
                            model.circuits.up_factors: factor_rows,
                            model.circuits.bias_factors: factor_rows,
                            model.circuits.factor_mix: selected,
                        })
                    optimizer.set_active_rows(active_rows)
        loss = nn.functional.cross_entropy(logits, batch.targets)
        stage_loss_weight = float(config.get("stage_loss_weight", 0.0))
        if stage_loss_weight and "step_logits" in route_stats:
            stage_losses = []
            for stage in range(min(route_stats["step_logits"].shape[1], batch.stage_targets.shape[1])):
                mask = batch.stage_mask[:, stage]
                if mask.any():
                    stage_losses.append(nn.functional.cross_entropy(
                        route_stats["step_logits"][mask, stage], batch.stage_targets[mask, stage]))
            if stage_losses:
                loss = loss + stage_loss_weight * torch.stack(stage_losses).mean()
        if hasattr(model, "adaptive_halting") and model.adaptive_halting:
            halt_targets = (torch.arange(model.internal_steps, device=device).unsqueeze(0)
                            >= (batch.depths.unsqueeze(1) - 1)).float()
            halt_loss = nn.functional.binary_cross_entropy_with_logits(
                route_stats["halt_logits"], halt_targets)
            loss = loss + float(config.get("halt_loss_weight", 0.1)) * halt_loss
            exit_loss_weight = float(config.get("exit_loss_weight", 0.0))
            if exit_loss_weight:
                row_indices = torch.arange(batch.inputs.shape[0], device=device)
                exit_logits = route_stats["step_logits"][row_indices, model.internal_steps - 1]
                loss = loss + exit_loss_weight * nn.functional.cross_entropy(exit_logits, batch.targets)
        if "router_entropy" in route_stats:
            loss = loss - 0.0001 * route_stats["router_entropy"]
        if coverage_enabled and "routing_coverage_loss" in route_stats:
            loss = loss + coverage_weight * route_stats["routing_coverage_loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == steps):
            print(f"step={step:04d} loss={losses[-1]:.4f}")
    elapsed = time.perf_counter() - start
    train_eval = evaluate(model, train_generator, args.examples_per_task, device)
    heldout_eval = evaluate(model, heldout_generator, args.examples_per_task, device)
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "model_name": config["model"],
        "seed": run_seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "steps": steps,
        "batch_size": config["batch_size"],
        "training_seconds": elapsed,
        "samples_per_second": steps * int(config["batch_size"]) / max(elapsed, 1e-9),
        "total_params": count_parameters(model),
        "stage_loss_weight": stage_loss_weight,
        "routing_coverage_weight": coverage_weight,
        "routing_warmup_steps": routing_warmup_steps,
        "optimizer": str(config.get("optimizer", "adamw")),
        "heldout_pairs": [list(pair) for pair in heldout_pairs],
        "evaluation_split": evaluation_split,
        "train_value_range": [train_value_min, train_value_max],
        "eval_value_range": [eval_value_min, eval_value_max],
        "train_combination_split": train_combination_split,
        "eval_combination_split": eval_combination_split,
        "train": train_eval,
        "heldout": heldout_eval,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
    }
    if hasattr(model, "parameter_report"):
        report.update(model.parameter_report())
        report.update({"routing_capacity": model.router.routing_capacity,
                       "routing_depth": model.router.active_depth})
    else:
        report.update({"active_params_estimate": count_parameters(model), "active_fraction": 1.0})
    if hasattr(optimizer, "report"):
        report.update(optimizer.report())
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report["output"] = str(output_path)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        report["checkpoint"] = str(checkpoint_path)
        checkpoint_config = dict(config)
        if hasattr(model, "router"):
            checkpoint_config["routing_capacity"] = model.router.routing_capacity
            checkpoint_config["routing_depth"] = model.router.active_depth
        torch.save({"model_state": model.state_dict(), "config": checkpoint_config,
                   "report": report}, checkpoint_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neural Engine on held-out operation compositions")
    parser.add_argument("--config", default="configs/ne_composition_v0.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="composition_smoke")
    parser.add_argument("--output", default="results/runs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None,
                        help="Initialize model weights from an existing checkpoint before training")
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
