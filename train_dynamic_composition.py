from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from data.dynamic_composition import DynamicCompositionGenerator
from neural_engine.dynamic_register import DynamicRegisterNeuralEngine
from neural_engine.instrumentation import count_parameters
from neural_engine.optim import LazyAdamW


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: dict[str, Any]) -> DynamicRegisterNeuralEngine:
    fields = (
        "vocab_size", "num_classes", "modulus", "max_ops", "seq_len", "d_model", "state_dim",
        "num_circuits", "circuit_rank", "router_branch", "router_depth",
        "candidate_pool", "active_circuits", "circuit_bank_mode", "factor_count",
        "factor_candidate_pool", "factor_capacity", "ordered_factor_slots", "circuit_mode",
        "route_exploration_prob",
        "input_reinjection_scale", "write_gate", "value_encoder_mode",
        "factor_mix_mode", "route_context_mode", "modular_prior",
        "state_layout",
        "predecessor_operation_context",
        "operation_adapter_rank", "operation_adapter_scale",
        "operation_adapter_gate",
        "operation_read_adapter_rank", "operation_read_adapter_scale",
        "operation_write_adapter_rank", "operation_write_adapter_scale",
        "operation_circuit_bank",
        "operation_router_keys",
        "operation_transition_rank", "operation_transition_scale",
        "numeric_state_dim", "numeric_state_scale",
        "modular_prior_mode",
        "modular_template_init",
        "circuit_residual_scale",
        "circuit_input_norm",
        "output_mode", "output_temperature", "output_scalar_bias",
        "macro_cell_count", "macro_cell_rank", "macro_cell_depth",
        "macro_router_branch", "macro_router_depth", "macro_candidate_pool",
        "active_macro_cells", "macro_cell_scale",
    )
    return DynamicRegisterNeuralEngine(**{
        key: config[key] for key in fields if key in config
    })


def make_optimizer(model: DynamicRegisterNeuralEngine, config: dict[str, Any]):
    if str(config.get("optimizer", "adamw")).lower() == "lazy_adamw":
        lazy_parameters = [
            parameter for name, parameter in model.named_parameters()
            if name.startswith("circuits.") or name == "router.keys"
        ]
        return LazyAdamW(
            model.parameters(), lr=config["learning_rate"],
            weight_decay=config["weight_decay"], lazy_parameters=lazy_parameters,
        )
    return torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )


@torch.no_grad()
def route_audit(
    model: DynamicRegisterNeuralEngine,
    selected_ids: torch.Tensor,
    depths: torch.Tensor,
) -> dict[str, Any]:
    """Summarize hard route traffic without changing the forward path."""
    num_circuits = int(model.router.num_circuits)

    def summarize(ids: torch.Tensor) -> dict[str, Any]:
        valid = ids.ge(0)
        flat = ids[valid].to(dtype=torch.long)
        if not flat.numel():
            return {
                "selected_count": 0,
                "unique_virtual_circuits": 0,
                "virtual_bank_utilization": 0.0,
                "dead_virtual_circuits": num_circuits,
            }
        counts = torch.bincount(flat, minlength=num_circuits).float()
        active_counts = counts[counts.gt(0)]
        probabilities = active_counts / active_counts.sum()
        result: dict[str, Any] = {
            "selected_count": int(flat.numel()),
            "unique_virtual_circuits": int(active_counts.numel()),
            "virtual_bank_utilization": float(active_counts.numel() / num_circuits),
            "dead_virtual_circuits": int(num_circuits - active_counts.numel()),
            "top_route_fraction": float(active_counts.max() / active_counts.sum()),
            "route_entropy": float((-(probabilities * probabilities.log()).sum()).cpu()),
        }
        if model.circuit_bank_mode == "factorized":
            first, second = model.router._factor_ids(flat)
            factors = torch.cat([first, second])
            factor_count = int(model.router.factor_count)
            factor_usage = torch.bincount(factors, minlength=factor_count).gt(0)
            result.update({
                "unique_factor_rows": int(factor_usage.sum()),
                "factor_bank_utilization": float(factor_usage.float().mean()),
                "dead_factor_rows": int(factor_count - factor_usage.sum()),
            })
        return result

    audit = summarize(selected_ids)
    audit["by_program_depth"] = {
        str(depth): summarize(selected_ids[depths.eq(depth)])
        for depth in sorted(int(value) for value in torch.unique(depths).cpu().tolist())
    }
    audit["by_execution_step"] = {
        str(step): summarize(selected_ids[:, step])
        for step in range(selected_ids.shape[1])
    }
    return audit


@torch.no_grad()
def macro_route_audit(
    model: DynamicRegisterNeuralEngine,
    selected_ids: torch.Tensor,
    depths: torch.Tensor,
) -> dict[str, Any]:
    """Summarize macro-cell traffic independently from the micro bank."""
    num_cells = int(model.macro_cell_count)

    def summarize(ids: torch.Tensor) -> dict[str, Any]:
        valid = ids.ge(0)
        flat = ids[valid].to(dtype=torch.long)
        if not flat.numel():
            return {
                "selected_count": 0,
                "unique_macro_cells": 0,
                "macro_bank_utilization": 0.0,
                "dead_macro_cells": num_cells,
            }
        counts = torch.bincount(flat, minlength=num_cells).float()
        active_counts = counts[counts.gt(0)]
        probabilities = active_counts / active_counts.sum()
        return {
            "selected_count": int(flat.numel()),
            "unique_macro_cells": int(active_counts.numel()),
            "macro_bank_utilization": float(active_counts.numel() / num_cells),
            "dead_macro_cells": int(num_cells - active_counts.numel()),
            "top_macro_fraction": float(active_counts.max() / active_counts.sum()),
            "macro_route_entropy": float(
                (-(probabilities * probabilities.log()).sum()).cpu()
            ),
        }

    audit = summarize(selected_ids)
    audit["by_program_depth"] = {
        str(depth): summarize(selected_ids[depths.eq(depth)])
        for depth in sorted(int(value) for value in torch.unique(depths).cpu().tolist())
    }
    audit["by_execution_step"] = {
        str(step): summarize(selected_ids[:, step])
        for step in range(selected_ids.shape[1])
    }
    return audit


@torch.no_grad()
def evaluate(
    model: DynamicRegisterNeuralEngine,
    generator: DynamicCompositionGenerator,
    device: torch.device,
    examples_per_depth: int,
) -> dict[str, Any]:
    model.eval()
    batch = generator.balanced_batch(examples_per_depth, device)
    logits, stats = model(batch.inputs)
    correct = logits.argmax(dim=-1).eq(batch.targets)
    per_depth = {}
    for depth in generator.allowed_depths:
        mask = batch.depths.eq(depth)
        per_depth[str(depth)] = float(correct[mask].float().mean().cpu())
    executed = stats["executed_steps"].float()
    return {
        "accuracy": float(correct.float().mean().cpu()),
        "loss": float(nn.functional.cross_entropy(logits, batch.targets).cpu()),
        "accuracy_by_depth": per_depth,
        "avg_executed_steps": float(executed.mean().cpu()),
        "active_step_fraction": float((executed / model.max_ops).mean().cpu()),
        "router_entropy": float(stats["router_entropy"].cpu()),
        "route_audit": route_audit(model, stats["selected_ids"], batch.depths),
        "macro_route_audit": (
            macro_route_audit(model, stats["macro_selected_ids"], batch.depths)
            if model.macro_cell_count
            else None
        ),
    }


def set_lazy_active_rows(
    model: DynamicRegisterNeuralEngine,
    optimizer: Any,
    selected_ids: torch.Tensor,
) -> None:
    if not hasattr(optimizer, "set_active_rows"):
        return
    selected = selected_ids.detach().reshape(-1)
    selected = selected[selected.ge(0)].unique()
    if not selected.numel():
        return
    if model.circuit_bank_mode == "factorized":
        first, second = model.router._factor_ids(selected)
        factor_rows = torch.cat([first, second]).unique()
        optimizer.set_active_rows({
            model.router.keys: factor_rows,
            model.circuits.down_factors: factor_rows,
            model.circuits.up_factors: factor_rows,
            model.circuits.bias_factors: factor_rows,
            model.circuits.factor_mix: selected,
        })
    else:
        optimizer.set_active_rows({
            model.router.keys: selected,
            model.circuits.down: selected,
            model.circuits.up: selected,
            model.circuits.bias: selected,
        })


def run(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_seed = int(config["seed"]) if args.seed is None else int(args.seed)
    seed_everything(run_seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = make_model(config).to(device)
    train_generator = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        seed=run_seed + 1,
        modulus=int(config.get("modulus", 64)),
        value_min=args.train_value_min,
        value_max=args.train_value_max,
        split="train" if args.heldout_depths else "all",
    )
    eval_generator = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        seed=run_seed + 2,
        modulus=int(config.get("modulus", 64)),
        value_min=args.eval_value_min,
        value_max=args.eval_value_max,
        split="heldout" if args.heldout_depths else "all",
    )
    optimizer = make_optimizer(model, config)
    steps = args.steps
    model.train()
    losses = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        batch_size = args.batch_size or int(config["batch_size"])
        batch = train_generator.task_balanced_batch(batch_size, device)
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
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        set_lazy_active_rows(model, optimizer, stats["selected_ids"])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == steps):
            print(f"step={step:05d} loss={losses[-1]:.5f}")
    elapsed = time.perf_counter() - start
    train_eval = evaluate(model, train_generator, device, args.examples_per_depth)
    eval_eval = evaluate(model, eval_generator, device, args.examples_per_depth)
    report = {
        "run_id": args.run_id,
        "model_name": config["model"],
        "seed": run_seed,
        "device": str(device),
        "steps": steps,
        "batch_size": args.batch_size or config["batch_size"],
        "training_seconds": elapsed,
        "train_depths": list(train_generator.allowed_depths),
        "eval_depths": list(eval_generator.allowed_depths),
        "train_value_range": [args.train_value_min, args.train_value_max],
        "eval_value_range": [args.eval_value_min, args.eval_value_max],
        "total_params": count_parameters(model),
        "train": train_eval,
        "evaluation": eval_eval,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
    }
    report.update(model.parameter_report())
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "config": config, "report": report}, checkpoint_path)
        report["checkpoint"] = str(checkpoint_path)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the attention-free dynamic register machine")
    parser.add_argument("--config", default="configs/ne_dynamic_20m.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="ne_dynamic_20m_smoke")
    parser.add_argument("--output", default="results/runs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--examples-per-depth", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--heldout-depths", action="store_true")
    parser.add_argument("--train-value-min", type=int, default=0)
    parser.add_argument("--train-value-max", type=int, default=63)
    parser.add_argument("--eval-value-min", type=int, default=0)
    parser.add_argument("--eval-value-max", type=int, default=63)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
