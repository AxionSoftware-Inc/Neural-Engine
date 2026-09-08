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

from baseline.transformer import DenseTransformerBaseline
from data.generator import Batch, SyntheticTaskGenerator, accuracy_by_depth, accuracy_by_task
from neural_engine.instrumentation import count_parameters
from neural_engine.model import NeuralEngineV0
from neural_engine.optim import LazyAdamW
from neural_engine.register_model import TypedRegisterNeuralEngine


def load_config(path: str, smoke: bool) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if smoke:
        config.update(d_model=128, state_dim=128, num_circuits=128, circuit_rank=8,
                      router_depth=3, candidate_pool=16, active_circuits=4,
                      internal_steps=2, nhead=4, num_layers=2, ff_dim=256, batch_size=64)
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: dict[str, Any]) -> nn.Module:
    if config["model"] == "baseline":
        fields = ("vocab_size", "num_classes", "seq_len", "d_model", "nhead", "num_layers", "ff_dim", "dropout")
        model_kwargs = {key: config[key] for key in fields}
        model_kwargs["numeric_value_encoding"] = config.get("numeric_value_encoding", False)
        return DenseTransformerBaseline(**model_kwargs)
    fields = ("vocab_size", "num_classes", "seq_len", "d_model", "state_dim", "num_circuits", "circuit_rank",
              "router_branch", "router_depth", "candidate_pool", "active_circuits", "internal_steps")
    model_kwargs = {key: config[key] for key in fields}
    model_kwargs["router_addresses"] = config.get("router_addresses", 1)
    model_kwargs["slot_count"] = config.get("slot_count", 0)
    model_kwargs["task_context"] = config.get("task_context", False)
    model_kwargs["task_context_update"] = config.get("task_context_update", True)
    model_kwargs["circuit_mode"] = config.get("circuit_mode", "parallel")
    model_kwargs["circuit_bank_mode"] = config.get("circuit_bank_mode", "independent")
    model_kwargs["shared_rank"] = config.get("shared_rank", 8)
    model_kwargs["numeric_value_encoding"] = config.get("numeric_value_encoding", False)
    model_kwargs["adaptive_halting"] = config.get("adaptive_halting", False)
    model_kwargs["halt_threshold"] = config.get("halt_threshold", 0.5)
    model_kwargs["routing_coverage_temperature"] = config.get("routing_coverage_temperature", 0.25)
    model_kwargs["input_reinjection"] = config.get("input_reinjection", 1.0)
    model_kwargs["input_reinjection_schedule"] = config.get("input_reinjection_schedule")
    model_kwargs["circuit_delta_scale"] = config.get("circuit_delta_scale", 1.0)
    model_kwargs["correction_gate_mode"] = config.get("correction_gate_mode", "none")
    model_kwargs["memory_write_mode"] = config.get("memory_write_mode", "none")
    model_kwargs["post_correction_residual_scale"] = config.get("post_correction_residual_scale", 0.0)
    model_kwargs["routing_reuse_weight"] = config.get("routing_reuse_weight", 0.0)
    model_kwargs["routing_reuse_start_level"] = config.get("routing_reuse_start_level", 0)
    model_kwargs["route_exploration_prob"] = config.get("route_exploration_prob", 0.0)
    model_kwargs["routing_capacity"] = config.get("routing_capacity")
    model_kwargs["routing_depth"] = config.get("routing_depth")
    model_kwargs["router_variant"] = config.get("router_variant", "global")
    model_kwargs["family_count"] = config.get("family_count", 2)
    model_kwargs["shared_fraction"] = config.get("shared_fraction", 0.125)
    model_kwargs["soft_routing_temperature"] = config.get("soft_routing_temperature", 0.0)
    model_kwargs["route_target_supervision"] = config.get("route_target_supervision", False)
    model_kwargs["typed_register_bridge"] = config.get("typed_register_bridge", False)
    model_kwargs["register_bridge_scale"] = config.get("register_bridge_scale", 1.0)
    model_kwargs["register_bridge_temperature"] = config.get("register_bridge_temperature", 1.0)
    model_kwargs["register_bridge_mode"] = config.get("register_bridge_mode", "soft")
    model_kwargs["register_slot_count"] = config.get("register_slot_count", 1)
    model_kwargs["register_slot_read_mode"] = config.get("register_slot_read_mode", "sum")
    model_kwargs["state_stage_head"] = config.get("state_stage_head", False)
    model_kwargs["operation_transition_rank"] = config.get("operation_transition_rank", 0)
    model_kwargs["operation_transition_scale"] = config.get("operation_transition_scale", 1.0)
    if config.get("architecture") == "typed_register":
        for key in ("task_context", "task_context_update", "adaptive_halting",
                    "halt_threshold", "routing_coverage_temperature",
                    "input_reinjection", "circuit_delta_scale", "correction_gate_mode",
                    "memory_write_mode", "post_correction_residual_scale", "router_variant",
                    "soft_routing_temperature", "route_target_supervision",
                    "typed_register_bridge", "register_bridge_scale",
                    "register_bridge_temperature", "register_bridge_mode",
                    "register_slot_count",
                    "register_slot_read_mode",
                    "state_stage_head",
                    "operation_transition_rank", "operation_transition_scale",
                    "routing_reuse_weight", "routing_reuse_start_level",
                    "input_reinjection_schedule"):
            model_kwargs.pop(key, None)
        model_kwargs["readout_mode"] = config.get("readout_mode", "routed")
        model_kwargs["route_query_mode"] = config.get("route_query_mode", "value_and_type")
        model_kwargs["route_context_dim"] = config.get("route_context_dim", 32)
        model_kwargs["pair_mode"] = config.get("pair_mode", "concat")
        model_kwargs["routing_mode"] = config.get("routing_mode", "global")
        model_kwargs["family_count"] = config.get("family_count", 9)
        model_kwargs["shared_fraction"] = config.get("shared_fraction", 0.125)
        model_kwargs["anchor_branch"] = config.get("anchor_branch", 4)
        model_kwargs["anchor_depth"] = config.get("anchor_depth", 2)
        model_kwargs["role_count"] = config.get("role_count", 9)
        model_kwargs["circuit_bank_mode"] = config.get("circuit_bank_mode", "independent")
        model_kwargs["shared_rank"] = config.get("shared_rank", 8)
        model_kwargs["factor_count"] = config.get("factor_count")
        model_kwargs["factor_candidate_pool"] = config.get("factor_candidate_pool")
        model_kwargs["typed_route_partitions"] = config.get("typed_route_partitions", False)
        model_kwargs["typed_route_shared"] = config.get("typed_route_shared", False)
        model_kwargs["operator_partition_count"] = config.get("operator_partition_count", 4)
        return TypedRegisterNeuralEngine(**model_kwargs)
    return NeuralEngineV0(**model_kwargs)


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    if optimizer_name == "lazy_adamw":
        lazy_parameters = [parameter for name, parameter in model.named_parameters()
                           if name.startswith("circuits.") or name == "router.keys"]
        if not lazy_parameters:
            raise ValueError("lazy_adamw requires a Neural Engine circuit/key parameter set")
        return LazyAdamW(model.parameters(), lr=config["learning_rate"],
                         weight_decay=config["weight_decay"],
                         lazy_parameters=lazy_parameters)
    raise ValueError(f"unknown optimizer: {optimizer_name}")


class BatchSource:
    def __init__(self, generator: SyntheticTaskGenerator, batch_size: int, device: torch.device,
                 task_balanced: bool = False, composition_strength: float = 0.0):
        self.generator = generator
        self.batch_size = batch_size
        self.device = device
        self.task_balanced = task_balanced
        self.composition_strength = composition_strength

    def batch(self) -> Batch:
        if self.composition_strength > 0:
            return self.generator.composition_batch(self.batch_size, self.device, self.composition_strength)
        if self.task_balanced:
            return self.generator.task_balanced_batch(self.batch_size, self.device)
        return self.generator.batch(self.batch_size, self.device)

    def balanced(self, examples_per_task: int = 32) -> Batch:
        return self.generator.balanced_batch(examples_per_task, self.device)


def controlled_task_route_ids(task_ids: torch.Tensor, internal_steps: int,
                              active_circuits: int, num_circuits: int) -> torch.Tensor:
    """Assign each task to a fixed group of independent circuits.

    This is the capacity-control arm of the scaling diagnostic.  It removes
    learned route selection while keeping the active circuit count fixed.  A
    task is mapped to one contiguous group; when the bank is too small for a
    one-to-one task mapping, tasks deliberately share a group.  The returned
    route has the same shape as ``NeuralEngineV0``'s replay hook.
    """
    if internal_steps < 1 or active_circuits < 1 or num_circuits < active_circuits:
        raise ValueError("invalid circuit dimensions for controlled task routing")
    group_count = max(1, num_circuits // active_circuits)
    group_ids = task_ids.remainder(group_count)
    slots = torch.arange(active_circuits, device=task_ids.device).view(1, -1)
    base = group_ids.view(-1, 1) * active_circuits
    selected = (base + slots).remainder(num_circuits)
    return selected.unsqueeze(1).expand(-1, internal_steps, -1).clone()


def apply_routing_schedule(model: nn.Module, schedule: list[dict[str, int]], step: int) -> None:
    """Expose a larger routing prefix at explicit training steps.

    The model is constructed with the initial ``routing_capacity`` and
    ``routing_depth`` from its config. Schedule events only change the
    router's reachable prefix/tree depth; they do not alter model weights or
    the inference-time active circuit budget.
    """
    if not isinstance(model, NeuralEngineV0):
        return
    for event in schedule:
        if int(event.get("step", -1)) != int(step):
            continue
        if "capacity" not in event:
            raise ValueError("routing schedule events require capacity")
        kwargs = {"capacity": int(event["capacity"])}
        if "depth" in event:
            kwargs["depth"] = int(event["depth"])
        model.router.set_routing_state(**kwargs)


@torch.no_grad()
def evaluate(model: nn.Module, source: BatchSource, batches: int = 8) -> dict[str, Any]:
    model.eval()
    losses, predictions, targets, task_ids, depths, selected_ids = [], [], [], [], [], []
    executed_step_values = []
    for _ in range(batches):
        batch = source.balanced(16 if batches <= 2 else 32)
        if isinstance(model, NeuralEngineV0):
            forced_ids = None
            if getattr(model, "routing_mode", "learned") == "controlled_task":
                forced_ids = controlled_task_route_ids(
                    batch.task_ids, model.internal_steps, model.active_circuits,
                    model.router.num_circuits)
            logits, route_stats = model(batch.inputs, adaptive=model.adaptive_inference,
                                        forced_selected_ids=forced_ids)
        else:
            logits, route_stats = model(batch.inputs)
        losses.append(float(nn.functional.cross_entropy(logits, batch.targets).cpu()))
        predictions.append(logits.argmax(dim=-1).cpu())
        targets.append(batch.targets.cpu())
        task_ids.append(batch.task_ids.cpu())
        depths.append(batch.depths.cpu())
        if "selected_ids" in route_stats:
            selected_ids.append(route_stats["selected_ids"].detach().cpu().reshape(-1))
        if "executed_steps" in route_stats:
            executed_step_values.append(route_stats["executed_steps"].detach().cpu())
    joined = Batch(inputs=torch.empty(0, dtype=torch.long), targets=torch.cat(targets),
                   task_ids=torch.cat(task_ids), depths=torch.cat(depths))
    pred = torch.cat(predictions)
    result: dict[str, Any] = {
        "val_loss": float(np.mean(losses)),
        "exact_accuracy": float(pred.eq(joined.targets).float().mean()),
        "task_accuracy": accuracy_by_task(pred, joined),
        "depth_accuracy": accuracy_by_depth(pred, joined),
    }
    if selected_ids and hasattr(model, "router"):
        routed = torch.cat(selected_ids)
        routed = routed[routed.ge(0)]
        counts = torch.bincount(routed, minlength=model.router.num_circuits).float()
        probabilities = counts / counts.sum().clamp_min(1)
        result.update({
            "circuits_used": int((counts > 0).sum()),
            "dead_circuit_fraction": float((counts == 0).float().mean()),
            "routing_entropy": float(-(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()),
            "routing_max_load_fraction": float(probabilities.max()),
        })
    if executed_step_values:
        executed = torch.cat(executed_step_values).float()
        depth_execution = {}
        for depth in torch.unique(joined.depths).tolist():
            mask = joined.depths.eq(depth)
            depth_execution[str(int(depth))] = float(executed[mask].mean())
        result.update({
            "avg_executed_steps": float(executed.mean()),
            "active_step_fraction": float(executed.mean() / getattr(model, "internal_steps", 1)),
            "executed_steps_by_depth": depth_execution,
        })
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config, args.smoke)
    if args.num_circuits is not None:
        config["num_circuits"] = args.num_circuits
    if args.active_circuits is not None:
        config["active_circuits"] = args.active_circuits
    if args.family_count is not None:
        config["family_count"] = args.family_count
    if args.soft_routing_temperature is not None:
        config["soft_routing_temperature"] = args.soft_routing_temperature
    if args.soft_routing_steps is not None:
        config["soft_routing_steps"] = args.soft_routing_steps
    if args.route_target_supervision:
        config["route_target_supervision"] = True
    if args.route_target_weight is not None:
        config["route_target_weight"] = args.route_target_weight
    if args.routing_mode is not None:
        config["routing_mode"] = args.routing_mode
    if args.seed is not None:
        config["seed"] = args.seed
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device)
    if isinstance(model, NeuralEngineV0):
        model.routing_mode = str(config.get("routing_mode", "learned"))
        if model.routing_mode not in {"learned", "controlled_task"}:
            raise ValueError("routing_mode must be 'learned' or 'controlled_task'")
    if args.init_checkpoint:
        initialization = torch.load(Path(args.init_checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(initialization.get("model_state", initialization))
    optimizer = make_optimizer(model, config)
    composition_strength = (args.composition_strength
                            if args.composition_strength > 0
                            else 1.0 if args.composition_train else 0.0)
    train_value_min = int(config.get("train_value_min", 0))
    train_value_max = int(config.get("train_value_max", 63))
    eval_value_min = int(config.get("eval_value_min", train_value_min))
    eval_value_max = int(config.get("eval_value_max", train_value_max))
    train_split = str(config.get("train_split", "all"))
    eval_split = str(config.get("eval_split", "all"))
    train_source = BatchSource(SyntheticTaskGenerator(
                               config["seq_len"], int(config["seed"]) + 1,
                               value_min=train_value_min, value_max=train_value_max,
                               split=train_split),
                               config["batch_size"], device, task_balanced=args.balanced_train,
                               composition_strength=composition_strength)
    eval_source = BatchSource(SyntheticTaskGenerator(
                              config["seq_len"], int(config["seed"]) + 2,
                              value_min=eval_value_min, value_max=eval_value_max,
                              split=eval_split), 256, device)
    steps = args.steps if args.steps is not None else (20 if args.smoke else 1000)
    model.train()
    start = time.perf_counter()
    losses: list[float] = []
    peak_vram = 0
    coverage_weight = float(config.get("routing_coverage_weight", 0.0))
    coverage_enabled = isinstance(model, NeuralEngineV0) and coverage_weight > 0.0
    routing_warmup_steps = int(config.get("routing_warmup_steps", 0))
    routing_schedule = list(config.get("routing_schedule", []))
    if routing_schedule and routing_warmup_steps:
        raise ValueError("routing_schedule and routing_warmup_steps are mutually exclusive")
    soft_routing_temperature = float(config.get("soft_routing_temperature", 0.0))
    soft_routing_steps = int(config.get("soft_routing_steps", 0))
    for step in range(1, steps + 1):
        if routing_schedule:
            apply_routing_schedule(model, routing_schedule, step)
        elif (isinstance(model, NeuralEngineV0) and routing_warmup_steps
              and step == routing_warmup_steps + 1):
            model.router.set_routing_state(
                capacity=model.router.num_circuits,
                depth=model.router.depth,
            )
        batch = train_source.batch()
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, NeuralEngineV0):
            if hasattr(model.router, "soft_routing_temperature"):
                model.router.soft_routing_temperature = (
                    soft_routing_temperature
                    if soft_routing_temperature > 0.0 and (soft_routing_steps <= 0 or step <= soft_routing_steps)
                    else 0.0)
            forced_ids = None
            if model.routing_mode == "controlled_task":
                forced_ids = controlled_task_route_ids(
                    batch.task_ids, model.internal_steps, model.active_circuits,
                    model.router.num_circuits)
            logits, route_stats = model(batch.inputs, adaptive=False,
                                        forced_selected_ids=forced_ids,
                                        coverage=coverage_enabled)
        else:
            logits, route_stats = model(batch.inputs)
        if hasattr(optimizer, "set_active_rows") and isinstance(model, NeuralEngineV0):
            selected = route_stats.get("selected_ids")
            if selected is not None:
                selected = selected.detach().reshape(-1).unique()
                optimizer.set_active_rows({
                    model.circuits.down: selected,
                    model.circuits.up: selected,
                    model.circuits.bias: selected,
                    model.router.keys: selected,
                })
        loss = nn.functional.cross_entropy(logits, batch.targets)
        stage_loss_weight = float(config.get("stage_loss_weight", 0.0))
        if stage_loss_weight and batch.stage_targets is not None and batch.stage_mask is not None:
            step_logits = route_stats.get("step_logits")
            if step_logits is not None:
                stage_losses = []
                for stage in range(min(step_logits.shape[1], batch.stage_targets.shape[1])):
                    mask = batch.stage_mask[:, stage]
                    if mask.any():
                        stage_losses.append(nn.functional.cross_entropy(
                            step_logits[mask, stage], batch.stage_targets[mask, stage]))
                if stage_losses:
                    loss = loss + stage_loss_weight * torch.stack(stage_losses).mean()
        if isinstance(model, NeuralEngineV0) and config.get("adaptive_halting", False):
            halt_targets = (torch.arange(model.internal_steps, device=device).unsqueeze(0)
                            >= (batch.depths.unsqueeze(1) - 1)).float()
            halt_loss = nn.functional.binary_cross_entropy_with_logits(
                route_stats["halt_logits"], halt_targets)
            loss = loss + float(config.get("halt_loss_weight", 0.1)) * halt_loss
            exit_loss_weight = float(config.get("exit_loss_weight", 0.0))
            if exit_loss_weight:
                exit_steps = (batch.depths - 1).clamp(0, model.internal_steps - 1)
                row_indices = torch.arange(batch.inputs.shape[0], device=device)
                exit_logits = route_stats["step_logits"][row_indices, exit_steps]
                loss = loss + exit_loss_weight * nn.functional.cross_entropy(exit_logits, batch.targets)
        route_final_target_weight = float(config.get("route_final_target_weight", 0.0))
        if route_final_target_weight and isinstance(model, NeuralEngineV0):
            route_deltas = route_stats.get("route_deltas")
            executed_mask = route_stats.get("executed_mask")
            if route_deltas is not None and executed_mask is not None:
                route_losses = []
                for stage in range(route_deltas.shape[1]):
                    mask = executed_mask[:, stage]
                    if mask.any():
                        route_losses.append(nn.functional.cross_entropy(
                            model.output(route_deltas[mask, stage]), batch.targets[mask]))
                if route_losses:
                    loss = loss + route_final_target_weight * torch.stack(route_losses).mean()
        if "router_entropy" in route_stats and model.routing_mode != "controlled_task":
            loss = loss - 0.0001 * route_stats["router_entropy"]
        if coverage_enabled and "routing_coverage_loss" in route_stats:
            loss = loss + coverage_weight * route_stats["routing_coverage_loss"]
        route_target_weight = float(config.get("route_target_weight", 0.0))
        if route_target_weight and "routing_target_loss" in route_stats:
            loss = loss + route_target_weight * route_stats["routing_target_loss"]
        routing_reuse_weight = float(config.get("routing_reuse_weight", 0.0))
        if routing_reuse_weight and "routing_reuse_loss" in route_stats:
            loss = loss + routing_reuse_weight * route_stats["routing_reuse_loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if device.type == "cuda":
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(device) // (1024 * 1024))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == steps):
            print(f"step={step:04d} loss={losses[-1]:.4f}")
    elapsed = time.perf_counter() - start
    validation = evaluate(model, eval_source, batches=2 if args.smoke else 8)
    heldout_min = config.get("heldout_value_min")
    heldout_max = config.get("heldout_value_max")
    heldout_split = str(config.get("heldout_split", "all"))
    if heldout_min is not None and heldout_max is not None and (heldout_split != "all" or heldout_min != eval_value_min or heldout_max != eval_value_max):
        heldout_source = BatchSource(SyntheticTaskGenerator(
                                     config["seq_len"], int(config["seed"]) + 3,
                                     value_min=int(heldout_min), value_max=int(heldout_max),
                                     split=heldout_split),
                                     256, device)
        heldout = evaluate(model, heldout_source, batches=2 if args.smoke else 8)
        validation.update({f"heldout_{key}": value for key, value in heldout.items()})
    report: dict[str, Any] = {
        "run_id": args.run_id, "model_name": config["model"], "seed": config["seed"], "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "steps": steps,
        "batch_size": config["batch_size"], "training_seconds": elapsed,
        "samples_per_second": steps * config["batch_size"] / max(elapsed, 1e-9), "peak_vram_mb": int(peak_vram),
        "task_balanced": bool(args.balanced_train), "composition_strength": composition_strength,
        "stage_loss_weight": float(config.get("stage_loss_weight", 0.0)),
        "halt_loss_weight": float(config.get("halt_loss_weight", 0.0)),
        "exit_loss_weight": float(config.get("exit_loss_weight", 0.0)),
        "route_final_target_weight": float(config.get("route_final_target_weight", 0.0)),
        "routing_coverage_weight": coverage_weight,
        "routing_reuse_weight": float(config.get("routing_reuse_weight", 0.0)),
        "routing_reuse_start_level": int(config.get("routing_reuse_start_level", 0)),
        "routing_coverage_temperature": float(config.get("routing_coverage_temperature", 0.25)),
        "routing_warmup_steps": routing_warmup_steps,
        "routing_schedule": routing_schedule,
        "routing_mode": str(config.get("routing_mode", "learned")),
        "soft_routing_temperature": float(config.get("soft_routing_temperature", 0.0)),
        "soft_routing_steps": soft_routing_steps,
        "route_target_weight": float(config.get("route_target_weight", 0.0)),
        "input_reinjection": float(config.get("input_reinjection", 1.0)),
        "input_reinjection_schedule": list(config.get("input_reinjection_schedule", [])),
        "memory_write_mode": str(config.get("memory_write_mode", "none")),
        "optimizer": str(config.get("optimizer", "adamw")),
        "train_value_range": [train_value_min, train_value_max],
        "eval_value_range": [eval_value_min, eval_value_max],
        "train_split": train_split, "eval_split": eval_split,
        "total_params": count_parameters(model), "train_loss_first": losses[0], "train_loss_last": losses[-1],
        **validation,
    }
    if isinstance(model, NeuralEngineV0):
        report.update(model.parameter_report())
        report.update({"active_circuits": model.active_circuits, "internal_steps": model.internal_steps,
                       "circuit_mode": model.circuit_mode, "task_context": model.use_task_context,
                       "router_variant": model.router_variant,
                       "family_count": model.family_count,
                       "adaptive_halting": model.adaptive_halting,
                       "router_type": type(model.router).__name__,
                       "routing_capacity": model.router.routing_capacity,
                       "routing_depth": model.router.active_depth,
                       "router_entropy": float(model._last_route["router_entropy"].detach().cpu())})
        if model.adaptive_halting and "avg_executed_steps" in report:
            shared_params = report["active_params_estimate"] - report["active_circuit_params"]
            average_active = shared_params + report["active_circuit_params"] * report["avg_executed_steps"]
            report.update({
                "average_active_params_estimate": average_active,
                "average_active_fraction": average_active / report["total_params"],
            })
    else:
        report.update({"active_params_estimate": count_parameters(model), "active_fraction": 1.0, "router_type": "dense"})
    if hasattr(optimizer, "report"):
        report.update(optimizer.report())
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        report["checkpoint"] = str(checkpoint_path)
        checkpoint_config = dict(config)
        if isinstance(model, NeuralEngineV0):
            checkpoint_config["routing_capacity"] = model.router.routing_capacity
            checkpoint_config["routing_depth"] = model.router.active_depth
        torch.save({"model_state": model.state_dict(), "config": checkpoint_config,
                   "report": report}, checkpoint_path)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neural Engine V0 or the dense baseline")
    parser.add_argument("--model", choices=("ne", "baseline"), default=None)
    parser.add_argument("--config", default="configs/ne_v0.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="local_smoke")
    parser.add_argument("--output", default="results/runs")
    parser.add_argument("--checkpoint", default=None,
                        help="Save model weights plus effective config/report to this path")
    parser.add_argument("--init-checkpoint", default=None,
                        help="Initialize model weights from an existing checkpoint before training")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--balanced-train", action="store_true",
                        help="Use an equal task mix in every training batch")
    parser.add_argument("--composition-train", action="store_true",
                        help="Oversample depth-2/3 tasks during training")
    parser.add_argument("--composition-strength", type=float, default=0.0,
                        help="Extra sampling weight per depth level (e.g. 0.5 gives 1:1.5:2)")
    parser.add_argument("--num-circuits", type=int, default=None,
                        help="Override the config circuit-bank size for scaling controls")
    parser.add_argument("--active-circuits", type=int, default=None,
                        help="Override the active circuit budget for scaling controls")
    parser.add_argument("--family-count", type=int, default=None,
                        help="Override semantic family count for structured routing")
    parser.add_argument("--soft-routing-temperature", type=float, default=None,
                        help="Training-only soft candidate mixture temperature; eval remains hard top-k")
    parser.add_argument("--soft-routing-steps", type=int, default=None,
                        help="Number of initial training steps using soft routing before hard top-k")
    parser.add_argument("--route-target-supervision", action="store_true",
                        help="Train the hierarchical router toward task-derived circuit-group targets")
    parser.add_argument("--route-target-weight", type=float, default=None,
                        help="Auxiliary route-target loss weight")
    parser.add_argument("--routing-mode", choices=("learned", "controlled_task"), default=None,
                        help="Use learned routing or fixed task-to-circuit allocation")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the config seed for multi-seed controls")
    args = parser.parse_args()
    if args.model == "baseline":
        args.config = "configs/transformer_30m.yaml"
    run(args)


if __name__ == "__main__":
    main()
