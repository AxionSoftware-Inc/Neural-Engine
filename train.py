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
        return DenseTransformerBaseline(**{key: config[key] for key in fields})
    fields = ("vocab_size", "num_classes", "seq_len", "d_model", "state_dim", "num_circuits", "circuit_rank",
              "router_branch", "router_depth", "candidate_pool", "active_circuits", "internal_steps")
    model_kwargs = {key: config[key] for key in fields}
    model_kwargs["router_addresses"] = config.get("router_addresses", 1)
    model_kwargs["slot_count"] = config.get("slot_count", 0)
    model_kwargs["task_context"] = config.get("task_context", False)
    model_kwargs["task_context_update"] = config.get("task_context_update", True)
    model_kwargs["circuit_mode"] = config.get("circuit_mode", "parallel")
    model_kwargs["numeric_value_encoding"] = config.get("numeric_value_encoding", False)
    model_kwargs["adaptive_halting"] = config.get("adaptive_halting", False)
    model_kwargs["halt_threshold"] = config.get("halt_threshold", 0.5)
    model_kwargs["routing_coverage_temperature"] = config.get("routing_coverage_temperature", 0.25)
    model_kwargs["input_reinjection"] = config.get("input_reinjection", 1.0)
    return NeuralEngineV0(**model_kwargs)


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


@torch.no_grad()
def evaluate(model: nn.Module, source: BatchSource, batches: int = 8) -> dict[str, Any]:
    model.eval()
    losses, predictions, targets, task_ids, depths, selected_ids = [], [], [], [], [], []
    executed_step_values = []
    for _ in range(batches):
        batch = source.balanced(16 if batches <= 2 else 32)
        if isinstance(model, NeuralEngineV0):
            logits, route_stats = model(batch.inputs, adaptive=model.adaptive_inference)
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
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
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
    for step in range(1, steps + 1):
        batch = train_source.batch()
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, NeuralEngineV0):
            logits, route_stats = model(batch.inputs, adaptive=False, coverage=coverage_enabled)
        else:
            logits, route_stats = model(batch.inputs)
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
        if "router_entropy" in route_stats:
            loss = loss - 0.0001 * route_stats["router_entropy"]
        if coverage_enabled and "routing_coverage_loss" in route_stats:
            loss = loss + coverage_weight * route_stats["routing_coverage_loss"]
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
        "routing_coverage_weight": coverage_weight,
        "routing_coverage_temperature": float(config.get("routing_coverage_temperature", 0.25)),
        "input_reinjection": float(config.get("input_reinjection", 1.0)),
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
                       "adaptive_halting": model.adaptive_halting,
                       "router_type": f"hierarchical-tree-{model.router.num_addresses}-address-local-pool",
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
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        report["checkpoint"] = str(checkpoint_path)
        torch.save({"model_state": model.state_dict(), "config": config, "report": report}, checkpoint_path)
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--balanced-train", action="store_true",
                        help="Use an equal task mix in every training batch")
    parser.add_argument("--composition-train", action="store_true",
                        help="Oversample depth-2/3 tasks during training")
    parser.add_argument("--composition-strength", type=float, default=0.0,
                        help="Extra sampling weight per depth level (e.g. 0.5 gives 1:1.5:2)")
    args = parser.parse_args()
    if args.model == "baseline":
        args.config = "configs/transformer_30m.yaml"
    run(args)


if __name__ == "__main__":
    main()
