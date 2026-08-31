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
    return NeuralEngineV0(**{key: config[key] for key in fields})


class BatchSource:
    def __init__(self, generator: SyntheticTaskGenerator, batch_size: int, device: torch.device,
                 task_balanced: bool = False):
        self.generator = generator
        self.batch_size = batch_size
        self.device = device
        self.task_balanced = task_balanced

    def batch(self) -> Batch:
        if self.task_balanced:
            return self.generator.task_balanced_batch(self.batch_size, self.device)
        return self.generator.batch(self.batch_size, self.device)

    def balanced(self, examples_per_task: int = 32) -> Batch:
        return self.generator.balanced_batch(examples_per_task, self.device)


@torch.no_grad()
def evaluate(model: nn.Module, source: BatchSource, batches: int = 8) -> dict[str, Any]:
    model.eval()
    losses, predictions, targets, task_ids, depths, selected_ids = [], [], [], [], [], []
    for _ in range(batches):
        batch = source.balanced(16 if batches <= 2 else 32)
        logits, route_stats = model(batch.inputs)
        losses.append(float(nn.functional.cross_entropy(logits, batch.targets).cpu()))
        predictions.append(logits.argmax(dim=-1).cpu())
        targets.append(batch.targets.cpu())
        task_ids.append(batch.task_ids.cpu())
        depths.append(batch.depths.cpu())
        if "selected_ids" in route_stats:
            selected_ids.append(route_stats["selected_ids"].detach().cpu().reshape(-1))
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
        counts = torch.bincount(routed, minlength=model.router.num_circuits).float()
        probabilities = counts / counts.sum().clamp_min(1)
        result.update({
            "circuits_used": int((counts > 0).sum()),
            "dead_circuit_fraction": float((counts == 0).float().mean()),
            "routing_entropy": float(-(probabilities[probabilities > 0] * probabilities[probabilities > 0].log()).sum()),
            "routing_max_load_fraction": float(probabilities.max()),
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
    train_source = BatchSource(SyntheticTaskGenerator(config["seq_len"], int(config["seed"]) + 1),
                               config["batch_size"], device, task_balanced=args.balanced_train)
    eval_source = BatchSource(SyntheticTaskGenerator(config["seq_len"], int(config["seed"]) + 2), 256, device)
    steps = args.steps if args.steps is not None else (20 if args.smoke else 1000)
    model.train()
    start = time.perf_counter()
    losses: list[float] = []
    peak_vram = 0
    for step in range(1, steps + 1):
        batch = train_source.batch()
        optimizer.zero_grad(set_to_none=True)
        logits, route_stats = model(batch.inputs)
        loss = nn.functional.cross_entropy(logits, batch.targets)
        if "router_entropy" in route_stats:
            loss = loss - 0.0001 * route_stats["router_entropy"]
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
    report: dict[str, Any] = {
        "run_id": args.run_id, "model_name": config["model"], "seed": config["seed"], "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "steps": steps,
        "batch_size": config["batch_size"], "training_seconds": elapsed,
        "samples_per_second": steps * config["batch_size"] / max(elapsed, 1e-9), "peak_vram_mb": int(peak_vram),
        "total_params": count_parameters(model), "train_loss_first": losses[0], "train_loss_last": losses[-1],
        **validation,
    }
    if isinstance(model, NeuralEngineV0):
        report.update(model.parameter_report())
        report.update({"active_circuits": model.active_circuits, "internal_steps": model.internal_steps,
                       "router_type": "hierarchical-tree-local-pool",
                       "router_entropy": float(model._last_route["router_entropy"].detach().cpu())})
    else:
        report.update({"active_params_estimate": count_parameters(model), "active_fraction": 1.0, "router_type": "dense"})
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--balanced-train", action="store_true",
                        help="Use an equal task mix in every training batch")
    args = parser.parse_args()
    if args.model == "baseline":
        args.config = "configs/transformer_30m.yaml"
    run(args)


if __name__ == "__main__":
    main()
