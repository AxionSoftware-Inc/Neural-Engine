from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data.dynamic_composition import DynamicCompositionGenerator
from data.generator import VALUE_TOKEN_OFFSET
from neural_engine.instrumentation import count_parameters
from neural_engine.mesoscopic_macro import MesoscopicMacroCellBank
from neural_engine.router import HierarchicalRouter


class MesoscopicCompositionModel(nn.Module):
    def __init__(self, max_ops: int = 8, modulus: int = 64,
                 state_dim: int = 384, num_cells: int = 64,
                 hidden_dim: int = 480, bilinear_rank: int = 128,
                 candidate_pool: int = 8, active_cells: int = 2,
                 router_branch: int = 4, router_depth: int = 3,
                 residual_scale: float = 0.1) -> None:
        super().__init__()
        self.max_ops = max_ops
        self.modulus = modulus
        self.state_dim = state_dim
        self.active_cells = active_cells
        self.residual_scale = residual_scale
        self.value_embedding = nn.Embedding(modulus, state_dim)
        self.operation_embedding = nn.Embedding(3, state_dim)
        self.step_embedding = nn.Parameter(torch.zeros(max_ops, state_dim))
        self.initial_writer = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, state_dim), nn.Tanh()
        )
        self.router = HierarchicalRouter(
            state_dim, num_cells, router_branch, router_depth,
            candidate_pool, active_cells,
        )
        self.cells = MesoscopicMacroCellBank(
            num_cells, state_dim, hidden_dim, bilinear_rank, residual_scale
        )
        self.output = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, modulus)
        )
        nn.init.normal_(self.step_embedding, std=0.02)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value_start = 1 + self.max_ops
        operations = (inputs[:, 1:value_start] - 2).clamp(0, 2)
        operation_mask = inputs[:, 1:value_start].ge(2)
        values = (inputs[:, value_start:value_start + self.max_ops + 1]
                  - VALUE_TOKEN_OFFSET).clamp(0, self.modulus - 1)
        value_states = self.value_embedding(values)
        state = self.initial_writer(value_states[:, 0])
        memory = value_states[:, 0]
        step_logits = []
        selected_steps = []
        selected_weights = []
        route_entropies = []
        for step in range(self.max_ops):
            operand = value_states[:, step + 1]
            query = state + memory + operand + self.operation_embedding(operations[:, step])
            query = query + self.step_embedding[step]
            selected, weights, route_stats = self.router(query)
            next_state, next_memory = self.cells(state, memory + operand, selected, weights)
            active = operation_mask[:, step].unsqueeze(-1)
            state = torch.where(active, next_state, state)
            memory = torch.where(active, next_memory, memory)
            step_logits.append(self.output(state))
            selected_steps.append(selected)
            selected_weights.append(weights)
            route_entropies.append(route_stats["router_entropy"])
        return step_logits[-1], {
            "step_logits": torch.stack(step_logits, dim=1),
            "selected_ids": torch.stack(selected_steps, dim=1),
            "selected_weights": torch.stack(selected_weights, dim=1),
            "router_entropy": torch.stack(route_entropies),
        }

    def parameter_report(self) -> dict[str, int | float]:
        total = count_parameters(self)
        active_body = self.cells.parameters_per_cell * self.active_cells
        return {
            "total_params": total,
            "macro_cell_count": self.cells.num_cells,
            "parameters_per_cell": self.cells.parameters_per_cell,
            "active_macro_cells": self.active_cells,
            "active_body_params": active_body,
            "active_fraction": active_body / max(total, 1),
        }


def evaluate(model: MesoscopicCompositionModel, generator: DynamicCompositionGenerator,
             examples_per_depth: int, device: torch.device) -> dict[str, Any]:
    model.eval()
    batch = generator.balanced_batch(examples_per_depth, device)
    with torch.no_grad():
        logits, stats = model(batch.inputs)
    correct = logits.argmax(dim=-1).eq(batch.targets)
    by_depth = {}
    for depth in generator.allowed_depths:
        mask = batch.depths.eq(depth)
        by_depth[str(depth)] = float(correct[mask].float().mean().cpu())
    selected = stats["selected_ids"]
    return {
        "accuracy": float(correct.float().mean().cpu()),
        "accuracy_by_depth": by_depth,
        "loss": float(nn.functional.cross_entropy(logits, batch.targets).cpu()),
        "router_entropy": float(stats["router_entropy"].mean().cpu()),
        "unique_cells": int(selected.unique().numel()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == "auto":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu" if args.device == "auto" else args.device)
    model = MesoscopicCompositionModel(
        max_ops=args.max_ops, modulus=args.modulus, state_dim=args.state_dim,
        num_cells=args.num_cells, hidden_dim=args.hidden_dim,
        bilinear_rank=args.bilinear_rank, candidate_pool=args.candidate_pool,
        active_cells=args.active_cells, router_branch=args.router_branch,
        router_depth=args.router_depth,
        residual_scale=args.residual_scale,
    ).to(device)
    train_generator = DynamicCompositionGenerator(
        max_ops=args.max_ops, train_max_ops=args.train_max_ops,
        seed=args.seed + 1, split="train", modulus=args.modulus,
    )
    eval_generator = DynamicCompositionGenerator(
        max_ops=args.max_ops, train_max_ops=args.train_max_ops,
        seed=args.seed + 2, split="heldout", modulus=args.modulus,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    losses = []
    start = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        batch = train_generator.task_balanced_batch(args.batch_size, device)
        logits, stats = model(batch.inputs)
        loss = nn.functional.cross_entropy(logits, batch.targets)
        stage_losses = []
        for stage in range(args.max_ops):
            mask = batch.stage_mask[:, stage]
            if mask.any():
                stage_losses.append(nn.functional.cross_entropy(
                    stats["step_logits"][mask, stage], batch.stage_targets[mask, stage]
                ))
        if stage_losses:
            loss = loss + args.stage_loss_weight * torch.stack(stage_losses).mean()
        loss = loss - 0.0001 * stats["router_entropy"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == args.steps):
            print(f"step={step:05d} loss={losses[-1]:.5f}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    report = {
        "run_id": args.run_id,
        "seed": args.seed,
        "device": str(device),
        "steps": args.steps,
        "training_seconds": elapsed,
        "train_depths": list(train_generator.allowed_depths),
        "eval_depths": list(eval_generator.allowed_depths),
        "train": evaluate(model, train_generator, args.examples_per_depth, device),
        "evaluation": evaluate(model, eval_generator, args.examples_per_depth, device),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "model": model.parameter_report(),
        "canonical": {
            "num_cells": args.num_cells,
            "state_dim": args.state_dim,
            "hidden_dim": args.hidden_dim,
            "bilinear_rank": args.bilinear_rank,
            "candidate_pool": args.candidate_pool,
            "active_cells": args.active_cells,
            "residual_scale": args.residual_scale,
        },
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Equal-active-budget mesoscopic MacroCell task")
    parser.add_argument("--max-ops", type=int, default=8)
    parser.add_argument("--train-max-ops", type=int, default=4)
    parser.add_argument("--modulus", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=384)
    parser.add_argument("--num-cells", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=480)
    parser.add_argument("--bilinear-rank", type=int, default=128)
    parser.add_argument("--candidate-pool", type=int, default=8)
    parser.add_argument("--active-cells", type=int, default=2)
    parser.add_argument("--router-branch", type=int, default=4)
    parser.add_argument("--router-depth", type=int, default=3)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--examples-per-depth", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--stage-loss-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="ne_mesoscopic_macro_64_equal_active")
    parser.add_argument("--output", default="results/runs/ne_mesoscopic_macro_64_equal_active.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
