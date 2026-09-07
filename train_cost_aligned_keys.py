"""Train only Native Engine router keys against frozen one-swap CE labels.

This is an offline policy-improvement diagnostic.  The circuit bank, recurrent
body, tree retrieval projections, and inference path stay frozen.  Calibration
labels are the real final CE after replacing one selected circuit.  At
inference the treatment still performs the original query-key scoring over the
same candidate pool, so no extra candidate circuit computation is introduced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_route_cost_surrogate import _collect_batch, _load_checkpoint, _make_batches
from neural_engine.model import NeuralEngineV0


def _collect_key_dataset(model: NeuralEngineV0,
                         batches: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
    query_rows = []
    candidate_rows = []
    cost_rows = []
    for inputs, targets in batches:
        _, labels, context = _collect_batch(model, inputs, targets)
        stats = context["stats"]
        query_rows.append(stats["query_states"].detach().cpu())
        candidate_rows.append(stats["candidate_ids"].detach().cpu())
        cost_rows.append(labels.detach().cpu())
    return torch.cat(query_rows), torch.cat(candidate_rows), torch.cat(cost_rows)


def _freeze_except_keys(model: NeuralEngineV0) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.router.keys.requires_grad_(True)


def _train_keys(model: NeuralEngineV0, queries: torch.Tensor,
                candidate_ids: torch.Tensor, costs: torch.Tensor,
                steps: int, batch_size: int, target_temperature: float,
                score_temperature: float, anchor_weight: float,
                learning_rate: float, seed: int) -> dict:
    if target_temperature <= 0 or score_temperature <= 0:
        raise ValueError("temperatures must be positive")
    device = next(model.parameters()).device
    queries = queries.to(device)
    candidate_ids = candidate_ids.to(device)
    costs = costs.to(device)
    initial_keys = model.router.keys.detach().clone()
    optimizer = torch.optim.AdamW([model.router.keys], lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 8101)
    loss_total = 0.0
    for _ in range(steps):
        indices = torch.randint(queries.shape[0], (batch_size,), generator=generator, device=device)
        query = queries[indices]
        ids = candidate_ids[indices]
        cost = costs[indices]
        keys = model.router.keys[ids]
        scores = torch.einsum("bsd,bspd->bsp", query, keys)
        scores = scores / (query.shape[-1] ** 0.5)
        target = F.softmax(-(cost - cost.min(dim=-1, keepdim=True).values)
                            / target_temperature, dim=-1)
        loss = -(target * F.log_softmax(scores / score_temperature, dim=-1)).sum(dim=-1).mean()
        if anchor_weight:
            loss = loss + anchor_weight * F.mse_loss(model.router.keys, initial_keys)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([model.router.keys], 1.0)
        optimizer.step()
        loss_total += float(loss.detach().cpu())
    delta = (model.router.keys.detach() - initial_keys).norm() / initial_keys.norm().clamp_min(1e-8)
    return {
        "training_steps": int(steps),
        "training_batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "mean_loss": loss_total / max(1, steps),
        "relative_key_update_l2": float(delta.cpu()),
    }


@torch.inference_mode()
def _evaluate(model: NeuralEngineV0,
              batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    ce = 0.0
    correct = 0
    count = 0
    routes = []
    for inputs, targets in batches:
        logits, stats = model(inputs, adaptive=False, collect_stats=True)
        ce += float(F.cross_entropy(logits, targets, reduction="sum").cpu())
        correct += int(logits.argmax(dim=-1).eq(targets).sum().cpu())
        count += int(targets.numel())
        routes.append(stats["selected_ids"].detach().cpu())
    route = torch.cat(routes).reshape(-1, model.active_circuits)
    return {
        "examples": count,
        "mean_ce": ce / count,
        "accuracy": correct / count,
        "unique_selected_circuits": int(route.unique().numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Native Engine keys on one-swap CE labels")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--target-temperature", type=float, default=0.05)
    parser.add_argument("--score-temperature", type=float, default=0.05)
    parser.add_argument("--anchor-weight", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    model, config = _load_checkpoint(checkpoint, device)
    train_batches = _make_batches(config, device, "train", args.train_batches, args.examples_per_task)
    eval_batches = _make_batches(config, device, "heldout", args.eval_batches, args.examples_per_task)
    train_queries, train_candidates, train_costs = _collect_key_dataset(model, train_batches)
    baseline = _evaluate(model, eval_batches)
    _freeze_except_keys(model)
    training = _train_keys(
        model, train_queries, train_candidates, train_costs,
        args.steps, args.batch_size, args.target_temperature,
        args.score_temperature, args.anchor_weight, args.learning_rate,
        int(config["seed"]),
    )
    treatment = _evaluate(model, eval_batches)
    result = {
        "checkpoint": str(checkpoint),
        "model": config.get("model"),
        "seed": int(config.get("seed", -1)),
        "device": str(device),
        "active_circuits": int(model.active_circuits),
        "candidate_pool": int(model.router.candidate_pool),
        "train_label_rows": int(train_queries.shape[0] * train_queries.shape[1] * train_candidates.shape[2]),
        "calibration": {
            "train_batches": int(args.train_batches),
            "eval_batches": int(args.eval_batches),
            "examples_per_task": int(args.examples_per_task),
            "target_temperature": float(args.target_temperature),
            "score_temperature": float(args.score_temperature),
            "anchor_weight": float(args.anchor_weight),
            "learning_rate": float(args.learning_rate),
        },
        "training": training,
        "baseline": baseline,
        "treatment": treatment,
        "delta": {
            "mean_ce": treatment["mean_ce"] - baseline["mean_ce"],
            "accuracy_pp": (treatment["accuracy"] - baseline["accuracy"]) * 100,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
