"""Distill final-CE route targets into the Native Engine retriever.

This is an opt-in retrieval-only diagnostic.  A frozen Native Engine is
probed with one-circuit substitutions from both the natural candidate pool and
the full-bank key top-k probe.  The best observed circuit is converted to its
contiguous active-group base and used as a teacher target for the hierarchical
tree/key router.  Circuit parameters, recurrent state, and the hard inference
budget stay frozen.

The experiment is deliberately conservative: it tests whether a target
aligned retrieval signal can improve the existing retriever at all.  It is not
an exhaustive subset oracle and it is not a production default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from analyze_route_neighborhood import _forced_losses
from benchmark_route_cost_surrogate import _load_checkpoint, _make_batches
from neural_engine.model import NeuralEngineV0
from neural_engine.router import HierarchicalRouter


@torch.inference_mode()
def _collect_targets(
    model: NeuralEngineV0,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    global_topk: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    query_rows = []
    target_rows = []
    total_examples = 0
    improved_examples = 0
    external_targets = 0
    target_gain = 0.0
    for inputs, targets in batches:
        natural_logits, stats = model(inputs, adaptive=False, collect_stats=True)
        natural_loss = F.cross_entropy(natural_logits, targets, reduction="none")
        natural_ids = stats["selected_ids"]
        natural_weights = stats["selected_weights"]
        natural_gains = stats["route_gains"]
        for step in range(model.internal_steps):
            query = stats["query_states"][:, step]
            candidates = stats["candidate_ids"][:, step]
            full_scores = torch.einsum(
                "bd,cd->bc", query, model.router.keys[:model.router.routing_capacity],
            ) / (query.shape[-1] ** 0.5)
            global_ids = full_scores.topk(
                min(global_topk, full_scores.shape[-1]), dim=-1,
            ).indices
            alternatives = torch.cat((candidates, global_ids), dim=-1)
            replace_slot = natural_weights[:, step].argmin(dim=-1)
            best_loss = natural_loss.clone()
            best_id = natural_ids[:, step].gather(
                1, replace_slot.unsqueeze(-1),
            ).squeeze(-1)
            candidate_width = candidates.shape[-1]
            for rank in range(alternatives.shape[-1]):
                losses = F.cross_entropy(
                    _forced_losses(
                        model,
                        inputs,
                        natural_ids,
                        natural_weights,
                        natural_gains,
                        query,
                        step,
                        alternatives[:, rank],
                        replace_slot,
                    ),
                    targets,
                    reduction="none",
                )
                improved = losses < best_loss
                best_loss = torch.where(improved, losses, best_loss)
                best_id = torch.where(improved, alternatives[:, rank], best_id)
            target_base = (
                best_id.div(model.active_circuits, rounding_mode="floor")
                * model.active_circuits
            ).clamp_max(model.router.routing_capacity - model.active_circuits)
            query_rows.append(query.detach().cpu())
            target_rows.append(target_base.detach().cpu())
            total_examples += int(targets.numel())
            improved_examples += int((best_loss < natural_loss - 1e-6).sum().cpu())
            target_gain += float((natural_loss - best_loss).sum().cpu())
            in_local = alternatives[:, :candidate_width].eq(best_id.unsqueeze(-1)).any(dim=-1)
            external_targets += int((~in_local).sum().cpu())
    return torch.cat(query_rows), torch.cat(target_rows), {
        "examples": total_examples,
        "steps": len(query_rows),
        "improved_fraction": improved_examples / max(1, total_examples),
        "mean_target_gain_ce": target_gain / max(1, total_examples),
        "external_target_fraction": external_targets / max(1, total_examples),
    }


def _freeze_except_retriever(model: NeuralEngineV0) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    router = model.router
    trainable = [router.level_projections, router.level_bias, router.keys]
    for parameter in trainable:
        parameter.requires_grad_(True)
    return trainable


def _train_retriever(
    model: NeuralEngineV0,
    queries: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict:
    device = next(model.parameters()).device
    queries = queries.to(device)
    targets = targets.to(device)
    trainable = _freeze_except_retriever(model)
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 9917)
    loss_total = 0.0
    model.eval()
    for _ in range(steps):
        indices = torch.randint(
            queries.shape[0], (batch_size,), generator=generator, device=device,
        )
        _, _, route_stats = model.router(
            queries[indices], target_bases=targets[indices], collect_stats=True,
        )
        loss = route_stats["routing_target_loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        loss_total += float(loss.detach().cpu())
    return {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "mean_target_loss": loss_total / max(1, steps),
    }


@torch.inference_mode()
def _evaluate(
    model: NeuralEngineV0,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict:
    total_loss = 0.0
    correct = 0
    examples = 0
    routes = []
    for inputs, targets in batches:
        logits, stats = model(inputs, adaptive=False, collect_stats=True)
        total_loss += float(F.cross_entropy(logits, targets, reduction="sum").cpu())
        correct += int(logits.argmax(dim=-1).eq(targets).sum().cpu())
        examples += int(targets.numel())
        routes.append(stats["selected_ids"].detach().cpu())
    route = torch.cat(routes).reshape(-1, model.active_circuits)
    return {
        "examples": examples,
        "mean_ce": total_loss / max(1, examples),
        "accuracy": correct / max(1, examples),
        "unique_selected_circuits": int(route.unique().numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill final-CE targets into Native Engine retrieval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-batches", type=int, default=2)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--global-topk", type=int, default=8)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, config = _load_checkpoint(Path(args.checkpoint), device)
    if not isinstance(model.router, HierarchicalRouter):
        raise ValueError("target retrieval distillation requires HierarchicalRouter")
    train_batches = _make_batches(
        config, device, "train", args.train_batches, args.examples_per_task,
    )
    eval_batches = _make_batches(
        config, device, "heldout", args.eval_batches, args.examples_per_task,
    )
    baseline = _evaluate(model, eval_batches)
    queries, target_bases, target_report = _collect_targets(
        model, train_batches, args.global_topk,
    )
    training = _train_retriever(
        model, queries, target_bases, args.steps, args.batch_size,
        args.learning_rate, int(config["seed"]),
    )
    treatment = _evaluate(model, eval_batches)
    result = {
        "checkpoint": str(args.checkpoint),
        "model": config.get("model"),
        "seed": int(config.get("seed", -1)),
        "device": str(device),
        "candidate_pool": int(model.router.candidate_pool),
        "active_circuits": int(model.active_circuits),
        "global_topk": int(args.global_topk),
        "target_report": target_report,
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
