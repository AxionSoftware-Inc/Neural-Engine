"""Prototype a cheap final-cost surrogate for Native Engine route selection.

The frozen Native Engine supplies calibration labels by evaluating one-circuit
route substitutions.  A small MLP then predicts those costs from information
available before circuit execution: the recurrent query, candidate key, the
current selected-route summary, a key score, and the internal-step identity.
Held-out evaluation uses the predicted candidate but measures its real final
CE with the frozen model.  This is an opt-in diagnostic, not a model default.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from analyze_route_neighborhood import _forced_losses
from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from neural_engine.router import HierarchicalRouter
from train import make_model


class CostSurrogate(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _load_checkpoint(path: Path, device: torch.device) -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("cost surrogate requires a NeuralEngineV0 checkpoint")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    if not isinstance(model.router, HierarchicalRouter):
        raise ValueError("cost surrogate currently targets HierarchicalRouter")
    return model, config


def _make_batches(config: dict, device: torch.device, split: str,
                  count: int, examples_per_task: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    prefix = "train" if split == "train" else "heldout"
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config["seed"]) + (101 if split == "train" else 3),
        value_min=int(config.get(prefix + "_value_min", 0)),
        value_max=int(config.get(prefix + "_value_max", 63)),
        split=split,
    )
    return [
        (batch.inputs, batch.targets)
        for batch in (
            generator.balanced_batch(examples_per_task, device)
            for _ in range(count)
        )
    ]


def _feature_tensor(model: NeuralEngineV0, stats: dict, step: int,
                    signature_projection: torch.Tensor | None = None) -> torch.Tensor:
    query = stats["query_states"][:, step]
    candidate_ids = stats["candidate_ids"][:, step]
    selected_ids = stats["selected_ids"][:, step]
    selected_keys = model.router.keys[selected_ids]
    route_summary = selected_keys.mean(dim=1)
    candidate_keys = model.router.keys[candidate_ids]
    key_score = torch.einsum("bd,bpd->bp", query, candidate_keys)
    key_score = key_score / (query.shape[-1] ** 0.5)
    step_one_hot = F.one_hot(
        torch.full((query.shape[0],), step, device=query.device),
        num_classes=model.internal_steps,
    ).to(dtype=query.dtype)
    repeated_query = query.unsqueeze(1).expand_as(candidate_keys)
    repeated_summary = route_summary.unsqueeze(1).expand_as(candidate_keys)
    repeated_step = step_one_hot.unsqueeze(1).expand(
        -1, candidate_keys.shape[1], -1,
    )
    features = [
        repeated_query, candidate_keys, repeated_summary,
        key_score.unsqueeze(-1),
    ]
    if signature_projection is not None:
        circuits = model.circuits
        if not all(hasattr(circuits, name) for name in ("down", "up", "bias")):
            raise ValueError("output signature requires an independent-style circuit bank")
        down = circuits.down[candidate_ids]
        up = circuits.up[candidate_ids]
        bias = circuits.bias[candidate_ids]
        hidden = torch.einsum("bd,bpdr->bpr", query, down)
        hidden = F.gelu(hidden)
        projected_up = torch.einsum("bprd,ds->bprs", up, signature_projection)
        signature = torch.einsum("bpr,bprs->bps", hidden, projected_up)
        signature = signature + torch.einsum(
            "bpd,ds->bps", bias, signature_projection,
        )
        features.append(signature)
    features.append(repeated_step)
    return torch.cat(tuple(features), dim=-1)


@torch.inference_mode()
def _collect_batch(model: NeuralEngineV0, inputs: torch.Tensor,
                   targets: torch.Tensor,
                   signature_projection: torch.Tensor | None = None,
                   ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    natural_logits, stats = model(inputs, adaptive=False, collect_stats=True)
    natural_loss = F.cross_entropy(natural_logits, targets, reduction="none")
    features = []
    labels = []
    for step in range(model.internal_steps):
        step_features = _feature_tensor(model, stats, step, signature_projection)
        query = stats["query_states"][:, step]
        candidates = stats["candidate_ids"][:, step]
        replace_slot = stats["selected_weights"][:, step].argmin(dim=-1)
        step_labels = []
        for rank in range(candidates.shape[-1]):
            losses = F.cross_entropy(
                _forced_losses(
                    model,
                    inputs,
                    stats["selected_ids"],
                    stats["selected_weights"],
                    stats["route_gains"],
                    query,
                    step,
                    candidates[:, rank],
                    replace_slot,
                ),
                targets,
                reduction="none",
            )
            step_labels.append(losses)
        features.append(step_features)
        labels.append(torch.stack(step_labels, dim=1))
    return torch.stack(features, dim=1), torch.stack(labels, dim=1), {
        "natural_loss": natural_loss,
        "stats": stats,
    }


def _collect_dataset(model: NeuralEngineV0,
                     batches: list[tuple[torch.Tensor, torch.Tensor]],
                     signature_projection: torch.Tensor | None = None,
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    feature_rows = []
    label_rows = []
    for inputs, targets in batches:
        features, labels, _ = _collect_batch(
            model, inputs, targets, signature_projection,
        )
        feature_rows.append(features.reshape(-1, features.shape[-1]).cpu())
        label_rows.append(labels.reshape(-1).cpu())
    return torch.cat(feature_rows), torch.cat(label_rows)


def _train_surrogate(features: torch.Tensor, labels: torch.Tensor,
                     device: torch.device, steps: int, batch_size: int,
                     seed: int) -> tuple[CostSurrogate, torch.Tensor, torch.Tensor, dict]:
    torch.manual_seed(seed + 7001)
    mean = features.mean(dim=0)
    std = features.std(dim=0).clamp_min(1e-5)
    normalized = (features - mean) / std
    model = CostSurrogate(normalized.shape[-1]).to(device)
    normalized = normalized.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    generator = torch.Generator(device=device).manual_seed(seed + 7002)
    loss_total = 0.0
    for _ in range(steps):
        indices = torch.randint(normalized.shape[0], (batch_size,), generator=generator, device=device)
        prediction = model(normalized[indices])
        loss = F.smooth_l1_loss(prediction, labels[indices], beta=0.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_total += float(loss.detach().cpu())
    return model, mean.to(device), std.to(device), {
        "training_steps": int(steps),
        "training_batch_size": int(batch_size),
        "mean_smooth_l1": loss_total / max(1, steps),
        "train_examples": int(features.shape[0]),
        "surrogate_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


@torch.inference_mode()
def _evaluate(model: NeuralEngineV0, surrogate: CostSurrogate,
              feature_mean: torch.Tensor, feature_std: torch.Tensor,
              batches: list[tuple[torch.Tensor, torch.Tensor]],
              signature_projection: torch.Tensor | None = None,
              ) -> dict:
    totals = {
        "examples": 0,
        "natural_loss": 0.0,
        "oracle_local_loss": 0.0,
        "predicted_loss": 0.0,
        "oracle_matches": 0,
        "predicted_improvement": 0,
    }
    step_rows = []
    for step in range(model.internal_steps):
        step_totals = {key: 0.0 for key in (
            "examples", "natural_loss", "oracle_local_loss", "predicted_loss",
            "oracle_matches", "predicted_improvement",
        )}
        for inputs, targets in batches:
            features, labels, context = _collect_batch(
                model, inputs, targets, signature_projection,
            )
            step_features = features[:, step]
            step_labels = labels[:, step]
            prediction = surrogate(
                (step_features.reshape(-1, step_features.shape[-1]) - feature_mean)
                / feature_std,
            ).reshape(step_features.shape[:2])
            predicted_rank = prediction.argmin(dim=-1)
            oracle_rank = step_labels.argmin(dim=-1)
            candidate_ids = context["stats"]["candidate_ids"][:, step]
            alternative = candidate_ids.gather(1, predicted_rank.unsqueeze(-1)).squeeze(-1)
            stats = context["stats"]
            query = stats["query_states"][:, step]
            replace_slot = stats["selected_weights"][:, step].argmin(dim=-1)
            predicted_loss = F.cross_entropy(
                _forced_losses(
                    model,
                    inputs,
                    stats["selected_ids"],
                    stats["selected_weights"],
                    stats["route_gains"],
                    query,
                    step,
                    alternative,
                    replace_slot,
                ),
                targets,
                reduction="none",
            )
            natural_loss = context["natural_loss"]
            oracle_local_loss = step_labels.min(dim=-1).values
            n = targets.numel()
            step_totals["examples"] += n
            step_totals["natural_loss"] += float(natural_loss.sum().cpu())
            step_totals["oracle_local_loss"] += float(oracle_local_loss.sum().cpu())
            step_totals["predicted_loss"] += float(predicted_loss.sum().cpu())
            step_totals["oracle_matches"] += int(predicted_rank.eq(oracle_rank).sum().cpu())
            step_totals["predicted_improvement"] += int((predicted_loss < natural_loss - 1e-6).sum().cpu())
        for key in totals:
            totals[key] += step_totals[key]
        natural = step_totals["natural_loss"] / step_totals["examples"]
        oracle = step_totals["oracle_local_loss"] / step_totals["examples"]
        predicted = step_totals["predicted_loss"] / step_totals["examples"]
        step_rows.append({
            "step": step,
            "examples": int(step_totals["examples"]),
            "natural_ce": natural,
            "oracle_local_ce": oracle,
            "predicted_route_ce": predicted,
            "oracle_local_gain_ce": natural - oracle,
            "surrogate_gain_ce": natural - predicted,
            "oracle_gain_recovery": ((natural - predicted) / (natural - oracle)
                                      if natural > oracle else 0.0),
            "top1_oracle_match": step_totals["oracle_matches"] / step_totals["examples"],
            "predicted_improvement_fraction": step_totals["predicted_improvement"] / step_totals["examples"],
        })
    return {
        "steps": step_rows,
        "overall": {
            "examples": int(totals["examples"]),
            "natural_ce": totals["natural_loss"] / totals["examples"],
            "oracle_local_ce": totals["oracle_local_loss"] / totals["examples"],
            "predicted_route_ce": totals["predicted_loss"] / totals["examples"],
            "oracle_local_gain_ce": (totals["natural_loss"] - totals["oracle_local_loss"]) / totals["examples"],
            "surrogate_gain_ce": (totals["natural_loss"] - totals["predicted_loss"]) / totals["examples"],
            "oracle_gain_recovery": ((totals["natural_loss"] - totals["predicted_loss"])
                                      / (totals["natural_loss"] - totals["oracle_local_loss"])
                                      if totals["natural_loss"] > totals["oracle_local_loss"] else 0.0),
            "top1_oracle_match": totals["oracle_matches"] / totals["examples"],
            "predicted_improvement_fraction": totals["predicted_improvement"] / totals["examples"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a frozen-bank route cost surrogate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-batches", type=int, default=3)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--surrogate-steps", type=int, default=1000)
    parser.add_argument("--surrogate-batch-size", type=int, default=4096)
    parser.add_argument("--signature-dim", type=int, default=0,
                        help="Add a fixed candidate output sketch to the surrogate feature")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    random.seed(17)
    np.random.seed(17)
    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    model, config = _load_checkpoint(checkpoint, device)
    train_batches = _make_batches(config, device, "train", args.train_batches, args.examples_per_task)
    eval_batches = _make_batches(config, device, "heldout", args.eval_batches, args.examples_per_task)
    signature_projection = None
    if args.signature_dim > 0:
        generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 9091)
        signature_projection = torch.randn(
            model.state_dim, args.signature_dim, generator=generator, device=device,
        ) / (model.state_dim ** 0.5)
    train_features, train_labels = _collect_dataset(
        model, train_batches, signature_projection,
    )
    surrogate, mean, std, train_report = _train_surrogate(
        train_features,
        train_labels,
        device,
        args.surrogate_steps,
        args.surrogate_batch_size,
        int(config["seed"]),
    )
    evaluation = _evaluate(
        model, surrogate, mean, std, eval_batches, signature_projection,
    )
    result = {
        "checkpoint": str(checkpoint),
        "model": config.get("model"),
        "seed": int(config.get("seed", -1)),
        "device": str(device),
        "train_batches": int(args.train_batches),
        "eval_batches": int(args.eval_batches),
        "examples_per_task": int(args.examples_per_task),
        "active_circuits": int(model.active_circuits),
        "candidate_pool": int(model.router.candidate_pool),
        "signature_dim": int(args.signature_dim),
        "calibration": train_report,
        "evaluation": evaluation,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
