from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from analyze_counterfactual_routes import build_counterfactual_rows
from data.generator import SyntheticTaskGenerator
from train import load_config, make_model, seed_everything


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(targets).float().mean().cpu())


def summarize_replay(base_batch, variant_batch, base_logits, variant_logits,
                     replayed_variant, replayed_base) -> dict[str, float]:
    base_loss = nn.functional.cross_entropy(base_logits, base_batch.targets)
    variant_loss = nn.functional.cross_entropy(variant_logits, variant_batch.targets)
    replayed_variant_loss = nn.functional.cross_entropy(replayed_variant, variant_batch.targets)
    replayed_base_loss = nn.functional.cross_entropy(replayed_base, base_batch.targets)
    return {
        "base_natural_accuracy": accuracy(base_logits, base_batch.targets),
        "variant_natural_accuracy": accuracy(variant_logits, variant_batch.targets),
        "variant_with_base_route_accuracy": accuracy(replayed_variant, variant_batch.targets),
        "base_with_variant_route_accuracy": accuracy(replayed_base, base_batch.targets),
        "variant_natural_loss": float(variant_loss.cpu()),
        "variant_with_base_route_loss": float(replayed_variant_loss.cpu()),
        "variant_loss_increase": float((replayed_variant_loss - variant_loss).cpu()),
        "base_natural_loss": float(base_loss.cpu()),
        "base_with_variant_route_loss": float(replayed_base_loss.cpu()),
        "base_loss_increase": float((replayed_base_loss - base_loss).cpu()),
        "variant_mean_abs_logit_delta": float((variant_logits - replayed_variant).abs().mean().cpu()),
        "base_mean_abs_logit_delta": float((base_logits - replayed_base).abs().mean().cpu()),
    }


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = load_config(args.config, smoke=False) if args.config else dict(payload["config"])
    if config["model"] == "baseline":
        raise ValueError("route replay requires a Neural Engine checkpoint")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload.get("model_state", payload))
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=args.seed,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split="all",
    )
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model": config["model"],
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "forced_route_mode": "fixed three-step replay; adaptive halting disabled for causal comparison",
        "counterfactuals": {},
    }
    for mode, title in (("operand", "single_operand"), ("task_token", "single_task_token")):
        base_rows, variant_rows, _ = build_counterfactual_rows(
            generator, args.examples_per_task, mode)
        base_batch = generator._make_batch(base_rows, device)
        variant_batch = generator._make_batch(variant_rows, device)
        base_logits, base_stats = model(base_batch.inputs, adaptive=False)
        variant_logits, variant_stats = model(variant_batch.inputs, adaptive=False)
        replayed_variant, _ = model(
            variant_batch.inputs, adaptive=False,
            forced_selected_ids=base_stats["selected_ids"],
            forced_selected_weights=base_stats["selected_weights"],
            forced_route_gains=base_stats["route_gains"],
        )
        replayed_base, _ = model(
            base_batch.inputs, adaptive=False,
            forced_selected_ids=variant_stats["selected_ids"],
            forced_selected_weights=variant_stats["selected_weights"],
            forced_route_gains=variant_stats["route_gains"],
        )
        result["counterfactuals"][title] = summarize_replay(
            base_batch, variant_batch, base_logits, variant_logits,
            replayed_variant, replayed_base)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether route changes are causally used")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1710)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rendered = json.dumps(analyze(args), indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
