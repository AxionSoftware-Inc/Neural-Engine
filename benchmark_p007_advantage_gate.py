"""P-007 final-loss advantage gate pilot.

The body and router stay frozen.  A tiny linear gate is fitted from
per-example final-CE advantage (correction on versus correction off), using
the natural ``[step_query, circuit_delta]`` features.  The learned gate is
then folded into the existing bounded correction-gate interface and evaluated
on a different balanced batch.  This is an opt-in diagnostic, not a default
training recipe.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data.generator import SyntheticTaskGenerator
from train import load_config, make_model, seed_everything


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    losses = nn.functional.cross_entropy(logits, targets, reduction="none")
    return {
        "accuracy": float(logits.argmax(dim=-1).eq(targets).float().mean().cpu()),
        "ce": float(losses.mean().cpu()),
    }


def _capture_calls(model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def hook(_module: nn.Module, hook_inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        if hook_inputs and isinstance(output, torch.Tensor):
            calls.append((hook_inputs[0].detach(), output.detach()))

    handle = model.circuits.register_forward_hook(hook)
    try:
        logits, _stats = model(inputs, adaptive=False)
    finally:
        handle.remove()
    return logits, calls


@torch.no_grad()
def _forward_at_scale(model: nn.Module, inputs: torch.Tensor, scale: float) -> torch.Tensor:
    original = float(model.circuit_delta_scale)
    model.circuit_delta_scale = float(scale)
    try:
        logits, _stats = model(inputs, adaptive=False)
    finally:
        model.circuit_delta_scale = original
    return logits


def _fold_standardized_linear(
    linear: nn.Linear,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a linear on standardized features back to raw-feature weights."""
    raw_weight = linear.weight.detach() / std.unsqueeze(0)
    raw_bias = linear.bias.detach() - (raw_weight * mean.unsqueeze(0)).sum(dim=1)
    return raw_weight, raw_bias


def _fit_advantage_gate(
    features: torch.Tensor,
    advantage: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    margin: float,
    negative_logit: float,
) -> dict[str, Any]:
    repeated_advantage = advantage.repeat(features.shape[0] // advantage.shape[0])
    labels = torch.where(repeated_advantage.ge(margin), 0.0, negative_logit)
    valid = repeated_advantage.abs().ge(margin)
    if not valid.any():
        raise ValueError("advantage margin excluded every calibration feature")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-5)
    normalized = (features - mean) / std
    linear = nn.Linear(features.shape[-1], 1, device=features.device)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    optimizer = torch.optim.Adam(linear.parameters(), lr=learning_rate)
    final_loss = 0.0
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = linear(normalized).squeeze(-1)
        loss = ((prediction[valid] - labels[valid]) ** 2).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    raw_weight, raw_bias = _fold_standardized_linear(linear, mean, std)
    with torch.no_grad():
        preactivation = linear(normalized).squeeze(-1)
        predicted_gate = 2.0 * torch.sigmoid(preactivation)
        target_gate = torch.where(labels.eq(0.0), 1.0, 0.0)
        valid_accuracy = float(
            predicted_gate[valid].sub(target_gate[valid]).abs().lt(0.5).float().mean().cpu()
        )
    return {
        "weight": raw_weight,
        "bias": raw_bias,
        "fit_loss": final_loss,
        "valid_fraction": float(valid.float().mean().cpu()),
        "positive_fraction": float(labels.eq(0.0).float().mean().cpu()),
        "gate_target_accuracy": valid_accuracy,
        "feature_mean": mean,
        "feature_std": std,
    }


def analyze_checkpoint(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = load_config(args.config, smoke=False) if args.config else dict(payload["config"])
    if config["model"] == "baseline":
        raise ValueError("P-007 advantage gate requires a Neural Engine checkpoint")
    seed_everything(int(config["seed"]))
    device = (torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
              else torch.device(args.device))
    state_dict = payload.get("model_state", payload)
    base = make_model(config).to(device).eval()
    base.load_state_dict(state_dict)
    checkpoint_report = payload.get("report", {})
    if "routing_capacity" in checkpoint_report or "routing_depth" in checkpoint_report:
        base.router.set_routing_state(
            capacity=int(checkpoint_report.get("routing_capacity", base.router.routing_capacity)),
            depth=int(checkpoint_report.get("routing_depth", base.router.active_depth)),
        )

    def make_batch(seed: int):
        generator = SyntheticTaskGenerator(
            config["seq_len"], seed=seed,
            value_min=int(config.get("eval_value_min", 0)),
            value_max=int(config.get("eval_value_max", 63)),
            split=str(config.get("eval_split", "all")),
        )
        return generator.balanced_batch(args.examples_per_task, device)

    calibration = make_batch(args.calibration_seed)
    evaluation = make_batch(args.eval_seed)
    calibration_logits, calls = _capture_calls(base, calibration.inputs)
    no_circuit_logits = _forward_at_scale(base, calibration.inputs, 0.0)
    natural_losses = nn.functional.cross_entropy(
        calibration_logits, calibration.targets, reduction="none",
    )
    no_circuit_losses = nn.functional.cross_entropy(
        no_circuit_logits, calibration.targets, reduction="none",
    )
    advantage = no_circuit_losses - natural_losses
    features = torch.cat([
        torch.cat((query, delta), dim=-1) for query, delta in calls
    ], dim=0)
    fit = _fit_advantage_gate(
        features, advantage, steps=args.fit_steps, learning_rate=args.learning_rate,
        margin=args.margin, negative_logit=args.negative_logit,
    )

    gated_config = dict(config)
    gated_config["correction_gate_mode"] = "route_bounded"
    gated = make_model(gated_config).to(device).eval()
    gated.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        gated.correction_gate.weight.copy_(fit["weight"])
        gated.correction_gate.bias.copy_(fit["bias"])
    natural_eval = _metrics(_forward_at_scale(base, evaluation.inputs, 1.0), evaluation.targets)
    no_correction_eval = _metrics(_forward_at_scale(base, evaluation.inputs, 0.0), evaluation.targets)
    gated_eval = _metrics(gated(evaluation.inputs, adaptive=False)[0], evaluation.targets)
    calibration_natural = _metrics(calibration_logits, calibration.targets)
    calibration_no_correction = _metrics(no_circuit_logits, calibration.targets)
    result = {
        "checkpoint": str(checkpoint_path),
        "model": config["model"],
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "calibration_seed": args.calibration_seed,
        "eval_seed": args.eval_seed,
        "fit_steps": args.fit_steps,
        "margin": args.margin,
        "negative_logit": args.negative_logit,
        "calibration_advantage": {
            "mean": float(advantage.mean().cpu()),
            "positive_fraction": float(advantage.gt(0).float().mean().cpu()),
            "negative_fraction": float(advantage.lt(0).float().mean().cpu()),
            "p10": float(advantage.quantile(0.10).cpu()),
            "p90": float(advantage.quantile(0.90).cpu()),
        },
        "gate_fit": {
            key: value for key, value in fit.items()
            if key not in {"weight", "bias", "feature_mean", "feature_std"}
        },
        "calibration": {
            "natural_scale_1": calibration_natural,
            "no_correction_scale_0": calibration_no_correction,
        },
        "evaluation": {
            "natural_scale_1": natural_eval,
            "no_correction_scale_0": no_correction_eval,
            "advantage_gate": gated_eval,
            "gate_minus_natural_accuracy_pp": (gated_eval["accuracy"] - natural_eval["accuracy"]) * 100.0,
            "gate_minus_natural_ce": gated_eval["ce"] - natural_eval["ce"],
        },
    }
    del base, gated, payload, calibration, evaluation
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="P-007 final-loss advantage gate pilot")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=1712)
    parser.add_argument("--eval-seed", type=int, default=2712)
    parser.add_argument("--fit-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=0.001)
    parser.add_argument("--negative-logit", type=float, default=-4.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = [analyze_checkpoint(args, Path(path)) for path in args.checkpoint]
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
