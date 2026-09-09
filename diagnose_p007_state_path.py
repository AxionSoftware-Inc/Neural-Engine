"""No-training P-007 diagnostic for circuit/state causal strength.

The diagnostic keeps a checkpoint fixed and measures four related quantities
on one balanced batch: circuit correction magnitude, final quality with the
correction disabled, route replay damage, and whether replay changes the
correction itself.  It is intentionally a measurement tool, not a training or
default-model patch.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from analyze_route_ablation import route_source_indices
from data.generator import SyntheticTaskGenerator
from train import load_config, make_model, seed_everything


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(logits.argmax(dim=-1).eq(targets).float().mean().cpu())


def _loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(nn.functional.cross_entropy(logits, targets).cpu())


def _route_candidate_recall(
    candidates: torch.Tensor, selected: torch.Tensor,
) -> float:
    valid = selected.ge(0)
    included = (
        selected.unsqueeze(-1).eq(candidates.unsqueeze(-2)).any(dim=-1)
        & valid
    )
    return float(included[valid].float().mean().cpu()) if valid.any() else 0.0


def _capture_circuit_calls(model: nn.Module) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], Any]:
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        if len(inputs) >= 1 and isinstance(output, torch.Tensor):
            calls.append((inputs[0].detach(), output.detach()))

    handle = model.circuits.register_forward_hook(hook)
    return calls, handle


@torch.no_grad()
def _run_with_capture(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    adaptive: bool,
    forced_selected_ids: torch.Tensor | None = None,
    forced_selected_weights: torch.Tensor | None = None,
    forced_route_gains: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    calls, handle = _capture_circuit_calls(model)
    try:
        logits, stats = model(
            inputs,
            adaptive=adaptive,
            forced_selected_ids=forced_selected_ids,
            forced_selected_weights=forced_selected_weights,
            forced_route_gains=forced_route_gains,
        )
    finally:
        handle.remove()
    return logits, stats, calls


def _summarize_calls(
    calls: list[tuple[torch.Tensor, torch.Tensor]],
    encoded: torch.Tensor,
) -> dict[str, Any]:
    if not calls:
        return {
            "captured_steps": 0,
            "delta_norm_mean": 0.0,
            "delta_to_encoded_norm_ratio_mean": 0.0,
            "delta_to_query_norm_ratio_mean": 0.0,
        }
    encoded_norm = encoded.norm(dim=-1).clamp_min(1e-8)
    delta_norms = []
    encoded_ratios = []
    query_ratios = []
    for query, delta in calls:
        delta_norm = delta.norm(dim=-1)
        query_norm = query.norm(dim=-1).clamp_min(1e-8)
        delta_norms.append(delta_norm)
        encoded_ratios.append(delta_norm / encoded_norm)
        query_ratios.append(delta_norm / query_norm)
    return {
        "captured_steps": len(calls),
        "delta_norm_mean": float(torch.cat(delta_norms).mean().cpu()),
        "delta_norm_p95": float(torch.cat(delta_norms).quantile(0.95).cpu()),
        "delta_to_encoded_norm_ratio_mean": float(torch.cat(encoded_ratios).mean().cpu()),
        "delta_to_encoded_norm_ratio_p95": float(torch.cat(encoded_ratios).quantile(0.95).cpu()),
        "delta_to_query_norm_ratio_mean": float(torch.cat(query_ratios).mean().cpu()),
        "delta_to_query_norm_ratio_p95": float(torch.cat(query_ratios).quantile(0.95).cpu()),
    }


def _swap_routes(
    stats: dict[str, torch.Tensor],
    task_ids: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = route_source_indices(task_ids, mode)
    return (
        stats["selected_ids"][source].clone(),
        stats["selected_weights"][source].clone(),
        stats["route_gains"][source].clone(),
    )


@torch.no_grad()
def analyze_checkpoint(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = load_config(args.config, smoke=False) if args.config else dict(payload["config"])
    if config["model"] == "baseline":
        raise ValueError("P-007 state-path diagnostic requires a Neural Engine checkpoint")
    seed_everything(int(config["seed"]))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload.get("model_state", payload))
    checkpoint_report = payload.get("report", {})
    if "routing_capacity" in checkpoint_report or "routing_depth" in checkpoint_report:
        model.router.set_routing_state(
            capacity=int(checkpoint_report.get("routing_capacity", model.router.routing_capacity)),
            depth=int(checkpoint_report.get("routing_depth", model.router.active_depth)),
        )
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=args.seed,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    batch = generator.balanced_batch(args.examples_per_task, device)
    encoded = model.encode(batch.inputs)
    natural_logits, natural_stats, natural_calls = _run_with_capture(
        model, batch.inputs, adaptive=False,
    )
    natural_acc = _accuracy(natural_logits, batch.targets)
    natural_loss = _loss(natural_logits, batch.targets)
    natural_path = _summarize_calls(natural_calls, encoded)

    original_scale = float(model.circuit_delta_scale)
    model.circuit_delta_scale = 0.0
    no_circuit_logits, _no_circuit_stats, no_circuit_calls = _run_with_capture(
        model, batch.inputs, adaptive=False,
    )
    model.circuit_delta_scale = original_scale
    no_circuit_acc = _accuracy(no_circuit_logits, batch.targets)
    no_circuit_loss = _loss(no_circuit_logits, batch.targets)
    no_circuit_path = _summarize_calls(no_circuit_calls, encoded)

    route_replay: dict[str, dict[str, float]] = {}
    for mode in ("global", "within_task"):
        forced_ids, forced_weights, forced_gains = _swap_routes(
            natural_stats, batch.task_ids, mode,
        )
        replay_logits, _replay_stats, replay_calls = _run_with_capture(
            model,
            batch.inputs,
            adaptive=False,
            forced_selected_ids=forced_ids,
            forced_selected_weights=forced_weights,
            forced_route_gains=forced_gains,
        )
        replay_acc = _accuracy(replay_logits, batch.targets)
        replay_loss = _loss(replay_logits, batch.targets)
        route_delta_norms = []
        for (_, natural_delta), (_, replay_delta) in zip(natural_calls, replay_calls):
            route_delta_norms.append((replay_delta - natural_delta).norm(dim=-1))
        route_replay[mode] = {
            "accuracy": replay_acc,
            "accuracy_drop": natural_acc - replay_acc,
            "loss": replay_loss,
            "loss_increase": replay_loss - natural_loss,
            "mean_circuit_delta_change_norm": float(torch.cat(route_delta_norms).mean().cpu())
            if route_delta_norms else 0.0,
            "mean_circuit_delta_change_ratio": float(
                torch.cat(route_delta_norms).mean().cpu()
                / torch.cat([
                    delta.norm(dim=-1) for _, delta in natural_calls
                ]).mean().clamp_min(1e-8).cpu()
            ) if route_delta_norms else 0.0,
        }

    selected = natural_stats["selected_ids"]
    candidates = natural_stats["candidate_ids"]
    selected_valid = selected.ge(0)
    unique_selected = torch.unique(selected[selected_valid]).numel() if selected_valid.any() else 0
    result = {
        "checkpoint": str(checkpoint_path),
        "model": config["model"],
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "fixed_execution": True,
        "natural": {
            "accuracy": natural_acc,
            "loss": natural_loss,
            **natural_path,
            "candidate_recall": _route_candidate_recall(candidates, selected),
            "selected_unique": int(unique_selected),
            "selected_bank_fraction": float(unique_selected / model.circuits.num_circuits),
        },
        "no_circuit_scale": {
            "accuracy": no_circuit_acc,
            "accuracy_drop": natural_acc - no_circuit_acc,
            "loss": no_circuit_loss,
            "loss_increase": no_circuit_loss - natural_loss,
            **no_circuit_path,
        },
        "route_replay": route_replay,
    }
    del model, payload, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose P-007 circuit/state causal strength")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1712)
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
