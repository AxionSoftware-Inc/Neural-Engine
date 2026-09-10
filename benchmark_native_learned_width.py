"""Train a small native width head from paired K=8/K=16 loss-gap labels.

The circuit bank and base router are frozen. The head is trained on a separate
calibration stream, then saved into a K=16 checkpoint whose inference path can
perform one-path K=8/K=16 dispatch. This is a first predictor screen, not a
claim that the oracle labels are exact recurrent-trajectory costs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from data.generator import SyntheticTaskGenerator
from train import BatchSource, make_model, seed_everything


CONDITIONS = {
    "uniform_all": {"value_min": 0, "value_max": 63, "split": "all"},
    "combination_heldout": {"value_min": 0, "value_max": 63, "split": "heldout"},
    "low_edge_values": {"value_min": 0, "value_max": 7, "split": "all"},
    "high_edge_values": {"value_min": 56, "value_max": 63, "split": "all"},
}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _collect_features(checkpoint: dict[str, Any], device: torch.device,
                      batches: int, lambda_target: float,
                      seed_offset: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    base_config = dict(checkpoint["config"])
    config8 = dict(base_config)
    config8["active_circuits"] = 8
    config8["dynamic_width_mode"] = "none"
    config16 = dict(base_config)
    config16["active_circuits"] = 16
    config16["dynamic_width_mode"] = "none"
    model8 = make_model(config8).to(device)
    model16 = make_model(config16).to(device)
    model8.load_state_dict(checkpoint["model_state"])
    model16.load_state_dict(checkpoint["model_state"])
    model8.eval()
    model16.eval()
    seed = int(base_config.get("seed", 17)) + seed_offset
    feature_parts = []
    label_parts = []
    counts: dict[str, int] = {}
    for condition_index, (name, condition) in enumerate(CONDITIONS.items()):
        generator = SyntheticTaskGenerator(
            seq_len=int(base_config["seq_len"]),
            seed=seed + condition_index * 1000,
            **condition,
        )
        source = BatchSource(generator, 256, device)
        count = 0
        for _ in range(batches):
            batch = source.balanced(32)
            with torch.no_grad():
                _, stats8 = model8(batch.inputs, adaptive=False)
                _, stats16 = model16(batch.inputs, adaptive=False)
            step_targets = batch.targets.unsqueeze(1).expand(-1, stats8["step_logits"].shape[1])
            loss8 = F.cross_entropy(
                stats8["step_logits"].reshape(-1, stats8["step_logits"].shape[-1]),
                step_targets.reshape(-1), reduction="none",
            ).reshape_as(step_targets)
            loss16 = F.cross_entropy(
                stats16["step_logits"].reshape(-1, stats16["step_logits"].shape[-1]),
                step_targets.reshape(-1), reduction="none",
            ).reshape_as(step_targets)
            # Width 16 is worth its extra half-budget when it lowers CE by
            # more than lambda * (1.0 - 0.5).  Pair both trajectory states so
            # the head sees the states it may encounter after either choice.
            labels = (loss8 - loss16 > float(lambda_target) * 0.5).float()
            query8 = stats8["query_states"]
            query16 = stats16["query_states"]
            feature_parts.extend((query8.reshape(-1, query8.shape[-1]).cpu(),
                                  query16.reshape(-1, query16.shape[-1]).cpu()))
            label_parts.extend((labels.reshape(-1).cpu(), labels.reshape(-1).cpu()))
            count += int(labels.numel())
        counts[name] = count
    del model8, model16
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(feature_parts), torch.cat(label_parts), counts


def _fit_head(model: torch.nn.Module, features: torch.Tensor, labels: torch.Tensor,
              epochs: int, learning_rate: float, seed: int) -> dict[str, float]:
    if getattr(model, "dynamic_width_head", None) is None:
        raise ValueError("model must be constructed with dynamic_width_mode='learned'")
    seed_everything(seed)
    split = max(1, int(features.shape[0] * 0.8))
    permutation = torch.randperm(features.shape[0])
    train_indices = permutation[:split]
    valid_indices = permutation[split:]
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    valid_features = features[valid_indices]
    valid_labels = labels[valid_indices]
    positive = train_labels.sum().clamp_min(1.0)
    negative = train_labels.numel() - positive
    pos_weight = (negative / positive).sqrt().clamp(1.0, 12.0)
    head = model.dynamic_width_head
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    batch_size = 2048
    for _ in range(epochs):
        order = torch.randperm(train_features.shape[0])
        for start in range(0, train_features.shape[0], batch_size):
            indices = order[start:start + batch_size]
            batch_features = train_features[indices].to(next(head.parameters()).device)
            batch_labels = train_labels[indices].to(batch_features.device)
            logits = head(batch_features).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logits, batch_labels, pos_weight=pos_weight.to(batch_features.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        train_scores = torch.sigmoid(
            head(train_features.to(next(head.parameters()).device)).squeeze(-1)).cpu()
        valid_scores = torch.sigmoid(
            head(valid_features.to(next(head.parameters()).device)).squeeze(-1)).cpu()
    return {
        "train_bce": float(F.binary_cross_entropy(train_scores, train_labels).item()),
        "valid_bce": float(F.binary_cross_entropy(valid_scores, valid_labels).item()),
        "train_accuracy": float(train_scores.ge(0.5).eq(train_labels.bool()).float().mean()),
        "valid_accuracy": float(valid_scores.ge(0.5).eq(valid_labels.bool()).float().mean()),
        "train_positive_rate": float(train_labels.mean()),
        "valid_positive_rate": float(valid_labels.mean()),
        "train_predicted_wide_rate": float(train_scores.ge(0.5).float().mean()),
        "valid_predicted_wide_rate": float(valid_scores.ge(0.5).float().mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    output: dict[str, Any] = {
        "experiment": "native_learned_dynamic_width_head",
        "device": str(device),
        "lambda_target": float(args.lambda_target),
        "calibration_batches_per_condition": args.batches,
        "epochs": args.epochs,
        "checkpoints": {},
    }
    for checkpoint_name in args.checkpoints:
        checkpoint_path = Path(checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        features, labels, counts = _collect_features(
            checkpoint, device, args.batches, args.lambda_target, args.seed_offset)
        config = dict(checkpoint["config"])
        config["active_circuits"] = 16
        config["dynamic_width_mode"] = "learned"
        config["dynamic_width_min"] = args.dynamic_width_min
        config["dynamic_width_threshold"] = args.threshold
        model = make_model(config).to(device)
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        if missing != ["dynamic_width_head.weight", "dynamic_width_head.bias"] or unexpected:
            raise RuntimeError(f"unexpected learned-width checkpoint keys: {missing}, {unexpected}")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.dynamic_width_head.parameters():
            parameter.requires_grad_(True)
        training = _fit_head(model, features, labels, args.epochs,
                             args.learning_rate, int(config.get("seed", 17)))
        output_path = Path(args.output_dir) / (
            f"{checkpoint_path.stem}_learned_width.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "training": training,
            "feature_count": int(features.shape[0]),
            "label_positive_rate": float(labels.mean()),
            "condition_label_counts": counts,
            "source_checkpoint": str(checkpoint_path),
        }
        torch.save({"model_state": model.state_dict(), "config": config,
                    "report": report}, output_path)
        output["checkpoints"][str(output_path)] = report
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--lambda-target", type=float, default=0.05)
    parser.add_argument("--dynamic-width-min", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed-offset", type=int, default=7000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="results/checkpoints")
    parser.add_argument("--output", default="results/diagnostic_native_learned_width_training_20260910.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
