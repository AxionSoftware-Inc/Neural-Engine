"""Measure whether DynamicRegister states linearly carry intermediate values.

This is an inference-only diagnostic.  A frozen checkpoint is run on a
calibration stream containing all depths, and scalar linear probes are fitted
from the captured pre-state, query, post-state, and output-readout state to
the true intermediate accumulator value.  The probes are then evaluated on
held-out depths.  Direct intermediate-head accuracy is reported beside them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.dynamic_composition import DynamicCompositionGenerator
from train_dynamic_composition import make_model


@torch.inference_mode()
def collect(
    model: torch.nn.Module,
    generator: DynamicCompositionGenerator,
    device: torch.device,
    batches: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    traces: dict[str, list[torch.Tensor]] = {
        "pre": [], "query": [], "post": [], "step": [],
        "targets": [], "mask": [], "stage_logits": [],
    }
    for _ in range(batches):
        batch = generator.balanced_batch(batch_size, device)
        _, stats = model(batch.inputs, collect_state_stats=True)
        traces["pre"].append(stats["pre_accumulator_states"].cpu())
        traces["query"].append(stats["query_states"].cpu())
        traces["post"].append(stats["post_accumulator_states"].cpu())
        traces["step"].append(stats["step_states"].cpu())
        traces["targets"].append(batch.stage_targets.cpu())
        traces["mask"].append(batch.stage_mask.cpu())
        traces["stage_logits"].append(stats["step_logits"].cpu())
    return {key: torch.cat(value) for key, value in traces.items()}


def fit_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> dict[str, float | int]:
    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    train_design = torch.cat((train_x, torch.ones(train_x.shape[0], 1)), dim=1)
    test_design = torch.cat((test_x, torch.ones(test_x.shape[0], 1)), dim=1)
    weights = torch.linalg.lstsq(train_design, train_y.unsqueeze(-1)).solution
    train_pred = (train_design @ weights).squeeze(-1)
    test_pred = (test_design @ weights).squeeze(-1)
    baseline = (train_y - train_y.mean()).square().mean().clamp_min(1e-8)
    test_mse = (test_pred - test_y).square().mean()
    centered_test = (test_y - test_y.mean()).square().mean().clamp_min(1e-8)
    correlation = torch.corrcoef(torch.stack((test_pred, test_y)))[0, 1]
    return {
        "train_examples": int(train_y.numel()),
        "heldout_examples": int(test_y.numel()),
        "train_rmse": float((train_pred - train_y).square().mean().sqrt()),
        "heldout_rmse": float(test_mse.sqrt()),
        "heldout_normalized_mse": float(test_mse / baseline),
        "heldout_r2": float(1.0 - test_mse / centered_test),
        "heldout_correlation": float(correlation),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    batches: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    model = make_model(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    target_offset = int(config.get("target_offset", 0))
    modulus = config.get("generator_modulus", config.get("modulus", 64))
    modulus = None if modulus is None else int(modulus)
    value_min = int(config.get("train_value_min", 0))
    value_max = int(config.get("train_value_max", 7))
    calibration = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        seed=seed,
        modulus=modulus,
        target_offset=target_offset,
        value_min=value_min,
        value_max=value_max,
        split="all",
    )
    heldout = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        seed=seed + 1,
        modulus=modulus,
        target_offset=target_offset,
        value_min=value_min,
        value_max=value_max,
        split="heldout",
    )
    train_data = collect(model, calibration, device, batches, batch_size)
    test_data = collect(model, heldout, device, batches, batch_size)
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "model": config.get("model"),
        "seed": int(payload.get("report", {}).get("seed", config.get("seed", -1))),
        "target_offset": target_offset,
        "value_range": [value_min, value_max],
        "calibration_depths": list(calibration.allowed_depths),
        "heldout_depths": list(heldout.allowed_depths),
        "stages": [],
    }
    for stage in range(int(config["max_ops"])):
        valid_train = train_data["mask"][:, stage]
        valid_test = test_data["mask"][:, stage]
        if not valid_train.any() or not valid_test.any():
            continue
        train_y = train_data["targets"][valid_train, stage].to(torch.float32) - target_offset
        test_y = test_data["targets"][valid_test, stage].to(torch.float32) - target_offset
        stage_result: dict[str, Any] = {
            "stage": stage,
            "heldout_examples": int(valid_test.sum()),
            "direct_stage_accuracy": float(
                test_data["stage_logits"][valid_test, stage].argmax(dim=-1)
                .eq(test_data["targets"][valid_test, stage]).float().mean()
            ),
            "probes": {},
        }
        for name in ("pre", "query", "post", "step"):
            stage_result["probes"][name] = fit_linear_probe(
                train_data[name][valid_train, stage], train_y,
                test_data[name][valid_test, stage], test_y,
            )
        result["stages"].append(stage_result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1700)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    result = {
        "device": str(device),
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "checkpoints": [
            evaluate_checkpoint(Path(path), device, args.batches, args.batch_size,
                                args.seed + index * 2)
            for index, path in enumerate(args.checkpoint)
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
