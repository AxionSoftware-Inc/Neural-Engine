"""Probe whether Native recurrent queries carry prior composition results.

This is an inference-only diagnostic.  For a depth-2/3 example, the query at
internal step ``s`` is the state after the preceding circuit execution.  A
small linear probe is trained on train-split queries to predict the preceding
stage target and evaluated on held-out operand combinations.  If the probe can
recover the value while the model's direct stage head cannot, the bottleneck is
the readout/circuit path; if neither can, the state transition is not carrying
the intermediate computation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from train import make_model


def _load_checkpoint(path: Path, device: torch.device) -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("state information diagnostic requires NeuralEngineV0")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, config


@torch.inference_mode()
def _collect(model: NeuralEngineV0, config: dict, device: torch.device,
             batches: int, examples_per_task: int, split: str) -> dict[str, torch.Tensor]:
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config.get("seed", 17)) + (41 if split == "train" else 42),
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=split,
    )
    queries = []
    stage_logits = []
    stage_targets = []
    stage_masks = []
    for _ in range(batches):
        batch = generator.balanced_batch(examples_per_task, device)
        _, stats = model(batch.inputs, adaptive=False, collect_stats=True)
        queries.append(stats["query_states"].cpu())
        stage_logits.append(stats["step_logits"].cpu())
        stage_targets.append(batch.stage_targets.cpu())
        stage_masks.append(batch.stage_mask.cpu())
    return {
        "queries": torch.cat(queries),
        "stage_logits": torch.cat(stage_logits),
        "stage_targets": torch.cat(stage_targets),
        "stage_masks": torch.cat(stage_masks),
    }


def _fit_probe(train_x: torch.Tensor, train_y: torch.Tensor,
               test_x: torch.Tensor, test_y: torch.Tensor,
               epochs: int, lr: float, nonlinear: bool = False) -> dict[str, float | int]:
    mean = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, keepdim=True).clamp_min(1e-4)
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    if nonlinear:
        probe = torch.nn.Sequential(
            torch.nn.Linear(train_x.shape[-1], 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 64),
        )
    else:
        probe = torch.nn.Linear(train_x.shape[-1], 64)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        loss = F.cross_entropy(probe(train_x), train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        predictions = probe(test_x).argmax(dim=-1)
        accuracy = predictions.eq(test_y).float().mean()
        train_accuracy = probe(train_x).argmax(dim=-1).eq(train_y).float().mean()
    return {
        "train_examples": int(train_y.numel()),
        "heldout_examples": int(test_y.numel()),
        "probe_train_accuracy": float(train_accuracy),
        "probe_heldout_accuracy": float(accuracy),
        "chance_accuracy": 1.0 / 64.0,
        "epochs": int(epochs),
        "nonlinear": bool(nonlinear),
    }


def _evaluate(model: NeuralEngineV0, train_data: dict[str, torch.Tensor],
              heldout_data: dict[str, torch.Tensor], epochs: int, lr: float) -> dict:
    result = {
        "carry_probes": [],
        "nonlinear_carry_probes": [],
        "direct_stage_accuracy": [],
    }
    max_steps = min(model.internal_steps, train_data["queries"].shape[1])
    for query_step in range(1, max_steps):
        target_stage = query_step - 1
        train_valid = train_data["stage_masks"][:, target_stage]
        heldout_valid = heldout_data["stage_masks"][:, target_stage]
        probe = _fit_probe(
            train_data["queries"][train_valid, query_step],
            train_data["stage_targets"][train_valid, target_stage],
            heldout_data["queries"][heldout_valid, query_step],
            heldout_data["stage_targets"][heldout_valid, target_stage],
            epochs, lr,
        )
        probe["query_step"] = query_step
        probe["target_stage"] = target_stage
        result["carry_probes"].append(probe)
        nonlinear_probe = _fit_probe(
            train_data["queries"][train_valid, query_step],
            train_data["stage_targets"][train_valid, target_stage],
            heldout_data["queries"][heldout_valid, query_step],
            heldout_data["stage_targets"][heldout_valid, target_stage],
            epochs, lr, nonlinear=True,
        )
        nonlinear_probe["query_step"] = query_step
        nonlinear_probe["target_stage"] = target_stage
        result["nonlinear_carry_probes"].append(nonlinear_probe)

        logits = heldout_data["stage_logits"][:, target_stage]
        result["direct_stage_accuracy"].append({
            "stage": target_stage,
            "examples": int(heldout_valid.sum()),
            "accuracy": float(logits[heldout_valid].argmax(dim=-1).eq(
                heldout_data["stage_targets"][heldout_valid, target_stage]
            ).float().mean()),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Native recurrent state information")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    result = {
        "device": str(device),
        "batches": int(args.batches),
        "examples_per_task": int(args.examples_per_task),
        "probe_epochs": int(args.epochs),
        "checkpoints": [],
    }
    for checkpoint_name in args.checkpoint:
        path = Path(checkpoint_name)
        model, config = _load_checkpoint(path, device)
        train_data = _collect(model, config, device, args.batches, args.examples_per_task, "train")
        heldout_data = _collect(model, config, device, args.batches, args.examples_per_task, "heldout")
        result["checkpoints"].append({
            "checkpoint": str(path),
            "model": config.get("model"),
            "seed": int(config.get("seed", -1)),
            "internal_steps": int(model.internal_steps),
            "evaluation": _evaluate(model, train_data, heldout_data, args.epochs, args.lr),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
