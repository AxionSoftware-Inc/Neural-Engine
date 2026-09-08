"""Measure intermediate-step supervision signals on Native checkpoints.

The evaluator does not train or alter a model.  It compares each recurrent
step's emitted class logits with the generator's deterministic intermediate
targets, grouped by task.  This locates composition failure inside the state
trajectory rather than treating the final accuracy as one undifferentiated
router score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.generator import SyntheticTaskGenerator
from data.tasks import TASK_BY_ID
from neural_engine.model import NeuralEngineV0
from train import make_model


def _load_checkpoint(path: Path, device: torch.device) -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("composition stage diagnostic requires NeuralEngineV0")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, config


def _make_batches(config: dict, device: torch.device, count: int,
                  examples_per_task: int, split: str) -> list:
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config.get("seed", 17)) + 3,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=split,
    )
    return [generator.balanced_batch(examples_per_task, device) for _ in range(count)]


@torch.inference_mode()
def _evaluate(model: NeuralEngineV0, batches: list) -> dict:
    stage_totals: dict[tuple[int, int], list[int]] = {}
    task_totals: dict[int, list[int]] = {}
    for batch in batches:
        logits, stats = model(batch.inputs, adaptive=False, collect_stats=True)
        final_predictions = logits.argmax(dim=-1)
        for task_id in torch.unique(batch.task_ids).tolist():
            mask = batch.task_ids.eq(task_id)
            slot = task_totals.setdefault(int(task_id), [0, 0])
            slot[0] += int(mask.sum().cpu())
            slot[1] += int(final_predictions[mask].eq(batch.targets[mask]).sum().cpu())
        step_predictions = stats["step_logits"].argmax(dim=-1)
        for stage in range(step_predictions.shape[1]):
            valid = batch.stage_mask[:, stage]
            for task_id in torch.unique(batch.task_ids[valid]).tolist():
                mask = valid & batch.task_ids.eq(task_id)
                slot = stage_totals.setdefault((int(task_id), stage), [0, 0])
                slot[0] += int(mask.sum().cpu())
                slot[1] += int(step_predictions[mask, stage].eq(
                    batch.stage_targets[mask, stage],
                ).sum().cpu())

    task_rows = []
    for task_id in sorted(task_totals):
        name = TASK_BY_ID[task_id].name
        stages = {}
        for (candidate_id, stage), values in sorted(stage_totals.items()):
            if candidate_id == task_id:
                stages[str(stage)] = {
                    "examples": values[0],
                    "accuracy": values[1] / values[0],
                }
        task_rows.append({
            "task_id": task_id,
            "task": name,
            "depth": int(TASK_BY_ID[task_id].depth),
            "final_accuracy": task_totals[task_id][1] / task_totals[task_id][0],
            "stage_accuracy": stages,
        })
    stage_rows = []
    for stage in range(model.internal_steps):
        values = [row for (task_id, candidate_stage), row in stage_totals.items()
                  if candidate_stage == stage]
        examples = sum(row[0] for row in values)
        correct = sum(row[1] for row in values)
        stage_rows.append({
            "stage": stage,
            "examples": examples,
            "accuracy": correct / examples if examples else None,
        })
    return {"task_rows": task_rows, "stage_rows": stage_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Native intermediate composition stages")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--split", choices=("all", "train", "heldout"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    result = {
        "device": str(device),
        "split": args.split,
        "batches": int(args.batches),
        "examples_per_task": int(args.examples_per_task),
        "checkpoints": [],
    }
    for checkpoint_name in args.checkpoint:
        path = Path(checkpoint_name)
        model, config = _load_checkpoint(path, device)
        batches = _make_batches(
            config, device, args.batches, args.examples_per_task, args.split,
        )
        result["checkpoints"].append({
            "checkpoint": str(path),
            "model": config.get("model"),
            "seed": int(config.get("seed", -1)),
            "internal_steps": int(model.internal_steps),
            "evaluation": _evaluate(model, batches),
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
