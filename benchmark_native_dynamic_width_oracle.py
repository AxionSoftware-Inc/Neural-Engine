"""Oracle screen for native per-example active width.

The same K=16 checkpoint is evaluated as fixed K=8 and fixed K=16. For each
example, an oracle chooses the final output with the lower cross-entropy after
adding a width penalty. This is an upper-bound diagnostic: it computes both
widths and therefore is not a deployable router.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.generator import accuracy_by_depth, accuracy_by_task
from train import BatchSource, make_model, seed_everything
from data.generator import SyntheticTaskGenerator


HARD_TASKS = {
    "reverse_sum",
    "lookup",
    "chain3",
    "compose_add_mul",
    "compose_if",
    "state_machine",
}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _per_example_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, targets, reduction="none")


def _evaluate_condition(model8: torch.nn.Module, model16: torch.nn.Module,
                        source: BatchSource, batches: int,
                        lambdas: list[float]) -> dict[str, Any]:
    logits8, logits16, targets, task_ids, depths = [], [], [], [], []
    for _ in range(batches):
        batch = source.balanced(32)
        with torch.no_grad():
            logits8.append(model8(batch.inputs)[0])
            logits16.append(model16(batch.inputs)[0])
        targets.append(batch.targets)
        task_ids.append(batch.task_ids)
        depths.append(batch.depths)
    logits8 = torch.cat(logits8)
    logits16 = torch.cat(logits16)
    targets = torch.cat(targets)
    task_ids = torch.cat(task_ids)
    depths = torch.cat(depths)
    ce8 = _per_example_ce(logits8, targets)
    ce16 = _per_example_ce(logits16, targets)
    pred8 = logits8.argmax(dim=-1)
    pred16 = logits16.argmax(dim=-1)

    def summarize(prediction: torch.Tensor, loss: torch.Tensor) -> dict[str, Any]:
        task_accuracy = accuracy_by_task(prediction.cpu(), type("BatchView", (), {
            "targets": targets.cpu(), "task_ids": task_ids.cpu(),
        })())
        depth_accuracy = accuracy_by_depth(prediction.cpu(), type("BatchView", (), {
            "targets": targets.cpu(), "depths": depths.cpu(),
        })())
        hard = [float(task_accuracy[name]) for name in HARD_TASKS if name in task_accuracy]
        return {
            "val_loss": float(loss.mean().cpu()),
            "exact_accuracy": float(prediction.eq(targets).float().mean().cpu()),
            "task_accuracy": task_accuracy,
            "depth_accuracy": depth_accuracy,
            "hard_task_mean_accuracy": sum(hard) / len(hard) if hard else None,
        }

    result: dict[str, Any] = {
        "fixed_k8": summarize(pred8, ce8),
        "fixed_k16": summarize(pred16, ce16),
        "variants": {},
    }
    for penalty in lambdas:
        # Width costs are fractions of the K=16 active circuit budget.
        cost8 = ce8 + float(penalty) * 0.5
        cost16 = ce16 + float(penalty)
        use16 = cost16 < cost8
        chosen_loss = torch.where(use16, ce16, ce8)
        chosen_logits = torch.where(use16.unsqueeze(-1), logits16, logits8)
        chosen_prediction = chosen_logits.argmax(dim=-1)
        task_accuracy = accuracy_by_task(chosen_prediction.cpu(), type("BatchView", (), {
            "targets": targets.cpu(), "task_ids": task_ids.cpu(),
        })())
        depth_accuracy = accuracy_by_depth(chosen_prediction.cpu(), type("BatchView", (), {
            "targets": targets.cpu(), "depths": depths.cpu(),
        })())
        hard = [float(task_accuracy[name]) for name in HARD_TASKS if name in task_accuracy]
        result["variants"][str(penalty)] = {
            "lambda": float(penalty),
            "active_width_fraction": float(torch.where(use16,
                                                        torch.ones_like(ce16),
                                                        torch.full_like(ce16, 0.5)).mean().cpu()),
            "wide_fraction": float(use16.float().mean().cpu()),
            "val_loss": float(chosen_loss.mean().cpu()),
            "exact_accuracy": float(chosen_prediction.eq(targets).float().mean().cpu()),
            "task_accuracy": task_accuracy,
            "depth_accuracy": depth_accuracy,
            "hard_task_mean_accuracy": sum(hard) / len(hard) if hard else None,
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    output: dict[str, Any] = {
        "experiment": "native_dynamic_width_final_loss_oracle",
        "device": str(device),
        "batches": args.batches,
        "lambdas": [float(value) for value in args.lambdas],
        "conditions": {},
        "checkpoints": {},
        "oracle_note": "both K=8 and K=16 are computed; route is not deployable",
    }
    conditions = {
        "uniform_all": {"value_min": 0, "value_max": 63, "split": "all"},
        "combination_heldout": {"value_min": 0, "value_max": 63, "split": "heldout"},
        "low_edge_values": {"value_min": 0, "value_max": 7, "split": "all"},
        "high_edge_values": {"value_min": 56, "value_max": 63, "split": "all"},
    }
    output["conditions"] = conditions
    for checkpoint_name in args.checkpoints:
        checkpoint_path = Path(checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config16 = dict(checkpoint["config"])
        config8 = dict(config16)
        config8["active_circuits"] = 8
        config8["dynamic_width_mode"] = "none"
        config16["active_circuits"] = 16
        config16["dynamic_width_mode"] = "none"
        model8 = make_model(config8).to(device)
        model16 = make_model(config16).to(device)
        model8.load_state_dict(checkpoint["model_state"])
        model16.load_state_dict(checkpoint["model_state"])
        model8.eval()
        model16.eval()
        seed = int(config16.get("seed", 17))
        checkpoint_result: dict[str, Any] = {
            "model_name": config16["model"],
            "seed": seed,
            "total_params": sum(parameter.numel() for parameter in model16.parameters()),
            "conditions": {},
        }
        for offset, (name, condition) in enumerate(conditions.items()):
            seed_everything(seed + 1000 + offset)
            generator = SyntheticTaskGenerator(
                seq_len=int(config16["seq_len"]), seed=seed + 3000 + offset,
                **condition,
            )
            checkpoint_result["conditions"][name] = _evaluate_condition(
                model8, model16, BatchSource(generator, 256, device),
                args.batches, [float(value) for value in args.lambdas],
            )
        output["checkpoints"][checkpoint_path.name] = checkpoint_result
        del model8, model16
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
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.01, 0.03, 0.05, 0.1, 0.2])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/diagnostic_native_dynamic_width_oracle_20260910.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
