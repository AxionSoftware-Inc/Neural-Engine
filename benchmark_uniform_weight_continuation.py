"""Test whether uniform selected-route weights improve Native training.

The treatment changes only the hard-route mixture weights to 1/K.  Circuit
selection, candidate pool, model body, training batches, and continuation
length match the natural-weight control.  Evaluation reports the treatment
under both uniform and natural weights to separate training benefit from
inference-weight compatibility.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from benchmark_nonlinear_route_scorer import _evaluate, _loss, _train_pair
from benchmark_route_cost_surrogate import _load_checkpoint, _make_batches
from neural_engine.model import NeuralEngineV0
from train import seed_everything
import torch


def _run(path: Path, args: argparse.Namespace, device: torch.device) -> dict:
    base, config = _load_checkpoint(path, device)
    if not isinstance(base, NeuralEngineV0):
        raise ValueError("uniform-weight continuation requires NeuralEngineV0")
    control = copy.deepcopy(base)
    treatment = copy.deepcopy(base)
    treatment.router.set_route_weight_mode("uniform")
    eval_batches = _make_batches(
        config, device, "heldout", args.eval_batches, args.eval_examples_per_task,
    )
    training = _train_pair(
        control, treatment, config, device, args.steps,
        soft_temperature=0.0, seed=int(config.get("seed", 17)),
    )
    control_eval = _evaluate(control, eval_batches)
    treatment_uniform_eval = _evaluate(treatment, eval_batches)
    treatment.router.set_route_weight_mode("natural")
    treatment_natural_eval = _evaluate(treatment, eval_batches)
    result = {
        "checkpoint": str(path),
        "model": config.get("model"),
        "seed": int(config.get("seed", -1)),
        "device": str(device),
        "active_circuits": int(treatment.active_circuits),
        "training": training,
        "control_natural": control_eval,
        "treatment_uniform": treatment_uniform_eval,
        "treatment_natural": treatment_natural_eval,
        "delta": {
            "uniform_mean_ce": treatment_uniform_eval["mean_ce"] - control_eval["mean_ce"],
            "uniform_accuracy_pp": (treatment_uniform_eval["accuracy"] - control_eval["accuracy"]) * 100,
            "natural_mean_ce": treatment_natural_eval["mean_ce"] - control_eval["mean_ce"],
            "natural_accuracy_pp": (treatment_natural_eval["accuracy"] - control_eval["accuracy"]) * 100,
        },
    }
    del base, control, treatment
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit uniform Native route-weight continuation")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--eval-examples-per-task", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seed_everything(17)
    device = torch.device(args.device)
    result = {
        "device": str(device),
        "steps": int(args.steps),
        "eval_batches": int(args.eval_batches),
        "eval_examples_per_task": int(args.eval_examples_per_task),
        "checkpoints": [],
    }
    for checkpoint_name in args.checkpoint:
        result["checkpoints"].append(_run(Path(checkpoint_name), args, device))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
