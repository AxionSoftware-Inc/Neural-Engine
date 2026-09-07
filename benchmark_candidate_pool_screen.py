"""Screen whether local candidate retrieval limits frozen-model quality.

The checkpoint, circuit bank, router weights, and evaluation batches stay
fixed.  Only the hierarchical router's candidate window is widened at
inference time.  This is deliberately a retrieval diagnostic, not a trained
model comparison: a gain means the existing window hides useful circuits;
flat or negative results move attention to selection, specialization, or the
recurrent output path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from neural_engine.router import HierarchicalRouter
from train import make_model


def _load_checkpoint(path: Path, device: torch.device) -> tuple[NeuralEngineV0, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    model = make_model(config)
    if not isinstance(model, NeuralEngineV0):
        raise ValueError("candidate pool screen requires a NeuralEngineV0 checkpoint")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    if not isinstance(model.router, HierarchicalRouter):
        raise ValueError("candidate pool screen currently targets HierarchicalRouter")
    return model, config


def _make_batches(config: dict, device: torch.device, batches: int,
                  examples_per_task: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config["seed"]) + 3,
        value_min=int(config.get("heldout_value_min", config.get("eval_value_min", 0))),
        value_max=int(config.get("heldout_value_max", config.get("eval_value_max", 63))),
        split=str(config.get("heldout_split", "all")),
    )
    return [
        (batch.inputs, batch.targets)
        for batch in (
            generator.balanced_batch(examples_per_task, device)
            for _ in range(batches)
        )
    ]


def _set_candidate_pool(model: NeuralEngineV0, candidate_pool: int) -> None:
    router = model.router
    if candidate_pool < model.active_circuits:
        raise ValueError("candidate pool must cover active circuits")
    if candidate_pool > router.routing_capacity:
        raise ValueError("candidate pool must fit the active routing capacity")
    if candidate_pool % router.num_addresses != 0:
        raise ValueError("candidate pool must divide across router addresses")
    router.candidate_pool = int(candidate_pool)
    router.candidates_per_address = candidate_pool // router.num_addresses


@torch.inference_mode()
def _evaluate(model: NeuralEngineV0, batches: list[tuple[torch.Tensor, torch.Tensor]],
              candidate_pool: int) -> dict:
    _set_candidate_pool(model, candidate_pool)
    loss_total = 0.0
    correct = 0
    examples = 0
    selected_sets: list[torch.Tensor] = []
    candidate_sets: list[torch.Tensor] = []
    for inputs, targets in batches:
        logits, stats = model(inputs, adaptive=False, collect_stats=True)
        loss_total += float(F.cross_entropy(logits, targets, reduction="sum").cpu())
        correct += int(logits.argmax(dim=-1).eq(targets).sum().cpu())
        examples += int(targets.numel())
        selected_sets.append(stats["selected_ids"].detach().cpu().reshape(-1))
        candidate_sets.append(stats["candidate_ids"].detach().cpu().reshape(-1))
    selected = torch.cat(selected_sets)
    candidates = torch.cat(candidate_sets)
    return {
        "candidate_pool": int(candidate_pool),
        "mean_ce": loss_total / examples,
        "accuracy": correct / examples,
        "unique_selected_circuits": int(selected.unique().numel()),
        "unique_candidate_circuits": int(candidates.unique().numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen hierarchical candidate-pool width")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--candidate-pools", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--examples-per-task", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    result = {
        "device": str(device),
        "batches": int(args.batches),
        "examples_per_task": int(args.examples_per_task),
        "candidate_pools": [int(value) for value in args.candidate_pools],
        "checkpoints": [],
    }
    for checkpoint_name in args.checkpoint:
        path = Path(checkpoint_name)
        model, config = _load_checkpoint(path, device)
        pools = [pool for pool in args.candidate_pools
                 if pool <= model.router.routing_capacity]
        batches = _make_batches(config, device, args.batches, args.examples_per_task)
        rows = []
        for pool in pools:
            rows.append(_evaluate(model, batches, pool))
        result["checkpoints"].append({
            "checkpoint": str(path),
            "model": config.get("model"),
            "seed": int(config.get("seed", -1)),
            "routing_capacity": int(model.router.routing_capacity),
            "active_circuits": int(model.active_circuits),
            "rows": rows,
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
