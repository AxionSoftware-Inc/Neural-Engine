from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.composition import CompositionalProgramGenerator
from train_dynamic_composition import make_model


@torch.no_grad()
def analyze(checkpoint_path: str, examples_per_task: int, device_name: str) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    heldout_pairs = tuple(tuple(pair) for pair in config.get("heldout_pairs", []))
    modulus_config = config.get("generator_modulus", config.get("modulus", 64))
    generator_modulus = None if modulus_config is None else int(modulus_config)
    generator = CompositionalProgramGenerator(
        seq_len=int(config["seq_len"]),
        seed=int(payload.get("report", {}).get("seed", config.get("seed", 17))) + 2,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split="heldout" if heldout_pairs else "all",
        heldout_pairs=heldout_pairs,
        combination_split=str(config.get("eval_combination_split", "all")),
        modulus=generator_modulus,
        target_offset=int(config.get("target_offset", 0)),
    )
    batch = generator.balanced_batch(examples_per_task, device)
    logits, stats = model(batch.inputs)
    correct = logits.argmax(dim=-1).eq(batch.targets)
    selected = stats["selected_ids"].detach().cpu()
    flat_selected = selected[selected.ge(0)]
    first, second = model.router._factor_ids(flat_selected.to(device))
    factor_ids = torch.cat((first, second)).cpu()
    factor_counts = torch.bincount(
        factor_ids, minlength=int(model.router.factor_count)
    ).float()
    virtual_counts = torch.bincount(
        flat_selected, minlength=int(model.router.num_circuits)
    ).float()

    task_names = {spec.task_id: spec.name for spec in generator.allowed_specs}
    per_task: dict[str, dict[str, Any]] = {}
    task_factor_sets: dict[str, set[int]] = {}
    for task_id, task_name in task_names.items():
        mask = batch.task_ids.detach().cpu().eq(task_id)
        task_selected = selected[mask]
        task_flat = task_selected[task_selected.ge(0)]
        task_first, task_second = model.router._factor_ids(task_flat.to(device))
        task_factors = torch.cat((task_first, task_second)).cpu().unique()
        task_virtual = task_flat.unique()
        task_factor_sets[task_name] = set(int(value) for value in task_factors.tolist())
        per_task[task_name] = {
            "accuracy": float(correct[mask.to(device)].float().mean().cpu()),
            "unique_virtual_circuits": int(task_virtual.numel()),
            "unique_factor_rows": int(task_factors.numel()),
            "new_factor_rows_after_154": int((task_factors >= 154).sum()),
        }

    task_factor_jaccard: dict[str, float] = {}
    task_items = list(task_factor_sets.items())
    for left_index in range(len(task_items)):
        left_name, left_set = task_items[left_index]
        for right_name, right_set in task_items[left_index + 1:]:
            union = left_set | right_set
            task_factor_jaccard[f"{left_name}__{right_name}"] = (
                len(left_set & right_set) / len(union) if union else 1.0
            )

    factor_prob = factor_counts / factor_counts.sum().clamp_min(1.0)
    nonzero = factor_prob[factor_prob > 0]
    factor_entropy = float(-(nonzero * nonzero.log()).sum())
    unique_virtual_per_example = [
        int(values[values.ge(0)].unique().numel()) for values in selected
    ]
    result = {
        "checkpoint": str(checkpoint_path),
        "model": config["model"],
        "device": str(device),
        "examples_per_task": examples_per_task,
        "heldout_accuracy": float(correct.float().mean().cpu()),
        "factor_count": int(model.router.factor_count),
        "virtual_circuit_capacity": int(model.router.num_circuits),
        "virtual_circuits_used": int((virtual_counts > 0).sum()),
        "factor_rows_used": int((factor_counts > 0).sum()),
        "new_factor_rows_used_after_154": int((factor_counts[154:] > 0).sum())
        if factor_counts.numel() > 154 else 0,
        "factor_entropy": factor_entropy,
        "factor_effective_rows": float(torch.exp(torch.tensor(factor_entropy))),
        "factor_max_load_fraction": float(factor_prob.max()),
        "mean_unique_virtual_per_example": (
            sum(unique_virtual_per_example) / max(len(unique_virtual_per_example), 1)
        ),
        "between_task_factor_jaccard": task_factor_jaccard,
        "per_task": per_task,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit factor and virtual-route utilization on composition checkpoints")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--examples-per-task", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = [analyze(path, args.examples_per_task, args.device) for path in args.checkpoint]
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
