from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.generator import PAD_TOKEN, TASK_TOKEN_OFFSET, VALUE_TOKEN_OFFSET, SyntheticTaskGenerator
from data.tasks import TASKS, TaskSpec
from neural_engine.model import NeuralEngineV0
from train import load_config, make_model, seed_everything


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def make_row(generator: SyntheticTaskGenerator, task: TaskSpec, values: list[int]):
    tokens = [TASK_TOKEN_OFFSET + task.task_id]
    tokens += [VALUE_TOKEN_OFFSET + value for value in values]
    tokens += [PAD_TOKEN] * (generator.seq_len - len(tokens))
    target = int(task.fn(values))
    stage_targets, stage_mask = generator._stage_targets(task, values, target)
    return tokens, target, task.task_id, task.depth, stage_targets, stage_mask


def build_counterfactual_rows(generator: SyntheticTaskGenerator, examples_per_task: int,
                              mode: str):
    base_rows = []
    variant_rows = []
    labels: list[tuple[str, int]] = []
    for task in TASKS:
        matching_tasks = [candidate for candidate in TASKS
                          if candidate.arity == task.arity and candidate.depth == task.depth
                          and candidate.task_id != task.task_id]
        if mode == "task_token" and not matching_tasks:
            continue
        for sample_index in range(examples_per_task):
            generated, _, _, _ = generator._one(task)
            values = [token - VALUE_TOKEN_OFFSET
                      for token in generated[1:1 + task.arity]]
            base_values = list(values)
            variant_values = list(values)
            if mode == "operand":
                changed_index = sample_index % task.arity
                domain_size = generator.value_max - generator.value_min + 1
                variant_values[changed_index] = generator.value_min + (
                    (variant_values[changed_index] - generator.value_min + 1) % domain_size)
                variant_task = task
            else:
                variant_task = matching_tasks[sample_index % len(matching_tasks)]
            base_rows.append(make_row(generator, task, base_values))
            variant_rows.append(make_row(generator, variant_task, variant_values))
            labels.append((task.name, task.depth))
    return base_rows, variant_rows, labels


def route_sets(model: NeuralEngineV0, batch) -> tuple[list[set[int]], torch.Tensor]:
    _, stats = model(batch.inputs, adaptive=model.adaptive_inference)
    selected = stats["selected_ids"].detach().cpu()
    sets = [set(int(value) for value in sample.flatten().tolist() if int(value) >= 0)
            for sample in selected]
    return sets, selected


def summarize_pairs(base_sets: list[set[int]], variant_sets: list[set[int]],
                    base_selected: torch.Tensor, variant_selected: torch.Tensor,
                    labels: list[tuple[str, int]]) -> dict[str, Any]:
    pair_jaccards = [jaccard(left, right) for left, right in zip(base_sets, variant_sets)]
    step_changed = []
    for base_sample, variant_sample in zip(base_selected, variant_selected):
        base_steps = [set(int(value) for value in step.tolist() if int(value) >= 0)
                      for step in base_sample]
        variant_steps = [set(int(value) for value in step.tolist() if int(value) >= 0)
                         for step in variant_sample]
        step_changed.append(sum(left != right for left, right in zip(base_steps, variant_steps))
                            / max(len(base_steps), 1))
    grouped: dict[str, list[float]] = {}
    for label, value in zip(labels, pair_jaccards):
        grouped.setdefault(label[0], []).append(value)
    return {
        "mean_route_jaccard": sum(pair_jaccards) / max(len(pair_jaccards), 1),
        "mean_route_change": 1.0 - sum(pair_jaccards) / max(len(pair_jaccards), 1),
        "fraction_with_any_step_change": sum(value > 0 for value in step_changed) / max(len(step_changed), 1),
        "mean_fraction_of_steps_changed": sum(step_changed) / max(len(step_changed), 1),
        "per_task": {
            task: {
                "mean_route_jaccard": sum(values) / len(values),
                "mean_route_change": 1.0 - sum(values) / len(values),
            }
            for task, values in grouped.items()
        },
    }


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = load_config(args.config, smoke=False) if args.config else dict(payload["config"])
    if config["model"] == "baseline":
        raise ValueError("counterfactual route analysis requires a Neural Engine checkpoint")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload.get("model_state", payload))
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=args.seed,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split="all",
    )
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model": config["model"],
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "counterfactuals": {},
    }
    for mode, title in (("operand", "single_operand"), ("task_token", "single_task_token")):
        base_rows, variant_rows, labels = build_counterfactual_rows(
            generator, args.examples_per_task, mode)
        base_batch = generator._make_batch(base_rows, device)
        variant_batch = generator._make_batch(variant_rows, device)
        base_sets, base_selected = route_sets(model, base_batch)
        variant_sets, variant_selected = route_sets(model, variant_batch)
        result["counterfactuals"][title] = summarize_pairs(
            base_sets, variant_sets, base_selected, variant_selected, labels)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure route sensitivity with controlled counterfactual inputs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1710)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rendered = json.dumps(analyze(args), indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
