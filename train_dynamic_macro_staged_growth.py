from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from data.dynamic_composition import DynamicCompositionGenerator
from neural_engine.dynamic_register import DynamicRegisterNeuralEngine
from neural_engine.macro_growth import expand_macro_model
from neural_engine.instrumentation import count_parameters
from train_dynamic_composition import evaluate, make_model, make_optimizer, seed_everything
from train_dynamic_macro_growth import train_steps


def make_generator(
    config: dict[str, Any],
    *,
    seed: int,
    value_min: int,
    value_max: int,
    split: str,
) -> DynamicCompositionGenerator:
    return DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        seed=seed,
        value_min=value_min,
        value_max=value_max,
        split=split,
    )


def evaluate_stage(
    model: DynamicRegisterNeuralEngine,
    config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    examples_per_depth: int,
    train_value_min: int,
    train_value_max: int,
    eval_value_min: int,
    eval_value_max: int,
    heldout_depths: bool,
) -> dict[str, Any]:
    split_train = "train" if heldout_depths else "all"
    split_eval = "heldout" if heldout_depths else "all"
    train_generator = make_generator(
        config, seed=seed + 3, value_min=train_value_min,
        value_max=train_value_max, split=split_train,
    )
    eval_generator = make_generator(
        config, seed=seed + 2, value_min=eval_value_min,
        value_max=eval_value_max, split=split_eval,
    )
    return {
        "train": evaluate(model, train_generator, device, examples_per_depth),
        "evaluation": evaluate(model, eval_generator, device, examples_per_depth),
    }


def load_parent(
    model: DynamicRegisterNeuralEngine,
    checkpoint: str | None,
) -> None:
    if checkpoint is None:
        return
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.parent_config, "r", encoding="utf-8") as handle:
        parent_config = yaml.safe_load(handle)
    with open(args.intermediate_config, "r", encoding="utf-8") as handle:
        intermediate_config = yaml.safe_load(handle)
    with open(args.target_config, "r", encoding="utf-8") as handle:
        target_config = yaml.safe_load(handle)
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    parent = make_model(parent_config).to(device)
    load_parent(parent, args.parent_checkpoint)
    if args.parent_checkpoint is None:
        parent_generator = make_generator(
            parent_config, seed=args.seed + 1,
            value_min=args.train_value_min, value_max=args.train_value_max,
            split="train" if args.heldout_depths else "all",
        )
        train_steps(
            parent, make_optimizer(parent, parent_config), parent_generator,
            parent_config, steps=args.parent_steps, batch_size=args.batch_size,
            device=device, log_every=args.log_every, label="parent16",
        )

    train_generator = make_generator(
        parent_config, seed=args.seed + 1,
        value_min=args.train_value_min, value_max=args.train_value_max,
        split="train" if args.heldout_depths else "all",
    )
    stages: list[dict[str, Any]] = []

    intermediate = make_model(intermediate_config).to(device)
    expand_macro_model(parent, intermediate, preserve_parent_route=False)
    intermediate_training = train_steps(
        intermediate, make_optimizer(intermediate, intermediate_config),
        train_generator, intermediate_config, steps=args.intermediate_steps,
        batch_size=args.batch_size, device=device, log_every=args.log_every,
        label="grown64",
    )
    intermediate_eval = evaluate_stage(
        intermediate, intermediate_config, seed=args.seed, device=device,
        examples_per_depth=args.examples_per_depth,
        train_value_min=args.train_value_min, train_value_max=args.train_value_max,
        eval_value_min=args.eval_value_min, eval_value_max=args.eval_value_max,
        heldout_depths=args.heldout_depths,
    )
    stages.append({
        "name": "grown64",
        "params": count_parameters(intermediate),
        "training": intermediate_training,
        **intermediate_eval,
        "parameter_report": intermediate.parameter_report(),
    })

    grown = make_model(target_config).to(device)
    expand_macro_model(intermediate, grown, preserve_parent_route=False)
    grown_training = train_steps(
        grown, make_optimizer(grown, target_config), train_generator,
        target_config, steps=args.final_steps, batch_size=args.batch_size,
        device=device, log_every=args.log_every, label="grown256",
    )
    grown_eval = evaluate_stage(
        grown, target_config, seed=args.seed, device=device,
        examples_per_depth=args.examples_per_depth,
        train_value_min=args.train_value_min, train_value_max=args.train_value_max,
        eval_value_min=args.eval_value_min, eval_value_max=args.eval_value_max,
        heldout_depths=args.heldout_depths,
    )
    stages.append({
        "name": "grown256",
        "params": count_parameters(grown),
        "training": grown_training,
        **grown_eval,
        "parameter_report": grown.parameter_report(),
    })

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    intermediate_checkpoint = checkpoint_dir / f"{args.run_id}_grown64.pt"
    final_checkpoint = checkpoint_dir / f"{args.run_id}_grown256.pt"
    torch.save({"model_state": intermediate.state_dict(), "config": intermediate_config}, intermediate_checkpoint)
    torch.save({"model_state": grown.state_dict(), "config": target_config}, final_checkpoint)

    report = {
        "run_id": args.run_id,
        "device": str(device),
        "seed": args.seed,
        "parent_model": parent_config["model"],
        "intermediate_model": intermediate_config["model"],
        "grown_model": target_config["model"],
        "parent_checkpoint": args.parent_checkpoint,
        "stages": stages,
        "intermediate_checkpoint": str(intermediate_checkpoint),
        "final_checkpoint": str(final_checkpoint),
    }
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged Macro-Cell parent growth")
    parser.add_argument("--parent-config", default="configs/ne_dynamic_20m_macro_v0.yaml")
    parser.add_argument("--intermediate-config", default="configs/ne_dynamic_20m_macro_v0_64.yaml")
    parser.add_argument("--target-config", default="configs/ne_dynamic_20m_macro_v0_256.yaml")
    parser.add_argument("--parent-checkpoint", default=None)
    parser.add_argument("--parent-steps", type=int, default=3000)
    parser.add_argument("--intermediate-steps", type=int, default=3000)
    parser.add_argument("--final-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="ne_dynamic_20m_macro_staged_parent16_grow64_grow256")
    parser.add_argument("--output", default="results/runs")
    parser.add_argument("--checkpoint-dir", default="results/checkpoints")
    parser.add_argument("--examples-per-depth", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--heldout-depths", action="store_true")
    parser.add_argument("--train-value-min", type=int, default=0)
    parser.add_argument("--train-value-max", type=int, default=63)
    parser.add_argument("--eval-value-min", type=int, default=0)
    parser.add_argument("--eval-value-max", type=int, default=63)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
