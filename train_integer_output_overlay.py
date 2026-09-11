from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from data.dynamic_composition import DynamicCompositionGenerator
from train_dynamic_composition import (
    evaluate,
    make_model,
    output_loss,
    seed_everything,
    validate_class_targets,
)


def run(args: argparse.Namespace) -> dict:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_seed = int(config["seed"]) if args.seed is None else int(args.seed)
    seed_everything(run_seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    model = make_model(config).to(device)
    init_checkpoint = torch.load(args.init_checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(
        init_checkpoint["model_state"], strict=False
    )
    allowed_prefixes = (
        "algebraic_integer_output_decoder.",
        "algebraic_integer_digit_embeddings.",
        "algebraic_integer_output_head.",
    )
    # Use state_dict keys rather than named_parameters(): FactorizedDigitOutput
    # exposes ``high``/``low`` aliases for its last digit head, and those alias
    # keys are present in a migrated state_dict even though named_parameters()
    # de-duplicates the underlying module.
    allowed_missing = {
        key for key in model.state_dict() if key.startswith(allowed_prefixes)
    }
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(
            f"unexpected checkpoint migration: missing={missing}, unexpected={unexpected}"
        )
    full_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules = [
        model.algebraic_integer_digit_embeddings,
        model.algebraic_integer_output_decoder,
        model.algebraic_integer_output_head,
    ]
    trainable = [
        parameter
        for module in trainable_modules
        for parameter in module.parameters()
    ]
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    modulus_config = config.get("generator_modulus", config.get("modulus", 64))
    generator_modulus = None if modulus_config is None else int(modulus_config)
    target_offset = int(config.get("target_offset", 0))
    train_generator = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        seed=run_seed + 1,
        modulus=generator_modulus,
        target_offset=target_offset,
        value_min=args.train_value_min,
        value_max=args.train_value_max,
        split="train" if args.heldout_depths else "all",
    )
    eval_generator = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        seed=run_seed + 2,
        modulus=generator_modulus,
        target_offset=target_offset,
        value_min=args.eval_value_min,
        value_max=args.eval_value_max,
        split="heldout" if args.heldout_depths else "all",
    )

    model.train()
    losses = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = train_generator.task_balanced_batch(args.batch_size, device)
        validate_class_targets(
            batch.targets, int(model.output[-1].out_features), "training targets"
        )
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(batch.inputs, return_full_logits=False)
        loss = output_loss(model, logits, stats, batch.targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, float(config["grad_clip"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == args.steps):
            print(f"step={step:05d} loss={losses[-1]:.5f}")

    elapsed = time.perf_counter() - started
    compact = bool(config.get("compact_factorized_eval", False))
    train_eval = evaluate(model, train_generator, device, args.examples_per_depth, compact)
    heldout_eval = evaluate(model, eval_generator, device, args.examples_per_depth, compact)
    report = {
        "run_id": args.run_id,
        "model_name": config["model"],
        "seed": run_seed,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "training_seconds": elapsed,
        "init_checkpoint": args.init_checkpoint,
        "train_depths": list(train_generator.allowed_depths),
        "eval_depths": list(eval_generator.allowed_depths),
        "train_value_range": [args.train_value_min, args.train_value_max],
        "eval_value_range": [args.eval_value_min, args.eval_value_max],
        "generator_modulus": generator_modulus,
        "target_offset": target_offset,
        "compact_factorized_eval": compact,
        "full_model_parameter_count": full_parameter_count,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train": train_eval,
        "evaluation": heldout_eval,
    }
    report.update(model.parameter_report())
    # ``parameter_report`` intentionally counts trainable parameters. Preserve
    # the full storage count separately because this overlay freezes the base
    # checkpoint by design.
    report["total_params"] = full_parameter_count
    report["frozen_parameter_count"] = full_parameter_count - report["trainable_parameter_count"]
    if args.checkpoint:
        report["checkpoint"] = str(Path(args.checkpoint))
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "config": config, "report": report},
            checkpoint_path,
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train only a multiply-specific integer output overlay"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default="results/runs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--examples-per-depth", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--heldout-depths", action="store_true")
    parser.add_argument("--train-value-min", type=int, default=0)
    parser.add_argument("--train-value-max", type=int, default=63)
    parser.add_argument("--eval-value-min", type=int, default=0)
    parser.add_argument("--eval-value-max", type=int, default=63)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
