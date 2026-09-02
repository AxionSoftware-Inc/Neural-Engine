from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.dynamic_composition import DynamicCompositionGenerator
from neural_engine.modular_templates import TrainableModularTemplateRegister


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, generator, examples_per_depth, device):
    model.eval()
    batch = generator.balanced_batch(examples_per_depth, device)
    logits, stats = model(batch.inputs)
    correct = logits.argmax(dim=-1).eq(batch.targets)
    return {
        "accuracy": float(correct.float().mean().cpu()),
        "loss": float(nn.functional.cross_entropy(logits, batch.targets).cpu()),
        "accuracy_by_depth": {
            str(depth): float(correct[batch.depths.eq(depth)].float().mean().cpu())
            for depth in generator.allowed_depths
        },
        "template_weights": stats["template_weights"].cpu().tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the modular template control")
    parser.add_argument("--max-ops", type=int, default=6)
    parser.add_argument("--train-max-ops", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--examples-per-depth", type=int, default=512)
    parser.add_argument("--train-value-min", type=int, default=0)
    parser.add_argument("--train-value-max", type=int, default=31)
    parser.add_argument("--eval-value-min", type=int, default=32)
    parser.add_argument("--eval-value-max", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="modular_template_control")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    model = TrainableModularTemplateRegister(max_ops=args.max_ops).to(device)
    train_generator = DynamicCompositionGenerator(
        max_ops=args.max_ops, train_max_ops=args.train_max_ops, seed=args.seed + 1,
        value_min=args.train_value_min, value_max=args.train_value_max, split="train",
    )
    eval_generator = DynamicCompositionGenerator(
        max_ops=args.max_ops, train_max_ops=args.train_max_ops, seed=args.seed + 2,
        value_min=args.eval_value_min, value_max=args.eval_value_max, split="heldout",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses = []
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        batch = train_generator.task_balanced_batch(args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(batch.inputs)
        loss = nn.functional.cross_entropy(logits, batch.targets)
        for stage in range(model.max_ops):
            mask = batch.stage_mask[:, stage]
            if mask.any():
                loss = loss + 0.5 * nn.functional.cross_entropy(
                    stats["step_logits"][mask, stage], batch.stage_targets[mask, stage]
                )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 1 or step % 500 == 0 or step == args.steps:
            print(f"step={step:05d} loss={losses[-1]:.6f}")

    result = {
        "run_id": args.run_id,
        "seed": args.seed,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "training_seconds": time.perf_counter() - started,
        "train": evaluate(model, train_generator, args.examples_per_depth, device),
        "evaluation": evaluate(model, eval_generator, args.examples_per_depth, device),
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        **model.parameter_report(),
    }
    print(json.dumps(result, indent=2))
    output = Path("results/runs") / f"{args.run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
