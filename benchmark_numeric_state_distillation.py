"""Distill an exact numeric teacher into the prior-free recurrent state.

The peak integer-overlay checkpoints contain an exact arithmetic packet.  This
screen uses their compact digit logits as a frozen teacher while the student
has the same body and output codec but no integer packet.  The teacher is used
only during training; student evaluation runs through its ordinary learned
state path.  This is deliberately an opt-in research benchmark, not a change
to the default model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from data.dynamic_composition import DynamicCompositionGenerator
from train_dynamic_composition import (
    evaluate,
    factorized_digit_targets,
    make_model,
    output_loss,
    seed_everything,
    set_lazy_active_rows,
    validate_class_targets,
)


def stage_loss(
    model: nn.Module,
    stats: dict[str, torch.Tensor],
    stage_targets: torch.Tensor,
    stage_mask: torch.Tensor,
) -> torch.Tensor:
    """Match the normal stage supervision used by train_dynamic_composition."""
    terms = []
    for stage in range(model.max_ops):
        mask = stage_mask[:, stage]
        if not mask.any():
            continue
        targets = stage_targets[mask, stage]
        if model.output_mode == "factorized_digits":
            digits = factorized_digit_targets(
                targets, model.output_digit_base, model.output_digit_count
            )
            terms.append(torch.stack([
                nn.functional.cross_entropy(logits[mask, stage], target)
                for logits, target in zip(stats["digit_logits"], digits)
            ]).sum())
        else:
            terms.append(nn.functional.cross_entropy(
                stats["step_logits"][mask, stage], targets
            ))
    if not terms:
        return stage_targets.new_zeros((), dtype=torch.float32)
    return torch.stack(terms).mean()


def digit_distillation_loss(
    student_stats: dict[str, torch.Tensor],
    teacher_stats: dict[str, torch.Tensor],
    stage_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Distill teacher digit distributions on every executed prefix stage."""
    if temperature <= 0.0:
        raise ValueError("teacher temperature must be positive")
    terms = []
    scale = temperature * temperature
    for student_logits, teacher_logits in zip(
        student_stats["digit_logits"], teacher_stats["digit_logits"]
    ):
        student_log_probs = nn.functional.log_softmax(
            student_logits / temperature, dim=-1
        )
        teacher_probs = nn.functional.softmax(
            teacher_logits / temperature, dim=-1
        )
        per_token = nn.functional.kl_div(
            student_log_probs, teacher_probs, reduction="none"
        ).sum(dim=-1) * scale
        mask = stage_mask.to(per_token.dtype)
        terms.append((per_token * mask).sum() / mask.sum().clamp_min(1.0))
    if not terms:
        return stage_mask.new_zeros((), dtype=torch.float32)
    return torch.stack(terms).mean()


def _load_model(path: Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    model = make_model(config).to(device)
    model.load_state_dict(payload["model_state"])
    return model, config, payload


def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    student, student_config, student_payload = _load_model(
        Path(args.student_checkpoint), device
    )
    teacher, teacher_config, _ = _load_model(
        Path(args.teacher_checkpoint), device
    )
    if student.output_mode != "factorized_digits":
        raise ValueError("student distillation requires factorized digit output")
    if teacher.output_mode != "factorized_digits":
        raise ValueError("teacher distillation requires factorized digit output")
    if (
        student.output_digit_base != teacher.output_digit_base
        or student.output_digit_count != teacher.output_digit_count
    ):
        raise ValueError("student and teacher digit codecs do not match")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.train()
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(student_config["learning_rate"]),
        weight_decay=float(student_config["weight_decay"]),
    )
    modulus_config = student_config.get(
        "generator_modulus", student_config.get("modulus", 64)
    )
    modulus = None if modulus_config is None else int(modulus_config)
    target_offset = int(student_config.get("target_offset", 0))
    train_generator = DynamicCompositionGenerator(
        max_ops=int(student_config["max_ops"]),
        train_max_ops=int(student_config.get("train_max_ops", student_config["max_ops"])),
        seed=args.seed + 1,
        modulus=modulus,
        target_offset=target_offset,
        value_min=args.train_value_min,
        value_max=args.train_value_max,
        split="train" if args.heldout_depths else "all",
    )
    eval_generator = DynamicCompositionGenerator(
        max_ops=int(student_config["max_ops"]),
        train_max_ops=int(student_config.get("train_max_ops", student_config["max_ops"])),
        seed=args.seed + 2,
        modulus=modulus,
        target_offset=target_offset,
        value_min=args.eval_value_min,
        value_max=args.eval_value_max,
        split="heldout" if args.heldout_depths else "all",
    )
    distill_weight = float(args.distill_weight)
    if distill_weight < 0.0:
        raise ValueError("distill weight must be non-negative")
    stage_weight = float(student_config.get("stage_loss_weight", 0.0))
    losses = []
    distill_losses = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = train_generator.task_balanced_batch(args.batch_size, device)
        validate_class_targets(
            batch.targets, int(student.output[-1].out_features), "training targets"
        )
        optimizer.zero_grad(set_to_none=True)
        student_logits, student_stats = student(
            batch.inputs, return_full_logits=False
        )
        with torch.no_grad():
            _, teacher_stats = teacher(batch.inputs, return_full_logits=False)
        loss = output_loss(student, student_logits, student_stats, batch.targets)
        if stage_weight and batch.stage_targets is not None and batch.stage_mask is not None:
            loss = loss + stage_weight * stage_loss(
                student, student_stats, batch.stage_targets, batch.stage_mask
            )
        distill = digit_distillation_loss(
            student_stats,
            teacher_stats,
            batch.stage_mask,
            args.teacher_temperature,
        )
        distill_losses.append(float(distill.detach().cpu()))
        loss = loss + distill_weight * distill
        loss = loss - 0.0001 * student_stats["router_entropy"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), float(student_config["grad_clip"]))
        set_lazy_active_rows(student, optimizer, student_stats["selected_ids"])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == args.steps):
            print(
                f"step={step:05d} loss={losses[-1]:.5f} "
                f"distill={distill_losses[-1]:.5f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    compact = bool(student_config.get("compact_factorized_eval", False))
    train_eval = evaluate(
        student, train_generator, device, args.examples_per_depth, compact
    )
    heldout_eval = evaluate(
        student, eval_generator, device, args.examples_per_depth, compact
    )
    report = {
        "run_id": args.run_id,
        "model_name": student_config["model"],
        "seed": args.seed,
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "training_seconds": elapsed,
        "student_checkpoint": args.student_checkpoint,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_model_name": teacher_config["model"],
        "student_has_integer_decoder": bool(
            getattr(student, "algebraic_integer_output_decoder_enabled", False)
        ),
        "teacher_has_integer_decoder": bool(
            getattr(teacher, "algebraic_integer_output_decoder_enabled", False)
        ),
        "teacher_temperature": args.teacher_temperature,
        "distill_weight": distill_weight,
        "train_depths": list(train_generator.allowed_depths),
        "eval_depths": list(eval_generator.allowed_depths),
        "train_value_range": [args.train_value_min, args.train_value_max],
        "eval_value_range": [args.eval_value_min, args.eval_value_max],
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "distill_loss_mean": sum(distill_losses) / len(distill_losses),
        "train": train_eval,
        "evaluation": heldout_eval,
    }
    report["checkpoint"] = args.checkpoint
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": student.state_dict(),
                "config": student_config,
                "report": report,
            },
            checkpoint_path,
        )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill an exact integer teacher into a prior-free Neural Engine"
    )
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default="results/runs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--examples-per-depth", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--heldout-depths", action="store_true")
    parser.add_argument("--train-value-min", type=int, default=0)
    parser.add_argument("--train-value-max", type=int, default=95)
    parser.add_argument("--eval-value-min", type=int, default=0)
    parser.add_argument("--eval-value-max", type=int, default=95)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--teacher-temperature", type=float, default=2.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
