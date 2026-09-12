from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from data.dynamic_composition import DynamicCompositionGenerator
from neural_engine.dynamic_register import DynamicRegisterNeuralEngine
from neural_engine.instrumentation import count_parameters
from neural_engine.optim import LazyAdamW


def validate_class_targets(
    targets: torch.Tensor, num_classes: int, label: str,
) -> None:
    minimum = int(targets.min().item())
    maximum = int(targets.max().item())
    if minimum < 0 or maximum >= num_classes:
        raise ValueError(
            f"{label} outside classifier range: min={minimum}, max={maximum}, "
            f"num_classes={num_classes}"
        )


def output_loss(
    model: DynamicRegisterNeuralEngine,
    logits: torch.Tensor,
    stats: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> torch.Tensor:
    """Use compact digit supervision when the output head is factorized."""
    if model.output_mode != "factorized_digits":
        return nn.functional.cross_entropy(logits, targets)
    return factorized_digit_loss(
        stats["digit_logits"], targets, model.output_digit_base
    )


def factorized_digit_targets(
    targets: torch.Tensor, digit_base: int, digit_count: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        (targets // (digit_base ** (digit_count - 1 - index))).remainder(digit_base)
        for index in range(digit_count)
    )


def factorized_digit_loss(
    digit_logits: tuple[torch.Tensor, ...],
    targets: torch.Tensor,
    digit_base: int,
) -> torch.Tensor:
    digit_targets = factorized_digit_targets(
        targets, digit_base, len(digit_logits)
    )
    return torch.stack(
        [
            nn.functional.cross_entropy(logits[:, -1], digit_target)
            for logits, digit_target in zip(digit_logits, digit_targets)
        ]
    ).sum()


def structured_scalar_contract_loss(
    model: DynamicRegisterNeuralEngine,
    stats: dict[str, torch.Tensor],
    stage_targets: torch.Tensor,
    stage_mask: torch.Tensor,
    target_offset: int,
    target_scale: float,
) -> torch.Tensor:
    """Supervise the learned scalar packet against raw intermediate values."""
    if not model.structured_scalar_state:
        return stage_targets.new_zeros((), dtype=torch.float32)
    values = stats["structured_scalar_states"]
    terms = []
    for stage in range(model.max_ops):
        mask = stage_mask[:, stage]
        if mask.any():
            predicted = values[mask, stage] / target_scale
            target = (
                stage_targets[mask, stage].to(predicted.dtype) - target_offset
            ) / target_scale
            terms.append(nn.functional.smooth_l1_loss(predicted, target))
    if not terms:
        return values.new_zeros(())
    return torch.stack(terms).mean()


def typed_digit_contract_loss(
    model: DynamicRegisterNeuralEngine,
    stats: dict[str, torch.Tensor],
    stage_targets: torch.Tensor,
    stage_mask: torch.Tensor,
) -> torch.Tensor:
    """Supervise the typed recurrent register on every executed stage digit."""
    if not model.typed_digit_state:
        return stage_targets.new_zeros((), dtype=torch.float32)
    digit_logits = stats["typed_digit_logits"]
    digit_targets = factorized_digit_targets(
        stage_targets, model.typed_digit_base, model.typed_digit_count
    )
    terms = []
    for stage in range(model.max_ops):
        mask = stage_mask[:, stage]
        if not mask.any():
            continue
        terms.extend(
            nn.functional.cross_entropy(
                logits[mask, stage], target[mask, stage]
            )
            for logits, target in zip(digit_logits, digit_targets)
        )
    if not terms:
        return stage_targets.new_zeros((), dtype=torch.float32)
    return torch.stack(terms).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: dict[str, Any]) -> DynamicRegisterNeuralEngine:
    fields = (
        "vocab_size", "num_classes", "modulus", "max_ops", "seq_len", "d_model", "state_dim",
        "num_circuits", "circuit_rank", "router_branch", "router_depth",
        "candidate_pool", "active_circuits", "circuit_bank_mode", "factor_count",
        "factor_candidate_pool", "factor_capacity", "ordered_factor_slots", "circuit_mode",
        "query_factor_mix_scale", "factor_pair_rank", "factor_pair_scale",
        "route_exploration_prob",
        "input_reinjection_scale", "write_gate", "value_encoder_mode",
        "value_encoder_modulus",
        "factor_mix_mode", "route_context_mode", "modular_prior",
        "state_layout",
        "state_update_mode", "state_residual_scale",
        "predecessor_operation_context",
        "operation_adapter_rank", "operation_adapter_scale",
        "operation_adapter_gate",
        "operation_read_adapter_rank", "operation_read_adapter_scale",
        "operation_write_adapter_rank", "operation_write_adapter_scale",
        "operation_write_adapter_mode",
        "operation_output_adapter_rank", "operation_output_adapter_scale",
        "operation_circuit_bank",
        "operation_router_keys",
        "operation_transition_rank", "operation_transition_scale",
        "operation_bilinear_transition_rank",
        "operation_bilinear_transition_scale",
        "structured_scalar_state", "structured_scalar_scale",
        "structured_scalar_read_scale",
        "structured_scalar_authoritative",
        "algebraic_state_mode", "algebraic_state_scale",
        "algebraic_output_bridge_scale",
        "algebraic_state_write_scale",
        "algebraic_state_authoritative_read",
        "algebraic_output_decoder",
        "algebraic_integer_output_decoder", "algebraic_integer_digit_dim",
        "algebraic_integer_output_decoder_mode",
        "algebraic_integer_output_head",
        "algebraic_integer_output_factor_rank",
        "algebraic_integer_output_digit_interaction_rank",
        "algebraic_integer_state_read_scale",
        "algebraic_state_value_scale", "algebraic_state_fourier_base",
        "operator_valued_product_encoder", "operator_valued_packet_width",
        "operator_valued_basis_count",
        "numeric_state_dim", "numeric_state_scale", "numeric_state_value_scale",
        "typed_digit_state", "typed_digit_dim", "typed_digit_base",
        "typed_digit_count", "typed_digit_scale", "typed_digit_value_offset",
        "typed_digit_operand_offset",
        "typed_digit_carry_chain",
        "modular_prior_mode",
        "modular_template_init",
        "circuit_residual_scale",
        "circuit_input_norm",
        "output_mode", "output_temperature", "output_scalar_bias", "output_digit_base",
        "output_factor_rank", "output_digit_count",
        "output_digit_interaction_rank",
        "output_digit_context_mode",
        "output_digit_geometry", "output_digit_temperature",
        "macro_cell_count", "macro_cell_rank", "macro_cell_depth",
        "macro_router_branch", "macro_router_depth", "macro_candidate_pool",
        "active_macro_cells", "macro_cell_scale",
    )
    return DynamicRegisterNeuralEngine(**{
        key: config[key] for key in fields if key in config
    })


def make_optimizer(model: DynamicRegisterNeuralEngine, config: dict[str, Any]):
    if str(config.get("optimizer", "adamw")).lower() == "lazy_adamw":
        lazy_parameters = [
            parameter for name, parameter in model.named_parameters()
            if name.startswith("circuits.") or name == "router.keys"
        ]
        return LazyAdamW(
            model.parameters(), lr=config["learning_rate"],
            weight_decay=config["weight_decay"], lazy_parameters=lazy_parameters,
        )
    return torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )


@torch.no_grad()
def route_audit(
    model: DynamicRegisterNeuralEngine,
    selected_ids: torch.Tensor,
    depths: torch.Tensor,
) -> dict[str, Any]:
    """Summarize hard route traffic without changing the forward path."""
    num_circuits = int(model.router.num_circuits)

    def summarize(ids: torch.Tensor) -> dict[str, Any]:
        valid = ids.ge(0)
        flat = ids[valid].to(dtype=torch.long)
        if not flat.numel():
            return {
                "selected_count": 0,
                "unique_virtual_circuits": 0,
                "virtual_bank_utilization": 0.0,
                "dead_virtual_circuits": num_circuits,
            }
        counts = torch.bincount(flat, minlength=num_circuits).float()
        active_counts = counts[counts.gt(0)]
        probabilities = active_counts / active_counts.sum()
        result: dict[str, Any] = {
            "selected_count": int(flat.numel()),
            "unique_virtual_circuits": int(active_counts.numel()),
            "virtual_bank_utilization": float(active_counts.numel() / num_circuits),
            "dead_virtual_circuits": int(num_circuits - active_counts.numel()),
            "top_route_fraction": float(active_counts.max() / active_counts.sum()),
            "route_entropy": float((-(probabilities * probabilities.log()).sum()).cpu()),
        }
        if model.circuit_bank_mode == "factorized":
            first, second = model.router._factor_ids(flat)
            factors = torch.cat([first, second])
            factor_count = int(model.router.factor_count)
            factor_usage = torch.bincount(factors, minlength=factor_count).gt(0)
            result.update({
                "unique_factor_rows": int(factor_usage.sum()),
                "factor_bank_utilization": float(factor_usage.float().mean()),
                "dead_factor_rows": int(factor_count - factor_usage.sum()),
            })
        return result

    audit = summarize(selected_ids)
    audit["by_program_depth"] = {
        str(depth): summarize(selected_ids[depths.eq(depth)])
        for depth in sorted(int(value) for value in torch.unique(depths).cpu().tolist())
    }
    audit["by_execution_step"] = {
        str(step): summarize(selected_ids[:, step])
        for step in range(selected_ids.shape[1])
    }
    return audit


@torch.no_grad()
def macro_route_audit(
    model: DynamicRegisterNeuralEngine,
    selected_ids: torch.Tensor,
    depths: torch.Tensor,
) -> dict[str, Any]:
    """Summarize macro-cell traffic independently from the micro bank."""
    num_cells = int(model.macro_cell_count)

    def summarize(ids: torch.Tensor) -> dict[str, Any]:
        valid = ids.ge(0)
        flat = ids[valid].to(dtype=torch.long)
        if not flat.numel():
            return {
                "selected_count": 0,
                "unique_macro_cells": 0,
                "macro_bank_utilization": 0.0,
                "dead_macro_cells": num_cells,
            }
        counts = torch.bincount(flat, minlength=num_cells).float()
        active_counts = counts[counts.gt(0)]
        probabilities = active_counts / active_counts.sum()
        return {
            "selected_count": int(flat.numel()),
            "unique_macro_cells": int(active_counts.numel()),
            "macro_bank_utilization": float(active_counts.numel() / num_cells),
            "dead_macro_cells": int(num_cells - active_counts.numel()),
            "top_macro_fraction": float(active_counts.max() / active_counts.sum()),
            "macro_route_entropy": float(
                (-(probabilities * probabilities.log()).sum()).cpu()
            ),
        }

    audit = summarize(selected_ids)
    audit["by_program_depth"] = {
        str(depth): summarize(selected_ids[depths.eq(depth)])
        for depth in sorted(int(value) for value in torch.unique(depths).cpu().tolist())
    }
    audit["by_execution_step"] = {
        str(step): summarize(selected_ids[:, step])
        for step in range(selected_ids.shape[1])
    }
    return audit


@torch.no_grad()
def evaluate(
    model: DynamicRegisterNeuralEngine,
    generator: DynamicCompositionGenerator,
    device: torch.device,
    examples_per_depth: int,
    compact_factorized: bool = False,
) -> dict[str, Any]:
    model.eval()
    batch = generator.balanced_batch(examples_per_depth, device)
    use_compact = compact_factorized and model.output_mode == "factorized_digits"
    logits, stats = model(batch.inputs, return_full_logits=not use_compact)
    validate_class_targets(
        batch.targets, int(model.output[-1].out_features), "evaluation targets"
    )
    if use_compact:
        digit_base = model.output_digit_base
        digit_logits = stats["digit_logits"]
        predictions = torch.zeros_like(batch.targets)
        for index, logits in enumerate(digit_logits):
            power = digit_base ** (len(digit_logits) - 1 - index)
            predictions = predictions + logits[:, -1].argmax(dim=-1) * power
        loss = factorized_digit_loss(digit_logits, batch.targets, digit_base)
        loss_mode = "factorized_digit_sum_compact"
    else:
        predictions = logits.argmax(dim=-1)
        loss = nn.functional.cross_entropy(logits, batch.targets)
        loss_mode = "reconstructed_class_logits"
    correct = predictions.eq(batch.targets)
    per_depth = {}
    for depth in generator.allowed_depths:
        mask = batch.depths.eq(depth)
        per_depth[str(depth)] = float(correct[mask].float().mean().cpu())
    executed = stats["executed_steps"].float()
    return {
        "accuracy": float(correct.float().mean().cpu()),
        "loss": float(loss.cpu()),
        "loss_mode": loss_mode,
        "accuracy_by_depth": per_depth,
        "avg_executed_steps": float(executed.mean().cpu()),
        "active_step_fraction": float((executed / model.max_ops).mean().cpu()),
        "router_entropy": float(stats["router_entropy"].cpu()),
        "route_audit": route_audit(model, stats["selected_ids"], batch.depths),
        "macro_route_audit": (
            macro_route_audit(model, stats["macro_selected_ids"], batch.depths)
            if model.macro_cell_count
            else None
        ),
    }


def set_lazy_active_rows(
    model: DynamicRegisterNeuralEngine,
    optimizer: Any,
    selected_ids: torch.Tensor,
) -> None:
    if not hasattr(optimizer, "set_active_rows"):
        return
    selected = selected_ids.detach().reshape(-1)
    selected = selected[selected.ge(0)].unique()
    if not selected.numel():
        return
    if model.circuit_bank_mode == "factorized":
        first, second = model.router._factor_ids(selected)
        factor_rows = torch.cat([first, second]).unique()
        optimizer.set_active_rows({
            model.router.keys: factor_rows,
            model.circuits.down_factors: factor_rows,
            model.circuits.up_factors: factor_rows,
            model.circuits.bias_factors: factor_rows,
            model.circuits.factor_mix: selected,
        })
    else:
        optimizer.set_active_rows({
            model.router.keys: selected,
            model.circuits.down: selected,
            model.circuits.up: selected,
            model.circuits.bias: selected,
        })


def run(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_seed = int(config["seed"]) if args.seed is None else int(args.seed)
    seed_everything(run_seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = make_model(config).to(device)
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
    value_curriculum = config.get("value_curriculum")
    normalized_curriculum = None
    if value_curriculum is not None:
        if not isinstance(value_curriculum, list) or not value_curriculum:
            raise ValueError("value_curriculum must be a non-empty list")
        normalized_curriculum = []
        previous_until = 0
        for stage in value_curriculum:
            if not isinstance(stage, dict):
                raise ValueError("each value_curriculum stage must be a mapping")
            until_step = int(stage["until_step"])
            value_min = int(stage.get("value_min", args.train_value_min))
            value_max = int(stage.get("value_max", args.train_value_max))
            if until_step <= previous_until:
                raise ValueError(
                    "value_curriculum until_step values must be strictly increasing"
                )
            if value_min > value_max:
                raise ValueError("value_curriculum value range must be ordered")
            normalized_curriculum.append({
                "until_step": until_step,
                "value_min": value_min,
                "value_max": value_max,
            })
            previous_until = until_step
        if normalized_curriculum[-1]["until_step"] < args.steps:
            raise ValueError(
                "value_curriculum must cover all requested training steps"
            )
        first_stage = normalized_curriculum[0]
        train_generator = DynamicCompositionGenerator(
            max_ops=int(config["max_ops"]),
            train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
            seed=run_seed + 1001,
            modulus=generator_modulus,
            target_offset=target_offset,
            value_min=first_stage["value_min"],
            value_max=first_stage["value_max"],
            split="train" if args.heldout_depths else "all",
        )
    curriculum_stage_index = 0
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
    optimizer = make_optimizer(model, config)
    steps = args.steps
    teacher_forcing_start = float(
        config.get("typed_digit_teacher_forcing_start", 0.0)
    )
    teacher_forcing_end = float(
        config.get("typed_digit_teacher_forcing_end", 0.0)
    )
    teacher_forcing_steps = int(
        config.get("typed_digit_teacher_forcing_steps", steps)
    )
    if not 0.0 <= teacher_forcing_start <= 1.0:
        raise ValueError("typed_digit_teacher_forcing_start must be in [0, 1]")
    if not 0.0 <= teacher_forcing_end <= 1.0:
        raise ValueError("typed_digit_teacher_forcing_end must be in [0, 1]")
    if teacher_forcing_steps < 1:
        raise ValueError("typed_digit_teacher_forcing_steps must be positive")
    if (teacher_forcing_start or teacher_forcing_end) and not model.typed_digit_state:
        raise ValueError(
            "typed digit teacher forcing requires typed_digit_state"
        )
    model.train()
    losses = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        if normalized_curriculum is not None:
            next_stage_index = next(
                index
                for index, stage in enumerate(normalized_curriculum)
                if step <= stage["until_step"]
            )
            if next_stage_index != curriculum_stage_index:
                stage = normalized_curriculum[next_stage_index]
                train_generator = DynamicCompositionGenerator(
                    max_ops=int(config["max_ops"]),
                    train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
                    seed=run_seed + 1001 + next_stage_index,
                    modulus=generator_modulus,
                    target_offset=target_offset,
                    value_min=stage["value_min"],
                    value_max=stage["value_max"],
                    split="train" if args.heldout_depths else "all",
                )
                curriculum_stage_index = next_stage_index
        batch_size = args.batch_size or int(config["batch_size"])
        batch = train_generator.task_balanced_batch(batch_size, device)
        num_classes = int(model.output[-1].out_features)
        validate_class_targets(batch.targets, num_classes, "training targets")
        validate_class_targets(batch.stage_targets, num_classes, "stage targets")
        teacher_forcing_fraction = min(
            1.0,
            max(0.0, (step - 1) / max(teacher_forcing_steps - 1, 1)),
        )
        teacher_forcing_probability = (
            teacher_forcing_start
            + teacher_forcing_fraction
            * (teacher_forcing_end - teacher_forcing_start)
        )
        teacher_targets = None
        if teacher_forcing_probability > 0.0:
            teacher_targets = batch.stage_targets - target_offset
        optimizer.zero_grad(set_to_none=True)
        logits, stats = model(
            batch.inputs,
            return_full_logits=(model.output_mode != "factorized_digits"),
            teacher_stage_targets=teacher_targets,
            teacher_forcing_probability=teacher_forcing_probability,
        )
        loss = output_loss(model, logits, stats, batch.targets)
        stage_weight = float(config.get("stage_loss_weight", 0.0))
        if stage_weight and batch.stage_targets is not None and batch.stage_mask is not None:
            stage_losses = []
            for stage in range(model.max_ops):
                mask = batch.stage_mask[:, stage]
                if mask.any():
                    stage_targets = batch.stage_targets[mask, stage]
                    if model.output_mode == "factorized_digits":
                        stage_digit_logits = tuple(
                            logits[mask, stage]
                            for logits in stats["digit_logits"]
                        )
                        stage_digit_targets = factorized_digit_targets(
                            stage_targets,
                            model.output_digit_base,
                            len(stage_digit_logits),
                        )
                        stage_losses.append(
                            torch.stack(
                                [
                                    nn.functional.cross_entropy(logits, target)
                                    for logits, target in zip(
                                        stage_digit_logits, stage_digit_targets
                                    )
                                ]
                            ).sum()
                        )
                    else:
                        stage_losses.append(nn.functional.cross_entropy(
                            stats["step_logits"][mask, stage], stage_targets
                        ))
            if stage_losses:
                loss = loss + stage_weight * torch.stack(stage_losses).mean()
        contract_weight = float(config.get("structured_scalar_contract_loss_weight", 0.0))
        if (
            contract_weight
            and batch.stage_targets is not None
            and batch.stage_mask is not None
        ):
            contract_scale = float(config.get("structured_scalar_target_scale", 1.0))
            if contract_scale <= 0.0:
                raise ValueError("structured_scalar_target_scale must be positive")
            loss = loss + contract_weight * structured_scalar_contract_loss(
                model,
                stats,
                batch.stage_targets,
                batch.stage_mask,
                target_offset,
                contract_scale,
            )
        typed_contract_weight = float(
            config.get("typed_digit_contract_loss_weight", 0.0)
        )
        if (
            typed_contract_weight
            and batch.stage_targets is not None
            and batch.stage_mask is not None
        ):
            loss = loss + typed_contract_weight * typed_digit_contract_loss(
                model,
                stats,
                batch.stage_targets,
                batch.stage_mask,
            )
        loss = loss - 0.0001 * stats["router_entropy"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        set_lazy_active_rows(model, optimizer, stats["selected_ids"])
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if args.log_every and (step == 1 or step % args.log_every == 0 or step == steps):
            print(f"step={step:05d} loss={losses[-1]:.5f}")
    elapsed = time.perf_counter() - start
    compact_factorized_eval = bool(config.get("compact_factorized_eval", False))
    train_eval = evaluate(
        model, train_generator, device, args.examples_per_depth,
        compact_factorized=compact_factorized_eval,
    )
    eval_eval = evaluate(
        model, eval_generator, device, args.examples_per_depth,
        compact_factorized=compact_factorized_eval,
    )
    report = {
        "run_id": args.run_id,
        "model_name": config["model"],
        "seed": run_seed,
        "device": str(device),
        "steps": steps,
        "batch_size": args.batch_size or config["batch_size"],
        "training_seconds": elapsed,
        "train_depths": list(train_generator.allowed_depths),
        "eval_depths": list(eval_generator.allowed_depths),
        "train_value_range": [args.train_value_min, args.train_value_max],
        "eval_value_range": [args.eval_value_min, args.eval_value_max],
        "train_value_curriculum": normalized_curriculum,
        "typed_digit_teacher_forcing": {
            "start": teacher_forcing_start,
            "end": teacher_forcing_end,
            "steps": teacher_forcing_steps,
        },
        "generator_modulus": generator_modulus,
        "target_offset": target_offset,
        "compact_factorized_eval": compact_factorized_eval,
        "total_params": count_parameters(model),
        "train": train_eval,
        "evaluation": eval_eval,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "structured_scalar_contract_loss_weight": float(
            config.get("structured_scalar_contract_loss_weight", 0.0)
        ),
        "structured_scalar_target_scale": float(
            config.get("structured_scalar_target_scale", 1.0)
        ),
        "typed_digit_contract_loss_weight": float(
            config.get("typed_digit_contract_loss_weight", 0.0)
        ),
    }
    report.update(model.parameter_report())
    output_path = Path(args.output) / f"{args.run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "config": config, "report": report}, checkpoint_path)
        report["checkpoint"] = str(checkpoint_path)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the attention-free dynamic register machine")
    parser.add_argument("--config", default="configs/ne_dynamic_20m.yaml")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default="ne_dynamic_20m_smoke")
    parser.add_argument("--output", default="results/runs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--examples-per-depth", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--heldout-depths", action="store_true")
    parser.add_argument("--train-value-min", type=int, default=0)
    parser.add_argument("--train-value-max", type=int, default=63)
    parser.add_argument("--eval-value-min", type=int, default=0)
    parser.add_argument("--eval-value-max", type=int, default=63)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
