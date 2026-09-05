"""Minimal taklif22 latent computational machine diagnostic.

The task deliberately keeps mutable facts outside the computation cells. The
model receives an entity query, an external fact table, and a variable-length
program. It must retrieve the fact, route each program token to a reusable
latent cell, update bounded working memory, and decode the final scalar.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


NUM_OPERATIONS = 4
PAD_OPERATION = NUM_OPERATIONS


def apply_operations(value: torch.Tensor, program: torch.Tensor, *, swapped: bool = False) -> torch.Tensor:
    """Apply the hidden synthetic rules; the model only sees operation tokens."""
    result = value
    for step in range(program.shape[1]):
        op = program[:, step]
        if swapped:
            # Operation-swap intervention: alter one computation rule while
            # leaving the external fact table unchanged.
            add = result - 0.15
        else:
            add = result + 0.10
        result = torch.where(op == 0, add, result)
        result = torch.where(op == 1, result * 0.80, result)
        result = torch.where(op == 2, 1.0 - result, result)
        result = torch.where(op == 3, result * result, result)
    return result


def operation_history(value: torch.Tensor, program: torch.Tensor, *, swapped: bool = False) -> torch.Tensor:
    result = value
    history = []
    for step in range(program.shape[1]):
        op = program[:, step]
        add = result - 0.15 if swapped else result + 0.10
        result = torch.where(op == 0, add, result)
        result = torch.where(op == 1, result * 0.80, result)
        result = torch.where(op == 2, 1.0 - result, result)
        result = torch.where(op == 3, result * result, result)
        history.append(result)
    return torch.stack(history, dim=1)


def sample_batch(
    batch_size: int,
    num_entities: int,
    max_steps: int,
    device: torch.device,
    *,
    min_steps: int,
    facts: torch.Tensor | None = None,
    swapped: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    entity_ids = torch.randint(num_entities, (batch_size,), device=device)
    if facts is None:
        fact_table = torch.empty(num_entities, device=device).uniform_(-0.5, 0.5)
    else:
        fact_table = facts.to(device=device)
    values = fact_table[entity_ids]
    lengths = torch.randint(min_steps, max_steps + 1, (batch_size,), device=device)
    positions = torch.arange(max_steps, device=device).unsqueeze(0)
    program = torch.randint(NUM_OPERATIONS, (batch_size, max_steps), device=device)
    program = torch.where(positions < lengths.unsqueeze(1), program, PAD_OPERATION)
    target = apply_operations(values, program, swapped=swapped)
    return entity_ids, program, fact_table, target


def sample_program_cases(
    num_batches: int,
    batch_size: int,
    num_entities: int,
    max_steps: int,
    device: torch.device,
    *,
    min_steps: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create fixed entity/program cases for paired interventions."""
    cases = []
    for _ in range(num_batches):
        entity_ids = torch.randint(num_entities, (batch_size,), device=device)
        lengths = torch.randint(min_steps, max_steps + 1, (batch_size,), device=device)
        positions = torch.arange(max_steps, device=device).unsqueeze(0)
        program = torch.randint(NUM_OPERATIONS, (batch_size, max_steps), device=device)
        program = torch.where(positions < lengths.unsqueeze(1), program, PAD_OPERATION)
        cases.append((entity_ids, program))
    return cases


class ComputationCell(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.transform(state)


class ScalarComputationCell(nn.Module):
    """A bounded scalar-register transform for the structured state ablation."""

    def __init__(self) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.transform(value)


class DenseOperationCell(nn.Module):
    """Compact dense control that receives the operation token directly."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(1 + latent_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, value: torch.Tensor, operation: torch.Tensor) -> torch.Tensor:
        return self.transform(torch.cat([value, operation], dim=-1))


class DenseRecurrentControl(nn.Module):
    """Matched compact recurrent control without sparse routing."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        memory_slots: int,
        state_update_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.memory_slots = memory_slots
        self.num_cells = 1
        self.structured_value_lane = True
        self.state_update_scale = state_update_scale
        self.value_encoder = nn.Sequential(nn.Linear(1, state_dim), nn.Tanh())
        self.operation_embedding = nn.Embedding(NUM_OPERATIONS + 1, latent_dim)
        self.cells = nn.ModuleList([DenseOperationCell(latent_dim)])
        self.write_projections = nn.ModuleList([
            nn.Linear(state_dim, state_dim) for _ in range(memory_slots)
        ])

    def forward(
        self,
        program: torch.Tensor,
        fact_values: torch.Tensor,
        entity_ids: torch.Tensor,
        *,
        hard_route: bool | None = None,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del hard_route
        value_lane = fact_values[entity_ids].unsqueeze(-1)
        state = self.value_encoder(value_lane)
        working = torch.zeros(
            program.shape[0], self.memory_slots, self.state_dim,
            device=program.device, dtype=state.dtype,
        )
        working[:, 0] = state
        histories = []
        routes = []
        scores = []
        for step in range(program.shape[1]):
            active = program[:, step] != PAD_OPERATION
            op_latent = self.operation_embedding(program[:, step])
            delta = self.cells[0](value_lane, op_latent).squeeze(-1)
            delta = delta * active
            value_lane = value_lane + self.state_update_scale * delta.unsqueeze(-1)
            state = self.value_encoder(value_lane)
            written = self.write_projections[step % self.memory_slots](state)
            working[:, step % self.memory_slots] = torch.where(
                active.unsqueeze(-1), written, working[:, step % self.memory_slots]
            )
            histories.append(value_lane.squeeze(-1))
            routes.append(torch.where(active, torch.zeros_like(active, dtype=torch.long), torch.full_like(active, -1, dtype=torch.long)))
            scores.append(torch.zeros(program.shape[0], 1, device=program.device, dtype=state.dtype))
        trace = {
            "route_ids": torch.stack(routes, dim=1),
            "route_scores": torch.stack(scores, dim=1),
            "final_state": state,
            "working_memory": working,
            "value_history": torch.stack(histories, dim=1),
        }
        return value_lane.squeeze(-1), trace if return_trace else {}


class LatentComputationalMachine(nn.Module):
    """Small state machine with separable external memory and computation."""

    def __init__(
        self,
        state_dim: int = 64,
        latent_dim: int = 32,
        memory_slots: int = 8,
        num_cells: int = NUM_OPERATIONS,
        route_temperature: float = 1.0,
        state_update_scale: float = 1.0,
        stabilize_state: bool = False,
        structured_value_lane: bool = False,
        value_clip: float = 0.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.memory_slots = memory_slots
        self.num_cells = num_cells
        self.route_temperature = route_temperature
        self.state_update_scale = state_update_scale
        self.stabilize_state = stabilize_state
        self.structured_value_lane = structured_value_lane
        self.value_clip = value_clip
        self.value_encoder = nn.Sequential(
            nn.Linear(1, state_dim),
            nn.Tanh(),
        )
        self.operation_embedding = nn.Embedding(NUM_OPERATIONS + 1, latent_dim)
        self.controller = nn.Sequential(
            nn.Linear(state_dim * 2 + latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        self.cell_keys = nn.Parameter(torch.randn(num_cells, latent_dim) * 0.05)
        cell_class = ScalarComputationCell if structured_value_lane else ComputationCell
        self.cells = nn.ModuleList([cell_class() if structured_value_lane else cell_class(state_dim) for _ in range(num_cells)])
        self.state_norm = nn.LayerNorm(state_dim)
        self.write_projections = nn.ModuleList([
            nn.Linear(state_dim, state_dim) for _ in range(memory_slots)
        ])
        self.decoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 1),
        )

    def forward(
        self,
        program: torch.Tensor,
        fact_values: torch.Tensor,
        entity_ids: torch.Tensor,
        *,
        hard_route: bool | None = None,
        return_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hard_route is None:
            hard_route = not self.training
        values = fact_values[entity_ids].unsqueeze(-1)
        value_lane = values
        state = self.value_encoder(value_lane)
        working = torch.zeros(
            program.shape[0], self.memory_slots, self.state_dim,
            device=program.device, dtype=state.dtype,
        )
        working[:, 0] = state
        route_ids = []
        route_scores = []
        value_history = []
        for step in range(program.shape[1]):
            active = program[:, step] != PAD_OPERATION
            op_latent = self.operation_embedding(program[:, step])
            pooled = working.mean(dim=1)
            instruction = self.controller(torch.cat([state, pooled, op_latent], dim=-1))
            scores = instruction @ self.cell_keys.t() / math.sqrt(self.latent_dim)
            scores = scores / self.route_temperature
            cell_input = value_lane if self.structured_value_lane else state
            outputs = torch.stack([cell(cell_input) for cell in self.cells], dim=1)
            if hard_route:
                selected = scores.argmax(dim=-1)
                weights = F.one_hot(selected, self.num_cells).to(outputs.dtype)
            else:
                selected = scores.argmax(dim=-1)
                weights = F.softmax(scores, dim=-1)
            delta = (outputs * weights.unsqueeze(-1)).sum(dim=1)
            delta = delta * active.unsqueeze(-1)
            if self.structured_value_lane:
                value_lane = value_lane + self.state_update_scale * delta
                if self.value_clip > 0:
                    value_lane = value_lane.clamp(-self.value_clip, self.value_clip)
                state = self.value_encoder(value_lane)
            else:
                state = state + self.state_update_scale * delta
                if self.stabilize_state:
                    state = self.state_norm(state)
            if self.structured_value_lane:
                value_history.append(value_lane.squeeze(-1))
            write_slot = step % self.memory_slots
            written = self.write_projections[write_slot](state)
            working[:, write_slot] = torch.where(
                active.unsqueeze(-1), written, working[:, write_slot]
            )
            route_ids.append(torch.where(active, selected, torch.full_like(selected, -1)))
            route_scores.append(scores)
        prediction = value_lane.squeeze(-1) if self.structured_value_lane else self.decoder(state).squeeze(-1)
        trace = {
            "route_ids": torch.stack(route_ids, dim=1),
            "route_scores": torch.stack(route_scores, dim=1),
            "final_state": state,
            "working_memory": working,
        }
        if self.structured_value_lane:
            trace["value_history"] = torch.stack(value_history, dim=1)
        if return_trace:
            return prediction, trace
        return prediction, {}


def train_machine(
    model: LatentComputationalMachine,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    batch_size: int,
    num_entities: int,
    train_max_steps: int,
    role_loss_weight: float,
    state_supervision_weight: float,
    log_every: int,
) -> list[dict[str, float]]:
    history = []
    model.train()
    for step in range(1, steps + 1):
        entity_ids, program, facts, target = sample_batch(
            batch_size, num_entities, train_max_steps, device, min_steps=2
        )
        prediction, trace = model(
            program,
            facts,
            entity_ids,
            return_trace=role_loss_weight > 0 or state_supervision_weight > 0,
        )
        task_loss = F.mse_loss(prediction, target)
        route_loss = prediction.new_zeros(())
        state_loss = prediction.new_zeros(())
        if role_loss_weight > 0:
            active = program != PAD_OPERATION
            role_targets = program.remainder(model.num_cells)
            route_loss = F.cross_entropy(
                trace["route_scores"].reshape(-1, model.num_cells),
                role_targets.reshape(-1),
                reduction="none",
            ).reshape_as(program)[active].mean()
        if state_supervision_weight > 0:
            if not model.structured_value_lane:
                raise ValueError("state supervision requires --structured-value-lane")
            active = program != PAD_OPERATION
            expected_history = operation_history(facts[entity_ids], program)
            state_loss = F.mse_loss(
                trace["value_history"][active], expected_history[active]
            )
        loss = (
            task_loss
            + role_loss_weight * route_loss
            + state_supervision_weight * state_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            history.append({
                "step": step,
                "task_loss": float(task_loss.detach().cpu()),
                "route_loss": float(route_loss.detach().cpu()),
                "state_loss": float(state_loss.detach().cpu()),
            })
    return history


def adapt_operation_swap(
    model: LatentComputationalMachine,
    device: torch.device,
    steps: int,
    batch_size: int,
    num_entities: int,
    max_steps: int,
    learning_rate: float,
    log_every: int,
) -> list[dict[str, float]]:
    """Adapt only cell 0 to a changed rule while external memory stays fixed."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.cells[0].parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.cells[0].parameters(), lr=learning_rate)
    history = []
    model.train()
    for step in range(1, steps + 1):
        entity_ids, program, facts, target = sample_batch(
            batch_size, num_entities, max_steps, device,
            min_steps=2, swapped=True,
        )
        prediction, _ = model(program, facts, entity_ids)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.cells[0].parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.eval()
    return history


def evaluate(
    model: LatentComputationalMachine,
    device: torch.device,
    *,
    facts: torch.Tensor,
    batch_size: int,
    num_entities: int,
    min_steps: int,
    max_steps: int,
    batches: int,
    swapped: bool = False,
    zero_memory: bool = False,
    cases: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> dict[str, object]:
    model.eval()
    losses = []
    absolute_errors = []
    route_trace = []
    with torch.no_grad():
        for batch_index in range(batches):
            if cases is None:
                entity_ids, program, fact_table, target = sample_batch(
                    batch_size, num_entities, max_steps, device,
                    min_steps=min_steps, facts=facts, swapped=swapped,
                )
            else:
                entity_ids, program = cases[batch_index]
                fact_table = facts
                target = apply_operations(
                    fact_table[entity_ids], program, swapped=swapped
                )
            if zero_memory:
                fact_table = torch.zeros_like(fact_table)
            prediction, trace = model(
                program, fact_table, entity_ids, hard_route=True, return_trace=True
            )
            losses.append(F.mse_loss(prediction, target).item())
            absolute_errors.append((prediction - target).abs().mean().item())
            route_trace.append(trace["route_ids"].cpu())
    routes = torch.cat(route_trace, dim=0)
    active = routes >= 0
    usage = torch.bincount(routes[active], minlength=model.num_cells).float()
    usage = usage / usage.sum().clamp_min(1)
    return {
        "mse": float(sum(losses) / len(losses)),
        "mae": float(sum(absolute_errors) / len(absolute_errors)),
        "route_usage": usage.tolist(),
        "route_ids": routes,
    }


def route_reuse(routes: torch.Tensor, programs: torch.Tensor) -> dict[str, float]:
    same = []
    different = []
    for first in range(len(programs)):
        for second in range(first + 1, len(programs)):
            mask = (programs[first] != PAD_OPERATION) & (programs[second] != PAD_OPERATION)
            if not bool(mask.any()):
                continue
            first_set = set(routes[first][mask].tolist())
            second_set = set(routes[second][mask].tolist())
            score = len(first_set & second_set) / max(len(first_set | second_set), 1)
            if torch.equal(programs[first], programs[second]):
                same.append(score)
            else:
                different.append(score)
    return {
        "same_program_jaccard": float(sum(same) / max(len(same), 1)),
        "different_program_jaccard": float(sum(different) / max(len(different), 1)),
        "same_pairs": float(len(same)),
        "different_pairs": float(len(different)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device("cuda" if args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    ) else "cpu")
    torch.manual_seed(args.seed)
    if args.model_kind == "dense-control":
        model = DenseRecurrentControl(
            state_dim=args.state_dim,
            latent_dim=args.latent_dim,
            memory_slots=args.memory_slots,
            state_update_scale=args.state_update_scale,
        ).to(device)
    else:
        model = LatentComputationalMachine(
            state_dim=args.state_dim,
            latent_dim=args.latent_dim,
            memory_slots=args.memory_slots,
            num_cells=args.num_cells,
            route_temperature=args.route_temperature,
            state_update_scale=args.state_update_scale,
            stabilize_state=args.stabilize_state,
            structured_value_lane=args.structured_value_lane,
            value_clip=args.value_clip,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    history = train_machine(
        model, optimizer, device, args.steps, args.batch_size, args.num_entities,
        args.train_max_steps, args.role_loss_weight,
        args.state_supervision_weight, args.log_every,
    )
    facts_a = torch.linspace(-0.5, 0.5, args.num_entities, device=device)
    facts_b = torch.linspace(0.45, -0.45, args.num_entities, device=device)
    ood_cases = sample_program_cases(
        args.eval_batches, args.batch_size, args.num_entities,
        args.eval_max_steps, device,
        min_steps=args.eval_min_steps,
    )
    in_range_cases = sample_program_cases(
        args.eval_batches, args.batch_size, args.num_entities,
        args.train_max_steps, device,
        min_steps=2,
    )
    eval_a = evaluate(
        model, device, facts=facts_a, batch_size=args.batch_size,
        num_entities=args.num_entities, min_steps=args.eval_min_steps,
        max_steps=args.eval_max_steps, batches=args.eval_batches, cases=ood_cases,
    )
    eval_in_range = evaluate(
        model, device, facts=facts_a, batch_size=args.batch_size,
        num_entities=args.num_entities, min_steps=2,
        max_steps=args.train_max_steps, batches=args.eval_batches, cases=in_range_cases,
    )
    eval_b = evaluate(
        model, device, facts=facts_b, batch_size=args.batch_size,
        num_entities=args.num_entities, min_steps=args.eval_min_steps,
        max_steps=args.eval_max_steps, batches=args.eval_batches, cases=ood_cases,
    )
    eval_b_in_range = evaluate(
        model, device, facts=facts_b, batch_size=args.batch_size,
        num_entities=args.num_entities, min_steps=2,
        max_steps=args.train_max_steps, batches=args.eval_batches,
        cases=in_range_cases,
    )
    eval_zero = evaluate(
        model, device, facts=facts_a, batch_size=args.batch_size,
        num_entities=args.num_entities, min_steps=args.eval_min_steps,
        max_steps=args.eval_max_steps, batches=args.eval_batches, cases=ood_cases,
        zero_memory=True,
    )
    operation_swap_before = evaluate(
        model, device, facts=facts_a, batch_size=args.batch_size,
        num_entities=args.num_entities, min_steps=2,
        max_steps=args.train_max_steps, batches=args.eval_batches,
        cases=in_range_cases, swapped=True,
    )
    operation_swap_history = []
    if args.operation_adapt_steps > 0:
        operation_swap_history = adapt_operation_swap(
            model, device, args.operation_adapt_steps, args.batch_size,
            args.num_entities, args.train_max_steps,
            args.operation_adapt_learning_rate, args.log_every,
        )
    operation_swap_after = evaluate(
        model, device, facts=facts_a, batch_size=args.batch_size,
        num_entities=args.num_entities, min_steps=2,
        max_steps=args.train_max_steps, batches=args.eval_batches,
        cases=in_range_cases, swapped=True,
    )
    # Diagnostic route reuse uses identical program batches with different
    # facts; the model is not given entity IDs after external retrieval.
    _, reuse_programs, _, _ = sample_batch(
        args.batch_size * args.eval_batches, args.num_entities,
        args.eval_max_steps, device, min_steps=args.eval_min_steps, facts=facts_a,
    )
    reuse_facts = facts_a
    reuse_entities = torch.arange(
        args.batch_size * args.eval_batches, device=device
    ) % args.num_entities
    with torch.no_grad():
        _, reuse_trace = model(
            reuse_programs, reuse_facts, reuse_entities,
            hard_route=True, return_trace=True,
        )
    reuse_metrics = route_reuse(reuse_trace["route_ids"].cpu(), reuse_programs.cpu())
    total_params = sum(parameter.numel() for parameter in model.parameters())
    cell_params = sum(parameter.numel() for parameter in model.cells[0].parameters())
    result = {
        "experiment": "taklif22_latent_computational_machine_v1",
        "model_kind": args.model_kind,
        "device": str(device),
        "seed": args.seed,
        "state_dim": args.state_dim,
        "latent_dim": args.latent_dim,
        "memory_slots": args.memory_slots,
        "num_cells": model.num_cells,
        "active_cells_per_step": 1,
        "num_entities": args.num_entities,
        "train_steps": args.steps,
        "train_max_steps": args.train_max_steps,
        "eval_min_steps": args.eval_min_steps,
        "eval_max_steps": args.eval_max_steps,
        "role_loss_weight": args.role_loss_weight,
        "state_supervision_weight": args.state_supervision_weight,
        "state_update_scale": args.state_update_scale,
        "stabilize_state": args.stabilize_state,
        "structured_value_lane": model.structured_value_lane,
        "value_clip": args.value_clip,
        "total_parameters": total_params,
        "one_active_cell_parameters": cell_params,
        "active_cell_fraction": cell_params / max(total_params, 1),
        "train_history": history,
        "fact_table_A_eval": {key: value for key, value in eval_a.items() if key != "route_ids"},
        "in_range_eval": {key: value for key, value in eval_in_range.items() if key != "route_ids"},
        "fact_table_B_eval": {key: value for key, value in eval_b.items() if key != "route_ids"},
        "fact_table_B_in_range_eval": {key: value for key, value in eval_b_in_range.items() if key != "route_ids"},
        "zero_memory_eval": {key: value for key, value in eval_zero.items() if key != "route_ids"},
        "operation_swap_before": {key: value for key, value in operation_swap_before.items() if key != "route_ids"},
        "operation_swap_after": {key: value for key, value in operation_swap_after.items() if key != "route_ids"},
        "operation_adapt_steps": args.operation_adapt_steps,
        "operation_adapt_history": operation_swap_history,
        "route_reuse": reuse_metrics,
        "quality_gates": {
            "fact_swap_finite": math.isfinite(float(eval_b["mse"])),
            "variable_depth_finite": math.isfinite(float(eval_a["mse"])),
            "memory_intervention_measured": True,
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-kind", choices=("latent", "dense-control"), default="latent")
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--num-cells", type=int, default=4)
    parser.add_argument("--num-entities", type=int, default=32)
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--state-update-scale", type=float, default=1.0)
    parser.add_argument("--stabilize-state", action="store_true")
    parser.add_argument("--structured-value-lane", action="store_true")
    parser.add_argument("--value-clip", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--train-max-steps", type=int, default=4)
    parser.add_argument("--eval-min-steps", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=6)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--role-loss-weight", type=float, default=0.05)
    parser.add_argument("--state-supervision-weight", type=float, default=0.0)
    parser.add_argument("--operation-adapt-steps", type=int, default=0)
    parser.add_argument("--operation-adapt-learning-rate", type=float, default=2e-3)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="results/runs/latent_computational_machine.json")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
