from __future__ import annotations

import argparse
import json
import math
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from torch import nn


class SemanticAddressModel(nn.Module):
    """Synthetic need -> coarse group -> local self-match address model."""

    def __init__(self, num_cells: int = 64, groups: int = 8,
                 cells_per_group: int = 8, descriptor_dim: int = 32,
                 coupled: bool = True) -> None:
        super().__init__()
        if num_cells != groups * cells_per_group:
            raise ValueError("num_cells must equal groups * cells_per_group")
        self.num_cells = num_cells
        self.groups = groups
        self.cells_per_group = cells_per_group
        self.descriptor_dim = descriptor_dim
        self.coupled = coupled
        self.need_encoder = nn.Sequential(
            nn.Linear(3 + 4, 64), nn.Tanh(), nn.Linear(64, descriptor_dim)
        )
        self.coarse_keys = nn.Parameter(torch.randn(groups, descriptor_dim) * 0.02)
        self.descriptors = nn.Parameter(torch.randn(num_cells, descriptor_dim) * 0.02)
        role_prototypes = torch.randn(3, descriptor_dim)
        role_prototypes = role_prototypes / role_prototypes.norm(dim=-1, keepdim=True)
        self.register_buffer("role_prototypes", role_prototypes)
        body_descriptors = torch.randn(num_cells, descriptor_dim)
        cell_roles = torch.full((num_cells,), -1, dtype=torch.long)
        for role in range(3):
            start = role * cells_per_group
            cell_roles[start:start + cells_per_group] = role
            body_descriptors[start:start + cells_per_group] = (
                role_prototypes[role] + 0.03 * torch.randn(cells_per_group, descriptor_dim)
            )
        body_descriptors = body_descriptors / body_descriptors.norm(dim=-1, keepdim=True)
        self.register_buffer("body_descriptors", body_descriptors)
        self.register_buffer("cell_roles", cell_roles)

    def encode_need(self, role: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        role_one_hot = nn.functional.one_hot(role, 3).to(value.dtype)
        angle = value.unsqueeze(-1) / 64.0 * (2.0 * math.pi)
        value_features = torch.cat((angle.sin(), angle.cos(), (2 * angle).sin(), (2 * angle).cos()), dim=-1)
        query = self.need_encoder(torch.cat((role_one_hot, value_features), dim=-1))
        return nn.functional.normalize(query, dim=-1)

    def scores(self, role: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.encode_need(role, value)
        coarse = query @ nn.functional.normalize(self.coarse_keys, dim=-1).t()
        fine = query @ nn.functional.normalize(self.descriptors, dim=-1).t()
        return coarse, fine

    def loss(self, role: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        coarse, fine = self.scores(role, value)
        coarse_loss = nn.functional.cross_entropy(coarse, role)
        target_cells = torch.arange(3, device=role.device).mul(self.cells_per_group)
        target_mask = role.unsqueeze(-1).eq(torch.arange(3, device=role.device))
        target_mask = torch.cat((target_mask, torch.zeros(
            role.shape[0], self.groups - 3, dtype=torch.bool, device=role.device
        )), dim=-1)
        cell_mask = target_mask.repeat_interleave(self.cells_per_group, dim=-1)
        target_scores = fine.masked_fill(~cell_mask, float("-inf"))
        fine_loss = -torch.logsumexp(target_scores, dim=-1) + torch.logsumexp(fine, dim=-1)
        descriptor_loss = 1.0 - nn.functional.cosine_similarity(
            self.descriptors, self.body_descriptors.detach(), dim=-1
        ).mean()
        return coarse_loss + fine_loss.mean() + (0.25 * descriptor_loss if self.coupled else 0.0)

    @torch.no_grad()
    def route(self, role: torch.Tensor, value: torch.Tensor, active: int = 2) -> dict[str, torch.Tensor]:
        coarse, fine = self.scores(role, value)
        groups = coarse.argmax(dim=-1)
        windows = groups.unsqueeze(-1) * self.cells_per_group + torch.arange(
            self.cells_per_group, device=role.device
        )
        local_scores = fine.gather(1, windows)
        local_top_scores, local_top = local_scores.topk(active, dim=-1)
        selected = windows.gather(1, local_top)
        return {
            "coarse": groups,
            "selected": selected,
            "local_scores": local_top_scores,
            "full_top": fine.topk(active, dim=-1).indices,
        }


def synthetic_batch(batch_size: int, device: torch.device, generator: torch.Generator
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    role = torch.randint(3, (batch_size,), generator=generator, device=device)
    value = torch.randint(64, (batch_size,), generator=generator, device=device).float()
    return role, value


@torch.no_grad()
def audit(model: SemanticAddressModel, device: torch.device) -> dict[str, Any]:
    roles = torch.arange(3, device=device).repeat_interleave(64)
    values = torch.arange(64, device=device).repeat(3).float()
    routed = model.route(roles, values)
    selected_roles = model.cell_roles[routed["selected"]]
    role_ok = selected_roles.eq(roles.unsqueeze(-1)).all(dim=-1)
    coarse_ok = routed["coarse"].eq(roles)
    recall = []
    reuse = []
    for role in range(3):
        role_selected = routed["selected"][roles.eq(role)]
        sets = [set(row.tolist()) for row in role_selected]
        pair_jaccard = [
            len(left & right) / max(1, len(left | right))
            for left, right in combinations(sets, 2)
        ]
        reuse.append(sum(pair_jaccard) / max(1, len(pair_jaccard)))
    for selected, full_top in zip(routed["selected"], routed["full_top"]):
        recall.append(len(set(selected.tolist()) & set(full_top.tolist())) / selected.numel())
    body_cos = nn.functional.cosine_similarity(
        model.descriptors, model.body_descriptors, dim=-1
    ).mean()
    return {
        "coarse_group_accuracy": float(coarse_ok.float().mean().cpu()),
        "all_active_cells_match_role": float(role_ok.float().mean().cpu()),
        "mean_full_scan_topk_recall": sum(recall) / len(recall),
        "same_role_route_jaccard": sum(reuse) / len(reuse),
        "descriptor_body_cosine": float(body_cos.cpu()),
        "route_dot_products": model.groups + model.cells_per_group,
        "full_scan_dot_products": model.num_cells,
    }


def run_variant(args: argparse.Namespace, coupled: bool, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(args.seed + (0 if coupled else 1000))
    model = SemanticAddressModel(coupled=coupled).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + (10 if coupled else 20))
    first_loss = None
    start = time.perf_counter()
    for _ in range(args.steps):
        role, value = synthetic_batch(args.batch_size, device, generator)
        loss = model.loss(role, value)
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    result = {
        "variant": "coupled_descriptor" if coupled else "free_descriptor",
        "steps": args.steps,
        "first_loss": first_loss,
        "final_loss": float(loss.detach().cpu()),
        "seconds": time.perf_counter() - start,
        "audit": audit(model, device),
    }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    results = [
        run_variant(args, True, device),
        run_variant(args, False, device),
    ]
    report = {
        "device": str(device),
        "num_cells": 64,
        "groups": 8,
        "cells_per_group": 8,
        "active_cells": 2,
        "results": results,
        "note": "Synthetic semantic-address audit; bodies are fixed behavior signatures."
                 " The local route evaluates 16 dot products instead of 64.",
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-describing semantic addressing audit")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/runs/semantic_addressing.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
