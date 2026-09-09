"""Sequential multi-layer Qwen circuit-bank transfer benchmark."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.profiler import record_function

from benchmark_qwen_parent_transplant import EVAL_TEXT, TRAIN_TEXT
from benchmark_qwen_two_layer_transplant import (
    CalibratedChild,
    InputCalibratedChild,
    MixedParentChild,
    NESwiGLUBlock,
    SwiGLUResidualChild,
    benchmark_forward,
    capture_batches,
    ce,
    evaluate_current,
    make_child,
    token_stream,
    train_child,
)


PROMPT_PARITY_TEXTS = (
    "Explain why a sparse neural network can save compute without changing every parameter.",
    "Solve this arithmetic problem step by step: 37 * 24 + 19.",
    "Give three practical risks when replacing a dense feed-forward layer with routed circuits.",
)


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must contain at least one distinct index")
    return layers


def parse_schedule(
    value: str | None,
    length: int,
    default: int | float | None,
    cast: type[int] | type[float] | None,
    label: str,
) -> list[int | float | None]:
    """Expand a scalar option or validate one value per replaced layer."""
    if value is None:
        return [default] * length
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != length:
        raise ValueError(
            f"{label} must contain exactly {length} comma-separated values"
        )
    return [None if item.lower() == "none" else cast(item) for item in items]


def load_text_file(path: str | None, default: str, label: str) -> str:
    """Load optional UTF-8 text while keeping the historical default."""
    if path is None:
        return default
    text_path = Path(path)
    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} text file is empty: {text_path}")
    return text


@torch.inference_mode()
def prompt_parity(
    model: torch.nn.Module,
    tokenizer,
    layers: list[torch.nn.Module],
    parents: list[torch.nn.Module],
    children: list[torch.nn.Module],
    max_new_tokens: int,
) -> list[dict[str, object]]:
    """Generate identical prompts with parent and sparse layer replacements."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        list(PROMPT_PARITY_TEXTS),
        return_tensors="pt",
        padding=True,
    ).to(next(model.parameters()).device)
    parent_outputs = []
    for layer, parent in zip(layers, parents):
        layer.mlp = parent
    parent_ids = model.generate(
        **encoded, do_sample=False, use_cache=True,
        max_new_tokens=max_new_tokens,
    )
    parent_outputs = tokenizer.batch_decode(
        parent_ids, skip_special_tokens=True,
    )
    for layer, child in zip(layers, children):
        layer.mlp = child
    sparse_ids = model.generate(
        **encoded, do_sample=False, use_cache=True,
        max_new_tokens=max_new_tokens,
    )
    sparse_outputs = tokenizer.batch_decode(
        sparse_ids, skip_special_tokens=True,
    )
    for layer, parent in zip(layers, parents):
        layer.mlp = parent
    return [
        {
            "prompt": prompt,
            "parent": parent_output,
            "sparse": sparse_output,
            "exact_text_match": parent_output == sparse_output,
        }
        for prompt, parent_output, sparse_output in zip(
            PROMPT_PARITY_TEXTS, parent_outputs, sparse_outputs,
        )
    ]


def _set_hard_train_modules(
    module: torch.nn.Module,
    enabled: bool,
) -> list[tuple[torch.nn.Module, bool]]:
    previous = []
    for nested in module.modules():
        if hasattr(nested, "hard_train"):
            previous.append((nested, bool(nested.hard_train)))
            nested.hard_train = enabled
    return previous


def make_transferred_qwen_child(
    parent: torch.nn.Module,
    calibration_rank: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Copy a Qwen3 gate/up/down SwiGLU into an attention-free child."""
    gate = parent.gate_proj
    up = parent.up_proj
    down = parent.down_proj
    if gate.bias is not None or up.bias is not None or down.bias is not None:
        raise ValueError("Qwen transfer currently expects bias-free projections")
    child = NESwiGLUBlock(
        int(gate.in_features), int(gate.out_features),
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        child.gate_projection.weight.copy_(gate.weight)
        child.value_projection.weight.copy_(up.weight)
        child.output_projection.weight.copy_(down.weight)
        child.gate_projection.bias.zero_()
        child.value_projection.bias.zero_()
        child.output_projection.bias.zero_()
    if calibration_rank > 0:
        child = CalibratedChild(
            child, int(gate.in_features), calibration_rank,
        ).to(device=device, dtype=dtype)
    return child


@torch.no_grad()
def activation_balanced_partition(
    parent: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    num_experts: int,
) -> list[torch.Tensor]:
    """Partition neurons by observed FFN contribution energy on calibration IO."""
    gate_weight = parent.gate_proj.weight
    up_weight = parent.up_proj.weight
    down_weight = parent.down_proj.weight
    inner_size = int(gate_weight.shape[0])
    score = torch.zeros(inner_size, device=device, dtype=torch.float32)
    down_norm_sq = down_weight.float().square().sum(dim=0)
    for batch in io_batches:
        inputs = batch["input"].to(device=device, dtype=dtype)
        gate = F.linear(inputs, gate_weight)
        value = F.linear(inputs, up_weight)
        neuron_value = F.silu(gate) * value
        score += neuron_value.float().square().mean(dim=(0, 1)) * down_norm_sq
    order = score.argsort(descending=True).tolist()
    loads = torch.zeros(num_experts, dtype=torch.float32)
    chunk = inner_size // num_experts
    counts = [0] * num_experts
    groups: list[list[int]] = [[] for _ in range(num_experts)]
    for neuron_id in order:
        eligible_loads = loads.clone()
        for expert_id, count in enumerate(counts):
            if count >= chunk:
                eligible_loads[expert_id] = float("inf")
        expert_id = int(eligible_loads.argmin().item())
        groups[expert_id].append(int(neuron_id))
        loads[expert_id] += score[neuron_id].cpu()
        counts[expert_id] += 1
    parent_device = gate_weight.device
    return [
        torch.tensor(group, device=parent_device, dtype=torch.long)
        for group in groups
    ]


@torch.no_grad()
def core_overlap_partition(
    parent: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    num_experts: int,
    core_fraction: float = 0.25,
) -> list[torch.Tensor]:
    """Repeat a high-energy core and distribute the remaining tail evenly.

    Each group keeps the same width as a disjoint partition.  The top
    ``core_fraction`` of neurons by observed output energy is copied into every
    group; the remaining slots are filled from a deterministic, interleaved
    tail.  This creates a fixed overlap codebook without random sampling or
    increasing active compute.
    """
    gate_weight = parent.gate_proj.weight
    up_weight = parent.up_proj.weight
    down_weight = parent.down_proj.weight
    inner_size = int(gate_weight.shape[0])
    if inner_size % num_experts:
        raise ValueError("Qwen intermediate size must divide evenly into experts")
    if not 0.0 < core_fraction < 1.0:
        raise ValueError("core_overlap core_fraction must be between zero and one")
    chunk = inner_size // num_experts
    score = torch.zeros(inner_size, device=device, dtype=torch.float32)
    down_norm_sq = down_weight.float().square().sum(dim=0)
    for batch in io_batches:
        inputs = batch["input"].to(device=device, dtype=dtype)
        neuron_value = F.silu(F.linear(inputs, gate_weight)) * F.linear(
            inputs, up_weight,
        )
        score += neuron_value.float().square().mean(dim=(0, 1)) * down_norm_sq
    order = score.argsort(descending=True)
    core_size = min(chunk - 1, max(1, int(round(chunk * core_fraction))))
    core = order[:core_size]
    tail = order[core_size:]
    tail_slots = chunk - core_size
    # Interleaving gives each group comparable tail energy while preserving a
    # deterministic codebook.  The unselected low-energy tail is intentional.
    groups = []
    for expert_id in range(num_experts):
        selected_tail = tail[expert_id::num_experts][:tail_slots]
        if selected_tail.numel() != tail_slots:
            raise RuntimeError("core-overlap tail could not fill all groups")
        groups.append(torch.cat((core, selected_tail), dim=0))
    parent_device = gate_weight.device
    return [group.to(device=parent_device, dtype=torch.long) for group in groups]


def sampled_overlap_partition(
    parent: torch.nn.Module,
    num_experts: int,
) -> list[torch.Tensor]:
    """Give each macro-cell an independent representative sample of neurons."""
    inner_size = int(parent.gate_proj.weight.shape[0])
    chunk = inner_size // num_experts
    parent_device = parent.gate_proj.weight.device
    return [
        torch.randperm(inner_size, device=parent_device)[:chunk]
        for _ in range(num_experts)
    ]


@torch.no_grad()
def stratified_overlap_partition(
    parent: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    num_experts: int,
) -> list[torch.Tensor]:
    """Overlap cells while matching the calibration contribution-score strata."""
    gate_weight = parent.gate_proj.weight
    up_weight = parent.up_proj.weight
    down_weight = parent.down_proj.weight
    inner_size = int(gate_weight.shape[0])
    chunk = inner_size // num_experts
    score = torch.zeros(inner_size, device=device, dtype=torch.float32)
    down_norm_sq = down_weight.float().square().sum(dim=0)
    for batch in io_batches:
        inputs = batch["input"].to(device=device, dtype=dtype)
        neuron_value = F.silu(F.linear(inputs, gate_weight)) * F.linear(
            inputs, up_weight,
        )
        score += neuron_value.float().square().mean(dim=(0, 1)) * down_norm_sq
    ordered = score.argsort(descending=True)
    strata = ordered.reshape(num_experts, chunk)
    per_stratum = chunk // num_experts
    cells = []
    for _ in range(num_experts):
        pieces = []
        for stratum in strata:
            pieces.append(stratum[torch.randperm(chunk, device=device)[:per_stratum]])
        cells.append(torch.cat(pieces, dim=0))
    return cells


@torch.no_grad()
def activation_cluster_partition(
    parent: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    num_experts: int,
    iterations: int = 6,
    max_tokens: int = 512,
) -> list[torch.Tensor]:
    """Group neurons with similar calibration-time SwiGLU activation signatures.

    Each neuron is represented by its activation coefficient over a deterministic
    token sample.  Cosine-normalized signatures are clustered with a balanced
    greedy assignment so every expert owns exactly the same number of neurons.
    The partition is intentionally deterministic for reproducible benchmark runs.
    """
    gate_weight = parent.gate_proj.weight
    up_weight = parent.up_proj.weight
    inner_size = int(gate_weight.shape[0])
    if inner_size % num_experts:
        raise ValueError("Qwen intermediate size must divide evenly into experts")
    signatures = []
    remaining_tokens = max(1, int(max_tokens))
    for batch in io_batches:
        if remaining_tokens <= 0:
            break
        inputs = batch["input"].to(device=device, dtype=dtype)
        activation = F.silu(F.linear(inputs, gate_weight)) * F.linear(
            inputs, up_weight,
        )
        flat = activation.float().reshape(-1, inner_size)
        take = min(int(flat.shape[0]), remaining_tokens)
        positions = torch.linspace(
            0, flat.shape[0] - 1, steps=take, device=device,
        ).long()
        signatures.append(flat.index_select(0, positions))
        remaining_tokens -= take
    if not signatures:
        raise ValueError("activation-cluster partition requires non-empty calibration IO")
    features = torch.cat(signatures, dim=0).transpose(0, 1).contiguous()
    features = F.normalize(features, p=2, dim=1, eps=1e-6)

    # Evenly spaced deterministic seeds avoid introducing a second random source
    # into a benchmark whose model/training seed is already controlled globally.
    seed_ids = torch.linspace(
        0, inner_size - 1, steps=num_experts, device=device,
    ).long()
    centers = features.index_select(0, seed_ids).clone()
    chunk = inner_size // num_experts
    for _ in range(max(1, int(iterations))):
        distances = 1.0 - features @ centers.transpose(0, 1)
        remaining = torch.full(
            (num_experts,), chunk, device=device, dtype=torch.int32,
        )
        unassigned = torch.ones(inner_size, device=device, dtype=torch.bool)
        assignments = torch.full(
            (inner_size,), -1, device=device, dtype=torch.long,
        )
        # Greedily take the globally closest available neuron/expert pair.  This
        # enforces exact cell sizes while preserving local signature similarity.
        for _ in range(inner_size):
            costs = distances.masked_fill(~unassigned[:, None], float("inf"))
            costs = costs.masked_fill(remaining[None, :] <= 0, float("inf"))
            flat_id = int(costs.argmin().item())
            neuron_id = flat_id // num_experts
            expert_id = flat_id % num_experts
            assignments[neuron_id] = expert_id
            unassigned[neuron_id] = False
            remaining[expert_id] -= 1
        updated = []
        for expert_id in range(num_experts):
            selected = features[assignments == expert_id]
            updated.append(F.normalize(selected.mean(dim=0), p=2, dim=0, eps=1e-6))
        centers = torch.stack(updated, dim=0)

    parent_device = gate_weight.device
    return [
        torch.where(assignments == expert_id)[0].to(device=parent_device)
        for expert_id in range(num_experts)
    ]


@torch.no_grad()
def contribution_cluster_partition(
    parent: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    num_experts: int,
    iterations: int = 6,
    max_tokens: int = 256,
    sketch_dim: int = 32,
) -> list[torch.Tensor]:
    """Cluster neurons by their output-space contribution signatures.

    ``activation_cluster_partition`` only compares the scalar SwiGLU
    coefficients.  Two neurons can have similar coefficients but point in
    unrelated directions after ``down_proj``.  This variant sketches the
    per-token contribution ``activation_j(x) * down_proj[:, j]`` before the
    same balanced clustering step.  The sketch keeps the calibration memory
    bounded while making the partition aware of the actual hidden-state
    output geometry.
    """
    gate_weight = parent.gate_proj.weight
    up_weight = parent.up_proj.weight
    down_weight = parent.down_proj.weight
    inner_size = int(gate_weight.shape[0])
    hidden_size = int(gate_weight.shape[1])
    if inner_size % num_experts:
        raise ValueError("Qwen intermediate size must divide evenly into experts")
    if sketch_dim < 1:
        raise ValueError("contribution-cluster sketch_dim must be positive")

    # Deterministic signed coordinate sketch: it introduces no extra random
    # source and preserves both positive and negative output directions.
    sketch_size = min(int(sketch_dim), hidden_size)
    sketch_ids = torch.linspace(
        0, hidden_size - 1, steps=sketch_size, device=device,
    ).long()
    sketch_signs = torch.where(
        torch.arange(sketch_size, device=device) % 2 == 0,
        torch.ones(sketch_size, device=device),
        -torch.ones(sketch_size, device=device),
    )
    down_sketch = down_weight.float().index_select(0, sketch_ids)
    down_sketch = down_sketch * sketch_signs[:, None]

    signatures = []
    remaining_tokens = max(1, int(max_tokens))
    for batch in io_batches:
        if remaining_tokens <= 0:
            break
        inputs = batch["input"].to(device=device, dtype=dtype)
        activation = F.silu(F.linear(inputs, gate_weight)) * F.linear(
            inputs, up_weight,
        )
        flat = activation.float().reshape(-1, inner_size)
        take = min(int(flat.shape[0]), remaining_tokens)
        positions = torch.linspace(
            0, flat.shape[0] - 1, steps=take, device=device,
        ).long()
        sampled = flat.index_select(0, positions)
        # [tokens, neurons, sketch] -> [neurons, tokens * sketch]
        contributions = sampled[:, :, None] * down_sketch.transpose(0, 1)[None, :, :]
        signatures.append(
            contributions.permute(1, 0, 2).reshape(inner_size, -1)
        )
        remaining_tokens -= take
    if not signatures:
        raise ValueError(
            "contribution-cluster partition requires non-empty calibration IO"
        )
    features = torch.cat(signatures, dim=1)
    features = F.normalize(features, p=2, dim=1, eps=1e-6)

    seed_ids = torch.linspace(
        0, inner_size - 1, steps=num_experts, device=device,
    ).long()
    centers = features.index_select(0, seed_ids).clone()
    chunk = inner_size // num_experts
    for _ in range(max(1, int(iterations))):
        distances = 1.0 - features @ centers.transpose(0, 1)
        remaining = torch.full(
            (num_experts,), chunk, device=device, dtype=torch.int32,
        )
        unassigned = torch.ones(inner_size, device=device, dtype=torch.bool)
        assignments = torch.full(
            (inner_size,), -1, device=device, dtype=torch.long,
        )
        for _ in range(inner_size):
            costs = distances.masked_fill(~unassigned[:, None], float("inf"))
            costs = costs.masked_fill(remaining[None, :] <= 0, float("inf"))
            flat_id = int(costs.argmin().item())
            neuron_id = flat_id // num_experts
            expert_id = flat_id % num_experts
            assignments[neuron_id] = expert_id
            unassigned[neuron_id] = False
            remaining[expert_id] -= 1
        updated = []
        for expert_id in range(num_experts):
            selected = features[assignments == expert_id]
            updated.append(F.normalize(selected.mean(dim=0), p=2, dim=0, eps=1e-6))
        centers = torch.stack(updated, dim=0)

    parent_device = gate_weight.device
    return [
        torch.where(assignments == expert_id)[0].to(device=parent_device)
        for expert_id in range(num_experts)
    ]


@torch.no_grad()
def contribution_diverse_partition(
    parent: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    num_experts: int,
) -> list[torch.Tensor]:
    """Distribute every output-space contribution cluster across all groups.

    ``contribution_cluster_partition`` makes groups locally coherent.  This
    complementary layout keeps the same disjoint compute budget but places an
    equal slice of every cluster in every group, so any active subset sees a
    broad mixture of functional directions instead of only half the clusters.
    """
    clusters = contribution_cluster_partition(
        parent, io_batches, device, dtype, num_experts,
    )
    inner_size = int(parent.gate_proj.weight.shape[0])
    chunk = inner_size // num_experts
    if chunk % num_experts:
        raise ValueError(
            "contribution-diverse requires group width divisible by num_experts"
        )
    per_cluster = chunk // num_experts
    groups = []
    for expert_id in range(num_experts):
        pieces = [
            cluster[expert_id * per_cluster:(expert_id + 1) * per_cluster]
            for cluster in clusters
        ]
        groups.append(torch.cat(pieces, dim=0))
    parent_device = parent.gate_proj.weight.device
    return [group.to(device=parent_device, dtype=torch.long) for group in groups]


class QwenSwiGLUSlice(torch.nn.Module):
    """One contiguous intermediate-neuron slice of a Qwen SwiGLU."""

    def __init__(
        self,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        hidden_size = int(gate_weight.shape[1])
        inner_size = int(gate_weight.shape[0])
        self.gate_projection = torch.nn.Linear(hidden_size, inner_size, bias=False)
        self.value_projection = torch.nn.Linear(hidden_size, inner_size, bias=False)
        self.output_projection = torch.nn.Linear(inner_size, hidden_size, bias=False)
        with torch.no_grad():
            self.gate_projection.weight.copy_(gate_weight)
            self.value_projection.weight.copy_(up_weight)
            self.output_projection.weight.copy_(down_weight)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_projection(hidden_states))
        return self.output_projection(gated * self.value_projection(hidden_states))


class TransferredRoutedQwenChild(torch.nn.Module):
    """Copied Qwen neurons partitioned into a hard-routed sparse bank."""

    def __init__(
        self,
        parent: torch.nn.Module,
        num_experts: int,
        active_experts: int,
        temperature: float,
        dispatch_mode: str,
        partition_mode: str,
        route_source: str,
        hard_route_scale: float | None,
        partition_indices: list[torch.Tensor] | None = None,
        router_hidden_size: int = 128,
        router_input: str = "hidden",
        pairwise_cost_parameterization: str = "components",
        router_sketch_dim: int = 8,
    ) -> None:
        super().__init__()
        if not 1 <= active_experts <= num_experts:
            raise ValueError("active_experts must be within num_experts")
        inner_size, hidden_size = parent.gate_proj.weight.shape
        if inner_size % num_experts:
            raise ValueError("Qwen intermediate size must divide evenly into experts")
        self.num_experts = int(num_experts)
        self.active_experts = int(active_experts)
        self.temperature = float(temperature)
        self.hard_train = False
        self.hard_train_blend = 0.0
        if partition_mode not in {
            "contiguous", "interleaved", "norm-balanced", "activation-balanced",
            "sampled-overlap", "stratified-overlap", "activation-cluster",
            "contribution-cluster", "contribution-diverse", "core-overlap",
        }:
            raise ValueError(
                "partition_mode must be contiguous, interleaved, norm-balanced, "
                "activation-balanced, sampled-overlap, stratified-overlap, "
                "activation-cluster, contribution-cluster, contribution-diverse, "
                "or core-overlap"
            )
        self.partition_mode = partition_mode
        if route_source not in {
            "router", "subset-router", "oracle-dot", "oracle-energy",
            "oracle-subset", "pairwise-cost-router",
        }:
            raise ValueError(
                "route_source must be router, subset-router, oracle-dot, "
                "oracle-energy, oracle-subset, or pairwise-cost-router"
            )
        self.route_source = route_source
        if router_hidden_size < 1:
            raise ValueError("router hidden size must be positive")
        if router_input not in {"hidden", "group-energy", "group-sketch"}:
            raise ValueError(
                "router input must be hidden, group-energy, or group-sketch"
            )
        self.router_hidden_size = int(router_hidden_size)
        self.router_input = router_input
        if router_sketch_dim < 1:
            raise ValueError("router sketch dimension must be positive")
        self.router_sketch_dim = int(router_sketch_dim)
        if pairwise_cost_parameterization not in {"components", "centered-basis"}:
            raise ValueError(
                "pairwise cost parameterization must be components or centered-basis"
            )
        self.pairwise_cost_parameterization = pairwise_cost_parameterization
        self.hard_route_scale = (
            (
                self.num_experts
                if self.active_experts == self.num_experts
                else self.num_experts / self.active_experts
            )
            if hard_route_scale is None else float(hard_route_scale)
        )
        if dispatch_mode not in {
            "grouped", "grouped-cached", "grouped-prepacked",
            "grouped-prepacked-fused", "grouped-fused",
            "grouped-tiled", "grouped-optimized", "grouped-adaptive",
            "grouped-adaptive-nozero", "grouped-adaptive-effective-output",
            "grouped-adaptive-atomic-pack",
            "grouped-adaptive-atomic-effective-output",
            "packed", "packed-fused", "packed-fp16",
            "fused", "token-loop",
        }:
            raise ValueError(
                "transferred sparse child supports grouped, grouped-cached, grouped-prepacked, "
                "grouped-prepacked-fused, grouped-fused, grouped-tiled, "
                "grouped-optimized, grouped-adaptive, grouped-adaptive-nozero, "
                "grouped-adaptive-effective-output, packed, packed-fused, "
                "grouped-adaptive-atomic-pack, grouped-adaptive-atomic-effective-output, "
                "packed, packed-fused, packed-fp16, "
                "fused, or token-loop"
            )
        self.dispatch_mode = dispatch_mode
        chunk = inner_size // num_experts
        norm_order = None
        if partition_mode == "norm-balanced":
            neuron_score = (
                parent.gate_proj.weight.detach().norm(dim=1)
                * parent.up_proj.weight.detach().norm(dim=1)
                * parent.down_proj.weight.detach().norm(dim=0)
            )
            norm_order = neuron_score.argsort(descending=True)
        if partition_mode == "activation-balanced":
            if partition_indices is None or len(partition_indices) != num_experts:
                raise ValueError(
                    "activation-balanced partition requires one index tensor per expert"
                )
        if partition_mode == "sampled-overlap":
            if partition_indices is None or len(partition_indices) != num_experts:
                raise ValueError(
                    "sampled-overlap partition requires one index tensor per expert"
                )
        if partition_mode == "stratified-overlap":
            if partition_indices is None or len(partition_indices) != num_experts:
                raise ValueError(
                    "stratified-overlap partition requires one index tensor per expert"
                )
        if partition_mode == "activation-cluster":
            if partition_indices is None or len(partition_indices) != num_experts:
                raise ValueError(
                    "activation-cluster partition requires one index tensor per expert"
                )
        if partition_mode == "contribution-cluster":
            if partition_indices is None or len(partition_indices) != num_experts:
                raise ValueError(
                    "contribution-cluster partition requires one index tensor per expert"
                )
        if partition_mode == "core-overlap":
            if partition_indices is None or len(partition_indices) != num_experts:
                raise ValueError(
                    "core-overlap partition requires one index tensor per expert"
                )
        if partition_mode == "contribution-diverse":
            if partition_indices is None or len(partition_indices) != num_experts:
                raise ValueError(
                    "contribution-diverse partition requires one index tensor per expert"
                )
        self.experts = torch.nn.ModuleList()
        for expert_id in range(num_experts):
            if partition_mode == "contiguous":
                indices = torch.arange(
                    expert_id * chunk, (expert_id + 1) * chunk,
                    device=parent.gate_proj.weight.device,
                )
            elif partition_mode == "interleaved":
                indices = torch.arange(
                    expert_id, inner_size, num_experts,
                    device=parent.gate_proj.weight.device,
                )
            elif partition_mode == "activation-balanced":
                indices = partition_indices[expert_id]
            elif partition_mode == "sampled-overlap":
                indices = partition_indices[expert_id]
            elif partition_mode == "stratified-overlap":
                indices = partition_indices[expert_id]
            elif partition_mode == "activation-cluster":
                indices = partition_indices[expert_id]
            elif partition_mode == "contribution-cluster":
                indices = partition_indices[expert_id]
            elif partition_mode == "core-overlap":
                indices = partition_indices[expert_id]
            elif partition_mode == "contribution-diverse":
                indices = partition_indices[expert_id]
            else:
                if norm_order is None:
                    raise RuntimeError("norm-balanced partition order was not built")
                indices = norm_order[expert_id::num_experts]
            self.experts.append(QwenSwiGLUSlice(
                parent.gate_proj.weight.index_select(0, indices).detach(),
                parent.up_proj.weight.index_select(0, indices).detach(),
                parent.down_proj.weight.index_select(1, indices).detach(),
            ))
        self.register_buffer(
            "group_gate_weight",
            torch.stack([expert.gate_projection.weight for expert in self.experts]),
            persistent=False,
        )
        self.register_buffer(
            "group_value_weight",
            torch.stack([expert.value_projection.weight for expert in self.experts]),
            persistent=False,
        )
        self.register_buffer(
            "group_output_weight",
            torch.stack([expert.output_projection.weight for expert in self.experts]),
            persistent=False,
        )
        self.register_buffer(
            "group_gate_value_weight",
            torch.cat((self.group_gate_weight, self.group_value_weight), dim=1),
            persistent=False,
        )
        if router_input == "group-sketch":
            coordinates = torch.arange(
                1, hidden_size + 1,
                device=parent.gate_proj.weight.device,
                dtype=torch.float32,
            ).unsqueeze(-1)
            frequencies = torch.arange(
                1, self.router_sketch_dim + 1,
                device=coordinates.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            sketch = torch.cos(
                coordinates * frequencies * torch.pi / hidden_size,
            )
            sketch = sketch / sketch.norm(dim=0, keepdim=True).clamp_min(1e-6)
            self.register_buffer(
                "router_sketch_projection", sketch, persistent=False,
            )
        if route_source in {
            "subset-router", "oracle-subset", "pairwise-cost-router",
        }:
            subset_ids = torch.tensor(
                list(combinations(range(num_experts), active_experts)),
                device=parent.gate_proj.weight.device, dtype=torch.long,
            )
            subset_membership = F.one_hot(
                subset_ids, num_classes=num_experts,
            ).sum(dim=1).float()
            self.register_buffer(
                "subset_membership",
                subset_membership,
                persistent=False,
            )
            router_input_size = (
                hidden_size if router_input == "hidden"
                else num_experts if router_input == "group-energy"
                else num_experts * self.router_sketch_dim
            )
            if route_source == "pairwise-cost-router":
                pair_ids = torch.tensor(
                    list(combinations(range(num_experts), 2)),
                    device=parent.gate_proj.weight.device, dtype=torch.long,
                )
                pair_membership = F.one_hot(
                    pair_ids, num_classes=num_experts,
                ).sum(dim=1).float()
                subset_pair_membership = (
                    subset_membership @ pair_membership.transpose(0, 1)
                ).eq(2).float()
                self.register_buffer(
                    "subset_pair_membership",
                    subset_pair_membership,
                    persistent=False,
                )
                component_membership = torch.cat(
                    (subset_membership, subset_pair_membership), dim=1,
                )
                if pairwise_cost_parameterization == "centered-basis":
                    centered_components = (
                        component_membership
                        - component_membership.mean(dim=0, keepdim=True)
                    )
                    left, singular, _ = torch.linalg.svd(
                        centered_components, full_matrices=False,
                    )
                    rank = max(
                        1,
                        int((singular > singular.max() * 1e-6).sum().item()),
                    )
                    self.register_buffer(
                        "pairwise_cost_basis", left[:, :rank], persistent=False,
                    )
                    router_output_size = rank
                else:
                    router_output_size = num_experts + pair_ids.shape[0]
                self.pairwise_cost_router = torch.nn.Sequential(
                    torch.nn.Linear(router_input_size, self.router_hidden_size),
                    torch.nn.SiLU(),
                    torch.nn.Linear(
                        self.router_hidden_size,
                        router_output_size,
                    ),
                )
                torch.nn.init.zeros_(self.pairwise_cost_router[-1].weight)
                torch.nn.init.zeros_(self.pairwise_cost_router[-1].bias)
            else:
                self.subset_router = torch.nn.Sequential(
                    torch.nn.Linear(router_input_size, self.router_hidden_size),
                    torch.nn.SiLU(),
                    torch.nn.Linear(self.router_hidden_size, subset_ids.shape[0]),
                )
                torch.nn.init.zeros_(self.subset_router[-1].weight)
                torch.nn.init.zeros_(self.subset_router[-1].bias)
        else:
            self.router = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, self.router_hidden_size),
                torch.nn.SiLU(),
                torch.nn.Linear(self.router_hidden_size, num_experts),
            )
            torch.nn.init.zeros_(self.router[-1].weight)
            torch.nn.init.zeros_(self.router[-1].bias)
        self.last_selected: torch.Tensor | None = None
        self.last_route_weights: torch.Tensor | None = None
        self.last_all_outputs: torch.Tensor | None = None
        self.last_selected_outputs: torch.Tensor | None = None
        self.last_active_expert_fraction = 1.0
        # Optional inference-only hook used by the cross-group correction
        # wrapper.  When set, grouped dispatch folds the correction into the
        # selected pair accumulation so the wrapper need not reorder and
        # project the selected outputs a second time.
        self.grouped_fused_correction: tuple[
            torch.Tensor, torch.Tensor,
        ] | None = None
        # Optional inference-only effective output projection. A correction
        # of the form W_out + mix_out @ mix_in @ W_out can be folded into the
        # grouped projection, removing the per-call low-rank correction pass.
        self.grouped_effective_output_weight: torch.Tensor | None = None
        self.grouped_uniform_accum = False
        # Optional inference-only BMM layout probe.  The native buffers keep
        # Linear's [out, in] layout; grouped BMM consumes their transposes.
        # Caching contiguous transposes lets cuBLAS see the exact [E, in, out]
        # operands without rebuilding a strided view on every call.  It is
        # deliberately opt-in because the extra copies cost device memory.
        self._grouped_prepacked_weights: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        ] | None = None
        self._grouped_prepacked_weights_key: tuple[object, ...] | None = None
        self._grouped_pair_metadata_cache: dict[
            tuple[int, int, torch.device],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._fp16_dispatch_weights: tuple[torch.Tensor, ...] | None = None
        # One-token dispatch is kept opt-in until a full-model numerical
        # equivalence/quality audit approves its different reduction order.
        self.single_token_fast_path = False
        # The fused router is an inference-only CUDA probe.  PyTorch remains
        # the default because trained-router tie/order and numerical parity
        # must be audited before changing serving behavior.
        self.single_token_router_backend = "torch"
        self.single_token_projection_backend = "einsum"
        # Optional inference-only override used when a linear correction is
        # folded into the selected output projection.
        self.single_token_output_weight: torch.Tensor | None = None
        # Optional inference-only hook installed by the cross-group wrapper
        # for a one-launch base-output plus low-rank correction dispatch.
        self.single_token_full_correction: tuple[
            torch.Tensor, torch.Tensor, str
        ] | None = None

    def _router_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.router_input == "hidden":
            return hidden_states
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        grouped_gate = torch.einsum(
            "nh,ech->nec", flat_hidden, self.group_gate_weight,
        )
        grouped_value = torch.einsum(
            "nh,ech->nec", flat_hidden, self.group_value_weight,
        )
        coefficient = F.silu(grouped_gate) * grouped_value
        if self.router_input == "group-energy":
            energy = coefficient.float().square().mean(dim=-1)
            output_scale = self.group_output_weight.float().square().mean(dim=(1, 2))
            features = torch.log1p(energy * output_scale.unsqueeze(0))
            return features.reshape(*hidden_states.shape[:-1], self.num_experts).to(
                dtype=hidden_states.dtype,
            )
        projection = self.router_sketch_projection.to(
            device=flat_hidden.device, dtype=coefficient.dtype,
        )
        sketch = torch.einsum(
            "nec,ehc,hd->ned",
            coefficient,
            self.group_output_weight.to(dtype=coefficient.dtype),
            projection,
        )
        return sketch.reshape(
            *hidden_states.shape[:-1],
            self.num_experts * self.router_sketch_dim,
        )

    def _subset_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return subset logits; pairwise cost routers return negative costs."""
        features = self._router_features(hidden_states)
        if self.route_source == "pairwise-cost-router":
            components = self.pairwise_cost_router(features)
            if self.pairwise_cost_parameterization == "centered-basis":
                costs = components @ self.pairwise_cost_basis.transpose(0, 1)
            else:
                singles = components[..., :self.num_experts]
                pairs = components[..., self.num_experts:]
                costs = (
                    singles @ self.subset_membership.transpose(0, 1)
                    + pairs @ self.subset_pair_membership.transpose(0, 1)
                )
            return -costs
        if hasattr(self, "subset_router"):
            return self.subset_router(features)
        # During paired exact-oracle evaluation a pairwise router is temporarily
        # relabeled as oracle-subset.  The exact oracle replaces these scores
        # later in forward, so a zero placeholder is sufficient here.
        return torch.zeros(
            *hidden_states.shape[:-1], self.subset_membership.shape[0],
            device=hidden_states.device, dtype=hidden_states.dtype,
        )

    def _single_token_route(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Optionally fuse the standard router and hard top-k for decode."""
        if (
            self.training
            or hidden_states.shape[-2] != 1
            or hidden_states.device.type != "cuda"
            or hidden_states.dtype != torch.float32
        ):
            return None
        if self.route_source == "router":
            if self.single_token_router_backend != "cuda-fused":
                return None
            router = self.router
        elif self.route_source == "subset-router":
            if self.single_token_router_backend != "cuda-fused-subset":
                return None
            router = self.subset_router
        else:
            return None
        if not isinstance(router, torch.nn.Sequential) or len(router) != 3:
            raise ValueError("cuda-fused router requires Linear-SiLU-Linear")

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        if self.route_source == "router":
            from neural_engine.qwen_router_dispatch import fused_router

            top_ids, weights = fused_router(
                flat_hidden,
                router[0].weight,
                router[0].bias,
                router[2].weight,
                router[2].bias,
                self.active_experts,
                self.temperature,
            )
        else:
            from neural_engine.qwen_router_dispatch import fused_subset_router

            top_ids, weights = fused_subset_router(
                flat_hidden,
                router[0].weight,
                router[0].bias,
                router[2].weight,
                router[2].bias,
                self.subset_membership,
                self.active_experts,
            )
        return (
            top_ids.reshape(*hidden_states.shape[:-1], self.active_experts),
            weights.reshape(*hidden_states.shape[:-1], self.active_experts),
        )

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        top_ids: torch.Tensor,
        weights: torch.Tensor,
        fused_projections: bool = False,
        cache_pair_metadata: bool = False,
        prepacked_weights: bool = False,
        tiled_projections: bool = False,
        uninitialized_pack: bool = False,
        atomic_pack: bool = False,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_ids = top_ids.reshape(-1, self.active_experts)
        if self.single_token_fast_path and hidden_states.shape[-2] == 1:
            return self._forward_single_token(
                hidden_states, flat_hidden, flat_ids,
                weights.reshape(-1, self.active_experts),
                fused_projections=fused_projections,
            )
        flat_weights = weights.reshape(-1, self.active_experts)
        with record_function("neural_engine.grouped.metadata"):
            if cache_pair_metadata:
                metadata_key = (
                    int(flat_ids.shape[0]), self.active_experts, flat_ids.device,
                )
                metadata = self._grouped_pair_metadata_cache.get(metadata_key)
                if metadata is None:
                    pair_indices = torch.arange(
                        flat_ids.numel(), device=flat_ids.device,
                    )
                    token_ids = pair_indices // self.active_experts
                    slots = pair_indices % self.active_experts
                    metadata = (pair_indices, token_ids, slots)
                    self._grouped_pair_metadata_cache[metadata_key] = metadata
                pair_indices, token_ids, slots = metadata
            else:
                pair_indices = torch.arange(
                    flat_ids.numel(), device=flat_ids.device,
                )
                token_ids = pair_indices // self.active_experts
                slots = pair_indices % self.active_experts
        with record_function("neural_engine.grouped.pack"):
            expert_ids = flat_ids[token_ids, slots]
            if atomic_pack:
                from neural_engine.qwen_atomic_pack import atomic_pack as pack_rows

                grouped_hidden, grouped_indices = pack_rows(
                    flat_hidden.contiguous(),
                    flat_ids.contiguous(),
                    self.num_experts,
                )
                max_count = flat_hidden.shape[0]
                sorted_experts = expert_ids
                sorted_token_ids = token_ids
                sorted_slots = slots
            else:
                sort_order = torch.argsort(expert_ids, stable=True)
                sorted_experts = expert_ids[sort_order]
                # ``bincount`` plus a host scalar read makes the grouped path
                # unusable inside CUDA Graph capture.  A token can contribute at most
                # once to each expert, so the number of flattened tokens is a safe
                # graph-stable upper bound.  Keep the tighter dynamic bound on eager
                # paths to avoid inflating prefill workspace.
                counts = torch.zeros(
                    self.num_experts, device=sorted_experts.device, dtype=torch.long,
                )
                counts.scatter_add_(
                    0, sorted_experts,
                    torch.ones_like(sorted_experts, dtype=torch.long),
                )
                if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                    max_count = flat_hidden.shape[0]
                else:
                    max_count = int(counts.max().item())
                starts = counts.cumsum(dim=0) - counts
                positions = torch.arange(
                    pair_indices.numel(), device=flat_ids.device,
                ) - starts[sorted_experts]
                grouped_indices = sorted_experts * max_count + positions
                grouped_hidden = (torch.empty if uninitialized_pack else torch.zeros)(
                    self.num_experts * max_count,
                    flat_hidden.shape[-1],
                    device=flat_hidden.device,
                    dtype=flat_hidden.dtype,
                )
                grouped_hidden.index_copy_(
                    0, grouped_indices, flat_hidden[token_ids[sort_order]],
                )
                grouped_hidden = grouped_hidden.reshape(
                    self.num_experts, max_count, flat_hidden.shape[-1],
                )
        with record_function("neural_engine.grouped.projections"):
            if tiled_projections:
                (
                    gate_weight,
                    value_weight,
                    output_weight,
                    _gate_value_weight,
                ) = self._get_grouped_prepacked_weights()
                from neural_engine.qwen_tiled_dispatch import tiled_grouped_dispatch

                grouped_output = tiled_grouped_dispatch(
                    grouped_hidden, gate_weight, value_weight, output_weight,
                )
            elif prepacked_weights:
                (
                    gate_weight,
                    value_weight,
                    output_weight,
                    gate_value_weight,
                ) = self._get_grouped_prepacked_weights()
            else:
                gate_weight = self.group_gate_weight.transpose(1, 2)
                value_weight = self.group_value_weight.transpose(1, 2)
                output_weight = (
                    self.grouped_effective_output_weight
                    if self.grouped_effective_output_weight is not None
                    else self.group_output_weight
                ).transpose(1, 2)
                gate_value_weight = self.group_gate_value_weight.transpose(1, 2)
            if not tiled_projections:
                if fused_projections:
                    group_size = self.group_gate_weight.shape[1]
                    grouped_gate_value = torch.bmm(
                        grouped_hidden, gate_value_weight,
                    )
                    grouped_gate = F.silu(grouped_gate_value[..., :group_size])
                    grouped_value = grouped_gate_value[..., group_size:]
                else:
                    grouped_gate = F.silu(torch.bmm(
                        grouped_hidden, gate_weight,
                    ))
                    grouped_value = torch.bmm(
                        grouped_hidden, value_weight,
                    )
                grouped_output = torch.bmm(
                    grouped_gate * grouped_value,
                    output_weight,
                )
        with record_function("neural_engine.grouped.select_correction"):
            selected_output = grouped_output.reshape(
                self.num_experts * max_count, flat_hidden.shape[-1],
            ).index_select(0, grouped_indices)
            if not atomic_pack:
                sorted_token_ids = token_ids[sort_order]
                sorted_slots = slots[sort_order]
            fused_correction = self.grouped_fused_correction
            if self.grouped_effective_output_weight is not None:
                # The effective output projection already includes the exact
                # per-expert low-rank correction. Accumulation can consume the
                # expert-major selected rows directly; no pair reorder or
                # second correction contraction is needed.
                self.last_selected_outputs = None
            elif fused_correction is None:
                selected_by_pair = torch.empty_like(selected_output)
                pair_slots = sorted_token_ids * self.active_experts + sorted_slots
                selected_by_pair.index_copy_(0, pair_slots, selected_output)
                self.last_selected_outputs = selected_by_pair.reshape(
                    flat_hidden.shape[0], self.active_experts, flat_hidden.shape[-1],
                ).reshape(
                    *hidden_states.shape[:-1], self.active_experts,
                    hidden_states.shape[-1],
                )
            else:
                # Apply the correction while the selected output is still in the
                # expert-major order produced by grouped dispatch.  This preserves
                # the selected-only contract and avoids the pair reorder plus
                # a second wrapper-side correction pass.
                mix_in, mix_out = fused_correction
                selected_mix_in = mix_in[sorted_experts]
                selected_mix_out = mix_out[sorted_experts]
                latent = torch.einsum(
                    "ph,prh->pr", selected_output, selected_mix_in,
                )
                selected_output = selected_output + torch.einsum(
                    "pr,phr->ph", latent, selected_mix_out,
                )
                self.last_selected_outputs = None
        with record_function("neural_engine.grouped.accumulate"):
            if self.grouped_uniform_accum and self.route_source == "subset-router":
                # The hard subset router assigns the same top-k score to every
                # member of the chosen subset.  Its softmax is therefore exactly
                # 1/K, and the accepted K=5 contract uses hard_route_scale=K.
                # Keep this shortcut opt-in because other route sources can have
                # non-uniform weights.
                contribution = selected_output
                accumulation_scale = self.hard_route_scale / self.active_experts
            else:
                contribution = selected_output * flat_weights[
                    sorted_token_ids, sorted_slots,
                ].unsqueeze(-1)
                accumulation_scale = self.hard_route_scale
            flat_output = torch.zeros_like(flat_hidden)
            flat_output.index_add_(0, sorted_token_ids, contribution)
            self.last_active_expert_fraction = pair_indices.numel() / max(
                flat_hidden.shape[0] * self.num_experts, 1
            )
        # The router selects contribution-heavy groups rather than a random
        # subset, so the empirical stable scale is E/K, not an unbiased E
        # estimator that over-corrects the selected high-energy groups.
        return accumulation_scale * flat_output.reshape_as(
            hidden_states,
        )

    def _get_grouped_prepacked_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return contiguous BMM operands for the opt-in grouped probe."""
        output_weight = (
            self.grouped_effective_output_weight
            if self.grouped_effective_output_weight is not None
            else self.group_output_weight
        )
        weights = (
            self.group_gate_weight,
            self.group_value_weight,
            output_weight,
            self.group_gate_value_weight,
        )
        key = tuple(
            item
            for weight in weights
            for item in (
                weight.device,
                weight.dtype,
                id(weight),
                None if weight.is_inference() else weight._version,
            )
        )
        if (
            self._grouped_prepacked_weights is None
            or self._grouped_prepacked_weights_key != key
        ):
            self._grouped_prepacked_weights = tuple(
                weight.transpose(1, 2).contiguous()
                for weight in weights
            )
            self._grouped_prepacked_weights_key = key
        return self._grouped_prepacked_weights

    def _forward_single_token(
        self,
        hidden_states: torch.Tensor,
        flat_hidden: torch.Tensor,
        flat_ids: torch.Tensor,
        flat_weights: torch.Tensor,
        *,
        fused_projections: bool = False,
    ) -> torch.Tensor:
        """Avoid sort/bincount/scatter for a single-token decode shape.

        The regular grouped path is optimized for many tokens per expert.  At
        one sequence token, its packing work costs more than the selected
        projections, including when the batch has multiple rows. Gathering
        only the K selected groups keeps the same hard route and output
        contract while using three small batched contractions.
        """
        if self.single_token_full_correction is not None:
            mix_in, mix_out, dispatch_backend = self.single_token_full_correction
            if dispatch_backend == "cuda-fused-token":
                from neural_engine.qwen_full_correction_dispatch import (
                    fused_token_correction_dispatch as dispatch,
                )
            else:
                from neural_engine.qwen_full_correction_dispatch import (
                    fused_correction_dispatch as dispatch,
                )
            selected, combined = dispatch(
                flat_hidden.contiguous(),
                flat_ids.contiguous(),
                flat_weights.contiguous(),
                self.group_gate_weight.contiguous(),
                self.group_value_weight.contiguous(),
                self.group_output_weight.contiguous(),
                mix_in,
                mix_out,
                self.hard_route_scale,
            )
            self.last_selected_outputs = selected.reshape(
                *hidden_states.shape[:-1], self.active_experts,
                hidden_states.shape[-1],
            )
            self.last_active_expert_fraction = flat_ids.numel() / max(
                flat_hidden.shape[0] * self.num_experts, 1
            )
            return combined.reshape_as(hidden_states)
        selected_gate = self.group_gate_weight[flat_ids]
        selected_value = self.group_value_weight[flat_ids]
        output_weight_bank = (
            self.group_output_weight
            if self.single_token_output_weight is None
            else self.single_token_output_weight
        )
        selected_output_weight = output_weight_bank[flat_ids]
        if fused_projections:
            selected_gate_value = self.group_gate_value_weight[flat_ids]
            if self.single_token_projection_backend == "bmm":
                pair_hidden = flat_hidden.unsqueeze(1).expand(
                    -1, self.active_experts, -1,
                ).reshape(-1, 1, flat_hidden.shape[-1])
                projected = torch.bmm(
                    pair_hidden,
                    selected_gate_value.reshape(
                        -1, selected_gate_value.shape[-2],
                        selected_gate_value.shape[-1],
                    ).transpose(1, 2),
                ).reshape(flat_hidden.shape[0], self.active_experts, -1)
            else:
                projected = torch.einsum(
                    "nh,nqgh->nqg", flat_hidden, selected_gate_value,
                )
            group_size = self.group_gate_weight.shape[1]
            gate = F.silu(projected[..., :group_size])
            value = projected[..., group_size:]
        else:
            if self.single_token_projection_backend == "bmm":
                pair_hidden = flat_hidden.unsqueeze(1).expand(
                    -1, self.active_experts, -1,
                ).reshape(-1, 1, flat_hidden.shape[-1])
                pair_gate = selected_gate.reshape(
                    -1, selected_gate.shape[-2], selected_gate.shape[-1],
                ).transpose(1, 2)
                pair_value = selected_value.reshape(
                    -1, selected_value.shape[-2], selected_value.shape[-1],
                ).transpose(1, 2)
                gate = F.silu(torch.bmm(pair_hidden, pair_gate)).reshape(
                    flat_hidden.shape[0], self.active_experts, -1,
                )
                value = torch.bmm(pair_hidden, pair_value).reshape(
                    flat_hidden.shape[0], self.active_experts, -1,
                )
            else:
                gate = F.silu(torch.einsum(
                    "nh,nqgh->nqg", flat_hidden, selected_gate,
                ))
                value = torch.einsum(
                    "nh,nqgh->nqg", flat_hidden, selected_value,
                )
        coefficient = gate * value
        if self.single_token_projection_backend == "bmm":
            selected = torch.bmm(
                coefficient.reshape(-1, 1, coefficient.shape[-1]),
                selected_output_weight.reshape(
                    -1, selected_output_weight.shape[-2],
                    selected_output_weight.shape[-1],
                ).transpose(1, 2),
            ).reshape(flat_hidden.shape[0], self.active_experts, -1)
        else:
            selected = torch.einsum(
                "nqg,nqhg->nqh", coefficient, selected_output_weight,
            )
        self.last_selected_outputs = selected.reshape(
            *hidden_states.shape[:-1], self.active_experts, hidden_states.shape[-1],
        )
        self.last_active_expert_fraction = flat_ids.numel() / max(
            flat_hidden.shape[0] * self.num_experts, 1
        )
        contribution = (selected * flat_weights.unsqueeze(-1)).sum(dim=1)
        return self.hard_route_scale * contribution.reshape_as(hidden_states)

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        top_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run selected token/group pairs with gathered batched matmuls.

        This keeps the selected-token-only contract while avoiding a massive
        per-token weight gather. It is a PyTorch fallback for environments
        without a fused Triton/CUDA kernel; timing is measured separately from
        grouped dispatch because the best layout is hardware-dependent.
        """
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_ids = top_ids.reshape(-1, self.active_experts)
        flat_weights = weights.reshape(-1, self.active_experts)
        selected_outputs = torch.empty(
            flat_ids.numel(), flat_hidden.shape[-1],
            device=flat_hidden.device, dtype=flat_hidden.dtype,
        )
        flat_output = torch.zeros_like(flat_hidden)
        pair_indices = torch.arange(flat_ids.numel(), device=flat_ids.device)
        token_ids = pair_indices // self.active_experts
        slots = pair_indices % self.active_experts
        expert_ids = flat_ids[token_ids, slots]
        for expert_id in range(self.num_experts):
            expert_token_ids = torch.where(expert_ids == expert_id)[0]
            if expert_token_ids.numel() == 0:
                continue
            selected_hidden = flat_hidden[token_ids[expert_token_ids]]
            coefficient = F.silu(F.linear(
                selected_hidden, self.group_gate_weight[expert_id],
            )) * F.linear(
                selected_hidden, self.group_value_weight[expert_id],
            )
            selected_output = F.linear(
                coefficient, self.group_output_weight[expert_id],
            )
            pair_ids = token_ids[expert_token_ids] * self.active_experts + slots[expert_token_ids]
            selected_outputs.index_copy_(0, pair_ids, selected_output)
            contribution = selected_output * flat_weights[
                token_ids[expert_token_ids], slots[expert_token_ids],
            ].unsqueeze(-1)
            flat_output.index_add_(0, token_ids[expert_token_ids], contribution)
        self.last_selected_outputs = selected_outputs.reshape(
            flat_hidden.shape[0], self.active_experts, flat_hidden.shape[-1],
        ).reshape(
            *hidden_states.shape[:-1], self.active_experts, hidden_states.shape[-1],
        )
        self.last_active_expert_fraction = token_ids.numel() / max(
            flat_hidden.shape[0] * self.num_experts, 1
        )
        return self.hard_route_scale * flat_output.reshape_as(hidden_states)

    def _forward_fused(
        self,
        hidden_states: torch.Tensor,
        top_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run selected groups through the optional single-launch CUDA kernel."""
        from neural_engine.qwen_fused_dispatch import fused_dispatch

        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        flat_ids = top_ids.reshape(-1, self.active_experts).contiguous()
        flat_weights = weights.reshape(-1, self.active_experts).contiguous()
        selected_outputs, flat_output = fused_dispatch(
            flat_hidden,
            flat_ids,
            flat_weights,
            self.group_gate_weight.contiguous(),
            self.group_value_weight.contiguous(),
            self.group_output_weight.contiguous(),
            self.hard_route_scale,
        )
        self.last_selected_outputs = selected_outputs.reshape(
            *hidden_states.shape[:-1], self.active_experts, hidden_states.shape[-1],
        )
        self.last_active_expert_fraction = flat_ids.numel() / max(
            flat_hidden.shape[0] * self.num_experts, 1
        )
        return flat_output.reshape_as(hidden_states)

    def _forward_packed_fused(
        self,
        hidden_states: torch.Tensor,
        top_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Packed dispatch with one gate+value GEMM per selected expert."""
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_ids = top_ids.reshape(-1, self.active_experts)
        flat_weights = weights.reshape(-1, self.active_experts)
        selected_outputs = torch.empty(
            flat_ids.numel(), flat_hidden.shape[-1],
            device=flat_hidden.device, dtype=flat_hidden.dtype,
        )
        flat_output = torch.zeros_like(flat_hidden)
        pair_indices = torch.arange(flat_ids.numel(), device=flat_ids.device)
        token_ids = pair_indices // self.active_experts
        slots = pair_indices % self.active_experts
        expert_ids = flat_ids[token_ids, slots]
        group_size = self.group_gate_weight.shape[1]
        for expert_id in range(self.num_experts):
            expert_token_ids = torch.where(expert_ids == expert_id)[0]
            if expert_token_ids.numel() == 0:
                continue
            selected_hidden = flat_hidden[token_ids[expert_token_ids]]
            gate_value = F.linear(
                selected_hidden, self.group_gate_value_weight[expert_id],
            )
            coefficient = (
                F.silu(gate_value[..., :group_size])
                * gate_value[..., group_size:]
            )
            selected_output = F.linear(
                coefficient, self.group_output_weight[expert_id],
            )
            pair_ids = token_ids[expert_token_ids] * self.active_experts + slots[
                expert_token_ids
            ]
            selected_outputs.index_copy_(0, pair_ids, selected_output)
            contribution = selected_output * flat_weights[
                token_ids[expert_token_ids], slots[expert_token_ids],
            ].unsqueeze(-1)
            flat_output.index_add_(0, token_ids[expert_token_ids], contribution)
        self.last_selected_outputs = selected_outputs.reshape(
            flat_hidden.shape[0], self.active_experts, flat_hidden.shape[-1],
        ).reshape(
            *hidden_states.shape[:-1], self.active_experts, hidden_states.shape[-1],
        )
        self.last_active_expert_fraction = token_ids.numel() / max(
            flat_hidden.shape[0] * self.num_experts, 1
        )
        return self.hard_route_scale * flat_output.reshape_as(hidden_states)

    def _forward_packed_fp16(
        self,
        hidden_states: torch.Tensor,
        top_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Packed dispatch using cached FP16 weights for Tensor Core inference."""
        if hidden_states.dtype != torch.float32 or hidden_states.device.type != "cuda":
            raise ValueError("packed-fp16 dispatch currently requires float32 CUDA input")
        if (
            self._fp16_dispatch_weights is None
            or self._fp16_dispatch_weights[0].device != hidden_states.device
        ):
            self._fp16_dispatch_weights = (
                self.group_gate_value_weight.to(dtype=torch.float16),
                self.group_output_weight.to(dtype=torch.float16),
            )
        gate_value_weight, output_weight = self._fp16_dispatch_weights
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_ids = top_ids.reshape(-1, self.active_experts)
        flat_weights = weights.reshape(-1, self.active_experts)
        selected_outputs = torch.empty(
            flat_ids.numel(), flat_hidden.shape[-1],
            device=flat_hidden.device, dtype=hidden_states.dtype,
        )
        flat_output = torch.zeros_like(flat_hidden)
        pair_indices = torch.arange(flat_ids.numel(), device=flat_ids.device)
        token_ids = pair_indices // self.active_experts
        slots = pair_indices % self.active_experts
        expert_ids = flat_ids[token_ids, slots]
        group_size = self.group_gate_weight.shape[1]
        for expert_id in range(self.num_experts):
            expert_token_ids = torch.where(expert_ids == expert_id)[0]
            if expert_token_ids.numel() == 0:
                continue
            selected_hidden = flat_hidden[token_ids[expert_token_ids]].half()
            gate_value = F.linear(selected_hidden, gate_value_weight[expert_id])
            coefficient = (
                F.silu(gate_value[..., :group_size])
                * gate_value[..., group_size:]
            )
            selected_output = F.linear(
                coefficient, output_weight[expert_id],
            ).float()
            pair_ids = token_ids[expert_token_ids] * self.active_experts + slots[
                expert_token_ids
            ]
            selected_outputs.index_copy_(0, pair_ids, selected_output)
            contribution = selected_output * flat_weights[
                token_ids[expert_token_ids], slots[expert_token_ids],
            ].unsqueeze(-1)
            flat_output.index_add_(0, token_ids[expert_token_ids], contribution)
        self.last_selected_outputs = selected_outputs.reshape(
            flat_hidden.shape[0], self.active_experts, flat_hidden.shape[-1],
        ).reshape(
            *hidden_states.shape[:-1], self.active_experts, hidden_states.shape[-1],
        )
        self.last_active_expert_fraction = token_ids.numel() / max(
            flat_hidden.shape[0] * self.num_experts, 1
        )
        return self.hard_route_scale * flat_output.reshape_as(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        soft_output = None
        hard_blend = 1.0
        fused_route = self._single_token_route(hidden_states)
        if self.route_source in {
            "subset-router", "oracle-subset", "pairwise-cost-router",
        }:
            if fused_route is None:
                subset_scores = self._subset_scores(hidden_states)
                if self.training:
                    hard_blend = float(self.hard_train_blend)
                if self.training and (not self.hard_train or hard_blend < 1.0):
                    scores = subset_scores @ self.subset_membership
                    outputs = torch.stack([
                        expert(hidden_states) for expert in self.experts
                    ], dim=-2)
                    weights = F.softmax(scores / self.temperature, dim=-1)
                    soft_output = self.num_experts * (
                        outputs * weights.unsqueeze(-1)
                    ).sum(dim=-2)
                    if not self.hard_train or hard_blend <= 0.0:
                        self.last_selected = scores.detach().argmax(dim=-1)
                        self.last_route_weights = weights
                        self.last_all_outputs = outputs
                        self.last_selected_outputs = None
                        self.last_active_expert_fraction = 1.0
                        return soft_output
                    # Continue below with the hard top-k path and blend its
                    # output with the soft operator during the transition.
                    hard_scores = subset_scores.argmax(dim=-1)
                    selected_membership = self.subset_membership[hard_scores]
                    scores = torch.where(
                        selected_membership.bool(),
                        torch.ones_like(selected_membership),
                        -torch.ones_like(selected_membership),
                    )
                else:
                    best_subset = subset_scores.argmax(dim=-1)
                    selected_membership = self.subset_membership[best_subset]
                    scores = torch.where(
                        selected_membership.bool(),
                        torch.ones_like(selected_membership),
                        -torch.ones_like(selected_membership),
                    )
            else:
                scores = None
        else:
            scores = None if fused_route is not None else self.router(hidden_states)
        oracle_outputs = None
        if not self.training and self.route_source in {
            "oracle-dot", "oracle-energy", "oracle-subset",
        }:
            oracle_outputs = torch.stack([
                expert(hidden_states) for expert in self.experts
            ], dim=-2)
            if self.route_source == "oracle-dot":
                full_output = oracle_outputs.sum(dim=-2)
                scores = (oracle_outputs * full_output.unsqueeze(-2)).sum(dim=-1)
            elif self.route_source == "oracle-energy":
                scores = oracle_outputs.square().mean(dim=-1)
            else:
                flat_outputs = oracle_outputs.float().reshape(
                    -1, self.num_experts, hidden_states.shape[-1],
                )
                flat_target = flat_outputs.sum(dim=1)
                gram = torch.einsum(
                    "neh,nfh->nef", flat_outputs, flat_outputs,
                )
                cross = torch.einsum(
                    "neh,nh->ne", flat_outputs, flat_target,
                )
                subset_ids = torch.tensor(
                    list(combinations(
                        range(self.num_experts), self.active_experts,
                    )), device=hidden_states.device, dtype=torch.long,
                )
                membership = F.one_hot(
                    subset_ids, num_classes=self.num_experts,
                ).sum(dim=1).float()
                pair_energy = torch.einsum(
                    "ce,nef,cf->nc", membership, gram, membership,
                )
                cross_energy = cross @ membership.transpose(0, 1)
                coefficient = self.hard_route_scale / self.active_experts
                errors = (
                    coefficient * coefficient * pair_energy
                    - 2.0 * coefficient * cross_energy
                    + flat_target.square().sum(dim=-1, keepdim=True)
                )
                best_subset = subset_ids[errors.argmin(dim=-1)]
                scores = torch.full(
                    (*hidden_states.shape[:-1], self.num_experts),
                    -1.0, device=hidden_states.device, dtype=hidden_states.dtype,
                )
                scores.scatter_(-1, best_subset.reshape(
                    *hidden_states.shape[:-1], self.active_experts,
                ), 1.0)
        if (
            self.training
            and not self.hard_train
            and self.route_source not in {
                "subset-router", "oracle-subset", "pairwise-cost-router",
            }
        ):
            outputs = torch.stack([
                expert(hidden_states) for expert in self.experts
            ], dim=-2)
            weights = F.softmax(scores / self.temperature, dim=-1)
            self.last_selected = scores.detach().argmax(dim=-1)
            self.last_route_weights = weights
            self.last_all_outputs = outputs
            self.last_selected_outputs = None
            self.last_active_expert_fraction = 1.0
            # At uniform routing, this exactly reconstructs the sum of slices.
            return self.num_experts * (outputs * weights.unsqueeze(-1)).sum(dim=-2)
        if fused_route is None:
            top_values, top_ids = scores.topk(self.active_experts, dim=-1)
            weights = F.softmax(top_values / self.temperature, dim=-1)
        else:
            top_ids, weights = fused_route
        self.last_selected = top_ids.detach()
        self.last_route_weights = weights
        self.last_all_outputs = None
        if oracle_outputs is not None:
            selected = torch.gather(
                oracle_outputs, -2,
                top_ids.unsqueeze(-1).expand(*top_ids.shape, hidden_states.shape[-1]),
            )
            self.last_selected_outputs = selected
            return self.hard_route_scale * (
                selected * weights.unsqueeze(-1)
            ).sum(dim=-2)
        if not self.training and self.dispatch_mode in {
            "grouped", "grouped-cached", "grouped-prepacked",
        }:
            return self._forward_grouped(
                hidden_states, top_ids, weights,
                cache_pair_metadata=self.dispatch_mode == "grouped-cached",
                prepacked_weights=self.dispatch_mode == "grouped-prepacked",
            )
        if not self.training and self.dispatch_mode == "grouped-optimized":
            return self._forward_grouped(
                hidden_states, top_ids, weights,
                fused_projections=True,
                cache_pair_metadata=True,
                prepacked_weights=True,
            )
        if not self.training and self.dispatch_mode in {
            "grouped-adaptive", "grouped-adaptive-nozero",
            "grouped-adaptive-effective-output",
            "grouped-adaptive-atomic-pack",
            "grouped-adaptive-atomic-effective-output",
        }:
            # Decode B=1 is launch/metadata bound; the extra cached layouts
            # only pay off once several rows can share the grouped work.
            use_optimized = (
                hidden_states.reshape(-1, hidden_states.shape[-1]).shape[0] > 1
            )
            return self._forward_grouped(
                hidden_states, top_ids, weights,
                fused_projections=use_optimized,
                cache_pair_metadata=use_optimized,
                prepacked_weights=use_optimized,
                uninitialized_pack=self.dispatch_mode == "grouped-adaptive-nozero",
                atomic_pack=self.dispatch_mode in {
                    "grouped-adaptive-atomic-pack",
                    "grouped-adaptive-atomic-effective-output",
                },
            )
        if not self.training and self.dispatch_mode in {
            "grouped-fused", "grouped-prepacked-fused",
        }:
            return self._forward_grouped(
                hidden_states, top_ids, weights,
                fused_projections=True,
                prepacked_weights=self.dispatch_mode == "grouped-prepacked-fused",
            )
        if not self.training and self.dispatch_mode == "grouped-tiled":
            return self._forward_grouped(
                hidden_states, top_ids, weights, tiled_projections=True,
            )
        if not self.training and self.dispatch_mode == "packed":
            return self._forward_packed(hidden_states, top_ids, weights)
        if not self.training and self.dispatch_mode == "packed-fused":
            return self._forward_packed_fused(hidden_states, top_ids, weights)
        if not self.training and self.dispatch_mode == "packed-fp16":
            return self._forward_packed_fp16(hidden_states, top_ids, weights)
        if not self.training and self.dispatch_mode == "fused":
            return self._forward_fused(hidden_states, top_ids, weights)
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_ids = top_ids.reshape(-1, self.active_experts)
        flat_weights = weights.reshape(-1, self.active_experts)
        flat_output = torch.zeros_like(flat_hidden)
        selected_outputs = torch.empty(
            flat_ids.numel(), flat_hidden.shape[-1],
            device=flat_hidden.device, dtype=flat_hidden.dtype,
        )
        selected_pairs = 0
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(flat_ids == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_output = expert(flat_hidden[token_ids])
            selected_outputs.index_copy_(
                0, token_ids * self.active_experts + slots, expert_output,
            )
            contribution = expert_output * flat_weights[token_ids, slots].unsqueeze(-1)
            flat_output.index_add_(0, token_ids, contribution)
            selected_pairs += int(token_ids.numel())
        self.last_selected_outputs = selected_outputs.reshape(
            flat_hidden.shape[0], self.active_experts, flat_hidden.shape[-1],
        ).reshape(*hidden_states.shape[:-1], self.active_experts, hidden_states.shape[-1])
        self.last_active_expert_fraction = selected_pairs / max(
            flat_hidden.shape[0] * self.num_experts, 1
        )
        # Keep token-loop hard training numerically identical to grouped
        # inference.  The scale is an explicit experiment parameter; using a
        # second implicit E/K rule here makes the correction learn one
        # operator and evaluation execute another.
        hard_output = self.hard_route_scale * flat_output.reshape_as(hidden_states)
        if soft_output is not None:
            return (1.0 - hard_blend) * soft_output + hard_blend * hard_output
        return hard_output


class SignedSubsetReconstructionQwenChild(torch.nn.Module):
    """Reconstruct selected group outputs with teacher-fitted signed weights.

    The base routed child uses a fixed ``E/K`` scale and a softmax over the
    selected groups.  That is unbiased only under a restrictive sampling
    assumption and cannot express cancellation between correlated group
    outputs.  This wrapper fits one small K-dimensional signed coefficient
    vector for every subset.  The hard path still evaluates only the selected
    groups; the change is solely in their reconstruction contract.
    """

    def __init__(self, base: TransferredRoutedQwenChild) -> None:
        super().__init__()
        if base.active_experts >= base.num_experts:
            raise ValueError("signed subset reconstruction requires sparse routing")
        if base.route_source not in {
            "subset-router", "oracle-subset", "pairwise-cost-router",
        }:
            raise ValueError(
                "signed subset reconstruction requires a subset-capable router"
            )
        self.base = base
        subset_ids = torch.tensor(
            list(combinations(range(base.num_experts), base.active_experts)),
            device=base.group_gate_weight.device,
            dtype=torch.long,
        )
        self.register_buffer("subset_ids", subset_ids, persistent=False)
        self.coefficients = torch.nn.Parameter(
            torch.zeros(subset_ids.shape[0], base.num_experts),
        )
        default = base.hard_route_scale / max(base.active_experts, 1)
        with torch.no_grad():
            self.coefficients.copy_(
                base.subset_membership.to(self.coefficients.dtype) * default
            )
        self.fit_ridge = 1e-3
        self.last_subset_index: torch.Tensor | None = None

    def _subset_index(self, selected: torch.Tensor) -> torch.Tensor:
        membership = F.one_hot(
            selected, num_classes=self.base.num_experts,
        ).sum(dim=-2).to(dtype=self.base.subset_membership.dtype)
        matches = membership.unsqueeze(-2).eq(
            self.base.subset_membership
        ).all(dim=-1)
        if not matches.any(dim=-1).all():
            raise RuntimeError("selected groups do not form a known subset")
        return matches.to(dtype=torch.float32).argmax(dim=-1)

    def _apply_selected(
        self,
        selected_outputs: torch.Tensor,
        selected: torch.Tensor,
        subset_index: torch.Tensor,
    ) -> torch.Tensor:
        subset_coefficients = self.coefficients[subset_index]
        selected_coefficients = torch.gather(
            subset_coefficients, -1, selected,
        )
        return (
            selected_outputs * selected_coefficients.unsqueeze(-1)
        ).sum(dim=-2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The exact oracle must score the *new* reconstruction contract.  It
        # intentionally evaluates the full bank, so this is an upper-bound
        # diagnostic and is never used as a deployment path.
        if not self.training and self.base.route_source == "oracle-subset":
            outputs = torch.stack([
                expert(hidden_states) for expert in self.base.experts
            ], dim=-2)
            flat_outputs = outputs.reshape(-1, self.base.num_experts, outputs.shape[-1])
            candidate_outputs = torch.einsum(
                "neh,ce->nch", flat_outputs, self.coefficients,
            )
            target = flat_outputs.sum(dim=1)
            errors = (candidate_outputs - target.unsqueeze(1)).square().mean(dim=-1)
            best_subset_index = errors.argmin(dim=-1)
            selected = self.subset_ids[best_subset_index]
            selected_output = candidate_outputs[
                torch.arange(flat_outputs.shape[0], device=hidden_states.device),
                best_subset_index,
            ]
            self.last_subset_index = best_subset_index.reshape(
                *hidden_states.shape[:-1],
            )
            self.base.last_selected = selected.reshape(
                *hidden_states.shape[:-1], self.base.active_experts,
            )
            self.base.last_route_weights = torch.full(
                (*hidden_states.shape[:-1], self.base.active_experts),
                1.0 / self.base.active_experts,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            self.base.last_all_outputs = None
            self.base.last_selected_outputs = torch.gather(
                outputs, -2,
                selected.reshape(
                    *hidden_states.shape[:-1], self.base.active_experts, 1,
                ).expand(*hidden_states.shape[:-1], self.base.active_experts, outputs.shape[-1]),
            )
            self.base.last_active_expert_fraction = (
                self.base.active_experts / self.base.num_experts
            )
            return selected_output.reshape_as(hidden_states)

        base_output = self.base(hidden_states)
        selected_outputs = self.base.last_selected_outputs
        selected = self.base.last_selected
        if selected_outputs is None or selected is None:
            # During the soft phase the base has no selected hard outputs.  The
            # wrapper is deliberately identity there; fitting is an offline
            # calibration step and future joint training can opt into hard mode.
            return base_output
        subset_index = self._subset_index(selected)
        self.last_subset_index = subset_index
        return self._apply_selected(selected_outputs, selected, subset_index)


class TransferredRoutedQwenNeuronChild(torch.nn.Module):
    """Copied Qwen neurons with token-level top-k execution."""

    def __init__(
        self,
        parent: torch.nn.Module,
        active_neurons: int,
        temperature: float,
        token_chunk_size: int = 64,
        route_source: str = "router",
        hard_route_scale: float | None = None,
    ) -> None:
        super().__init__()
        inner_size, hidden_size = parent.gate_proj.weight.shape
        if not 1 <= active_neurons <= inner_size:
            raise ValueError("active_neurons must be within the Qwen intermediate size")
        if parent.gate_proj.bias is not None or parent.up_proj.bias is not None:
            raise ValueError("Qwen transfer currently expects bias-free projections")
        self.num_neurons = int(inner_size)
        self.active_neurons = int(active_neurons)
        self.temperature = float(temperature)
        if route_source not in {"router", "oracle-dot", "oracle-energy"}:
            raise ValueError(
                "neuron route_source must be router, oracle-dot, or oracle-energy"
            )
        self.route_source = route_source
        self.hard_train = False
        self.token_chunk_size = int(token_chunk_size)
        self.hard_route_scale = (
            (
                self.num_neurons
                if self.active_neurons == self.num_neurons
                else self.num_neurons / self.active_neurons
            )
            if hard_route_scale is None else float(hard_route_scale)
        )
        if self.token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")
        self.register_buffer(
            "gate_weight", parent.gate_proj.weight.detach(), persistent=False,
        )
        self.register_buffer(
            "value_weight", parent.up_proj.weight.detach(), persistent=False,
        )
        self.register_buffer(
            "output_weight", parent.down_proj.weight.detach(), persistent=False,
        )
        self.router = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, inner_size),
        )
        torch.nn.init.zeros_(self.router[-1].weight)
        torch.nn.init.zeros_(self.router[-1].bias)
        self.last_selected: torch.Tensor | None = None
        self.last_active_expert_fraction = 1.0

    def _forward_hard(
        self,
        hidden_states: torch.Tensor,
        top_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_ids = top_ids.reshape(-1, self.active_neurons)
        flat_weights = weights.reshape(-1, self.active_neurons)
        flat_output = torch.empty_like(flat_hidden)
        output_weight = self.output_weight.transpose(0, 1)
        for start in range(0, flat_hidden.shape[0], self.token_chunk_size):
            stop = min(start + self.token_chunk_size, flat_hidden.shape[0])
            chunk_hidden = flat_hidden[start:stop]
            chunk_ids = flat_ids[start:stop]
            selected_gate = self.gate_weight[chunk_ids]
            selected_value = self.value_weight[chunk_ids]
            gate = torch.einsum("nh,nkh->nk", chunk_hidden, selected_gate)
            value = torch.einsum("nh,nkh->nk", chunk_hidden, selected_value)
            coefficients = F.silu(gate) * value * flat_weights[start:stop]
            selected_output = output_weight[chunk_ids]
            flat_output[start:stop] = torch.einsum(
                "nk,nkh->nh", coefficients, selected_output,
            )
        self.last_active_expert_fraction = flat_ids.numel() / max(
            flat_hidden.shape[0] * self.num_neurons, 1
        )
        return self.hard_route_scale * flat_output.reshape_as(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.router(hidden_states)
        if not self.training and self.route_source != "router":
            gate = F.linear(hidden_states, self.gate_weight)
            value = F.linear(hidden_states, self.value_weight)
            coefficient = F.silu(gate) * value
            if self.route_source == "oracle-energy":
                scores = coefficient.float().square() * (
                    self.output_weight.float().square().sum(dim=0)
                )
            else:
                full_output = F.linear(coefficient, self.output_weight)
                contribution_alignment = F.linear(
                    full_output, self.output_weight.transpose(0, 1),
                )
                scores = coefficient.float() * contribution_alignment.float()
        if self.training and not self.hard_train:
            gate = F.linear(hidden_states, self.gate_weight)
            value = F.linear(hidden_states, self.value_weight)
            weights = F.softmax(scores / self.temperature, dim=-1)
            self.last_selected = scores.detach().argmax(dim=-1)
            self.last_active_expert_fraction = 1.0
            return self.num_neurons * F.linear(
                F.silu(gate) * value * weights, self.output_weight,
            )
        top_values, top_ids = scores.topk(self.active_neurons, dim=-1)
        weights = F.softmax(top_values / self.temperature, dim=-1)
        self.last_selected = top_ids.detach()
        return self._forward_hard(hidden_states, top_ids, weights)


class LearnedLatentBasisQwenChild(torch.nn.Module):
    """A learned shared nonlinear basis with sparse output decoders.

    This is intentionally independent of copied Qwen neuron slices.  A small
    shared nonlinear feature vector is decoded by a bank of latent basis
    outputs, and only the routed decoder subset is executed in the hard path.
    """

    def __init__(
        self,
        hidden_size: int,
        num_basis: int,
        active_basis: int,
        rank: int,
        temperature: float,
        hard_route_scale: float | None,
    ) -> None:
        super().__init__()
        if not 1 <= active_basis <= num_basis:
            raise ValueError("active_basis must be within num_basis")
        if hidden_size < 1 or rank < 1:
            raise ValueError("hidden_size and rank must be positive")
        self.hidden_size = int(hidden_size)
        self.num_basis = int(num_basis)
        self.active_basis = int(active_basis)
        self.rank = int(rank)
        self.temperature = float(temperature)
        self.hard_route_scale = (
            self.num_basis / self.active_basis
            if hard_route_scale is None else float(hard_route_scale)
        )
        self.hard_train = False
        self.gate_projection = torch.nn.Linear(hidden_size, rank, bias=False)
        self.value_projection = torch.nn.Linear(hidden_size, rank, bias=False)
        self.basis_output = torch.nn.Parameter(
            torch.empty(num_basis, hidden_size, rank),
        )
        torch.nn.init.normal_(self.gate_projection.weight, std=hidden_size ** -0.5)
        torch.nn.init.normal_(self.value_projection.weight, std=hidden_size ** -0.5)
        torch.nn.init.normal_(self.basis_output, std=rank ** -0.5)
        self.router = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, num_basis),
        )
        torch.nn.init.zeros_(self.router[-1].weight)
        torch.nn.init.zeros_(self.router[-1].bias)
        self.last_selected: torch.Tensor | None = None
        self.last_active_expert_fraction = 1.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        latent = F.silu(self.gate_projection(hidden_states)) * self.value_projection(
            hidden_states,
        )
        scores = self.router(hidden_states)
        if self.training and not self.hard_train:
            basis_outputs = torch.einsum(
                "...r,bhr->...bh", latent, self.basis_output,
            )
            weights = F.softmax(scores / self.temperature, dim=-1)
            self.last_selected = scores.detach().argmax(dim=-1)
            self.last_active_expert_fraction = 1.0
            return self.num_basis * (
                basis_outputs * weights.unsqueeze(-1)
            ).sum(dim=-2)
        top_values, top_ids = scores.topk(self.active_basis, dim=-1)
        weights = F.softmax(top_values / self.temperature, dim=-1)
        self.last_selected = top_ids.detach()
        selected_output = torch.einsum(
            "...r,...khr->...kh", latent, self.basis_output[top_ids],
        )
        self.last_active_expert_fraction = (
            self.active_basis / max(self.num_basis, 1)
        )
        return self.hard_route_scale * (
            selected_output * weights.unsqueeze(-1)
        ).sum(dim=-2)


class SharedBasisRoutedQwenChild(torch.nn.Module):
    """Sparse Qwen groups plus a routed, shared nonlinear correction basis.

    The copied group slices preserve the cheap transferred path.  A compact
    nonlinear basis is evaluated once per token and each selected group mixes
    that basis into the output, allowing selected groups to compensate for
    omitted group interactions without evaluating all parent neurons.
    """

    def __init__(self, base: TransferredRoutedQwenChild, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("shared basis rank must be positive")
        hidden_size = int(base.group_gate_weight.shape[-1])
        self.base = base
        self.shared_gate = torch.nn.Linear(hidden_size, rank, bias=False)
        self.shared_value = torch.nn.Linear(hidden_size, rank, bias=False)
        self.expert_mix = torch.nn.Parameter(
            torch.zeros(base.num_experts, hidden_size, rank),
        )
        self.hard_train = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        route_weights = self.base.last_route_weights
        selected = self.base.last_selected
        if route_weights is None or selected is None:
            raise RuntimeError("base route state was not populated")
        basis = F.silu(self.shared_gate(hidden_states)) * self.shared_value(
            hidden_states,
        )
        expert_corrections = torch.einsum(
            "...r,ehr->...eh", basis, self.expert_mix,
        )
        if self.base.training and not self.base.hard_train:
            correction = (
                expert_corrections * route_weights.unsqueeze(-1)
            ).sum(dim=-2)
            if not self.replace_base_output:
                correction = self.base.num_experts * correction
        else:
            selected_corrections = torch.gather(
                expert_corrections, -2,
                selected.unsqueeze(-1).expand(
                    *selected.shape, hidden_states.shape[-1],
                ),
            )
            correction = (
                selected_corrections * route_weights.unsqueeze(-1)
            ).sum(dim=-2)
            if not self.replace_base_output:
                correction = self.base.hard_route_scale * correction
        return base_output + correction


class CrossGroupOutputMixRoutedQwenChild(torch.nn.Module):
    """Route copied groups and mix each selected output through a low-rank map.

    Unlike an input-only residual cell, this correction is conditioned on the
    actual transferred group output.  It can therefore learn a teacher-derived
    map from a selected group's contribution toward the omitted groups while
    retaining selected-group-only execution in the hard path.
    """

    def __init__(
        self,
        base: TransferredRoutedQwenChild,
        rank: int,
        replace_base_output: bool = False,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("cross-group mix rank must be positive")
        hidden_size = int(base.group_gate_weight.shape[-1])
        self.base = base
        self.mix_in = torch.nn.Parameter(
            torch.empty(base.num_experts, rank, hidden_size),
        )
        self.mix_out = torch.nn.Parameter(
            torch.zeros(base.num_experts, hidden_size, rank),
        )
        torch.nn.init.normal_(self.mix_in, std=hidden_size ** -0.5)
        self.replace_base_output = bool(replace_base_output)
        self.hard_train = False
        # The vectorized gather is faster for tiny decode-like batches, while
        # the indexed weights become prohibitive for prefill. Count both
        # gathered projection tensors against a conservative memory bound.
        self.max_dense_gather_bytes = 128 * 1024 * 1024
        self.correction_dispatch_backend = "vectorized"
        self._fused_output_weight_cache: torch.Tensor | None = None
        self._fused_output_weight_cache_key: tuple[object, ...] | None = None
        self.grouped_fused_correction_backend = "grouped-fused-correction"

    def _fused_output_weight(self) -> torch.Tensor:
        """Build W_out + mix_out @ mix_in @ W_out for frozen inference."""
        output_weight = self.base.group_output_weight
        key = (
            output_weight.device,
            output_weight.dtype,
            output_weight._version,
            self.mix_in.device,
            self.mix_in.dtype,
            self.mix_in._version,
            self.mix_out.device,
            self.mix_out.dtype,
            self.mix_out._version,
        )
        if (
            self._fused_output_weight_cache is None
            or self._fused_output_weight_cache_key != key
        ):
            with torch.no_grad():
                correction_weight = torch.einsum(
                    "eor,eri,eig->eog",
                    self.mix_out,
                    self.mix_in,
                    output_weight,
                )
                self._fused_output_weight_cache = (
                    output_weight + correction_weight
                ).contiguous()
            self._fused_output_weight_cache_key = key
        return self._fused_output_weight_cache

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        use_fused_output = (
            self.correction_dispatch_backend == "effective-output"
            and not self.replace_base_output
            and hidden_states.shape[-2] == 1
        )
        use_fused_full = (
            self.correction_dispatch_backend in {
                "cuda-fused-full", "cuda-fused-token",
            }
            and not self.replace_base_output
            and not self.base.training
            and hidden_states.shape[-2] == 1
        )
        use_grouped_fused_correction = (
            self.correction_dispatch_backend == self.grouped_fused_correction_backend
            and not self.replace_base_output
            and not self.base.training
            and self.base.dispatch_mode in {
                "grouped", "grouped-fused", "grouped-cached",
                "grouped-prepacked", "grouped-prepacked-fused",
                "grouped-tiled", "grouped-optimized",
                "grouped-adaptive", "grouped-adaptive-nozero",
                "grouped-adaptive-effective-output",
                "grouped-adaptive-atomic-pack",
                "grouped-adaptive-atomic-effective-output",
            }
            and not self.base.single_token_fast_path
        )
        use_grouped_effective_output = (
            self.correction_dispatch_backend == "grouped-effective-output"
            and not self.replace_base_output
            and not self.base.training
            and self.base.dispatch_mode in {
                "grouped-adaptive-effective-output",
                "grouped-adaptive-atomic-effective-output",
            }
            and not self.base.single_token_fast_path
        )
        if use_fused_output:
            self.base.single_token_output_weight = self._fused_output_weight()
        else:
            self.base.single_token_output_weight = None
        self.base.single_token_full_correction = (
            (
                self.mix_in, self.mix_out, self.correction_dispatch_backend
            ) if use_fused_full else None
        )
        self.base.grouped_effective_output_weight = (
            self._fused_output_weight() if use_grouped_effective_output else None
        )
        self.base.grouped_fused_correction = (
            (self.mix_in, self.mix_out) if use_grouped_fused_correction else None
        )
        base_output = self.base(hidden_states)
        route_weights = self.base.last_route_weights
        selected = self.base.last_selected
        if route_weights is None or selected is None:
            raise RuntimeError("base route state was not populated")
        if (
            use_fused_output
            or use_fused_full
            or use_grouped_fused_correction
            or use_grouped_effective_output
        ):
            return base_output
        if self.base.training and not self.base.hard_train:
            all_outputs = self.base.last_all_outputs
            if all_outputs is None:
                raise RuntimeError("soft route outputs were not populated")
            latent = torch.einsum(
                "...eh,erh->...er", all_outputs, self.mix_in,
            )
            expert_corrections = torch.einsum(
                "...er,ehr->...eh", latent, self.mix_out,
            )
            correction = self.base.num_experts * (
                expert_corrections * route_weights.unsqueeze(-1)
            ).sum(dim=-2)
        else:
            selected_outputs = self.base.last_selected_outputs
            if selected_outputs is None:
                raise RuntimeError("hard route outputs were not populated")
            if (
                self.correction_dispatch_backend == "cuda-fused"
                and selected_outputs.shape[-3] == 1
            ):
                from neural_engine.qwen_correction_dispatch import correction_dispatch

                correction = correction_dispatch(
                    selected_outputs.reshape(
                        -1, selected_outputs.shape[-2], selected_outputs.shape[-1],
                    ),
                    selected.reshape(-1, selected.shape[-1]),
                    route_weights.reshape(-1, route_weights.shape[-1]),
                    self.mix_in,
                    self.mix_out,
                    self.base.hard_route_scale,
                ).reshape(*selected_outputs.shape[:-2], selected_outputs.shape[-1])
                return correction if self.replace_base_output else base_output + correction
            gather_elements = (
                selected_outputs.shape[0]
                * selected_outputs.shape[1]
                * selected_outputs.shape[2]
                * self.mix_in.shape[1]
                * self.mix_in.shape[2]
            )
            gather_bytes = 2 * gather_elements * selected_outputs.element_size()
            if gather_bytes <= self.max_dense_gather_bytes:
                # For small decode-like inputs this vectorized path avoids
                # launching one projection pair per expert. The byte guard
                # prevents the old multi-hundred-MB/GB gather during prefill.
                selected_mix_in = self.mix_in[selected]
                selected_mix_out = self.mix_out[selected]
                if selected_outputs.shape[-3] == 1:
                    # Decode has one sequence token. Flattening batch×K into
                    # independent tiny GEMMs avoids the generic ellipsis
                    # einsum dispatch while preserving the exact contraction.
                    batch_size = selected_outputs.shape[0]
                    active_experts = selected_outputs.shape[-2]
                    hidden_size = selected_outputs.shape[-1]
                    rank = selected_mix_in.shape[-2]
                    flat_outputs = selected_outputs.reshape(
                        -1, 1, hidden_size,
                    )
                    flat_mix_in = selected_mix_in.reshape(
                        -1, rank, hidden_size,
                    )
                    latent = torch.bmm(
                        flat_outputs, flat_mix_in.transpose(1, 2),
                    ).reshape(batch_size, 1, active_experts, rank)
                    flat_mix_out = selected_mix_out.reshape(
                        -1, hidden_size, rank,
                    )
                    selected_corrections = torch.bmm(
                        latent.reshape(-1, 1, rank),
                        flat_mix_out.transpose(1, 2),
                    ).reshape(
                        batch_size, 1, active_experts, hidden_size,
                    )
                else:
                    latent = torch.einsum(
                        "...kh,...krh->...kr", selected_outputs, selected_mix_in,
                    )
                    selected_corrections = torch.einsum(
                        "...kr,...khr->...kh", latent, selected_mix_out,
                    )
                correction = self.base.hard_route_scale * (
                    selected_corrections * route_weights.unsqueeze(-1)
                ).sum(dim=-2)
            else:
                # Avoid materializing mix_in[selected] and mix_out[selected]
                # as [tokens, K, rank, hidden] tensors. The expert-packed
                # implementation computes the same projections per selected
                # expert and accumulates directly into the token output.
                flat_selected_outputs = selected_outputs.reshape(
                    -1, selected_outputs.shape[-2], selected_outputs.shape[-1],
                )
                flat_selected = selected.reshape(
                    -1, selected.shape[-1],
                )
                flat_route_weights = route_weights.reshape(
                    -1, route_weights.shape[-1],
                )
                flat_correction = torch.zeros_like(
                    flat_selected_outputs[..., 0, :],
                )
                for expert_id in range(self.base.num_experts):
                    token_ids, slots = torch.where(flat_selected == expert_id)
                    if token_ids.numel() == 0:
                        continue
                    latent = F.linear(
                        flat_selected_outputs[token_ids, slots],
                        self.mix_in[expert_id],
                    )
                    selected_correction = F.linear(
                        latent, self.mix_out[expert_id],
                    )
                    contribution = selected_correction * flat_route_weights[
                        token_ids, slots,
                    ].unsqueeze(-1)
                    flat_correction.index_add_(0, token_ids, contribution)
                correction = self.base.hard_route_scale * flat_correction.reshape(
                    *selected_outputs.shape[:-2], selected_outputs.shape[-1],
                )
        return correction if self.replace_base_output else base_output + correction


class ResidualCoresetRoutedQwenChild(torch.nn.Module):
    """Sparse copied groups plus a signed coefficient and compact residual.

    The copied groups remain the main path.  A zero-start coefficient head can
    add signed, input-conditioned corrections for selected groups, while the
    always-on low-rank SwiGLU branch models residual structure that no selected
    group can represent by itself.  The soft path remains an exact transfer at
    initialization because the signed coefficients are centered and zero-start
    and the residual output projection is zero-start.
    """

    def __init__(
        self,
        base: TransferredRoutedQwenChild,
        rank: int,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("residual coreset rank must be positive")
        hidden_size = int(base.group_gate_weight.shape[-1])
        self.base = base
        self.coefficient_router = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, base.num_experts),
        )
        torch.nn.init.zeros_(self.coefficient_router[-1].weight)
        torch.nn.init.zeros_(self.coefficient_router[-1].bias)
        self.residual_gate = torch.nn.Linear(hidden_size, rank, bias=False)
        self.residual_value = torch.nn.Linear(hidden_size, rank, bias=False)
        self.residual_output = torch.nn.Linear(rank, hidden_size, bias=False)
        torch.nn.init.zeros_(self.residual_output.weight)
        self.hard_train = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        selected = self.base.last_selected
        if selected is None:
            raise RuntimeError("base route state was not populated")
        coefficient_delta = self.coefficient_router(hidden_states)
        if self.base.training and not self.base.hard_train:
            all_outputs = self.base.last_all_outputs
            if all_outputs is None:
                raise RuntimeError("soft route outputs were not populated")
            centered_delta = coefficient_delta - coefficient_delta.mean(
                dim=-1, keepdim=True,
            )
            signed_correction = (
                all_outputs * centered_delta.unsqueeze(-1)
            ).sum(dim=-2)
        else:
            selected_outputs = self.base.last_selected_outputs
            if selected_outputs is None:
                raise RuntimeError("hard route outputs were not populated")
            selected_delta = torch.gather(
                coefficient_delta, -1, selected,
            )
            signed_correction = (
                selected_outputs * selected_delta.unsqueeze(-1)
            ).sum(dim=-2)
        residual = self.residual_output(
            F.silu(self.residual_gate(hidden_states))
            * self.residual_value(hidden_states),
        )
        return base_output + signed_correction + residual


class OutputContractRoutedQwenChild(torch.nn.Module):
    """Sparse child with a bounded per-channel output-statistics contract."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base
        route_base = next(
            nested for nested in base.modules()
            if isinstance(nested, TransferredRoutedQwenChild)
        )
        hidden_size = int(route_base.group_gate_weight.shape[-1])
        self.register_buffer("base_mean", torch.zeros(hidden_size))
        self.register_buffer("output_mean", torch.zeros(hidden_size))
        self.register_buffer("scale", torch.ones(hidden_size))
        self.fitted = False

    @torch.no_grad()
    def fit(self, io_batches: list[dict[str, torch.Tensor]], device: torch.device) -> None:
        was_training = self.base.training
        self.base.eval()
        base_outputs = []
        target_outputs = []
        for batch in io_batches:
            inputs = batch["input"].to(
                device=device,
                dtype=next(self.base.parameters()).dtype,
            )
            base_outputs.append(self.base(inputs).float().reshape(-1, inputs.shape[-1]))
            target_outputs.append(batch["output"].to(device=device).float().reshape(-1, inputs.shape[-1]))
        base_flat = torch.cat(base_outputs, dim=0)
        target_flat = torch.cat(target_outputs, dim=0)
        base_mean = base_flat.mean(dim=0)
        base_std = base_flat.std(dim=0, unbiased=False)
        target_mean = target_flat.mean(dim=0)
        target_std = target_flat.std(dim=0, unbiased=False)
        scale = (target_std / base_std.clamp_min(1e-4)).clamp(0.25, 4.0)
        self.base_mean.copy_(base_mean.to(self.base_mean.dtype))
        self.output_mean.copy_(target_mean.to(self.output_mean.dtype))
        self.scale.copy_(scale.to(self.scale.dtype))
        self.fitted = True
        self.base.train(was_training)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        if not self.fitted:
            return base_output
        return self.output_mean + (
            base_output - self.base_mean
        ) * self.scale


@torch.no_grad()
def initialize_teacher_group_decoders(
    child: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    ridge: float = 1e-3,
) -> None:
    """Fit low-rank group-output decoders against frozen teacher MLP outputs."""
    mixer = next(
        nested for nested in child.modules()
        if isinstance(nested, CrossGroupOutputMixRoutedQwenChild)
    )
    if not mixer.replace_base_output:
        raise ValueError("teacher group decoder requires replace_base_output")
    base = mixer.base
    hidden_size = int(base.group_gate_weight.shape[-1])
    num_experts = base.num_experts
    gram = torch.zeros(
        num_experts, hidden_size, hidden_size, device=device, dtype=torch.float32,
    )
    cross = torch.zeros_like(gram)
    base.eval()
    for batch in io_batches:
        inputs = batch["input"].to(device=device, dtype=dtype)
        target = batch["output"].to(device=device, dtype=torch.float32)
        outputs = torch.stack([
            expert(inputs) for expert in base.experts
        ], dim=-2).float()
        flat_outputs = outputs.reshape(-1, num_experts, hidden_size).transpose(0, 1)
        flat_target = target.reshape(-1, hidden_size)
        gram += torch.einsum("enh,enk->ehk", flat_outputs, flat_outputs)
        cross += torch.einsum("enh,nk->ehk", flat_outputs, flat_target)
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
    identity = torch.eye(hidden_size, device=device, dtype=torch.float32)
    solved = torch.linalg.solve(
        gram + (ridge * scale).view(num_experts, 1, 1) * identity,
        cross,
    )
    left, singular, right = torch.linalg.svd(solved, full_matrices=False)
    rank = min(mixer.mix_in.shape[1], singular.shape[-1])
    root = singular[:, :rank].clamp_min(0).sqrt()
    mixer.mix_in.zero_()
    mixer.mix_out.zero_()
    mixer.mix_in[:, :rank].copy_(root.unsqueeze(-1) * right[:, :rank])
    mixer.mix_out[:, :, :rank].copy_(left[:, :, :rank] * root.unsqueeze(1))


@torch.no_grad()
def initialize_teacher_group_residuals(
    child: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    ridge: float = 1e-3,
) -> None:
    """Fit group decoders to hard-route omitted residuals from teacher IO."""
    mixer = next(
        nested for nested in child.modules()
        if isinstance(nested, CrossGroupOutputMixRoutedQwenChild)
    )
    if mixer.replace_base_output:
        raise ValueError("residual initialization requires additive group mixing")
    base = mixer.base
    hidden_size = int(base.group_gate_weight.shape[-1])
    num_experts = base.num_experts
    active_experts = base.active_experts
    hard_scale = float(base.hard_route_scale)
    gram = torch.zeros(
        num_experts, hidden_size, hidden_size, device=device, dtype=torch.float32,
    )
    cross = torch.zeros_like(gram)
    base.eval()
    for batch in io_batches:
        inputs = batch["input"].to(device=device, dtype=dtype)
        target = batch["output"].to(device=device, dtype=torch.float32)
        outputs = torch.stack([
            expert(inputs) for expert in base.experts
        ], dim=-2).float()
        flat_outputs = outputs.reshape(-1, num_experts, hidden_size)
        flat_target = target.reshape(-1, hidden_size)
        importance = flat_outputs.square().mean(dim=-1)
        top_values, top_ids = importance.topk(active_experts, dim=-1)
        weights = F.softmax(top_values / base.temperature, dim=-1)
        selected = torch.gather(
            flat_outputs, 1,
            top_ids.unsqueeze(-1).expand(-1, -1, hidden_size),
        )
        base_selected = hard_scale * (
            selected * weights.unsqueeze(-1)
        ).sum(dim=1)
        residual_target = (flat_target - base_selected) / max(hard_scale, 1e-6)
        for expert_id in range(num_experts):
            mask = top_ids == expert_id
            if not mask.any():
                continue
            selected_rows = selected[mask]
            gram[expert_id] += selected_rows.transpose(0, 1) @ selected_rows
            residual_rows = residual_target.unsqueeze(1).expand(
                -1, active_experts, -1,
            )[mask]
            cross[expert_id] += selected_rows.transpose(0, 1) @ residual_rows
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
    identity = torch.eye(hidden_size, device=device, dtype=torch.float32)
    solved = torch.linalg.solve(
        gram + (ridge * scale).view(num_experts, 1, 1) * identity,
        cross,
    )
    left, singular, right = torch.linalg.svd(solved, full_matrices=False)
    rank = min(mixer.mix_in.shape[1], singular.shape[-1])
    root = singular[:, :rank].clamp_min(0).sqrt()
    mixer.mix_in.zero_()
    mixer.mix_out.zero_()
    mixer.mix_in[:, :rank].copy_(root.unsqueeze(-1) * right[:, :rank])
    mixer.mix_out[:, :, :rank].copy_(left[:, :, :rank] * root.unsqueeze(1))


@torch.no_grad()
def initialize_signed_subset_reconstruction(
    child: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    ridge: float = 1e-3,
) -> None:
    """Fit one signed K-group reconstruction vector per subset.

    The fit is a tiny ridge regression in the selected-group output space:
    ``sum_k coefficient[k] * group_output[k] ~= teacher_output``.  It does
    not introduce a dense decoder or evaluate omitted groups on the hard path.
    """
    reconstructor = next(
        nested for nested in child.modules()
        if isinstance(nested, SignedSubsetReconstructionQwenChild)
    )
    base = reconstructor.base
    if not io_batches:
        raise ValueError("signed subset reconstruction requires calibration IO")
    num_subsets = int(reconstructor.subset_ids.shape[0])
    active = base.active_experts
    gram = torch.zeros(
        num_subsets, active, active, device=device, dtype=torch.float32,
    )
    cross = torch.zeros(
        num_subsets, active, device=device, dtype=torch.float32,
    )
    was_training = base.training
    base.eval()
    for batch in io_batches:
        inputs = batch["input"].to(device=device, dtype=dtype)
        outputs = torch.stack([
            expert(inputs) for expert in base.experts
        ], dim=-2).float()
        outputs = outputs.reshape(-1, base.num_experts, outputs.shape[-1])
        target = batch["output"].to(device=device, dtype=torch.float32)
        target = target.reshape(-1, outputs.shape[-1])
        for subset_index, subset in enumerate(reconstructor.subset_ids):
            selected = outputs.index_select(1, subset)
            gram[subset_index] += torch.einsum(
                "nkh,nlh->kl", selected, selected,
            )
            cross[subset_index] += torch.einsum(
                "nkh,nh->k", selected, target,
            )
    identity = torch.eye(active, device=device, dtype=torch.float32)
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-6)
    solved = torch.linalg.solve(
        gram + (ridge * scale).view(num_subsets, 1, 1) * identity,
        cross.unsqueeze(-1),
    ).squeeze(-1)
    fitted = torch.zeros_like(reconstructor.coefficients)
    fitted.scatter_(1, reconstructor.subset_ids, solved)
    reconstructor.coefficients.copy_(fitted.to(reconstructor.coefficients.dtype))
    reconstructor.fit_ridge = float(ridge)
    reconstructor.last_fit_negative_fraction = float(
        (solved < 0).float().mean().item()
    )
    reconstructor.last_fit_coefficient_rms = float(
        solved.square().mean().sqrt().item()
    )
    base.train(was_training)


def make_transferred_routed_qwen_child(
    parent: torch.nn.Module,
    num_experts: int,
    active_experts: int,
    routing_temperature: float,
    calibration_rank: int,
    calibration_source: str,
    calibration_mode: str,
    dispatch_mode: str,
    partition_mode: str,
    route_source: str,
    hard_route_scale: float | None,
    device: torch.device,
    dtype: torch.dtype,
    partition_io: list[dict[str, torch.Tensor]] | None = None,
    router_hidden_size: int = 128,
    router_input: str = "hidden",
    pairwise_cost_parameterization: str = "components",
    core_overlap_fraction: float = 0.25,
    router_sketch_dim: int = 8,
) -> torch.nn.Module:
    partition_indices = None
    if partition_mode == "activation-balanced":
        if partition_io is None:
            raise ValueError("activation-balanced partition requires calibration IO")
        partition_indices = activation_balanced_partition(
            parent, partition_io, device, dtype, num_experts,
        )
    elif partition_mode == "sampled-overlap":
        partition_indices = sampled_overlap_partition(parent, num_experts)
    elif partition_mode == "stratified-overlap":
        if partition_io is None:
            raise ValueError("stratified-overlap partition requires calibration IO")
        partition_indices = stratified_overlap_partition(
            parent, partition_io, device, dtype, num_experts,
        )
    elif partition_mode == "activation-cluster":
        if partition_io is None:
            raise ValueError("activation-cluster partition requires calibration IO")
        partition_indices = activation_cluster_partition(
            parent, partition_io, device, dtype, num_experts,
        )
    elif partition_mode == "contribution-cluster":
        if partition_io is None:
            raise ValueError("contribution-cluster partition requires calibration IO")
        partition_indices = contribution_cluster_partition(
            parent, partition_io, device, dtype, num_experts,
        )
    elif partition_mode == "contribution-diverse":
        if partition_io is None:
            raise ValueError("contribution-diverse partition requires calibration IO")
        partition_indices = contribution_diverse_partition(
            parent, partition_io, device, dtype, num_experts,
        )
    elif partition_mode == "core-overlap":
        if partition_io is None:
            raise ValueError("core-overlap partition requires calibration IO")
        partition_indices = core_overlap_partition(
            parent, partition_io, device, dtype, num_experts,
            core_fraction=core_overlap_fraction,
        )
    child = TransferredRoutedQwenChild(
        parent, num_experts, active_experts, routing_temperature, dispatch_mode,
        partition_mode, route_source, hard_route_scale,
        partition_indices,
        router_hidden_size,
        router_input,
        pairwise_cost_parameterization,
        router_sketch_dim,
    ).to(device=device, dtype=dtype)
    if calibration_rank > 0 or calibration_mode == "signed-subset":
        if calibration_mode == "shared-basis":
            child = SharedBasisRoutedQwenChild(
                child, int(calibration_rank),
            ).to(device=device, dtype=dtype)
        elif calibration_mode in {"cross-group", "teacher-group-residual"}:
            child = CrossGroupOutputMixRoutedQwenChild(
                child, int(calibration_rank),
            ).to(device=device, dtype=dtype)
        elif calibration_mode == "teacher-group-decoder":
            child = CrossGroupOutputMixRoutedQwenChild(
                child, int(calibration_rank), replace_base_output=True,
            ).to(device=device, dtype=dtype)
        elif calibration_mode == "swiglu":
            child = SwiGLUResidualChild(
                child, int(parent.gate_proj.in_features), calibration_rank,
            ).to(device=device, dtype=dtype)
        elif calibration_mode == "residual-coreset":
            child = ResidualCoresetRoutedQwenChild(
                child, int(calibration_rank),
            ).to(device=device, dtype=dtype)
        elif calibration_mode == "output-contract":
            if calibration_rank > 0:
                child = CrossGroupOutputMixRoutedQwenChild(
                    child, int(calibration_rank),
                ).to(device=device, dtype=dtype)
            child = OutputContractRoutedQwenChild(child).to(
                device=device, dtype=dtype,
            )
        elif calibration_mode == "signed-subset":
            child = SignedSubsetReconstructionQwenChild(child).to(
                device=device, dtype=dtype,
            )
        else:
            calibration_class = {
                "base-output": CalibratedChild,
                "input": InputCalibratedChild,
            }[calibration_source]
            child = calibration_class(
                child, int(parent.gate_proj.in_features), calibration_rank,
            ).to(device=device, dtype=dtype)
    return child


def make_learned_latent_basis_qwen_child(
    hidden_size: int,
    num_basis: int,
    active_basis: int,
    rank: int,
    routing_temperature: float,
    hard_route_scale: float | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    return LearnedLatentBasisQwenChild(
        hidden_size, num_basis, active_basis, rank, routing_temperature,
        hard_route_scale,
    ).to(device=device, dtype=dtype)


def make_transferred_routed_qwen_neuron_child(
    parent: torch.nn.Module,
    active_neurons: int,
    routing_temperature: float,
    calibration_rank: int,
    calibration_source: str,
    calibration_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    route_source: str = "router",
    hard_route_scale: float | None = None,
) -> torch.nn.Module:
    child = TransferredRoutedQwenNeuronChild(
        parent, active_neurons, routing_temperature, route_source=route_source,
        hard_route_scale=hard_route_scale,
    ).to(device=device, dtype=dtype)
    if calibration_rank > 0:
        if calibration_mode == "swiglu":
            child = SwiGLUResidualChild(
                child, int(parent.gate_proj.in_features), calibration_rank,
            ).to(device=device, dtype=dtype)
        else:
            calibration_class = {
                "base-output": CalibratedChild,
                "input": InputCalibratedChild,
            }[calibration_source]
            child = calibration_class(
                child, int(parent.gate_proj.in_features), calibration_rank,
            ).to(device=device, dtype=dtype)
    return child


def train_importance_router(
    child: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    log_every: int,
    target_mode: str = "energy",
    target_temperature: float = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
) -> list[dict[str, float]]:
    """Distill group importance or exact best-subset labels into the router."""
    base = next(
        nested for nested in child.modules()
        if isinstance(nested, TransferredRoutedQwenChild)
    )
    signed_reconstructor = next(
        (
            nested for nested in child.modules()
            if isinstance(nested, SignedSubsetReconstructionQwenChild)
        ),
        None,
    )
    if base.active_experts == base.num_experts:
        # With every group active, routing is the identity and the copied
        # FFN already reconstructs the parent exactly.  There is no useful
        # subset label or router gradient in this case.
        return []
    all_parameters = list(base.parameters())
    previous_requires_grad = [parameter.requires_grad for parameter in all_parameters]
    for parameter in all_parameters:
        parameter.requires_grad_(False)
    if base.route_source in {
        "subset-router", "oracle-subset", "pairwise-cost-router",
    }:
        if target_mode not in {
            "subset", "subset-soft", "final-subset-soft",
            "pairwise-regret", "pairwise-regret-tail",
        }:
            raise ValueError("subset router requires subset or regret supervision")
        router_parameters = list(
            (
                base.pairwise_cost_router
                if base.route_source == "pairwise-cost-router"
                else base.subset_router
            ).parameters()
        )
    else:
        router_parameters = list(base.router.parameters())
    for parameter in router_parameters:
        parameter.requires_grad_(True)
    if optimizer is None:
        optimizer = torch.optim.AdamW(router_parameters, lr=learning_rate)
    history = []
    base.eval()
    try:
        for step in range(1, steps + 1):
            batch = io_batches[(step - 1) % len(io_batches)]
            inputs = batch["input"].to(device=device, dtype=dtype)
            subset_target = None
            subset_regrets = None
            with torch.no_grad():
                if base.route_source == "pairwise-cost-router":
                    _, outputs = routing_group_outputs(child, inputs)
                else:
                    outputs = torch.stack([
                        expert(inputs) for expert in base.experts
                    ], dim=-2).float()
                if target_mode == "dot":
                    full_output = outputs.sum(dim=-2)
                    importance = (
                        outputs * full_output.unsqueeze(-2)
                    ).sum(dim=-1)
                    target = F.softmax(importance, dim=-1)
                elif target_mode == "energy":
                    importance = outputs.square().mean(dim=-1)
                    target = F.softmax(
                        torch.log(importance + 1e-8), dim=-1,
                    )
                elif target_mode in {
                    "subset", "subset-soft", "final-subset-soft",
                    "pairwise-regret", "pairwise-regret-tail",
                }:
                    if (
                        target_mode == "final-subset-soft"
                        and base.route_source != "pairwise-cost-router"
                    ):
                        if not isinstance(child, CrossGroupOutputMixRoutedQwenChild):
                            raise ValueError(
                                "final-subset-soft requires cross-group child"
                            )
                        latent = torch.einsum(
                            "...eh,erh->...er", outputs, child.mix_in,
                        )
                        corrections = torch.einsum(
                            "...er,ehr->...eh", latent, child.mix_out,
                        )
                        outputs = outputs + corrections
                    flat_outputs = outputs.reshape(
                        -1, base.num_experts, outputs.shape[-1],
                    )
                    flat_target = (
                        batch["output"].to(device=device, dtype=torch.float32)
                        .reshape(-1, outputs.shape[-1])
                    )
                    subset_ids = torch.tensor(
                        list(combinations(
                            range(base.num_experts), base.active_experts,
                        )), device=device, dtype=torch.long,
                    )
                    if signed_reconstructor is not None:
                        # The router must see the same signed reconstruction
                        # contract that hard inference will execute.
                        candidate_outputs = torch.einsum(
                            "neh,ce->nch",
                            flat_outputs,
                            signed_reconstructor.coefficients,
                        )
                        errors = (
                            candidate_outputs - flat_target.unsqueeze(1)
                        ).square().mean(dim=-1)
                        membership = base.subset_membership
                    else:
                        gram = torch.einsum(
                            "neh,nfh->nef", flat_outputs, flat_outputs,
                        )
                        cross = torch.einsum(
                            "neh,nh->ne", flat_outputs, flat_target,
                        )
                        membership = F.one_hot(
                            subset_ids, num_classes=base.num_experts,
                        ).sum(dim=1).float()
                        pair_energy = torch.einsum(
                            "ce,nef,cf->nc", membership, gram, membership,
                        )
                        cross_energy = cross @ membership.transpose(0, 1)
                        coefficient = base.hard_route_scale / base.active_experts
                        errors = (
                            coefficient * coefficient * pair_energy
                            - 2.0 * coefficient * cross_energy
                            + flat_target.square().sum(dim=-1, keepdim=True)
                        )
                    best_subset_index = errors.argmin(dim=-1)
                    best_subset = subset_ids[best_subset_index]
                    if target_mode == "subset":
                        subset_target = best_subset_index
                        target = F.one_hot(
                            best_subset, num_classes=base.num_experts,
                        ).sum(dim=1).float().reshape(
                            *inputs.shape[:-1], base.num_experts,
                        )
                    else:
                        if target_temperature <= 0:
                            raise ValueError("router target temperature must be positive")
                        centered_errors = errors - errors.min(dim=-1, keepdim=True).values
                        subset_regrets = (
                            centered_errors / max(outputs.shape[-1], 1)
                            if target_mode in {
                                "pairwise-regret", "pairwise-regret-tail",
                            }
                            else centered_errors
                        )
                        subset_target = F.softmax(
                            -centered_errors / target_temperature, dim=-1,
                        )
                        target = None
                else:
                    raise ValueError(
                        "router target must be energy, dot, subset, subset-soft, "
                        "final-subset-soft, pairwise-regret, or "
                        "pairwise-regret-tail"
                    )
            scores = (
                base._subset_scores(inputs).float()
                if base.route_source in {
                    "subset-router", "oracle-subset", "pairwise-cost-router",
                }
                else base.router(inputs).float()
            )
            if target_mode in {
                "pairwise-regret", "pairwise-regret-tail",
            }:
                if base.route_source != "pairwise-cost-router":
                    raise ValueError(
                        "pairwise-regret targets require pairwise-cost-router"
                    )
                if subset_regrets is None:
                    raise RuntimeError("pairwise regrets were not computed")
                if target_temperature <= 0:
                    raise ValueError("router target temperature must be positive")
                flat_scores = scores.reshape(-1, scores.shape[-1])
                predicted_probabilities = F.softmax(
                    flat_scores / target_temperature, dim=-1,
                )
                per_token_regret = (
                    predicted_probabilities * subset_regrets
                ).sum(dim=-1)
                expected_regret = per_token_regret.mean()
                if target_mode == "pairwise-regret-tail":
                    tail_count = max(
                        1, (per_token_regret.shape[0] + 3) // 4,
                    )
                    regret_loss = expected_regret + 0.5 * torch.topk(
                        per_token_regret, tail_count,
                    ).values.mean()
                else:
                    regret_loss = expected_regret
                soft_target = F.softmax(
                    -subset_regrets / target_temperature, dim=-1,
                )
                auxiliary_loss = -(
                    soft_target
                    * F.log_softmax(flat_scores, dim=-1)
                ).sum(dim=-1).mean()
                loss = regret_loss + 0.1 * auxiliary_loss
            elif target_mode in {"subset", "subset-soft", "final-subset-soft"}:
                if base.route_source in {
                    "subset-router", "oracle-subset", "pairwise-cost-router",
                }:
                    if subset_target is None:
                        raise RuntimeError("subset target was not computed")
                    if target_mode == "subset":
                        loss = F.cross_entropy(
                            scores.reshape(-1, scores.shape[-1]),
                            subset_target.reshape(-1),
                        )
                    else:
                        loss = -(
                            subset_target
                            * F.log_softmax(scores.reshape(-1, scores.shape[-1]), dim=-1)
                        ).sum(dim=-1).mean()
                else:
                    raise ValueError(
                        "subset supervision requires a subset-router child"
                    )
            else:
                loss = F.kl_div(
                    F.log_softmax(scores, dim=-1), target, reduction="batchmean",
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite router importance loss at step {step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router_parameters, max_grad_norm)
            optimizer.step()
            if step == 1 or step % log_every == 0 or step == steps:
                history.append({"step": step, "loss": float(loss.detach().cpu())})
    finally:
        for parameter, previous in zip(all_parameters, previous_requires_grad):
            parameter.requires_grad_(previous)
    return history


@torch.no_grad()
def routing_group_outputs(
    child: torch.nn.Module,
    inputs: torch.Tensor,
) -> tuple[TransferredRoutedQwenChild, torch.Tensor]:
    """Return per-group outputs, including a trained cross-group correction."""
    base = next(
        nested for nested in child.modules()
        if isinstance(nested, TransferredRoutedQwenChild)
    )
    outputs = torch.stack([
        expert(inputs) for expert in base.experts
    ], dim=-2).float()
    mixer = next(
        (
            nested for nested in child.modules()
            if isinstance(nested, CrossGroupOutputMixRoutedQwenChild)
        ),
        None,
    )
    if mixer is not None:
        latent = torch.einsum(
            "...eh,erh->...er", outputs, mixer.mix_in,
        )
        corrections = torch.einsum(
            "...er,ehr->...eh", latent, mixer.mix_out,
        )
        outputs = corrections if mixer.replace_base_output else outputs + corrections
    return base, outputs


def capture_cascade_router_batches(
    model: torch.nn.Module,
    tokenizer,
    calibration_text: str,
    batch_size: int,
    sequence_length: int,
    train_batches: int,
    device: torch.device,
    layer_indices: list[int],
    layers: list[torch.nn.Module],
    parents: list[torch.nn.Module],
) -> list[list[dict[str, torch.Tensor]]]:
    """Capture current-cascade inputs with the original parent as each target.

    Previous sparse layers remain active while one layer is temporarily
    restored to its dense parent.  This gives the router the hidden states
    produced by the current learned cascade while preserving the dense FFN
    target for that exact input.
    """
    captured = []
    for layer_index, layer, parent in zip(layer_indices, layers, parents):
        previous_mlp = layer.mlp
        layer.mlp = parent
        try:
            captured.append(capture_batches(
                model, tokenizer, calibration_text, batch_size,
                sequence_length, train_batches, device, layer_index,
            ))
        finally:
            layer.mlp = previous_mlp
    return captured


def _pairwise_router_parameters(
    child: torch.nn.Module,
) -> list[torch.nn.Parameter]:
    base = next(
        nested for nested in child.modules()
        if isinstance(nested, TransferredRoutedQwenChild)
    )
    if base.route_source != "pairwise-cost-router":
        raise ValueError("router refit requires pairwise-cost-router children")
    return list(base.pairwise_cost_router.parameters())


def refit_pairwise_router_cascade(
    model: torch.nn.Module,
    tokenizer,
    calibration_text: str,
    batch_size: int,
    sequence_length: int,
    train_batches: int,
    device: torch.device,
    dtype: torch.dtype,
    layer_indices: list[int],
    layers: list[torch.nn.Module],
    parents: list[torch.nn.Module],
    children: list[torch.nn.Module],
    initial_io: list[list[dict[str, torch.Tensor]]],
    mode: str,
    steps: int,
    rounds: int,
    learning_rate: float,
    max_grad_norm: float,
    log_every: int,
    target_temperature: float,
    target_mode: str = "pairwise-regret",
) -> list[list[dict[str, float]]]:
    """Refit pairwise routers on static or aggregated cascade inputs.

    The aggregate mode preserves optimizer state between rounds.  Each new
    rollout is mixed 50/50 with an equal-size sample from the accumulated old
    pool, while all child and correction weights remain frozen.
    """
    if mode not in {"static", "aggregate"}:
        raise ValueError("router refit mode must be static or aggregate")
    if len(children) != len(initial_io) or len(children) != len(layers):
        raise ValueError("router refit children, IO, and layers must align")
    if steps < 1 or rounds < 1:
        raise ValueError("router refit steps and rounds must be positive")

    optimizers = [
        torch.optim.AdamW(
            _pairwise_router_parameters(child),
            lr=learning_rate,
        )
        for child in children
    ]
    histories = [[] for _ in children]
    if mode == "static":
        for index, (child, io_batches, optimizer) in enumerate(
            zip(children, initial_io, optimizers),
        ):
            histories[index].extend(train_importance_router(
                child, io_batches, device, dtype, steps,
                learning_rate, max_grad_norm, log_every,
                target_mode, target_temperature, optimizer,
            ))
        return histories

    old_pools = [list(io_batches) for io_batches in initial_io]
    for round_index in range(rounds):
        if round_index == 0:
            round_io = initial_io
        else:
            new_io = capture_cascade_router_batches(
                model, tokenizer, calibration_text, batch_size,
                sequence_length, train_batches, device, layer_indices,
                layers, parents,
            )
            round_io = []
            for pool, new_batches in zip(old_pools, new_io):
                if not pool or not new_batches:
                    raise ValueError("router aggregation received empty batches")
                old_sample = [
                    pool[index % len(pool)]
                    for index in range(len(new_batches))
                ]
                mixed = []
                for new_batch, old_batch in zip(new_batches, old_sample):
                    mixed.extend((new_batch, old_batch))
                round_io.append(mixed)
                pool.extend(new_batches)
        for index, (child, io_batches, optimizer) in enumerate(
            zip(children, round_io, optimizers),
        ):
            histories[index].extend(train_importance_router(
                child, io_batches, device, dtype, steps,
                learning_rate, max_grad_norm, log_every,
                target_mode, target_temperature, optimizer,
            ))
    return histories


@torch.no_grad()
def routing_diagnostics(
    child: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Measure learned-vs-best-subset routing on held-out child inputs."""
    base = next(
        nested for nested in child.modules()
        if isinstance(nested, TransferredRoutedQwenChild)
    )
    signed_reconstructor = next(
        (
            nested for nested in child.modules()
            if isinstance(nested, SignedSubsetReconstructionQwenChild)
        ),
        None,
    )
    if base.route_source not in {
        "subset-router", "oracle-subset", "pairwise-cost-router",
    }:
        return {}
    exact_matches = 0
    total_tokens = 0
    learned_mse = 0.0
    oracle_mse = 0.0
    learned_optimal_mse = 0.0
    oracle_optimal_mse = 0.0
    learned_optimal_scale = 0.0
    oracle_optimal_scale = 0.0
    subset_regret_values = []
    for batch in io_batches:
        inputs = batch["input"].to(device=device, dtype=dtype)
        target = batch["output"].to(device=device, dtype=torch.float32)
        _, outputs = routing_group_outputs(child, inputs)
        flat_outputs = outputs.reshape(-1, base.num_experts, outputs.shape[-1])
        flat_target = target.reshape(-1, outputs.shape[-1])
        subset_ids = torch.tensor(
            list(combinations(
                range(base.num_experts), base.active_experts,
            )), device=device, dtype=torch.long,
        )
        if signed_reconstructor is not None:
            candidate_outputs = torch.einsum(
                "neh,ce->nch", flat_outputs,
                signed_reconstructor.coefficients,
            )
            errors = (
                candidate_outputs - flat_target.unsqueeze(1)
            ).square().sum(dim=-1)
            membership = base.subset_membership
            coefficient = 1.0
        else:
            gram = torch.einsum("neh,nfh->nef", flat_outputs, flat_outputs)
            cross = torch.einsum("neh,nh->ne", flat_outputs, flat_target)
            membership = F.one_hot(
                subset_ids, num_classes=base.num_experts,
            ).sum(dim=1).float()
            pair_energy = torch.einsum(
                "ce,nef,cf->nc", membership, gram, membership,
            )
            cross_energy = cross @ membership.transpose(0, 1)
            coefficient = base.hard_route_scale / base.active_experts
            errors = (
                coefficient * coefficient * pair_energy
                - 2.0 * coefficient * cross_energy
                + flat_target.square().sum(dim=-1, keepdim=True)
            )
        best_subset_index = errors.argmin(dim=-1)
        best_membership = membership[best_subset_index]
        if base.route_source in {"subset-router", "pairwise-cost-router"}:
            predicted_subset_index = base._subset_scores(inputs).argmax(dim=-1)
            predicted_subset_index = predicted_subset_index.reshape(-1)
        else:
            predicted_subset_index = best_subset_index
        predicted_membership = membership[predicted_subset_index]
        if signed_reconstructor is not None:
            learned_output = candidate_outputs[
                torch.arange(flat_outputs.shape[0], device=device),
                predicted_subset_index,
            ]
            oracle_output = candidate_outputs[
                torch.arange(flat_outputs.shape[0], device=device),
                best_subset_index,
            ]
            learned_sum = learned_output
            oracle_sum = oracle_output
        else:
            learned_sum = (
                flat_outputs * predicted_membership.unsqueeze(-1)
            ).sum(dim=1)
            oracle_sum = (
                flat_outputs * best_membership.unsqueeze(-1)
            ).sum(dim=1)
            learned_output = coefficient * learned_sum
            oracle_output = coefficient * oracle_sum
        learned_error = errors.gather(
            1, predicted_subset_index.unsqueeze(1),
        ).squeeze(1)
        oracle_error = errors.min(dim=1).values
        subset_regret_values.append(
            ((learned_error - oracle_error) /
             max(flat_outputs.shape[-1], 1)).detach().cpu()
        )
        learned_denominator = learned_sum.square().sum(dim=-1).clamp_min(1e-8)
        oracle_denominator = oracle_sum.square().sum(dim=-1).clamp_min(1e-8)
        learned_scale = (
            learned_sum * flat_target
        ).sum(dim=-1) / learned_denominator
        oracle_scale = (
            oracle_sum * flat_target
        ).sum(dim=-1) / oracle_denominator
        learned_scale = learned_scale.clamp_min(0.0)
        oracle_scale = oracle_scale.clamp_min(0.0)
        learned_optimal_output = learned_scale.unsqueeze(-1) * learned_sum
        oracle_optimal_output = oracle_scale.unsqueeze(-1) * oracle_sum
        exact_matches += int(
            (predicted_subset_index == best_subset_index).sum().item()
        )
        total_tokens += int(best_subset_index.numel())
        learned_mse += float(
            F.mse_loss(learned_output, flat_target, reduction="sum").item()
        )
        oracle_mse += float(
            F.mse_loss(oracle_output, flat_target, reduction="sum").item()
        )
        learned_optimal_mse += float(
            F.mse_loss(
                learned_optimal_output, flat_target, reduction="sum",
            ).item()
        )
        oracle_optimal_mse += float(
            F.mse_loss(
                oracle_optimal_output, flat_target, reduction="sum",
            ).item()
        )
        learned_optimal_scale += float(learned_scale.sum().item())
        oracle_optimal_scale += float(oracle_scale.sum().item())
    denominator = max(total_tokens * int(base.group_gate_weight.shape[-1]), 1)
    learned_mse /= denominator
    oracle_mse /= denominator
    learned_optimal_mse /= denominator
    oracle_optimal_mse /= denominator
    learned_optimal_scale /= max(total_tokens, 1)
    oracle_optimal_scale /= max(total_tokens, 1)
    subset_regret_p95 = float(torch.quantile(
        torch.cat(subset_regret_values), 0.95,
    ).item()) if subset_regret_values else 0.0
    return {
        "exact_subset_match": exact_matches / max(total_tokens, 1),
        "learned_mse": learned_mse,
        "oracle_mse": oracle_mse,
        "subset_regret": learned_mse - oracle_mse,
        "subset_regret_p95": subset_regret_p95,
        "learned_optimal_scalar_mse": learned_optimal_mse,
        "oracle_optimal_scalar_mse": oracle_optimal_mse,
        "learned_optimal_scalar_gain": learned_mse - learned_optimal_mse,
        "oracle_optimal_scalar_gain": oracle_mse - oracle_optimal_mse,
        "learned_optimal_scalar_mean": learned_optimal_scale,
        "oracle_optimal_scalar_mean": oracle_optimal_scale,
    }


def train_neuron_importance_router(
    child: torch.nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    log_every: int,
) -> list[dict[str, float]]:
    """Pretrain a neuron router against teacher contribution energy."""
    base = next(
        nested for nested in child.modules()
        if isinstance(nested, TransferredRoutedQwenNeuronChild)
    )
    all_parameters = list(base.parameters())
    previous_requires_grad = [parameter.requires_grad for parameter in all_parameters]
    for parameter in all_parameters:
        parameter.requires_grad_(False)
    router_parameters = list(base.router.parameters())
    for parameter in router_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(router_parameters, lr=learning_rate)
    output_norm_sq = base.output_weight.float().square().sum(dim=0)
    history = []
    base.eval()
    try:
        for step in range(1, steps + 1):
            batch = io_batches[(step - 1) % len(io_batches)]
            inputs = batch["input"].to(device=device, dtype=dtype)
            with torch.no_grad():
                gate = F.linear(inputs, base.gate_weight)
                value = F.linear(inputs, base.value_weight)
                coefficient = F.silu(gate) * value
                importance = coefficient.float().square() * output_norm_sq
                target = F.softmax(
                    torch.log(importance + 1e-8), dim=-1,
                )
            scores = base.router(inputs).float()
            loss = F.kl_div(
                F.log_softmax(scores, dim=-1), target, reduction="batchmean",
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite neuron router loss at step {step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router_parameters, max_grad_norm)
            optimizer.step()
            if step == 1 or step % log_every == 0 or step == steps:
                history.append({"step": step, "loss": float(loss.detach().cpu())})
    finally:
        for parameter, previous in zip(all_parameters, previous_requires_grad):
            parameter.requires_grad_(previous)
    return history


def _freeze_sparse_copied_and_router_parameters(
    module: torch.nn.Module,
) -> None:
    """Keep transferred cells and route decisions fixed during refinement."""
    for nested in module.modules():
        if isinstance(nested, TransferredRoutedQwenChild):
            for parameter in nested.experts.parameters():
                parameter.requires_grad_(False)
            router = getattr(nested, "subset_router", None)
            if router is None:
                router = getattr(nested, "router", None)
            if router is not None:
                for parameter in router.parameters():
                    parameter.requires_grad_(False)
        elif isinstance(nested, TransferredRoutedQwenNeuronChild):
            for parameter in nested.router.parameters():
                parameter.requires_grad_(False)


def capture_layer_outputs(
    model: torch.nn.Module,
    input_batches: list[torch.Tensor],
    layer: torch.nn.Module,
) -> list[torch.Tensor]:
    """Capture a layer's post-block hidden state without changing the model."""
    outputs = []
    for ids in input_batches:
        captured: dict[str, torch.Tensor] = {}

        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            value = output[0] if isinstance(output, tuple) else output
            captured["output"] = value.detach()

        handle = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
        finally:
            handle.remove()
        if "output" not in captured:
            raise RuntimeError("layer output hook did not capture a value")
        outputs.append(captured["output"].to(device="cpu", dtype=torch.float16))
    return outputs


def run_with_layer_output(
    model: torch.nn.Module,
    ids: torch.Tensor,
    layer: torch.nn.Module,
) -> torch.Tensor:
    """Run a differentiable model forward and return one layer's output."""
    captured: dict[str, torch.Tensor] = {}

    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        captured["output"] = output[0] if isinstance(output, tuple) else output

    handle = layer.register_forward_hook(hook)
    try:
        model(input_ids=ids, use_cache=False)
    finally:
        handle.remove()
    if "output" not in captured:
        raise RuntimeError("layer output hook did not capture a value")
    return captured["output"]


def hard_route_block_refine(
    model: torch.nn.Module,
    layers: list[torch.nn.Module],
    children: list[torch.nn.Module],
    local_io_batches: list[list[dict[str, torch.Tensor]]],
    input_batches: list[torch.Tensor],
    teacher_block_outputs: list[torch.Tensor],
    device: torch.device,
    steps: int,
    learning_rate: float,
    local_weight: float,
    block_weight: float,
    max_grad_norm: float,
    log_every: int,
) -> list[dict[str, float]]:
    """Train a hard two-layer block while retaining local child fidelity."""
    if len(layers) != 2 or len(children) != 2:
        raise ValueError("hard-route block refinement currently requires two layers")
    if len(input_batches) != len(teacher_block_outputs):
        raise ValueError("block inputs and teacher outputs must have equal length")
    for layer, child in zip(layers, children):
        layer.mlp = child
    model_parameters = list(model.parameters())
    previous_requires_grad = [parameter.requires_grad for parameter in model_parameters]
    for parameter in model_parameters:
        parameter.requires_grad_(False)
    for child in children:
        for parameter in child.parameters():
            parameter.requires_grad_(True)
        _freeze_sparse_copied_and_router_parameters(child)
    trainable_parameters = [
        parameter for child in children for parameter in child.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("hard-route block has no trainable correction parameters")
    previous_hard_train = []
    for child in children:
        previous_hard_train.extend(_set_hard_train_modules(child, True))
        child.train()
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    history = []
    try:
        for step in range(1, steps + 1):
            index = (step - 1) % len(input_batches)
            ids = input_batches[index]
            student_block = run_with_layer_output(model, ids, layers[-1]).float()
            teacher_block = teacher_block_outputs[index].to(
                device=device, dtype=torch.float32,
            )
            block_scale = teacher_block.square().mean().clamp_min(1e-6)
            block_loss = F.mse_loss(student_block, teacher_block) / block_scale
            local_loss = torch.zeros((), device=device, dtype=torch.float32)
            for child, child_batches in zip(children, local_io_batches):
                local_batch = child_batches[index]
                local_input = local_batch["input"].to(device=device, dtype=student_block.dtype)
                local_target = local_batch["output"].to(device=device, dtype=torch.float32)
                local_prediction = child(local_input).float()
                local_scale = local_target.square().mean().clamp_min(1e-6)
                local_loss = local_loss + (
                    F.mse_loss(local_prediction, local_target) / local_scale
                )
            loss = local_weight * local_loss + block_weight * block_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite block loss at step {step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
            optimizer.step()
            if step == 1 or step % log_every == 0 or step == steps:
                history.append({
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "local_loss": float(local_loss.detach().cpu()),
                    "block_loss": float(block_loss.detach().cpu()),
                })
    finally:
        for parameter, previous in zip(model_parameters, previous_requires_grad):
            parameter.requires_grad_(previous)
        for nested, previous in previous_hard_train:
            nested.hard_train = previous
    for child in children:
        child.eval()
    return history


def joint_logit_refine_many(
    model: torch.nn.Module,
    layers: list[torch.nn.Module],
    children: list[torch.nn.Module],
    input_batches: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    device: torch.device,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    log_every: int,
    temperature: float,
) -> list[dict[str, float]]:
    """Refine a complete sparse cascade against frozen teacher logits."""
    for layer, child in zip(layers, children):
        layer.mlp = child
    model_parameters = list(model.parameters())
    previous_requires_grad = [parameter.requires_grad for parameter in model_parameters]
    previous_hard_train = []
    for child in children:
        previous_hard_train.extend(_set_hard_train_modules(child, True))
    for parameter in model_parameters:
        parameter.requires_grad_(False)
    for child in children:
        for parameter in child.parameters():
            parameter.requires_grad_(True)
        _freeze_sparse_copied_and_router_parameters(child)
    child_parameters = [
        parameter for child in children for parameter in child.parameters()
        if parameter.requires_grad
    ]
    if not child_parameters:
        for child in children:
            child.eval()
        return []
    optimizer = torch.optim.AdamW(child_parameters, lr=learning_rate)
    history = []
    for child in children:
        child.train()
    try:
        for step in range(1, steps + 1):
            index = (step - 1) % len(input_batches)
            ids = input_batches[index]
            target = teacher_logits[index].to(device=device, dtype=torch.float32)
            student = model(input_ids=ids, use_cache=False).logits.float()
            target_probs = torch.softmax(target / temperature, dim=-1)
            student_log_probs = F.log_softmax(student / temperature, dim=-1)
            loss = F.kl_div(
                student_log_probs,
                target_probs,
                reduction="batchmean",
            ) * (temperature * temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite joint loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(child_parameters, max_grad_norm)
            optimizer.step()
            if step == 1 or step % log_every == 0 or step == steps:
                history.append({"step": step, "loss": float(loss.detach().cpu())})
    finally:
        for parameter, previous in zip(model_parameters, previous_requires_grad):
            parameter.requires_grad_(previous)
        for nested, previous in previous_hard_train:
            nested.hard_train = previous
    for child in children:
        child.eval()
    return history


def layerwise_logit_refine_many(
    model: torch.nn.Module,
    layers: list[torch.nn.Module],
    parents: list[torch.nn.Module],
    children: list[torch.nn.Module],
    input_batches: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    device: torch.device,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    log_every: int,
    temperature: float,
) -> list[list[dict[str, float]]]:
    """Refine one sparse child at a time against the final teacher logits.

    Earlier layers use their already-trained sparse children, the active layer
    uses its sparse child, and later layers remain dense parents.  This keeps
    the optimization signal task-aware while preventing a joint optimizer from
    trading errors between multiple sparse interfaces at once.
    """
    if steps < 1:
        raise ValueError("layerwise refinement steps must be positive")
    if len(input_batches) != len(teacher_logits):
        raise ValueError("layerwise inputs and teacher logits must have equal length")
    for layer, parent in zip(layers, parents):
        layer.mlp = parent
    model_parameters = list(model.parameters())
    previous_requires_grad = [parameter.requires_grad for parameter in model_parameters]
    history: list[list[dict[str, float]]] = []
    try:
        for index, (layer, child) in enumerate(zip(layers, children)):
            for previous_layer, previous_child in zip(
                layers[:index], children[:index],
            ):
                previous_layer.mlp = previous_child
            layer.mlp = child
            for later_layer, later_parent in zip(
                layers[index + 1:], parents[index + 1:],
            ):
                later_layer.mlp = later_parent
            for parameter in model_parameters:
                parameter.requires_grad_(False)
            for parameter in child.parameters():
                parameter.requires_grad_(True)
            _freeze_sparse_copied_and_router_parameters(child)
            child_parameters = [
                parameter for parameter in child.parameters()
                if parameter.requires_grad
            ]
            if not child_parameters:
                child.eval()
                history.append([])
                continue
            previous_hard_train = _set_hard_train_modules(child, True)
            optimizer = torch.optim.AdamW(child_parameters, lr=learning_rate)
            child_history: list[dict[str, float]] = []
            model.eval()
            child.train()
            try:
                for step in range(1, steps + 1):
                    batch_index = (step - 1) % len(input_batches)
                    ids = input_batches[batch_index]
                    target = teacher_logits[batch_index].to(
                        device=device, dtype=torch.float32,
                    )
                    student = model(
                        input_ids=ids, use_cache=False,
                    ).logits.float()
                    target_probs = torch.softmax(target / temperature, dim=-1)
                    student_log_probs = F.log_softmax(
                        student / temperature, dim=-1,
                    )
                    loss = F.kl_div(
                        student_log_probs,
                        target_probs,
                        reduction="batchmean",
                    ) * (temperature * temperature)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite layerwise loss at layer {index}, step {step}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        child_parameters, max_grad_norm,
                    )
                    optimizer.step()
                    if (
                        step == 1 or step % log_every == 0 or step == steps
                    ):
                        child_history.append({
                            "step": step,
                            "loss": float(loss.detach().cpu()),
                        })
            finally:
                for nested, previous in previous_hard_train:
                    nested.hard_train = previous
            child.eval()
            history.append(child_history)
    finally:
        for parameter, previous in zip(model_parameters, previous_requires_grad):
            parameter.requires_grad_(previous)
    for layer, child in zip(layers, children):
        layer.mlp = child
    return history


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install requirements-transfer.txt first") from exc

    layer_indices = parse_layers(args.layers)
    active_experts_schedule = parse_schedule(
        args.active_experts_schedule, len(layer_indices), args.active_experts,
        int, "active-experts-schedule",
    )
    calibration_rank_schedule = parse_schedule(
        args.calibration_rank_schedule, len(layer_indices), args.calibration_rank,
        int, "calibration-rank-schedule",
    )
    hard_route_scale_schedule = parse_schedule(
        args.hard_route_scale_schedule, len(layer_indices), args.hard_route_scale,
        float, "hard-route-scale-schedule",
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
    )
    calibration_text = load_text_file(
        args.calibration_text_file, TRAIN_TEXT, "calibration",
    )
    eval_text = load_text_file(args.eval_text_file, EVAL_TEXT, "evaluation")
    train_ids = token_stream(
        tokenizer, calibration_text, args.batch_size,
        args.sequence_length * args.train_batches, device,
    ).reshape(args.train_batches, args.batch_size, args.sequence_length)
    eval_ids = token_stream(
        tokenizer, eval_text, args.batch_size,
        args.sequence_length * args.eval_batches, device,
    ).reshape(args.eval_batches, args.batch_size, args.sequence_length)
    with torch.no_grad():
        teacher_logits_gpu = [
            model(input_ids=ids, use_cache=False).logits.detach()
            for ids in eval_ids
        ]
    teacher_ce = sum(
        ce(logits.float(), ids)
        for logits, ids in zip(teacher_logits_gpu, eval_ids)
    ) / len(eval_ids)
    # The vocabulary logits are large; keep them off the GPU while children
    # are trained and copy only an evaluation batch back when needed.
    teacher_logits = [
        logits.to(device="cpu", dtype=torch.float16)
        for logits in teacher_logits_gpu
    ]
    layerwise_input_batches = []
    layerwise_teacher_logits = []
    if args.layerwise_steps > 0:
        layerwise_input_batches = list(train_ids)
        with torch.no_grad():
            layerwise_teacher_logits_gpu = [
                model(input_ids=ids, use_cache=False).logits.detach()
                for ids in layerwise_input_batches
            ]
        layerwise_teacher_logits = [
            logits.to(device="cpu", dtype=torch.float16)
            for logits in layerwise_teacher_logits_gpu
        ]
    joint_input_batches = []
    joint_teacher_logits = []
    if args.joint_steps > 0:
        joint_count = min(args.joint_calibration_batches, len(train_ids))
        if joint_count < 1:
            raise ValueError("joint-calibration-batches must be positive")
        joint_batch_size = (
            args.joint_batch_size
            if args.joint_batch_size is not None else args.batch_size
        )
        if joint_batch_size < 1:
            raise ValueError("joint-batch-size must be positive")
        joint_ids = token_stream(
            tokenizer, calibration_text, joint_batch_size,
            args.sequence_length * joint_count, device,
        ).reshape(joint_count, joint_batch_size, args.sequence_length)
        with torch.no_grad():
            joint_teacher_logits_gpu = [
                model(input_ids=ids, use_cache=False).logits.detach()
                for ids in joint_ids
            ]
        joint_input_batches = list(joint_ids)
        joint_teacher_logits = [
            logits.to(device="cpu", dtype=torch.float16)
            for logits in joint_teacher_logits_gpu
        ]

    layers = [model.model.layers[index] for index in layer_indices]
    parents = [layer.mlp for layer in layers]
    block_teacher_outputs = []
    if args.block_distill_steps > 0:
        if len(layer_indices) != 2 or layer_indices[1] != layer_indices[0] + 1:
            raise ValueError(
                "block distillation currently requires two consecutive layers"
            )
        block_teacher_outputs = capture_layer_outputs(
            model, list(train_ids), layers[-1],
        )
    hidden_size = int(model.config.hidden_size)
    children = []
    histories = []
    router_histories = []
    post_router_histories = []
    routing_metrics = []
    local_eval_mse = []
    child_train_io = []
    child_eval_io = []

    for layer_index, layer in zip(layer_indices, layers):
        train_io = capture_batches(
            model, tokenizer, calibration_text, args.batch_size,
            args.sequence_length, args.train_batches, device, layer_index,
        )
        eval_io = capture_batches(
            model, tokenizer, eval_text, args.batch_size,
            args.sequence_length, args.eval_batches, device, layer_index,
        )
        child_train_io.append(train_io)
        if args.child_kind == "qwen-transfer":
            child = make_transferred_qwen_child(
                parents[len(children)], args.calibration_rank, device, dtype,
            )
            histories.append([])
            router_histories.append([])
            post_router_histories.append([])
        elif args.child_kind == "qwen-transfer-sparse":
            child_index = len(children)
            child = make_transferred_routed_qwen_child(
                parents[child_index], args.num_experts,
                int(active_experts_schedule[child_index]),
                args.routing_temperature, int(calibration_rank_schedule[child_index]),
                args.calibration_source, args.calibration_mode,
                args.dispatch_mode,
                args.partition_mode, args.route_source,
                hard_route_scale_schedule[child_index], device, dtype,
                partition_io=train_io,
                router_hidden_size=args.router_hidden_size,
                router_input=args.router_input,
                pairwise_cost_parameterization=args.pairwise_cost_parameterization,
                core_overlap_fraction=args.core_overlap_fraction,
                router_sketch_dim=args.router_sketch_dim,
            )
            if args.calibration_mode == "teacher-group-decoder":
                initialize_teacher_group_decoders(
                    child, train_io, device, dtype,
                )
            elif args.calibration_mode == "teacher-group-residual":
                initialize_teacher_group_residuals(
                    child, train_io, device, dtype,
                )
            elif args.calibration_mode == "signed-subset":
                initialize_signed_subset_reconstruction(
                    child, train_io, device, dtype,
                )
            router_histories.append(train_importance_router(
                child, train_io, device, dtype, args.router_supervision_steps,
                args.learning_rate, args.max_grad_norm, args.log_every,
                args.router_target,
                args.router_target_temperature,
            ))
            frozen_subset_router = []
            if args.route_source in {
                "subset-router", "oracle-subset", "pairwise-cost-router",
            }:
                route_base = next(
                    nested for nested in child.modules()
                    if isinstance(nested, TransferredRoutedQwenChild)
                )
                router_module = (
                    route_base.pairwise_cost_router
                    if args.route_source == "pairwise-cost-router"
                    else route_base.subset_router
                )
                for parameter in router_module.parameters():
                    frozen_subset_router.append(
                        (parameter, bool(parameter.requires_grad))
                    )
                    parameter.requires_grad_(False)
            try:
                if any(parameter.requires_grad for parameter in child.parameters()):
                    histories.append(train_child(
                        child, train_io, device, dtype, args.steps,
                        args.learning_rate, args.max_grad_norm, args.log_every,
                        args.hard_train_steps,
                        args.hard_learning_rate,
                        args.hard_transition_steps,
                    ))
                else:
                    child.eval()
                    histories.append([])
            finally:
                for parameter, previous in frozen_subset_router:
                    parameter.requires_grad_(previous)
            if args.calibration_mode == "output-contract":
                contract = next(
                    nested for nested in child.modules()
                    if isinstance(nested, OutputContractRoutedQwenChild)
                )
                contract.fit(train_io, device)
            if (
                args.post_router_supervision_steps > 0
                and args.route_source in {
                    "subset-router", "oracle-subset", "pairwise-cost-router",
                }
            ):
                # The child correction is now fixed.  Recompute exact
                # best-subset labels against the final copied/corrected child
                # before the last router fit; pre-training labels can become
                # stale after hard child optimization.
                post_router_histories.append(train_importance_router(
                    child, train_io, device, dtype,
                    args.post_router_supervision_steps,
                    args.learning_rate, args.max_grad_norm, args.log_every,
                    (
                        args.post_router_target
                        if args.post_router_target is not None
                        else args.router_target
                    ),
                    args.router_target_temperature,
                ))
            else:
                post_router_histories.append([])
        elif args.child_kind == "qwen-transfer-neuron-sparse":
            child = make_transferred_routed_qwen_neuron_child(
                parents[len(children)], args.active_neurons,
                args.routing_temperature, args.calibration_rank,
                args.calibration_source, args.calibration_mode, device, dtype,
                args.route_source, hard_route_scale_schedule[len(children)],
            )
            router_histories.append(train_neuron_importance_router(
                child, train_io, device, dtype, args.router_supervision_steps,
                args.learning_rate, args.max_grad_norm, args.log_every,
            ))
            histories.append(train_child(
                child, train_io, device, dtype, args.steps,
                args.learning_rate, args.max_grad_norm, args.log_every,
                args.hard_train_steps,
                args.hard_learning_rate,
                args.hard_transition_steps,
            ))
            post_router_histories.append([])
        elif args.child_kind == "qwen-latent-basis":
            child_index = len(children)
            child = make_learned_latent_basis_qwen_child(
                hidden_size, args.num_experts,
                int(active_experts_schedule[child_index]),
                args.calibration_rank, args.routing_temperature,
                hard_route_scale_schedule[child_index], device, dtype,
            )
            histories.append(train_child(
                child, train_io, device, dtype, args.steps,
                args.learning_rate, args.max_grad_norm, args.log_every,
                args.hard_train_steps,
                args.hard_learning_rate,
                args.hard_transition_steps,
            ))
            router_histories.append([])
            post_router_histories.append([])
        else:
            child = make_child(
                hidden_size, args.inner_size, args.child_kind,
                args.calibration_rank, args.num_experts, args.active_experts,
                args.routing_temperature, device, dtype, args.dispatch_mode,
                not args.child_no_norm,
            )
            histories.append(train_child(
                child, train_io, device, dtype, args.steps,
                args.learning_rate, args.max_grad_norm, args.log_every,
                args.hard_train_steps,
                args.hard_learning_rate,
                args.hard_transition_steps,
            ))
            router_histories.append([])
            post_router_histories.append([])
        with torch.no_grad():
            mse = sum(
                F.mse_loss(
                    child(batch["input"].to(device=device, dtype=dtype)).float(),
                    batch["output"].to(device=device).float(),
                ).item()
                for batch in eval_io
            ) / len(eval_io)
        local_eval_mse.append(mse)
        routing_metrics.append(routing_diagnostics(child, eval_io, device, dtype))
        children.append(child)
        child_eval_io.append(eval_io)
        # The next child is calibrated on the representation produced by all
        # previously trained children, matching the eventual cascade.
        layer.mlp = child

    block_history = []
    if args.block_distill_steps > 0:
        if args.child_kind != "qwen-transfer-sparse":
            raise ValueError(
                "block distillation currently requires qwen-transfer-sparse"
            )
        block_history = hard_route_block_refine(
            model, layers, children, child_train_io, list(train_ids),
            block_teacher_outputs, device, args.block_distill_steps,
            args.block_distill_learning_rate, args.block_local_weight,
            args.block_distill_weight, args.max_grad_norm, args.log_every,
        )
    joint_history = []
    layerwise_history = []
    if args.layerwise_steps > 0:
        layerwise_history = layerwise_logit_refine_many(
            model, layers, parents, children,
            layerwise_input_batches, layerwise_teacher_logits, device,
            args.layerwise_steps, args.layerwise_learning_rate,
            args.max_grad_norm, args.log_every, args.joint_temperature,
        )
    if args.joint_steps > 0:
        joint_history = joint_logit_refine_many(
            model, layers, children, joint_input_batches, joint_teacher_logits, device,
            args.joint_steps, args.joint_learning_rate, args.max_grad_norm,
            args.log_every, args.joint_temperature,
        )

    pre_router_refit_routing_metrics = list(routing_metrics)
    router_refit_history = []
    if args.router_refit_mode != "none":
        if args.child_kind != "qwen-transfer-sparse":
            raise ValueError(
                "router refit currently requires qwen-transfer-sparse"
            )
        if args.route_source != "pairwise-cost-router":
            raise ValueError(
                "router refit currently requires pairwise-cost-router"
            )
        router_refit_history = refit_pairwise_router_cascade(
            model, tokenizer, calibration_text, args.batch_size,
            args.sequence_length, args.train_batches, device, dtype,
            layer_indices, layers, parents, children, child_train_io,
            args.router_refit_mode, args.router_refit_steps,
            args.router_refit_rounds, args.router_refit_learning_rate,
            args.max_grad_norm, args.log_every,
            args.router_target_temperature,
            args.router_refit_target,
        )
        routing_metrics = [
            routing_diagnostics(child, eval_io, device, dtype)
            for child, eval_io in zip(children, child_eval_io)
        ]

    variants = []
    for alpha in args.alphas:
        for layer, parent, child in zip(layers, parents, children):
            layer.mlp = MixedParentChild(parent, child, alpha)
        variants.append(evaluate_current(
            model, list(eval_ids), teacher_logits, teacher_ce,
            f"shared_alpha_{alpha:g}",
        ))
    for layer, parent in zip(layers, parents):
        layer.mlp = parent

    leave_one_child_variants = []
    if args.leave_one_child_ablation and len(children) > 1:
        for ablated_index, (ablated_layer, ablated_parent) in enumerate(
            zip(layers, parents)
        ):
            for index, (layer, parent, child) in enumerate(
                zip(layers, parents, children)
            ):
                layer.mlp = parent if index == ablated_index else child
            leave_one_child_variants.append(evaluate_current(
                model, list(eval_ids), teacher_logits, teacher_ce,
                f"leave_out_layer_{layer_indices[ablated_index]}",
            ))
        for layer, parent in zip(layers, parents):
            layer.mlp = parent

    paired_oracle_variants = []
    if args.paired_oracle_routing:
        route_bases = []
        for child in children:
            route_base = next(
                nested for nested in child.modules()
                if isinstance(nested, TransferredRoutedQwenChild)
            )
            if route_base.route_source not in {
                "subset-router", "oracle-subset", "pairwise-cost-router",
            }:
                raise ValueError(
                    "paired oracle routing requires subset-router children"
                )
            route_bases.append(route_base)
            route_base.route_source = "oracle-subset"
        for layer, child in zip(layers, children):
            layer.mlp = child
        paired_oracle_variants.append(evaluate_current(
            model, list(eval_ids), teacher_logits, teacher_ce,
            "paired_oracle_routing_alpha_0",
        ))
        for route_base in route_bases:
            route_base.route_source = args.route_source
        for layer, parent in zip(layers, parents):
            layer.mlp = parent

    prompt_results = []
    if args.prompt_parity:
        prompt_results = prompt_parity(
            model, tokenizer, layers, parents, children,
            args.generation_tokens,
        )

    timing_model = model
    if args.torch_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("this PyTorch build has no torch.compile")
        timing_model = torch.compile(
            model, mode=args.torch_compile_mode, dynamic=False,
        )
    parent_timing = benchmark_forward(
        timing_model, list(eval_ids), device,
        args.timing_warmup, args.timing_iterations,
    )
    for layer, child in zip(layers, children):
        layer.mlp = child
    sparse_timing = benchmark_forward(
        timing_model, list(eval_ids), device,
        args.timing_warmup, args.timing_iterations,
    )
    for layer, parent in zip(layers, parents):
        layer.mlp = parent

    alpha_zero = next(
        item for item in variants if item["variant"] == "shared_alpha_0"
    )
    child_params = [sum(parameter.numel() for parameter in child.parameters()) for child in children]
    child_buffer_scalars = [
        sum(buffer.numel() for buffer in child.buffers())
        for child in children
    ]
    child_storage_scalars = [
        parameter_count + buffer_count
        for parameter_count, buffer_count in zip(
            child_params, child_buffer_scalars,
        )
    ]
    parent_params = [sum(parameter.numel() for parameter in parent.parameters()) for parent in parents]
    if args.child_kind == "qwen-transfer":
        effective_child_inner_size = int(parents[0].gate_proj.out_features)
    elif args.child_kind == "qwen-transfer-sparse":
        effective_child_inner_size = int(
            parents[0].gate_proj.out_features // args.num_experts
        )
    elif args.child_kind == "qwen-transfer-neuron-sparse":
        effective_child_inner_size = 1
    elif args.child_kind == "qwen-latent-basis":
        effective_child_inner_size = args.calibration_rank
    else:
        effective_child_inner_size = args.inner_size
    result = {
        "experiment": "qwen_multi_layer_attention_free_parent_transplant",
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "layers": layer_indices,
        "hidden_size": hidden_size,
        "child_inner_size": effective_child_inner_size,
        "child_kind": args.child_kind,
        "calibration_rank": args.calibration_rank,
        "calibration_rank_schedule": calibration_rank_schedule,
        "calibration_source": args.calibration_source,
        "calibration_mode": args.calibration_mode,
        "num_experts": args.num_experts,
        "active_experts": args.active_experts,
        "active_experts_schedule": active_experts_schedule,
        "active_neurons": args.active_neurons,
        "hard_route_expected_expert_fraction": (
            args.active_neurons / parents[0].gate_proj.out_features
            if args.child_kind == "qwen-transfer-neuron-sparse"
            else [
                float(active_experts) / args.num_experts
                for active_experts in active_experts_schedule
            ]
            if args.child_kind in {
                "routed", "qwen-transfer-sparse", "qwen-latent-basis",
            }
            else 1.0
        ),
        "hard_route_dispatch": (
            f"selected-token-only:{args.dispatch_mode}"
            if args.child_kind in {
                "routed", "qwen-transfer-sparse", "qwen-transfer-neuron-sparse",
                "qwen-latent-basis",
            }
            else "single-child"
        ),
        "dispatch_mode": args.dispatch_mode,
        "partition_mode": args.partition_mode,
        "route_source": args.route_source,
        "router_hidden_size": args.router_hidden_size,
        "router_input": args.router_input,
        "router_sketch_dim": args.router_sketch_dim,
        "pairwise_cost_parameterization": args.pairwise_cost_parameterization,
        "router_target": args.router_target,
        "post_router_target": args.post_router_target,
        "router_target_temperature": args.router_target_temperature,
        "router_refit_mode": args.router_refit_mode,
        "router_refit_steps": args.router_refit_steps,
        "router_refit_rounds": args.router_refit_rounds,
        "router_refit_learning_rate": args.router_refit_learning_rate,
        "router_refit_target": args.router_refit_target,
        "hard_route_scale": args.hard_route_scale,
        "hard_route_scale_schedule": hard_route_scale_schedule,
        "child_internal_norm": (
            not args.child_no_norm
            if args.child_kind == "routed" else None
        ),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "train_batches": args.train_batches,
        "calibration_text_file": args.calibration_text_file,
        "calibration_text_chars": len(calibration_text),
        "eval_text_file": args.eval_text_file,
        "eval_text_chars": len(eval_text),
        "eval_batches": args.eval_batches,
        "distillation_steps_per_child": args.steps,
        "hard_train_steps_per_child": args.hard_train_steps,
        "hard_learning_rate": args.hard_learning_rate,
        "hard_transition_steps_per_child": args.hard_transition_steps,
        "router_supervision_steps_per_child": args.router_supervision_steps,
        "post_router_supervision_steps_per_child": args.post_router_supervision_steps,
        "joint_distillation_steps": args.joint_steps,
        "joint_training_corpus": "calibration" if args.joint_steps > 0 else None,
        "joint_calibration_batches": args.joint_calibration_batches,
        "joint_batch_size": args.joint_batch_size,
        "joint_learning_rate": args.joint_learning_rate,
        "joint_temperature": args.joint_temperature,
        "block_distillation_steps": args.block_distill_steps,
        "block_distillation_learning_rate": args.block_distill_learning_rate,
        "block_local_weight": args.block_local_weight,
        "block_distillation_weight": args.block_distill_weight,
        "block_train_history": block_history,
        "layerwise_distillation_steps": args.layerwise_steps,
        "layerwise_learning_rate": args.layerwise_learning_rate,
        "layerwise_train_history": layerwise_history,
        "teacher_ce": teacher_ce,
        "child_train_history": histories,
        "router_train_history": router_histories,
        "post_router_train_history": post_router_histories,
        "pre_router_refit_routing_metrics": pre_router_refit_routing_metrics,
        "router_refit_train_history": router_refit_history,
        "routing_metrics": routing_metrics,
        "joint_train_history": joint_history,
        "prompt_parity": prompt_results,
        "child_local_eval_mse": local_eval_mse,
        "parent_scalar_params_each": parent_params,
        "child_scalar_params_each": child_params,
        "child_buffer_scalars_each": child_buffer_scalars,
        "child_storage_scalars_each": child_storage_scalars,
        "child_parameter_fraction_each": [
            child_count / max(parent_count, 1)
            for child_count, parent_count in zip(child_params, parent_params)
        ],
        "child_storage_fraction_each": [
            child_count / max(parent_count, 1)
            for child_count, parent_count in zip(
                child_storage_scalars, parent_params,
            )
        ],
        "variants": variants,
        "leave_one_child_variants": leave_one_child_variants,
        "paired_oracle_variants": paired_oracle_variants,
        "timing": {
            "parent": parent_timing,
            "sparse_bank": sparse_timing,
            "bank_over_parent_mean_ratio": (
                sparse_timing["mean_ms_per_batch"]
                / max(parent_timing["mean_ms_per_batch"], 1e-9)
            ),
            "warmup": args.timing_warmup,
            "iterations": args.timing_iterations,
            "torch_compile": args.torch_compile,
            "torch_compile_mode": args.torch_compile_mode if args.torch_compile else None,
            "note": "end-to-end Qwen forward with all selected layers replaced",
        },
        "quality_gate": {
            "criterion": "shared alpha=0 CE delta <= 0.05 and all outputs finite",
            "passed": bool(
                float(alpha_zero["ce_delta"]) <= args.max_ce_delta
                and torch.isfinite(torch.tensor(float(alpha_zero["ce"])))
            ),
            "max_ce_delta": args.max_ce_delta,
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layers", default="23,24,25,26")
    parser.add_argument("--inner-size", type=int, default=384)
    parser.add_argument(
        "--child-kind",
        choices=(
            "gelu", "swiglu", "routed", "qwen-transfer", "qwen-transfer-sparse",
            "qwen-transfer-neuron-sparse", "qwen-latent-basis",
        ),
        default="routed",
    )
    parser.add_argument("--calibration-rank", type=int, default=8)
    parser.add_argument(
        "--calibration-rank-schedule", default=None,
        help="optional comma-separated correction rank per replaced layer",
    )
    parser.add_argument(
        "--calibration-source", choices=("base-output", "input"),
        default="base-output",
        help="hidden-state source for the low-rank correction",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=(
            "low-rank", "swiglu", "shared-basis", "cross-group",
            "teacher-group-decoder", "teacher-group-residual", "residual-coreset",
            "output-contract", "signed-subset",
        ),
        default="low-rank",
        help="correction type for transferred sparse children",
    )
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument(
        "--active-experts-schedule", default=None,
        help="optional comma-separated active expert count per replaced layer",
    )
    parser.add_argument(
        "--active-neurons", type=int, default=768,
        help="active Qwen neurons for qwen-transfer-neuron-sparse",
    )
    parser.add_argument("--routing-temperature", type=float, default=1.0)
    parser.add_argument(
        "--dispatch-mode",
        choices=(
            "grouped", "grouped-fused", "packed", "packed-fused", "packed-fp16", "fused",
            "token-loop",
        ),
        default="grouped",
    )
    parser.add_argument(
        "--partition-mode",
        choices=(
            "contiguous", "interleaved", "norm-balanced", "activation-balanced",
            "sampled-overlap", "stratified-overlap", "activation-cluster",
            "contribution-cluster", "contribution-diverse", "core-overlap",
        ),
        default="contiguous",
        help="layout of copied parent neurons inside expert groups",
    )
    parser.add_argument(
        "--core-overlap-fraction", type=float, default=0.25,
        help="fraction of each group repeated in core-overlap mode",
    )
    parser.add_argument(
        "--route-source",
        choices=(
            "router", "subset-router", "oracle-dot", "oracle-energy",
            "oracle-subset", "pairwise-cost-router",
        ), default="router",
        help="learned router or diagnostic parent-contribution oracle at eval",
    )
    parser.add_argument(
        "--router-hidden-size", type=int, default=128,
        help="hidden width of the learned group/subset router",
    )
    parser.add_argument(
        "--router-input",
        choices=("hidden", "group-energy", "group-sketch"),
        default="hidden",
        help="features provided to the learned subset router",
    )
    parser.add_argument(
        "--router-sketch-dim", type=int, default=8,
        help="signed output sketch dimensions per group for group-sketch input",
    )
    parser.add_argument(
        "--pairwise-cost-parameterization",
        choices=("components", "centered-basis"),
        default="components",
        help="coordinates used by the pairwise cost-router output head",
    )
    parser.add_argument(
        "--hard-route-scale", type=float, default=None,
        help="override the sparse hard-route output scale (default E/K)",
    )
    parser.add_argument(
        "--hard-route-scale-schedule", default=None,
        help="optional comma-separated hard-route scale per replaced layer",
    )
    parser.add_argument("--child-no-norm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument(
        "--calibration-text-file", default=None,
        help="optional UTF-8 text corpus for child/router calibration",
    )
    parser.add_argument(
        "--eval-text-file", default=None,
        help="optional UTF-8 held-out text corpus for evaluation",
    )
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--hard-train-steps", type=int, default=0,
        help="final child-training steps using the hard top-k route",
    )
    parser.add_argument(
        "--hard-learning-rate", type=float, default=None,
        help="optional learning rate after switching to hard top-k routing",
    )
    parser.add_argument(
        "--hard-transition-steps", type=int, default=0,
        help="soft-to-hard route blend steps inside the hard-training phase",
    )
    parser.add_argument(
        "--router-supervision-steps", type=int, default=0,
        help="steps distilling frozen expert contribution importance into the router",
    )
    parser.add_argument(
        "--post-router-supervision-steps", type=int, default=0,
        help="extra exact-subset router steps after child correction training",
    )
    parser.add_argument(
        "--router-refit-mode",
        choices=("none", "static", "aggregate"),
        default="none",
        help="final pairwise-router refit on static or current-cascade data",
    )
    parser.add_argument(
        "--router-refit-steps", type=int, default=300,
        help="steps per static refit or aggregation round",
    )
    parser.add_argument(
        "--router-refit-rounds", type=int, default=3,
        help="number of aggregation rounds; static mode uses one pass",
    )
    parser.add_argument(
        "--router-refit-learning-rate", type=float, default=1e-4,
        help="learning rate for final router refit/aggregation",
    )
    parser.add_argument(
        "--router-refit-target",
        choices=("pairwise-regret", "pairwise-regret-tail"),
        default="pairwise-regret",
        help="objective used by the final pairwise-router refit",
    )
    parser.add_argument(
        "--router-target",
        choices=(
            "energy", "dot", "subset", "subset-soft", "final-subset-soft",
            "pairwise-regret",
        ), default="energy",
        help="calibration target for group router supervision",
    )
    parser.add_argument(
        "--post-router-target",
        choices=(
            "subset", "subset-soft", "final-subset-soft", "pairwise-regret",
        ),
        default=None,
        help="optional target override for post-child router supervision",
    )
    parser.add_argument(
        "--router-target-temperature", type=float, default=1.0,
        help="temperature for cost-aware subset-soft router supervision",
    )
    parser.add_argument("--joint-steps", type=int, default=0)
    parser.add_argument(
        "--joint-calibration-batches", type=int, default=4,
        help="number of calibration batches used for leakage-free joint refinement",
    )
    parser.add_argument(
        "--joint-batch-size", type=int, default=None,
        help="optional smaller batch size for memory-safe joint refinement",
    )
    parser.add_argument("--joint-learning-rate", type=float, default=1e-4)
    parser.add_argument("--joint-temperature", type=float, default=2.0)
    parser.add_argument(
        "--block-distill-steps", type=int, default=0,
        help="hard two-layer block refinement steps",
    )
    parser.add_argument(
        "--block-distill-learning-rate", type=float, default=1e-5,
        help="learning rate for hard two-layer block refinement",
    )
    parser.add_argument(
        "--block-local-weight", type=float, default=1.0,
        help="normalized local child-loss weight during block refinement",
    )
    parser.add_argument(
        "--block-distill-weight", type=float, default=1.0,
        help="normalized two-layer block-loss weight during block refinement",
    )
    parser.add_argument(
        "--layerwise-steps", type=int, default=0,
        help="task-loss refinement steps for each sparse child in cascade order",
    )
    parser.add_argument(
        "--layerwise-learning-rate", type=float, default=1e-5,
        help="learning rate for per-layer task-loss refinement",
    )
    parser.add_argument("--prompt-parity", action="store_true")
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-ce-delta", type=float, default=0.05)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument(
        "--leave-one-child-ablation", action="store_true",
        help="evaluate each replaced layer with its original parent restored",
    )
    parser.add_argument(
        "--paired-oracle-routing", action="store_true",
        help="re-evaluate the same trained children with exact best-subset routing",
    )
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--timing-iterations", type=int, default=30)
    parser.add_argument(
        "--torch-compile", action="store_true",
        help="opt-in torch.compile timing path; compilation overhead is excluded",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_multi_layer_transplant.json")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
