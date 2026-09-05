"""Sequential multi-layer Qwen circuit-bank transfer benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

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
    if len(layers) < 2 or len(set(layers)) != len(layers):
        raise ValueError("layers must contain at least two distinct indices")
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
        if partition_mode not in {
            "contiguous", "interleaved", "norm-balanced", "activation-balanced",
            "sampled-overlap", "stratified-overlap", "activation-cluster",
        }:
            raise ValueError(
                "partition_mode must be contiguous, interleaved, norm-balanced, "
                "activation-balanced, sampled-overlap, stratified-overlap, "
                "or activation-cluster"
            )
        self.partition_mode = partition_mode
        if route_source not in {"router", "oracle-dot", "oracle-energy"}:
            raise ValueError("route_source must be router, oracle-dot, or oracle-energy")
        self.route_source = route_source
        self.hard_route_scale = (
            self.num_experts / self.active_experts
            if hard_route_scale is None else float(hard_route_scale)
        )
        if dispatch_mode not in {"grouped", "token-loop"}:
            raise ValueError("transferred sparse child supports grouped or token-loop")
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
        self.router = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, num_experts),
        )
        torch.nn.init.zeros_(self.router[-1].weight)
        torch.nn.init.zeros_(self.router[-1].bias)
        self.last_selected: torch.Tensor | None = None
        self.last_route_weights: torch.Tensor | None = None
        self.last_all_outputs: torch.Tensor | None = None
        self.last_selected_outputs: torch.Tensor | None = None
        self.last_active_expert_fraction = 1.0

    def _forward_grouped(
        self,
        hidden_states: torch.Tensor,
        top_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_ids = top_ids.reshape(-1, self.active_experts)
        flat_weights = weights.reshape(-1, self.active_experts)
        pair_indices = torch.arange(flat_ids.numel(), device=flat_ids.device)
        token_ids = pair_indices // self.active_experts
        slots = pair_indices % self.active_experts
        expert_ids = flat_ids[token_ids, slots]
        sort_order = torch.argsort(expert_ids, stable=True)
        sorted_experts = expert_ids[sort_order]
        counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        max_count = int(counts.max().item())
        starts = counts.cumsum(dim=0) - counts
        positions = torch.arange(
            pair_indices.numel(), device=flat_ids.device,
        ) - starts[sorted_experts]
        grouped_indices = sorted_experts * max_count + positions
        grouped_hidden = torch.zeros(
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
        grouped_gate = F.silu(torch.bmm(
            grouped_hidden, self.group_gate_weight.transpose(1, 2),
        ))
        grouped_value = torch.bmm(
            grouped_hidden, self.group_value_weight.transpose(1, 2),
        )
        grouped_output = torch.bmm(
            grouped_gate * grouped_value,
            self.group_output_weight.transpose(1, 2),
        )
        selected_output = grouped_output.reshape(
            self.num_experts * max_count, flat_hidden.shape[-1],
        ).index_select(0, grouped_indices)
        sorted_token_ids = token_ids[sort_order]
        sorted_slots = slots[sort_order]
        selected_by_pair = torch.empty_like(selected_output)
        pair_slots = sorted_token_ids * self.active_experts + sorted_slots
        selected_by_pair.index_copy_(0, pair_slots, selected_output)
        self.last_selected_outputs = selected_by_pair.reshape(
            flat_hidden.shape[0], self.active_experts, flat_hidden.shape[-1],
        ).reshape(*hidden_states.shape[:-1], self.active_experts, hidden_states.shape[-1])
        contribution = selected_output * flat_weights[
            sorted_token_ids, sorted_slots,
        ].unsqueeze(-1)
        flat_output = torch.zeros_like(flat_hidden)
        flat_output.index_add_(0, sorted_token_ids, contribution)
        self.last_active_expert_fraction = pair_indices.numel() / max(
            flat_hidden.shape[0] * self.num_experts, 1
        )
        # The router selects contribution-heavy groups rather than a random
        # subset, so the empirical stable scale is E/K, not an unbiased E
        # estimator that over-corrects the selected high-energy groups.
        return self.hard_route_scale * flat_output.reshape_as(
            hidden_states,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.router(hidden_states)
        oracle_outputs = None
        if not self.training and self.route_source in {"oracle-dot", "oracle-energy"}:
            oracle_outputs = torch.stack([
                expert(hidden_states) for expert in self.experts
            ], dim=-2)
            if self.route_source == "oracle-dot":
                full_output = oracle_outputs.sum(dim=-2)
                scores = (oracle_outputs * full_output.unsqueeze(-2)).sum(dim=-1)
            else:
                scores = oracle_outputs.square().mean(dim=-1)
        if self.training and not self.hard_train:
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
        top_values, top_ids = scores.topk(self.active_experts, dim=-1)
        weights = F.softmax(top_values / self.temperature, dim=-1)
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
        if not self.training and self.dispatch_mode == "grouped":
            return self._forward_grouped(hidden_states, top_ids, weights)
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
        # Top-k weights are normalized over the selected groups; rescale to
        # estimate the full intermediate-neuron sum from the active subset.
        return (
            self.num_experts / self.active_experts
            * flat_output.reshape_as(hidden_states)
        )


class TransferredRoutedQwenNeuronChild(torch.nn.Module):
    """Copied Qwen neurons with token-level top-k execution."""

    def __init__(
        self,
        parent: torch.nn.Module,
        active_neurons: int,
        temperature: float,
        token_chunk_size: int = 64,
        route_source: str = "router",
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
        return (
            self.num_neurons / self.active_neurons
            * flat_output.reshape_as(hidden_states)
        )

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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        route_weights = self.base.last_route_weights
        selected = self.base.last_selected
        if route_weights is None or selected is None:
            raise RuntimeError("base route state was not populated")
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
            selected_mix_in = self.mix_in[selected]
            selected_mix_out = self.mix_out[selected]
            latent = torch.einsum(
                "...kh,...krh->...kr", selected_outputs, selected_mix_in,
            )
            selected_corrections = torch.einsum(
                "...kr,...khr->...kh", latent, selected_mix_out,
            )
            correction = self.base.hard_route_scale * (
                selected_corrections * route_weights.unsqueeze(-1)
            ).sum(dim=-2)
        return correction if self.replace_base_output else base_output + correction


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
    child = TransferredRoutedQwenChild(
        parent, num_experts, active_experts, routing_temperature, dispatch_mode,
        partition_mode, route_source, hard_route_scale,
        partition_indices,
    ).to(device=device, dtype=dtype)
    if calibration_rank > 0:
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
) -> torch.nn.Module:
    child = TransferredRoutedQwenNeuronChild(
        parent, active_neurons, routing_temperature, route_source=route_source,
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
) -> list[dict[str, float]]:
    """Distill frozen group contribution importance into the cheap router."""
    base = next(
        nested for nested in child.modules()
        if isinstance(nested, TransferredRoutedQwenChild)
    )
    all_parameters = list(base.parameters())
    previous_requires_grad = [parameter.requires_grad for parameter in all_parameters]
    for parameter in all_parameters:
        parameter.requires_grad_(False)
    router_parameters = list(base.router.parameters())
    for parameter in router_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(router_parameters, lr=learning_rate)
    history = []
    base.eval()
    try:
        for step in range(1, steps + 1):
            batch = io_batches[(step - 1) % len(io_batches)]
            inputs = batch["input"].to(device=device, dtype=dtype)
            with torch.no_grad():
                outputs = torch.stack([
                    expert(inputs) for expert in base.experts
                ], dim=-2).float()
                if target_mode == "dot":
                    full_output = outputs.sum(dim=-2)
                    importance = (
                        outputs * full_output.unsqueeze(-2)
                    ).sum(dim=-1)
                elif target_mode == "energy":
                    importance = outputs.square().mean(dim=-1)
                else:
                    raise ValueError("router target must be energy or dot")
                target = F.softmax(
                    importance if target_mode == "dot"
                    else torch.log(importance + 1e-8),
                    dim=-1,
                )
            scores = base.router(inputs).float()
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
    child_parameters = [parameter for child in children for parameter in child.parameters()]
    previous_requires_grad = [parameter.requires_grad for parameter in model_parameters]
    previous_hard_train = []
    for child in children:
        previous_hard_train.extend(_set_hard_train_modules(child, True))
    for parameter in model_parameters:
        parameter.requires_grad_(False)
    for parameter in child_parameters:
        parameter.requires_grad_(True)
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
    hidden_size = int(model.config.hidden_size)
    children = []
    histories = []
    router_histories = []
    local_eval_mse = []

    for layer_index, layer in zip(layer_indices, layers):
        train_io = capture_batches(
            model, tokenizer, calibration_text, args.batch_size,
            args.sequence_length, args.train_batches, device, layer_index,
        )
        eval_io = capture_batches(
            model, tokenizer, eval_text, args.batch_size,
            args.sequence_length, args.eval_batches, device, layer_index,
        )
        if args.child_kind == "qwen-transfer":
            child = make_transferred_qwen_child(
                parents[len(children)], args.calibration_rank, device, dtype,
            )
            histories.append([])
            router_histories.append([])
        elif args.child_kind == "qwen-transfer-sparse":
            child_index = len(children)
            child = make_transferred_routed_qwen_child(
                parents[child_index], args.num_experts,
                int(active_experts_schedule[child_index]),
                args.routing_temperature, args.calibration_rank,
                args.calibration_source, args.calibration_mode,
                args.dispatch_mode,
                args.partition_mode, args.route_source,
                hard_route_scale_schedule[child_index], device, dtype,
                partition_io=train_io,
            )
            if args.calibration_mode == "teacher-group-decoder":
                initialize_teacher_group_decoders(
                    child, train_io, device, dtype,
                )
            elif args.calibration_mode == "teacher-group-residual":
                initialize_teacher_group_residuals(
                    child, train_io, device, dtype,
                )
            router_histories.append(train_importance_router(
                child, train_io, device, dtype, args.router_supervision_steps,
                args.learning_rate, args.max_grad_norm, args.log_every,
                args.router_target,
            ))
            histories.append(train_child(
                child, train_io, device, dtype, args.steps,
                args.learning_rate, args.max_grad_norm, args.log_every,
                args.hard_train_steps,
                args.hard_learning_rate,
            ))
        elif args.child_kind == "qwen-transfer-neuron-sparse":
            child = make_transferred_routed_qwen_neuron_child(
                parents[len(children)], args.active_neurons,
                args.routing_temperature, args.calibration_rank,
                args.calibration_source, args.calibration_mode, device, dtype,
                args.route_source,
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
            ))
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
            ))
            router_histories.append([])
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
            ))
            router_histories.append([])
        with torch.no_grad():
            mse = sum(
                F.mse_loss(
                    child(batch["input"].to(device=device, dtype=dtype)).float(),
                    batch["output"].to(device=device).float(),
                ).item()
                for batch in eval_io
            ) / len(eval_io)
        local_eval_mse.append(mse)
        children.append(child)
        # The next child is calibrated on the representation produced by all
        # previously trained children, matching the eventual cascade.
        layer.mlp = child

    joint_history = []
    if args.joint_steps > 0:
        joint_history = joint_logit_refine_many(
            model, layers, children, joint_input_batches, joint_teacher_logits, device,
            args.joint_steps, args.joint_learning_rate, args.max_grad_norm,
            args.log_every, args.joint_temperature,
        )

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

    prompt_results = []
    if args.prompt_parity:
        prompt_results = prompt_parity(
            model, tokenizer, layers, parents, children,
            args.generation_tokens,
        )

    parent_timing = benchmark_forward(
        model, list(eval_ids), device,
        args.timing_warmup, args.timing_iterations,
    )
    for layer, child in zip(layers, children):
        layer.mlp = child
    sparse_timing = benchmark_forward(
        model, list(eval_ids), device,
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
        "router_supervision_steps_per_child": args.router_supervision_steps,
        "joint_distillation_steps": args.joint_steps,
        "joint_training_corpus": "calibration" if args.joint_steps > 0 else None,
        "joint_calibration_batches": args.joint_calibration_batches,
        "joint_batch_size": args.joint_batch_size,
        "joint_learning_rate": args.joint_learning_rate,
        "joint_temperature": args.joint_temperature,
        "teacher_ce": teacher_ce,
        "child_train_history": histories,
        "router_train_history": router_histories,
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
        "timing": {
            "parent": parent_timing,
            "sparse_bank": sparse_timing,
            "bank_over_parent_mean_ratio": (
                sparse_timing["mean_ms_per_batch"]
                / max(parent_timing["mean_ms_per_batch"], 1e-9)
            ),
            "warmup": args.timing_warmup,
            "iterations": args.timing_iterations,
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
        "--calibration-source", choices=("base-output", "input"),
        default="base-output",
        help="hidden-state source for the low-rank correction",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=(
            "low-rank", "swiglu", "shared-basis", "cross-group",
            "teacher-group-decoder", "teacher-group-residual",
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
        "--dispatch-mode", choices=("grouped", "packed", "token-loop"),
        default="grouped",
    )
    parser.add_argument(
        "--partition-mode",
        choices=(
            "contiguous", "interleaved", "norm-balanced", "activation-balanced",
            "sampled-overlap", "stratified-overlap", "activation-cluster",
        ),
        default="contiguous",
        help="layout of copied parent neurons inside expert groups",
    )
    parser.add_argument(
        "--route-source",
        choices=("router", "oracle-dot", "oracle-energy"), default="router",
        help="learned router or diagnostic parent-contribution oracle at eval",
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
        "--router-supervision-steps", type=int, default=0,
        help="steps distilling frozen expert contribution importance into the router",
    )
    parser.add_argument(
        "--router-target", choices=("energy", "dot"), default="energy",
        help="calibration target for group router supervision",
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
    parser.add_argument("--prompt-parity", action="store_true")
    parser.add_argument("--generation-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-ce-delta", type=float, default=0.05)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--timing-iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_multi_layer_transplant.json")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
