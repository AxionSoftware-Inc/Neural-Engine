from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@dataclass
class StepStats:
    active_parameters: int
    active_circuits: int
    router_decisions: int
    router_entropy: float


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(value)


def parameter_bytes(module: nn.Module) -> int:
    """Return the storage size of trainable parameters in bytes."""
    return sum(parameter.numel() * parameter.element_size()
               for parameter in module.parameters() if parameter.requires_grad)


def estimate_neural_engine_macs(model: nn.Module, executed_steps: float,
                                value_tokens: float = 0.0) -> dict[str, float | int]:
    """Estimate inference MACs and parameter traffic for a Neural Engine path.

    This is an analytical model, not a profiler result. It counts matrix
    multiply accumulate pairs in the encoder, hierarchical router, selected
    low-rank circuits, GRU update, and output head. Elementwise operations,
    indexing, softmax, top-k, and kernel launch overhead are intentionally not
    counted; the benchmark reports wall-clock measurements separately.
    """
    state_dim = int(model.state_dim)
    d_model = int(model.token_embedding.embedding_dim)
    slot_count = int(getattr(model, "slot_count", 0))
    encoder_input = d_model * slot_count if slot_count else d_model
    output_layer = model.output[-1]
    num_classes = int(output_layer.out_features)
    router = model.router
    circuit_bank = model.circuits[0] if isinstance(model.circuits, nn.ModuleList) else model.circuits
    circuit_rank = int(circuit_bank.rank)

    encoder_macs = encoder_input * state_dim
    initial_state_macs = state_dim * state_dim
    value_macs = 0
    if getattr(model, "value_encoder", None) is not None:
        value_macs = int(value_tokens * model.value_encoder.in_features * model.value_encoder.out_features)
    fixed_macs = encoder_macs + initial_state_macs + value_macs

    router_macs = (router.num_addresses * router.depth * state_dim * router.branch
                   + router.candidate_pool * state_dim)
    circuit_macs = router.active_circuits * 2 * state_dim * circuit_rank
    gru_macs = 6 * state_dim * state_dim
    memory_write_macs = (2 * state_dim * state_dim
                         if getattr(model, "memory_write", None) is not None else 0)
    output_width = 1 if getattr(model, "output_mode", "learned") == "scalar_gaussian" else num_classes
    output_macs = state_dim * output_width
    per_step_macs = router_macs + circuit_macs + gru_macs + memory_write_macs + output_macs
    full_macs = fixed_macs + int(model.internal_steps) * per_step_macs
    active_macs = fixed_macs + float(executed_steps) * per_step_macs

    parameter_report = model.parameter_report()
    total_params = int(parameter_report["total_params"])
    active_params = int(parameter_report["active_params_estimate"])
    candidate_key_params = router.candidate_pool * state_dim
    shared_params = active_params - int(parameter_report["active_circuit_params"])
    fixed_path_params = shared_params - candidate_key_params
    # The router's tree projections, GRU, output head, and selected candidate
    # keys are touched on every executed step. Circuit blocks are the only
    # large capacity component that changes with the route.
    router_step_params = (router.num_addresses * router.depth * state_dim * router.branch
                          + router.candidate_pool * state_dim)
    step_shared_params = (router_step_params + 6 * state_dim * state_dim
                          + num_classes * state_dim + 2 * num_classes
                          + state_dim)
    if getattr(model, "memory_write", None) is not None:
        step_shared_params += 2 * state_dim * state_dim + state_dim
    path_read_params = fixed_path_params + float(executed_steps) * (
        step_shared_params + int(parameter_report["active_circuit_params"]))
    return {
        "estimated_fixed_macs_per_sample": fixed_macs,
        "estimated_macs_per_executed_step": per_step_macs,
        "estimated_full_macs_per_sample": full_macs,
        "estimated_active_macs_per_sample": active_macs,
        "estimated_active_macs_fraction": active_macs / max(full_macs, 1),
        "estimated_total_parameter_bytes": total_params * 4,
        "estimated_unique_active_parameter_bytes": active_params * 4,
        "estimated_parameter_read_bytes_per_sample": path_read_params * 4,
        "estimated_parameter_read_fraction_vs_dense": path_read_params / max(total_params, 1),
        "estimated_executed_steps": float(executed_steps),
    }


def estimate_transformer_macs(config: dict[str, Any]) -> dict[str, float | int]:
    """Estimate dense Transformer MACs for the same sequence and config.

    The estimate includes QKV/output projections, attention score/value
    products, and the two feed-forward projections per layer. Norms, biases,
    activations, embeddings, and masking are omitted consistently.
    """
    seq_len = int(config["seq_len"])
    d_model = int(config["d_model"])
    num_layers = int(config["num_layers"])
    ff_dim = int(config["ff_dim"])
    num_classes = int(config["num_classes"])
    attention_macs = 4 * seq_len * d_model * d_model + 2 * seq_len * seq_len * d_model
    feed_forward_macs = 2 * seq_len * d_model * ff_dim
    total_macs = num_layers * (attention_macs + feed_forward_macs) + d_model * num_classes
    parameter_count = (
        int(config["vocab_size"]) * d_model
        + seq_len * d_model
        + num_layers * (4 * d_model * d_model + 2 * d_model * ff_dim
                        + ff_dim + 9 * d_model)
        + 2 * d_model
        + d_model * num_classes + num_classes
    )
    return {
        "estimated_dense_macs_per_sample": total_macs,
        "estimated_dense_parameter_bytes": parameter_count * 4,
    }
