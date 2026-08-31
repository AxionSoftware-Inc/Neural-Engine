from __future__ import annotations

import torch
from torch import nn

from .circuits import MicroCircuitBank
from .instrumentation import count_parameters
from .router import HierarchicalRouter
from .state import PersistentState


class NeuralEngineV0(nn.Module):
    """Non-Transformer recurrent state + hierarchical routing + micro-circuits."""

    def __init__(self, vocab_size: int = 128, num_classes: int = 64, seq_len: int = 32,
                 d_model: int = 384, state_dim: int = 384, num_circuits: int = 2048,
                 circuit_rank: int = 16, router_branch: int = 8, router_depth: int = 4,
                 candidate_pool: int = 32, active_circuits: int = 8, internal_steps: int = 3,
                 router_addresses: int = 1, slot_count: int = 0, task_context: bool = False,
                 task_context_update: bool = True, circuit_mode: str = "parallel"):
        super().__init__()
        if circuit_mode not in {"parallel", "serial"}:
            raise ValueError("circuit_mode must be 'parallel' or 'serial'")
        self.state_dim = state_dim
        self.active_circuits = active_circuits
        self.internal_steps = internal_steps
        self.slot_count = slot_count
        self.use_task_context = task_context
        self.task_context_update = task_context_update
        self.circuit_mode = circuit_mode
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, d_model))
        # Multiplicative position conditioning binds a token to its slot before
        # pooling; this preserves operand order without attention.
        self.position_scale = nn.Parameter(torch.zeros(seq_len, d_model))
        self.position_bias = nn.Parameter(torch.zeros(seq_len, d_model))
        encoder_input = d_model * slot_count if slot_count else d_model
        self.encoder = nn.Sequential(nn.LayerNorm(encoder_input), nn.Linear(encoder_input, state_dim), nn.GELU())
        self.state = PersistentState(state_dim, state_dim)
        self.step_embedding = nn.Parameter(torch.zeros(internal_steps, state_dim))
        self.task_context_embedding = nn.Embedding(16, state_dim) if task_context else None
        self.router = HierarchicalRouter(state_dim, num_circuits, router_branch, router_depth,
                                         candidate_pool, active_circuits, router_addresses)
        self.circuits = MicroCircuitBank(num_circuits, state_dim, circuit_rank)
        self.output = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, num_classes))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.position_scale, std=0.01)
        nn.init.normal_(self.position_bias, std=0.01)
        nn.init.normal_(self.step_embedding, std=0.02)
        self._last_route: dict[str, torch.Tensor] = {}

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.token_embedding(inputs)
        positions = self.position_embedding[: inputs.shape[1]]
        scale = self.position_scale[: inputs.shape[1]]
        bias = self.position_bias[: inputs.shape[1]]
        tokens = tokens * (1.0 + scale) + positions + bias
        mask = inputs.ne(0).unsqueeze(-1)
        tokens = tokens * mask
        if self.slot_count:
            if inputs.shape[1] < self.slot_count:
                raise ValueError("inputs are shorter than configured slot_count")
            encoded_input = tokens[:, :self.slot_count].reshape(inputs.shape[0], -1)
        else:
            encoded_input = tokens.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.encoder(encoded_input)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encode(inputs)
        state = self.state.initialize(encoded)
        task_context = None
        if self.task_context_embedding is not None:
            task_ids = (inputs[:, 0] - 1).clamp(0, self.task_context_embedding.num_embeddings - 1)
            task_context = self.task_context_embedding(task_ids)
        selected_steps = []
        entropies = []
        step_logits = []
        for step in range(self.internal_steps):
            # A distinct query per recurrent step encourages compositional
            # paths instead of routing every step from the same representation.
            step_query = state + self.step_embedding[step]
            if task_context is not None:
                step_query = step_query + task_context
            selected, weights, route_stats = self.router(step_query)
            if self.circuit_mode == "serial":
                circuit_delta = self.circuits.forward_serial(step_query, selected, weights)
            else:
                circuit_delta = self.circuits(step_query, selected, weights)
            delta = circuit_delta * route_stats["route_gain"].unsqueeze(-1)
            update = delta + encoded + self.step_embedding[step]
            if task_context is not None and self.task_context_update:
                update = update + task_context
            state = self.state.step(state, update)
            selected_steps.append(selected)
            entropies.append(route_stats["router_entropy"])
            step_logits.append(self.output(state))
        logits = step_logits[-1]
        stats = {
            "active_circuits": torch.tensor(self.active_circuits, device=inputs.device),
            "internal_steps": torch.tensor(self.internal_steps, device=inputs.device),
            "router_entropy": torch.stack(entropies).mean(),
            "selected_ids": torch.stack(selected_steps, dim=1),
            "step_logits": torch.stack(step_logits, dim=1),
        }
        self._last_route = stats
        return logits, stats

    def parameter_report(self) -> dict[str, int | float]:
        total = count_parameters(self)
        shared = count_parameters(self.token_embedding) + self.position_embedding.numel()
        shared += self.position_scale.numel() + self.position_bias.numel() + count_parameters(self.encoder)
        shared += count_parameters(self.state) + self.step_embedding.numel()
        shared += self.router.level_projections.numel() + self.router.level_bias.numel()
        shared += count_parameters(self.output)
        if self.task_context_embedding is not None:
            shared += count_parameters(self.task_context_embedding)
        one_circuit = self.circuits.down[0].numel() + self.circuits.up[0].numel() + self.circuits.bias[0].numel()
        candidate_key_params = self.router.keys[0].numel() * self.router.candidate_pool
        active = shared + candidate_key_params + one_circuit * self.active_circuits
        return {
            "total_params": total,
            "active_params_estimate": active,
            "active_fraction": active / total,
            "active_circuit_params": one_circuit * self.active_circuits,
        }
