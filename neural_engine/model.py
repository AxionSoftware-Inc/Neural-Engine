from __future__ import annotations

import math

import torch
from torch import nn

from .circuits import MicroCircuitBank
from .instrumentation import count_parameters
from .router import HierarchicalRouter
from .state import PersistentState


VALUE_TOKEN_OFFSET = 32
VALUE_MODULUS = 64
VALUE_HARMONICS = (1, 2, 4, 8, 16, 32)


class NeuralEngineV0(nn.Module):
    """Non-Transformer recurrent state + hierarchical routing + micro-circuits."""

    def __init__(self, vocab_size: int = 128, num_classes: int = 64, seq_len: int = 32,
                 d_model: int = 384, state_dim: int = 384, num_circuits: int = 2048,
                 circuit_rank: int = 16, router_branch: int = 8, router_depth: int = 4,
                 candidate_pool: int = 32, active_circuits: int = 8, internal_steps: int = 3,
                 router_addresses: int = 1, slot_count: int = 0, task_context: bool = False,
                 task_context_update: bool = True, circuit_mode: str = "parallel",
                 numeric_value_encoding: bool = False, adaptive_halting: bool = False,
                 halt_threshold: float = 0.5, routing_coverage_temperature: float = 0.25,
                 input_reinjection: float = 1.0, memory_write_mode: str = "none"):
        super().__init__()
        if circuit_mode not in {"parallel", "serial"}:
            raise ValueError("circuit_mode must be 'parallel' or 'serial'")
        if memory_write_mode not in {"none", "gated"}:
            raise ValueError("memory_write_mode must be 'none' or 'gated'")
        if not 0.0 < halt_threshold < 1.0:
            raise ValueError("halt_threshold must be between 0 and 1")
        self.state_dim = state_dim
        self.active_circuits = active_circuits
        self.internal_steps = internal_steps
        self.slot_count = slot_count
        self.use_task_context = task_context
        self.task_context_update = task_context_update
        self.circuit_mode = circuit_mode
        self.numeric_value_encoding = numeric_value_encoding
        self.adaptive_halting = adaptive_halting
        self.adaptive_inference = adaptive_halting
        self.halt_threshold = halt_threshold
        self.routing_coverage_temperature = routing_coverage_temperature
        self.input_reinjection = input_reinjection
        self.memory_write_mode = memory_write_mode
        embedding_vocab = 16 if numeric_value_encoding else vocab_size
        self.token_embedding = nn.Embedding(embedding_vocab, d_model, padding_idx=0)
        self.value_encoder = nn.Linear(1 + 2 * len(VALUE_HARMONICS), d_model) if numeric_value_encoding else None
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, d_model))
        # Multiplicative position conditioning binds a token to its slot before
        # pooling; this preserves operand order without attention.
        self.position_scale = nn.Parameter(torch.zeros(seq_len, d_model))
        self.position_bias = nn.Parameter(torch.zeros(seq_len, d_model))
        encoder_input = d_model * slot_count if slot_count else d_model
        self.encoder = nn.Sequential(nn.LayerNorm(encoder_input), nn.Linear(encoder_input, state_dim), nn.GELU())
        self.state = PersistentState(state_dim, state_dim)
        self.memory_write = (nn.Linear(2 * state_dim, state_dim)
                             if memory_write_mode == "gated" else None)
        self.step_embedding = nn.Parameter(torch.zeros(internal_steps, state_dim))
        self.task_context_embedding = nn.Embedding(16, state_dim) if task_context else None
        self.halt_head = nn.Linear(state_dim, 1) if adaptive_halting else None
        self.router = HierarchicalRouter(state_dim, num_circuits, router_branch, router_depth,
                                         candidate_pool, active_circuits, router_addresses)
        self.circuits = MicroCircuitBank(num_circuits, state_dim, circuit_rank)
        self.output = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, num_classes))
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.position_scale, std=0.01)
        nn.init.normal_(self.position_bias, std=0.01)
        nn.init.normal_(self.step_embedding, std=0.02)
        if self.memory_write is not None:
            # Start close to the existing GRU path; training can learn to
            # preserve the old state when a write would overwrite useful work.
            nn.init.zeros_(self.memory_write.weight)
            nn.init.constant_(self.memory_write.bias, 5.0)
        self._last_route: dict[str, torch.Tensor] = {}

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        embedding_inputs = inputs.clamp_max(self.token_embedding.num_embeddings - 1)
        tokens = self.token_embedding(embedding_inputs)
        if self.value_encoder is not None:
            values = (inputs.float() - VALUE_TOKEN_OFFSET).clamp(0, VALUE_MODULUS - 1)
            angles = values.unsqueeze(-1) * (2.0 * math.pi / VALUE_MODULUS)
            features = [values.unsqueeze(-1) / (VALUE_MODULUS - 1)]
            for harmonic in VALUE_HARMONICS:
                features.append(torch.sin(angles * harmonic))
                features.append(torch.cos(angles * harmonic))
            numeric_tokens = self.value_encoder(torch.cat(features, dim=-1))
            value_mask = inputs.ge(VALUE_TOKEN_OFFSET) & inputs.lt(VALUE_TOKEN_OFFSET + VALUE_MODULUS)
            tokens = torch.where(value_mask.unsqueeze(-1), numeric_tokens, tokens)
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

    def forward(self, inputs: torch.Tensor, adaptive: bool | None = None,
                forced_selected_ids: torch.Tensor | None = None,
                forced_selected_weights: torch.Tensor | None = None,
                forced_route_gains: torch.Tensor | None = None,
                coverage: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the model, optionally replaying a previously recorded route.

        The forced-route arguments are an analysis hook for causal route
        replay. Normal training and inference leave them unset. When supplied,
        the router is still evaluated for control statistics, but the selected
        circuit IDs (and, when supplied, their weights/gains) come from the
        recorded route.
        """
        use_adaptive = self.adaptive_inference if adaptive is None else adaptive
        if use_adaptive and self.halt_head is None:
            raise ValueError("adaptive inference requires adaptive_halting=True")
        encoded = self.encode(inputs)
        state = self.state.initialize(encoded)
        task_context = None
        if self.task_context_embedding is not None:
            task_ids = (inputs[:, 0] - 1).clamp(0, self.task_context_embedding.num_embeddings - 1)
            task_context = self.task_context_embedding(task_ids)
        batch_size = inputs.shape[0]
        num_classes = self.output[-1].out_features
        selected_steps = []
        coverage_losses = []
        selected_weights = torch.zeros(batch_size, self.internal_steps, self.active_circuits,
                                       device=inputs.device)
        route_gains = torch.ones(batch_size, self.internal_steps, device=inputs.device)
        step_entropies = torch.zeros(batch_size, self.internal_steps, device=inputs.device)
        executed_mask = torch.zeros(batch_size, self.internal_steps, dtype=torch.bool, device=inputs.device)
        step_logits = torch.zeros(batch_size, self.internal_steps, num_classes, device=inputs.device)
        halt_logits = torch.zeros(batch_size, self.internal_steps, device=inputs.device)
        last_logits = torch.zeros(batch_size, num_classes, device=inputs.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=inputs.device)
        if forced_selected_ids is not None:
            expected_shape = (batch_size, self.internal_steps, self.active_circuits)
            if tuple(forced_selected_ids.shape) != expected_shape:
                raise ValueError(f"forced_selected_ids must have shape {expected_shape}")
            if (forced_selected_ids < 0).any():
                raise ValueError("forced_selected_ids must contain valid circuit IDs for every step")
            if forced_selected_weights is not None and tuple(forced_selected_weights.shape) != expected_shape:
                raise ValueError(f"forced_selected_weights must have shape {expected_shape}")
            if forced_route_gains is not None and tuple(forced_route_gains.shape) != (batch_size, self.internal_steps):
                raise ValueError(f"forced_route_gains must have shape {(batch_size, self.internal_steps)}")
            forced_selected_ids = forced_selected_ids.to(device=inputs.device)
            if forced_selected_weights is not None:
                forced_selected_weights = forced_selected_weights.to(device=inputs.device)
            if forced_route_gains is not None:
                forced_route_gains = forced_route_gains.to(device=inputs.device)
        for step in range(self.internal_steps):
            active_indices = active.nonzero(as_tuple=False).squeeze(-1)
            selected_step = torch.full((batch_size, self.active_circuits), -1,
                                       dtype=torch.long, device=inputs.device)
            if active_indices.numel() == 0:
                selected_steps.append(selected_step)
                step_logits[:, step] = last_logits
                continue
            active_state = state[active_indices]
            # A distinct query per recurrent step encourages compositional
            # paths instead of routing every step from the same representation.
            step_query = active_state + self.step_embedding[step]
            if task_context is not None:
                step_query = step_query + task_context[active_indices]
            selected, weights, route_stats = self.router(
                step_query, coverage=coverage,
                coverage_temperature=self.routing_coverage_temperature)
            route_gain = route_stats["route_gain"]
            if "routing_coverage_loss" in route_stats:
                coverage_losses.append(route_stats["routing_coverage_loss"])
            if forced_selected_ids is not None:
                selected = forced_selected_ids[active_indices, step].to(device=inputs.device)
                if forced_selected_weights is None:
                    weights = torch.full((active_indices.numel(), self.active_circuits),
                                         1.0 / self.active_circuits, device=inputs.device)
                else:
                    weights = forced_selected_weights[active_indices, step].to(device=inputs.device)
                route_gain = (torch.ones_like(route_gain)
                              if forced_route_gains is None
                              else forced_route_gains[active_indices, step].to(device=inputs.device))
                if self.circuit_mode == "serial":
                    circuit_delta = self.circuits.forward_serial(step_query, selected, weights)
                else:
                    circuit_delta = self.circuits(step_query, selected, weights)
            elif self.circuit_mode == "serial":
                circuit_delta = self.circuits.forward_serial(step_query, selected, weights)
            else:
                circuit_delta = self.circuits(step_query, selected, weights)
            delta = circuit_delta * route_gain.unsqueeze(-1)
            update = (delta + self.input_reinjection * encoded[active_indices]
                      + self.step_embedding[step])
            if task_context is not None and self.task_context_update:
                update = update + task_context[active_indices]
            proposal_state = self.state.step(active_state, update)
            if self.memory_write is not None:
                write_input = torch.cat([active_state, update], dim=-1)
                write_gate = torch.sigmoid(self.memory_write(write_input))
                updated_state = active_state + write_gate * (proposal_state - active_state)
            else:
                updated_state = proposal_state
            if active_indices.numel() == batch_size:
                state = updated_state
            else:
                next_state = state.clone()
                next_state[active_indices] = updated_state
                state = next_state
            selected_step[active_indices] = selected
            selected_steps.append(selected_step)
            selected_weights[active_indices, step] = weights
            route_gains[active_indices, step] = route_gain
            executed_mask[active_indices, step] = True
            step_entropies[active_indices, step] = route_stats["router_entropy"]
            updated_logits = self.output(updated_state)
            next_logits = last_logits.clone()
            next_logits[active_indices] = updated_logits
            last_logits = next_logits
            step_logits[:, step] = last_logits
            if self.halt_head is not None:
                updated_halt_logits = self.halt_head(updated_state).squeeze(-1)
                halt_logits[active_indices, step] = updated_halt_logits
                if use_adaptive:
                    should_halt = torch.sigmoid(updated_halt_logits) >= self.halt_threshold
                    if step == self.internal_steps - 1:
                        should_halt = torch.ones_like(should_halt, dtype=torch.bool)
                    next_active = active.clone()
                    next_active[active_indices[should_halt]] = False
                    active = next_active
        stats = {
            "active_circuits": torch.tensor(self.active_circuits, device=inputs.device),
            "internal_steps": torch.tensor(self.internal_steps, device=inputs.device),
            "router_entropy": step_entropies.sum() / executed_mask.sum().clamp_min(1),
            "selected_ids": torch.stack(selected_steps, dim=1),
            "selected_weights": selected_weights,
            "route_gains": route_gains,
            "step_logits": step_logits,
            "halt_logits": halt_logits,
            "executed_steps": executed_mask.sum(dim=1),
            "executed_mask": executed_mask,
        }
        if coverage_losses:
            stats["routing_coverage_loss"] = torch.stack(coverage_losses).mean()
        self._last_route = stats
        return last_logits, stats

    def parameter_report(self) -> dict[str, int | float]:
        total = count_parameters(self)
        shared = count_parameters(self.token_embedding) + self.position_embedding.numel()
        if self.value_encoder is not None:
            shared += count_parameters(self.value_encoder)
        shared += self.position_scale.numel() + self.position_bias.numel() + count_parameters(self.encoder)
        shared += count_parameters(self.state) + self.step_embedding.numel()
        shared += self.router.level_projections.numel() + self.router.level_bias.numel()
        shared += count_parameters(self.output)
        if self.task_context_embedding is not None:
            shared += count_parameters(self.task_context_embedding)
        if self.halt_head is not None:
            shared += count_parameters(self.halt_head)
        if self.memory_write is not None:
            shared += count_parameters(self.memory_write)
        one_circuit = self.circuits.down[0].numel() + self.circuits.up[0].numel() + self.circuits.bias[0].numel()
        candidate_key_params = self.router.keys[0].numel() * self.router.candidate_pool
        active = shared + candidate_key_params + one_circuit * self.active_circuits
        return {
            "total_params": total,
            "active_params_estimate": active,
            "active_fraction": active / total,
            "active_circuit_params": one_circuit * self.active_circuits,
        }
