from __future__ import annotations

import torch
from torch import nn

from .circuits import FactorizedMicroCircuitBank, MicroCircuitBank
from .encoding import (
    FixedFourierValueEncoder,
    HybridFourierValueEncoder,
    VALUE_HARMONICS,
    VALUE_MODULUS,
    VALUE_TOKEN_OFFSET,
    encode_tokens,
)
from .instrumentation import count_parameters
from .modular_templates import TrainableModularTemplateRegister
from .router import FactorizedRouter, HierarchicalRouter


class DynamicRegisterNeuralEngine(nn.Module):
    """Attention-free recurrent register machine with sparse circuit routing.

    A program is scanned left-to-right.  The accumulator and the next operand
    are composed with the primitive operation embedding, routed through a
    small candidate set, and written back to the accumulator.  The number of
    executed register updates is input-dependent; there is no self-attention,
    Transformer block, or dense all-bank score.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        num_classes: int = 64,
        max_ops: int = 6,
        seq_len: int | None = None,
        d_model: int = 384,
        state_dim: int = 384,
        num_circuits: int = 1408,
        circuit_rank: int = 16,
        router_branch: int = 8,
        router_depth: int = 4,
        candidate_pool: int = 32,
        active_circuits: int = 8,
        circuit_bank_mode: str = "factorized",
        factor_count: int | None = None,
        factor_candidate_pool: int | None = None,
        circuit_mode: str = "serial",
        route_exploration_prob: float = 0.05,
        input_reinjection_scale: float = 0.0,
        write_gate: bool = False,
        value_encoder_mode: str = "learned",
        factor_mix_mode: str = "per_address",
        route_context_mode: str = "full",
        modular_prior: bool = False,
        modular_prior_mode: str = "fixed",
    ) -> None:
        super().__init__()
        if max_ops < 1:
            raise ValueError("max_ops must be positive")
        expected_seq_len = 1 + max_ops + (max_ops + 1)
        if seq_len is None:
            seq_len = expected_seq_len
        if seq_len < expected_seq_len:
            raise ValueError("seq_len is too short for the dynamic program layout")
        if circuit_mode not in {"parallel", "serial"}:
            raise ValueError("circuit_mode must be parallel or serial")
        if circuit_bank_mode not in {"independent", "factorized"}:
            raise ValueError("circuit_bank_mode must be independent or factorized")
        if not 0.0 <= route_exploration_prob <= 1.0:
            raise ValueError("route_exploration_prob must be between zero and one")
        if value_encoder_mode not in {"learned", "fixed_fourier", "hybrid_fourier"}:
            raise ValueError(
                "value_encoder_mode must be learned, fixed_fourier, or hybrid_fourier"
            )
        if route_context_mode not in {"full", "operation_step"}:
            raise ValueError("route_context_mode must be full or operation_step")
        if modular_prior_mode not in {"fixed", "templates"}:
            raise ValueError("modular_prior_mode must be fixed or templates")

        self.max_ops = max_ops
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.internal_steps = max_ops
        self.circuit_mode = circuit_mode
        self.circuit_bank_mode = circuit_bank_mode
        self.route_exploration_prob = route_exploration_prob
        self.input_reinjection_scale = float(input_reinjection_scale)
        self.write_gate_enabled = bool(write_gate)
        self.value_encoder_mode = value_encoder_mode
        self.factor_mix_mode = factor_mix_mode
        self.route_context_mode = route_context_mode
        self.modular_prior_enabled = bool(modular_prior)
        self.modular_prior_mode = modular_prior_mode
        self.adaptive_halting = False
        self.adaptive_inference = False
        self.value_start = 1 + max_ops

        embedding_vocab = 16
        self.token_embedding = nn.Embedding(embedding_vocab, d_model, padding_idx=0)
        if value_encoder_mode == "fixed_fourier":
            self.value_encoder = FixedFourierValueEncoder(d_model)
        elif value_encoder_mode == "hybrid_fourier":
            self.value_encoder = HybridFourierValueEncoder(d_model)
        else:
            self.value_encoder = nn.Linear(1 + 2 * len(VALUE_HARMONICS), d_model)
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, d_model))
        self.position_scale = nn.Parameter(torch.zeros(seq_len, d_model))
        self.position_bias = nn.Parameter(torch.zeros(seq_len, d_model))

        self.operand_encoder = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, state_dim), nn.GELU()
        )
        self.initial_writer = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, state_dim), nn.Tanh()
        )
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(2 * state_dim),
            nn.Linear(2 * state_dim, state_dim),
            nn.GELU(),
        )
        self.product_encoder = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, state_dim), nn.GELU()
        )
        self.operation_embedding = nn.Embedding(3, state_dim)
        self.step_embedding = nn.Parameter(torch.zeros(max_ops, state_dim))
        self.register_writer = nn.Sequential(
            nn.LayerNorm(2 * state_dim),
            nn.Linear(2 * state_dim, state_dim),
            nn.Tanh(),
        )
        if self.modular_prior_enabled:
            if self.modular_prior_mode == "fixed":
                left = torch.arange(VALUE_MODULUS).view(-1, 1)
                right = torch.arange(VALUE_MODULUS).view(1, -1)
                transition = torch.stack((
                    (left + right).remainder(VALUE_MODULUS),
                    (left - right).remainder(VALUE_MODULUS),
                    (left * right).remainder(VALUE_MODULUS),
                ))
                self.register_buffer("modular_transition", transition, persistent=False)
            else:
                self.modular_template_logits = nn.Parameter(4.0 * torch.eye(3))
            self.modular_projection = nn.Sequential(
                nn.LayerNorm(VALUE_MODULUS),
                nn.Linear(VALUE_MODULUS, state_dim),
                nn.Tanh(),
            )
        if self.write_gate_enabled:
            self.write_gate = nn.Sequential(
                nn.LayerNorm(2 * state_dim),
                nn.Linear(2 * state_dim, state_dim),
                nn.Sigmoid(),
            )
        self.route_context = nn.Sequential(
            nn.LayerNorm(state_dim), nn.Linear(state_dim, state_dim), nn.Tanh()
        )

        if circuit_bank_mode == "factorized":
            self.router = FactorizedRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, 1,
                factor_count=factor_count,
                factor_candidate_pool=factor_candidate_pool,
            )
            self.circuits = FactorizedMicroCircuitBank(
                num_circuits, state_dim, circuit_rank, factor_count, factor_mix_mode
            )
        else:
            self.router = HierarchicalRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, 1
            )
            self.circuits = MicroCircuitBank(num_circuits, state_dim, circuit_rank)
        self.output = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, num_classes))

        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.position_scale, std=0.01)
        nn.init.normal_(self.position_bias, std=0.01)
        nn.init.normal_(self.step_embedding, std=0.02)
        nn.init.normal_(self.operation_embedding.weight, std=0.02)

    def encode_program(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] < self.value_start + self.max_ops + 1:
            raise ValueError("inputs are shorter than the configured program layout")
        tokens = encode_tokens(inputs, self.token_embedding, self.value_encoder)
        positions = self.position_embedding[: inputs.shape[1]]
        scale = self.position_scale[: inputs.shape[1]]
        bias = self.position_bias[: inputs.shape[1]]
        tokens = tokens * (1.0 + scale) + positions + bias
        return tokens * inputs.ne(0).unsqueeze(-1)

    def _apply_circuits(
        self,
        query: torch.Tensor,
        selected: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.circuit_mode == "serial":
            return self.circuits.forward_serial(query, selected, weights)
        return self.circuits(query, selected, weights)

    def forward(
        self,
        inputs: torch.Tensor,
        adaptive: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del adaptive
        batch_size = inputs.shape[0]
        device = inputs.device
        encoded = self.encode_program(inputs)
        operation_tokens = inputs[:, 1:1 + self.max_ops]
        operation_ids = (operation_tokens - 2).clamp(0, 2)
        operation_mask = operation_tokens.ge(2)
        operands = encoded[:, self.value_start:self.value_start + self.max_ops + 1]
        operand_states = self.operand_encoder(operands)
        operand_mask = inputs[:, self.value_start:self.value_start + self.max_ops + 1].ne(0)
        input_context = (
            (operand_states * operand_mask.unsqueeze(-1)).sum(dim=1)
            / operand_mask.sum(dim=1, keepdim=True).clamp_min(1).to(operand_states.dtype)
        )
        accumulator = self.initial_writer(operand_states[:, 0])
        if self.modular_prior_enabled:
            initial_values = (
                inputs[:, self.value_start] - VALUE_TOKEN_OFFSET
            ).clamp(0, VALUE_MODULUS - 1)
            if self.modular_prior_mode == "fixed":
                modular_accumulator = initial_values
            else:
                modular_state = nn.functional.one_hot(
                    initial_values, VALUE_MODULUS
                ).to(accumulator.dtype)

        selected_steps = []
        selected_weights = torch.zeros(
            batch_size, self.max_ops, self.router.active_circuits, device=device
        )
        step_entropies = torch.zeros(batch_size, self.max_ops, device=device)
        executed_mask = torch.zeros(
            batch_size, self.max_ops, dtype=torch.bool, device=device
        )
        step_logits = []

        for step in range(self.max_ops):
            active_indices = operation_mask[:, step].nonzero(as_tuple=False).squeeze(-1)
            selected_step = torch.full(
                (batch_size, self.router.active_circuits), -1,
                dtype=torch.long, device=device
            )
            if active_indices.numel():
                active_accumulator = accumulator[active_indices]
                active_operand = operand_states[active_indices, step + 1]
                pair = self.pair_encoder(torch.cat([active_accumulator, active_operand], dim=-1))
                pair = pair + self.product_encoder(active_accumulator * active_operand)
                query = (
                    pair
                    + self.operation_embedding(operation_ids[active_indices, step])
                    + self.step_embedding[step]
                )
                if self.input_reinjection_scale:
                    query = query + self.input_reinjection_scale * input_context[active_indices]
                if self.modular_prior_enabled:
                    if self.modular_prior_mode == "fixed":
                        modular_features = nn.functional.one_hot(
                            modular_accumulator[active_indices], VALUE_MODULUS
                        ).to(query.dtype)
                    else:
                        modular_features = modular_state[active_indices].to(query.dtype)
                    query = query + self.modular_projection(modular_features)
                query = query + 0.25 * self.route_context(query)
                route_query = query
                if self.route_context_mode == "operation_step":
                    route_query = (
                        self.operation_embedding(operation_ids[active_indices, step])
                        + self.step_embedding[step]
                    )
                    route_query = route_query + 0.25 * self.route_context(route_query)
                selected, weights, route_stats = self.router(
                    route_query,
                    exploration_prob=(self.route_exploration_prob if self.training else 0.0),
                )
                delta = self._apply_circuits(query, selected, weights)
                candidate = self.register_writer(
                    torch.cat([active_accumulator, query + delta], dim=-1)
                )
                if self.write_gate_enabled:
                    gate_input = torch.cat([active_accumulator, query + delta], dim=-1)
                    gate = self.write_gate(gate_input)
                    updated = gate * candidate + (1.0 - gate) * active_accumulator
                else:
                    updated = candidate
                next_accumulator = accumulator.clone()
                next_accumulator[active_indices] = updated
                accumulator = next_accumulator
                if self.modular_prior_enabled:
                    operand_values = (
                        inputs[active_indices, self.value_start + step + 1]
                        - VALUE_TOKEN_OFFSET
                    ).clamp(0, VALUE_MODULUS - 1)
                    if self.modular_prior_mode == "fixed":
                        modular_updated = self.modular_transition[
                            operation_ids[active_indices, step],
                            modular_accumulator[active_indices],
                            operand_values,
                        ]
                        next_modular_accumulator = modular_accumulator.clone()
                        next_modular_accumulator[active_indices] = modular_updated
                        modular_accumulator = next_modular_accumulator
                    else:
                        modular_primitives = torch.stack((
                            TrainableModularTemplateRegister._add_state(
                                modular_state[active_indices], operand_values),
                            TrainableModularTemplateRegister._subtract_state(
                                modular_state[active_indices], operand_values),
                            TrainableModularTemplateRegister._multiply_state(
                                modular_state[active_indices], operand_values),
                        ), dim=1)
                        template_weights = nn.functional.softmax(
                            self.modular_template_logits[
                                operation_ids[active_indices, step]
                            ], dim=-1
                        )
                        modular_updated = torch.einsum(
                            "bt,btv->bv", template_weights, modular_primitives
                        )
                        next_modular_state = modular_state.clone()
                        next_modular_state[active_indices] = modular_updated
                        modular_state = next_modular_state
                selected_step[active_indices] = selected
                selected_weights[active_indices, step] = weights
                step_entropies[active_indices, step] = route_stats["router_entropy"]
                executed_mask[active_indices, step] = True
            selected_steps.append(selected_step)
            step_state = accumulator
            if self.modular_prior_enabled:
                if self.modular_prior_mode == "fixed":
                    step_features = nn.functional.one_hot(
                        modular_accumulator, VALUE_MODULUS
                    ).to(accumulator.dtype)
                else:
                    step_features = modular_state
                step_state = step_state + self.modular_projection(step_features)
            step_logits.append(self.output(step_state))

        stats = {
            "active_circuits": torch.tensor(self.router.active_circuits, device=device),
            "internal_steps": torch.tensor(self.max_ops, device=device),
            "router_entropy": step_entropies.sum() / executed_mask.sum().clamp_min(1),
            "selected_ids": torch.stack(selected_steps, dim=1),
            "selected_weights": selected_weights,
            "step_logits": torch.stack(step_logits, dim=1),
            "executed_steps": executed_mask.sum(dim=1),
            "executed_mask": executed_mask,
        }
        self._last_route = stats
        return stats["step_logits"][:, -1], stats

    def parameter_report(self) -> dict[str, int | float | str]:
        total = count_parameters(self)
        shared = (
            count_parameters(self.token_embedding)
            + count_parameters(self.value_encoder)
            + self.position_embedding.numel()
            + self.position_scale.numel()
            + self.position_bias.numel()
            + count_parameters(self.operand_encoder)
            + count_parameters(self.initial_writer)
            + count_parameters(self.pair_encoder)
            + count_parameters(self.product_encoder)
            + count_parameters(self.operation_embedding)
            + self.step_embedding.numel()
            + count_parameters(self.register_writer)
            + count_parameters(self.route_context)
            + count_parameters(self.output)
        )
        if self.write_gate_enabled:
            shared += count_parameters(self.write_gate)
        if self.modular_prior_enabled:
            shared += count_parameters(self.modular_projection)
            if self.modular_prior_mode == "templates":
                shared += self.modular_template_logits.numel()
        if self.circuit_bank_mode == "factorized":
            factor_row = (
                self.circuits.down_factors[0].numel()
                + self.circuits.up_factors[0].numel()
                + self.circuits.bias_factors[0].numel()
                + self.circuits.factor_mix[0].numel()
            )
            active_circuit_params = factor_row * self.router.active_circuits * 2
            candidate_params = self.router.keys[0].numel() * self.router.factor_candidate_pool * 2
        else:
            one_circuit = (
                self.circuits.down[0].numel()
                + self.circuits.up[0].numel()
                + self.circuits.bias[0].numel()
            )
            active_circuit_params = one_circuit * self.router.active_circuits
            candidate_params = self.router.keys[0].numel() * self.router.candidate_pool
        return {
            "total_params": total,
            "active_params_estimate": shared + candidate_params + active_circuit_params,
            "active_fraction": (shared + candidate_params + active_circuit_params) / total,
            "active_circuit_params": active_circuit_params,
            "max_ops": self.max_ops,
            "circuit_bank_mode": self.circuit_bank_mode,
            "routing_mode": "factorized" if self.circuit_bank_mode == "factorized" else "hierarchical",
            "input_reinjection_scale": self.input_reinjection_scale,
            "write_gate": self.write_gate_enabled,
            "value_encoder_mode": self.value_encoder_mode,
            "factor_mix_mode": self.factor_mix_mode,
            "route_context_mode": self.route_context_mode,
            "modular_prior": self.modular_prior_enabled,
            "modular_prior_mode": self.modular_prior_mode,
        }
