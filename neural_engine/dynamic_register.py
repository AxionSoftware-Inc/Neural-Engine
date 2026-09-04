from __future__ import annotations

import math

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
from .macro_cells import MacroCellBank
from .modular_templates import (
    modular_add_state,
    modular_multiply_state,
    modular_subtract_state,
)
from .router import FactorizedRouter, HierarchicalRouter


class ScalarGaussianOutput(nn.Module):
    """Decode a continuous class coordinate into ordered class logits."""

    def __init__(
        self, input_dim: int, num_classes: int, temperature: float,
        initial_bias: float,
    ) -> None:
        super().__init__()
        self.scalar = nn.Linear(input_dim, 1)
        with torch.no_grad():
            self.scalar.bias.fill_(initial_bias)
        self.register_buffer(
            "class_positions", torch.arange(num_classes, dtype=torch.float32),
            persistent=False,
        )
        self.out_features = int(num_classes)
        self.temperature = float(temperature)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        coordinate = self.scalar(states)
        distances = coordinate - self.class_positions.to(states.dtype)
        return -0.5 * distances.square() / (self.temperature ** 2)


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
        modulus: int = VALUE_MODULUS,
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
        factor_capacity: int | None = None,
        circuit_mode: str = "serial",
        route_exploration_prob: float = 0.05,
        input_reinjection_scale: float = 0.0,
        write_gate: bool = False,
        value_encoder_mode: str = "learned",
        factor_mix_mode: str = "per_address",
        route_context_mode: str = "full",
        state_layout: str = "flat",
        predecessor_operation_context: bool = False,
        operation_adapter_rank: int = 0,
        operation_adapter_scale: float = 1.0,
        operation_adapter_gate: bool = False,
        operation_read_adapter_rank: int = 0,
        operation_read_adapter_scale: float = 1.0,
        operation_write_adapter_rank: int = 0,
        operation_write_adapter_scale: float = 1.0,
        operation_circuit_bank: bool = False,
        operation_router_keys: bool = False,
        operation_transition_rank: int = 0,
        operation_transition_scale: float = 1.0,
        numeric_state_dim: int = 0,
        numeric_state_scale: float = 1.0,
        modular_prior: bool = False,
        modular_prior_mode: str = "fixed",
        modular_template_init: str = "identity",
        circuit_residual_scale: float = 1.0,
        circuit_input_norm: bool = False,
        output_mode: str = "learned",
        output_temperature: float = 16.0,
        output_scalar_bias: float = 0.0,
        macro_cell_count: int = 0,
        macro_cell_rank: int = 8,
        macro_cell_depth: int = 4,
        macro_router_branch: int = 4,
        macro_router_depth: int | None = None,
        macro_candidate_pool: int = 4,
        active_macro_cells: int = 1,
        macro_cell_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if max_ops < 1:
            raise ValueError("max_ops must be positive")
        if modulus < 2:
            raise ValueError("modulus must be at least two")
        if modular_prior and num_classes != modulus:
            raise ValueError("num_classes must equal modulus when modular_prior is enabled")
        expected_seq_len = 1 + max_ops + (max_ops + 1)
        if seq_len is None:
            seq_len = expected_seq_len
        if seq_len < expected_seq_len:
            raise ValueError("seq_len is too short for the dynamic program layout")
        if circuit_mode not in {"parallel", "serial"}:
            raise ValueError("circuit_mode must be parallel or serial")
        if circuit_bank_mode not in {"independent", "factorized"}:
            raise ValueError("circuit_bank_mode must be independent or factorized")
        if operation_router_keys and circuit_bank_mode != "factorized":
            raise ValueError("operation_router_keys requires factorized routing")
        if not 0.0 <= route_exploration_prob <= 1.0:
            raise ValueError("route_exploration_prob must be between zero and one")
        if value_encoder_mode not in {"learned", "fixed_fourier", "hybrid_fourier"}:
            raise ValueError(
                "value_encoder_mode must be learned, fixed_fourier, or hybrid_fourier"
            )
        if route_context_mode not in {"full", "operation_step", "hybrid"}:
            raise ValueError(
                "route_context_mode must be full, operation_step, or hybrid"
            )
        if state_layout not in {"flat", "dual_slot"}:
            raise ValueError("state_layout must be flat or dual_slot")
        if state_layout == "dual_slot" and state_dim % 2:
            raise ValueError("state_dim must be even for dual_slot state layout")
        if operation_adapter_rank < 0:
            raise ValueError("operation_adapter_rank must be non-negative")
        if operation_adapter_scale < 0.0:
            raise ValueError("operation_adapter_scale must be non-negative")
        if operation_read_adapter_rank < 0:
            raise ValueError("operation_read_adapter_rank must be non-negative")
        if operation_read_adapter_scale < 0.0:
            raise ValueError("operation_read_adapter_scale must be non-negative")
        if operation_write_adapter_rank < 0:
            raise ValueError("operation_write_adapter_rank must be non-negative")
        if operation_write_adapter_scale < 0.0:
            raise ValueError("operation_write_adapter_scale must be non-negative")
        if operation_transition_rank < 0:
            raise ValueError("operation_transition_rank must be non-negative")
        if operation_transition_scale < 0.0:
            raise ValueError("operation_transition_scale must be non-negative")
        if numeric_state_dim < 0:
            raise ValueError("numeric_state_dim must be non-negative")
        if numeric_state_scale < 0.0:
            raise ValueError("numeric_state_scale must be non-negative")
        if modular_prior_mode not in {"fixed", "templates"}:
            raise ValueError("modular_prior_mode must be fixed or templates")
        if modular_template_init not in {"identity", "random"}:
            raise ValueError("modular_template_init must be identity or random")
        if circuit_residual_scale < 0.0:
            raise ValueError("circuit_residual_scale must be non-negative")
        if output_mode not in {"learned", "scalar_gaussian"}:
            raise ValueError("output_mode must be learned or scalar_gaussian")
        if output_temperature <= 0.0:
            raise ValueError("output_temperature must be positive")
        if macro_cell_count < 0:
            raise ValueError("macro_cell_count must be non-negative")
        if macro_cell_count:
            if macro_router_branch < 2:
                raise ValueError("macro_router_branch must be at least two")
            if macro_candidate_pool < 1 or macro_candidate_pool > macro_cell_count:
                raise ValueError("macro_candidate_pool must fit the macro bank")
            if active_macro_cells < 1 or active_macro_cells > macro_candidate_pool:
                raise ValueError(
                    "active_macro_cells must fit the macro candidate pool"
                )
        if macro_cell_scale < 0.0:
            raise ValueError("macro_cell_scale must be non-negative")

        self.max_ops = max_ops
        self.modulus = int(modulus)
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
        self.state_layout = state_layout
        self.predecessor_operation_context = bool(predecessor_operation_context)
        self.operation_adapter_rank = int(operation_adapter_rank)
        self.operation_adapter_scale = float(operation_adapter_scale)
        self.operation_adapter_gate_enabled = bool(operation_adapter_gate)
        self.operation_read_adapter_rank = int(operation_read_adapter_rank)
        self.operation_read_adapter_scale = float(operation_read_adapter_scale)
        self.operation_write_adapter_rank = int(operation_write_adapter_rank)
        self.operation_write_adapter_scale = float(operation_write_adapter_scale)
        self.operation_circuit_bank = bool(operation_circuit_bank)
        self.operation_router_keys = bool(operation_router_keys)
        self.operation_transition_rank = int(operation_transition_rank)
        self.operation_transition_scale = float(operation_transition_scale)
        self.numeric_state_dim = int(numeric_state_dim)
        self.numeric_state_scale = float(numeric_state_scale)
        self.modular_prior_enabled = bool(modular_prior)
        self.modular_prior_mode = modular_prior_mode
        self.modular_template_init = modular_template_init
        self.circuit_residual_scale = float(circuit_residual_scale)
        self.circuit_input_norm_enabled = bool(circuit_input_norm)
        self.output_mode = output_mode
        self.output_temperature = float(output_temperature)
        self.output_scalar_bias = float(output_scalar_bias)
        self.macro_cell_count = int(macro_cell_count)
        self.macro_cell_rank = int(macro_cell_rank)
        self.macro_cell_depth = int(macro_cell_depth)
        self.macro_router_branch = int(macro_router_branch)
        self.macro_candidate_pool = int(macro_candidate_pool)
        self.active_macro_cells = int(active_macro_cells)
        self.macro_cell_scale = float(macro_cell_scale)
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
        if self.predecessor_operation_context:
            self.predecessor_operation_embedding = nn.Embedding(4, state_dim)
        self.step_embedding = nn.Parameter(torch.zeros(max_ops, state_dim))
        if state_layout == "dual_slot":
            self.slot_dim = state_dim // 2
            self.slot_writers = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(2 * self.slot_dim),
                    nn.Linear(2 * self.slot_dim, self.slot_dim),
                    nn.Tanh(),
                )
                for _ in range(2)
            ])
        else:
            self.slot_dim = state_dim
            self.register_writer = nn.Sequential(
                nn.LayerNorm(2 * state_dim),
                nn.Linear(2 * state_dim, state_dim),
                nn.Tanh(),
            )
        if self.numeric_state_dim:
            self.numeric_value_encoder = nn.Sequential(
                nn.Linear(1, self.numeric_state_dim), nn.Tanh()
            )
            self.numeric_operation_embedding = nn.Embedding(
                3, self.numeric_state_dim
            )
            self.numeric_transition = nn.Sequential(
                nn.Linear(3 * self.numeric_state_dim, 2 * self.numeric_state_dim),
                nn.GELU(),
                nn.Linear(2 * self.numeric_state_dim, self.numeric_state_dim),
                nn.Tanh(),
            )
            self.numeric_state_projection = nn.Sequential(
                nn.LayerNorm(self.numeric_state_dim),
                nn.Linear(self.numeric_state_dim, state_dim),
                nn.Tanh(),
            )
        if self.operation_adapter_rank:
            self.operation_adapter_down = nn.Parameter(torch.empty(
                3, state_dim, self.operation_adapter_rank
            ))
            self.operation_adapter_up = nn.Parameter(torch.empty(
                3, self.operation_adapter_rank, state_dim
            ))
            self.operation_adapter_bias = nn.Parameter(torch.zeros(3, state_dim))
            nn.init.normal_(self.operation_adapter_down, std=0.02)
            nn.init.normal_(self.operation_adapter_up, std=0.02)
            if self.operation_adapter_gate_enabled:
                self.operation_adapter_gate = nn.Parameter(torch.zeros(()))
        if self.operation_read_adapter_rank:
            self.operation_read_adapter_down = nn.Parameter(torch.empty(
                3, state_dim, self.operation_read_adapter_rank
            ))
            self.operation_read_adapter_up = nn.Parameter(torch.empty(
                3, self.operation_read_adapter_rank, state_dim
            ))
            self.operation_read_adapter_bias = nn.Parameter(torch.zeros(3, state_dim))
            nn.init.normal_(self.operation_read_adapter_down, std=0.02)
            nn.init.normal_(self.operation_read_adapter_up, std=0.02)
        if self.operation_write_adapter_rank:
            self.operation_write_adapter_down = nn.Parameter(torch.empty(
                3, state_dim, self.operation_write_adapter_rank
            ))
            self.operation_write_adapter_up = nn.Parameter(torch.empty(
                3, self.operation_write_adapter_rank, state_dim
            ))
            self.operation_write_adapter_bias = nn.Parameter(torch.zeros(3, state_dim))
            nn.init.normal_(self.operation_write_adapter_down, std=0.02)
            nn.init.normal_(self.operation_write_adapter_up, std=0.02)
        if self.operation_transition_rank:
            self.operation_transition_down = nn.Parameter(torch.empty(
                3, state_dim, self.operation_transition_rank
            ))
            self.operation_transition_up = nn.Parameter(torch.empty(
                3, self.operation_transition_rank, state_dim
            ))
            self.operation_transition_bias = nn.Parameter(torch.zeros(3, state_dim))
            nn.init.normal_(self.operation_transition_down, std=0.02)
            nn.init.normal_(self.operation_transition_up, std=0.02)
        if self.modular_prior_enabled:
            if self.modular_prior_mode == "fixed":
                left = torch.arange(self.modulus).view(-1, 1)
                right = torch.arange(self.modulus).view(1, -1)
                transition = torch.stack((
                    (left + right).remainder(self.modulus),
                    (left - right).remainder(self.modulus),
                    (left * right).remainder(self.modulus),
                ))
                self.register_buffer("modular_transition", transition, persistent=False)
            else:
                self.modular_template_logits = nn.Parameter(torch.empty(3, 3))
                if modular_template_init == "identity":
                    with torch.no_grad():
                        self.modular_template_logits.copy_(4.0 * torch.eye(3))
                else:
                    nn.init.normal_(self.modular_template_logits, std=0.02)
            self.modular_projection = nn.Sequential(
                nn.LayerNorm(self.modulus),
                nn.Linear(self.modulus, state_dim),
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
                factor_capacity=factor_capacity,
                operation_key_bank=operation_router_keys,
            )

            def make_circuit_bank() -> nn.Module:
                return FactorizedMicroCircuitBank(
                    num_circuits, state_dim, circuit_rank, factor_count, factor_mix_mode
                )
        else:
            self.router = HierarchicalRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, 1
            )

            def make_circuit_bank() -> nn.Module:
                return MicroCircuitBank(num_circuits, state_dim, circuit_rank)

        self.circuits = (
            nn.ModuleList([make_circuit_bank() for _ in range(3)])
            if self.operation_circuit_bank else make_circuit_bank()
        )
        self.circuit_input_norm = (
            nn.LayerNorm(state_dim) if self.circuit_input_norm_enabled else None
        )
        if self.macro_cell_count:
            if macro_router_depth is None:
                depth = 1
                leaves = macro_router_branch
                while leaves < self.macro_cell_count:
                    depth += 1
                    leaves *= macro_router_branch
                macro_router_depth = depth
            if macro_router_depth < 1:
                raise ValueError("macro_router_depth must be positive")
            self.macro_router_depth = int(macro_router_depth)
            self.macro_router = HierarchicalRouter(
                state_dim,
                self.macro_cell_count,
                macro_router_branch,
                self.macro_router_depth,
                macro_candidate_pool,
                active_macro_cells,
                1,
            )
            self.macro_cell_bank = MacroCellBank(
                self.macro_cell_count,
                state_dim,
                macro_cell_rank,
                macro_cell_depth,
            )
        else:
            self.macro_router_depth = 0
        if output_mode == "scalar_gaussian":
            self.output = nn.Sequential(
                nn.LayerNorm(state_dim),
                ScalarGaussianOutput(
                    state_dim, num_classes, output_temperature, output_scalar_bias
                ),
            )
        else:
            self.output = nn.Sequential(
                nn.LayerNorm(state_dim), nn.Linear(state_dim, num_classes)
            )

        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.position_scale, std=0.01)
        nn.init.normal_(self.position_bias, std=0.01)
        nn.init.normal_(self.step_embedding, std=0.02)
        nn.init.normal_(self.operation_embedding.weight, std=0.02)
        if self.predecessor_operation_context:
            nn.init.normal_(self.predecessor_operation_embedding.weight, std=0.02)

    def encode_program(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] < self.value_start + self.max_ops + 1:
            raise ValueError("inputs are shorter than the configured program layout")
        tokens = encode_tokens(
            inputs, self.token_embedding, self.value_encoder,
            value_modulus=self.modulus,
        )
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
        operation_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.operation_circuit_bank:
            if operation_ids is None:
                raise ValueError("operation_ids are required for operation circuit banks")
            result = torch.zeros_like(query)
            for operation_id, bank in enumerate(self.circuits):
                mask = operation_ids.eq(operation_id)
                if mask.any():
                    if self.circuit_mode == "serial":
                        result[mask] = bank.forward_serial(
                            query[mask], selected[mask], weights[mask]
                        )
                    else:
                        result[mask] = bank(query[mask], selected[mask], weights[mask])
            return result
        if self.circuit_mode == "serial":
            return self.circuits.forward_serial(query, selected, weights)
        return self.circuits(query, selected, weights)

    def _operation_adapter(
        self, pair: torch.Tensor, operation_ids: torch.Tensor
    ) -> torch.Tensor:
        down = torch.einsum(
            "bd,bdr->br", pair, self.operation_adapter_down[operation_ids]
        )
        adapted = torch.einsum(
            "br,brd->bd", down, self.operation_adapter_up[operation_ids]
        ) + self.operation_adapter_bias[operation_ids]
        return nn.functional.gelu(adapted)

    def _operation_read_adapter(
        self, state: torch.Tensor, operation_ids: torch.Tensor
    ) -> torch.Tensor:
        down = torch.einsum(
            "bd,bdr->br", state,
            self.operation_read_adapter_down[operation_ids]
        )
        adapted = torch.einsum(
            "br,brd->bd", down,
            self.operation_read_adapter_up[operation_ids]
        ) + self.operation_read_adapter_bias[operation_ids]
        return nn.functional.gelu(adapted)

    def _operation_write_adapter(
        self, state: torch.Tensor, operation_ids: torch.Tensor
    ) -> torch.Tensor:
        down = torch.einsum(
            "bd,bdr->br", state,
            self.operation_write_adapter_down[operation_ids]
        )
        adapted = torch.einsum(
            "br,brd->bd", down,
            self.operation_write_adapter_up[operation_ids]
        ) + self.operation_write_adapter_bias[operation_ids]
        return nn.functional.gelu(adapted)

    def _operation_transition(
        self, state: torch.Tensor, operation_ids: torch.Tensor
    ) -> torch.Tensor:
        down = torch.einsum(
            "bd,bdr->br", state,
            self.operation_transition_down[operation_ids]
        )
        adapted = torch.einsum(
            "br,brd->bd", down,
            self.operation_transition_up[operation_ids]
        ) + self.operation_transition_bias[operation_ids]
        return nn.functional.gelu(adapted)

    def _write_state(
        self, accumulator: torch.Tensor, write_input: torch.Tensor
    ) -> torch.Tensor:
        if self.state_layout == "flat":
            return self.register_writer(torch.cat([accumulator, write_input], dim=-1))
        accumulator_slots = accumulator.chunk(2, dim=-1)
        write_slots = write_input.chunk(2, dim=-1)
        updated_slots = [
            writer(torch.cat([accumulator_slot, write_slot], dim=-1))
            for writer, accumulator_slot, write_slot in zip(
                self.slot_writers, accumulator_slots, write_slots
            )
        ]
        return torch.cat(updated_slots, dim=-1)

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
        if self.numeric_state_dim:
            numeric_values = (
                inputs[:, self.value_start:self.value_start + self.max_ops + 1]
                - VALUE_TOKEN_OFFSET
            ).clamp(0, self.modulus - 1).to(operand_states.dtype)
            numeric_operands = self.numeric_value_encoder(
                numeric_values.unsqueeze(-1) / float(max(self.modulus - 1, 1))
            )
            numeric_state = numeric_operands[:, 0]
        input_context = (
            (operand_states * operand_mask.unsqueeze(-1)).sum(dim=1)
            / operand_mask.sum(dim=1, keepdim=True).clamp_min(1).to(operand_states.dtype)
        )
        accumulator = self.initial_writer(operand_states[:, 0])
        if self.modular_prior_enabled:
            initial_values = (
                inputs[:, self.value_start] - VALUE_TOKEN_OFFSET
            ).clamp(0, self.modulus - 1)
            if self.modular_prior_mode == "fixed":
                modular_accumulator = initial_values
            else:
                modular_state = nn.functional.one_hot(
                    initial_values, self.modulus
                ).to(accumulator.dtype)

        selected_steps = []
        macro_selected_steps = []
        selected_weights = torch.zeros(
            batch_size, self.max_ops, self.router.active_circuits, device=device
        )
        macro_selected_weights = torch.zeros(
            batch_size,
            self.max_ops,
            self.active_macro_cells if self.macro_cell_count else 0,
            device=device,
        )
        step_entropies = torch.zeros(batch_size, self.max_ops, device=device)
        macro_step_entropies = torch.zeros(batch_size, self.max_ops, device=device)
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
            macro_selected_step = torch.full(
                (batch_size, self.active_macro_cells if self.macro_cell_count else 0),
                -1,
                dtype=torch.long,
                device=device,
            )
            if active_indices.numel():
                active_accumulator = accumulator[active_indices]
                active_operand = operand_states[active_indices, step + 1]
                current_operation_ids = operation_ids[active_indices, step]
                read_accumulator = active_accumulator
                if self.operation_read_adapter_rank:
                    read_accumulator = read_accumulator + self.operation_read_adapter_scale * (
                        self._operation_read_adapter(
                            active_accumulator, current_operation_ids
                        )
                    )
                pair = self.pair_encoder(torch.cat([read_accumulator, active_operand], dim=-1))
                pair = pair + self.product_encoder(read_accumulator * active_operand)
                query = (
                    pair
                    + self.operation_embedding(current_operation_ids)
                    + self.step_embedding[step]
                )
                if self.predecessor_operation_context:
                    if step == 0:
                        previous_operation_ids = torch.full_like(
                            current_operation_ids, 3
                        )
                    else:
                        previous_operation_ids = operation_ids[active_indices, step - 1]
                        previous_operation_ids = torch.where(
                            operation_mask[active_indices, step - 1],
                            previous_operation_ids,
                            torch.full_like(previous_operation_ids, 3),
                        )
                    query = query + self.predecessor_operation_embedding(
                        previous_operation_ids
                    )
                if self.numeric_state_dim:
                    numeric_operation = self.numeric_operation_embedding(
                        current_operation_ids
                    )
                    numeric_input = torch.cat((
                        numeric_state[active_indices],
                        numeric_operands[active_indices, step + 1],
                        numeric_operation,
                    ), dim=-1)
                    numeric_candidate = numeric_state[active_indices] + self.numeric_transition(
                        numeric_input
                    )
                    query = query + self.numeric_state_scale * self.numeric_state_projection(
                        numeric_candidate
                    )
                if self.operation_adapter_rank:
                    adapter_scale = self.operation_adapter_scale
                    if self.operation_adapter_gate_enabled:
                        adapter_scale = adapter_scale * torch.tanh(
                            self.operation_adapter_gate
                        )
                    query = query + adapter_scale * self._operation_adapter(
                        pair, current_operation_ids
                    )
                if self.input_reinjection_scale:
                    query = query + self.input_reinjection_scale * input_context[active_indices]
                if self.modular_prior_enabled:
                    if self.modular_prior_mode == "fixed":
                        modular_features = nn.functional.one_hot(
                            modular_accumulator[active_indices], self.modulus
                        ).to(query.dtype)
                    else:
                        modular_features = modular_state[active_indices].to(query.dtype)
                    query = query + self.modular_projection(modular_features)
                query = query + 0.25 * self.route_context(query)
                route_query = query
                if self.route_context_mode == "operation_step":
                    route_query = (
                        self.operation_embedding(current_operation_ids)
                        + self.step_embedding[step]
                    )
                    route_query = route_query + 0.25 * self.route_context(route_query)
                elif self.route_context_mode == "hybrid":
                    route_query = route_query + 0.25 * (
                        self.operation_embedding(current_operation_ids)
                        + self.step_embedding[step]
                    )
                if self.macro_cell_count:
                    macro_selected, macro_weights, macro_route_stats = self.macro_router(
                        route_query,
                        exploration_prob=(
                            self.route_exploration_prob if self.training else 0.0
                        ),
                    )
                    macro_delta = self.macro_cell_bank(
                        query, macro_selected, macro_weights
                    )
                    query = query + self.macro_cell_scale * macro_delta
                    macro_selected_step[active_indices] = macro_selected
                    macro_selected_weights[active_indices, step] = macro_weights
                    macro_step_entropies[active_indices, step] = macro_route_stats[
                        "router_entropy"
                    ]
                selected, weights, route_stats = self.router(
                    route_query,
                    exploration_prob=(self.route_exploration_prob if self.training else 0.0),
                    operation_ids=(current_operation_ids
                                   if self.operation_router_keys else None),
                )
                if self.circuit_residual_scale:
                    circuit_query = (
                        self.circuit_input_norm(query)
                        if self.circuit_input_norm is not None else query
                    )
                    delta = self.circuit_residual_scale * self._apply_circuits(
                        circuit_query, selected, weights,
                        current_operation_ids,
                    )
                else:
                    delta = torch.zeros_like(query)
                write_input = query + delta
                if self.operation_transition_rank:
                    write_input = write_input + self.operation_transition_scale * (
                        self._operation_transition(
                            write_input, current_operation_ids
                        )
                    )
                candidate = self._write_state(active_accumulator, write_input)
                if self.operation_write_adapter_rank:
                    candidate = candidate + self.operation_write_adapter_scale * (
                        self._operation_write_adapter(
                            candidate, current_operation_ids
                        )
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
                if self.numeric_state_dim:
                    next_numeric_state = numeric_state.clone()
                    next_numeric_state[active_indices] = numeric_candidate
                    numeric_state = next_numeric_state
                if self.modular_prior_enabled:
                    operand_values = (
                        inputs[active_indices, self.value_start + step + 1]
                        - VALUE_TOKEN_OFFSET
                    ).clamp(0, self.modulus - 1)
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
                            modular_add_state(
                                modular_state[active_indices], operand_values,
                                self.modulus),
                            modular_subtract_state(
                                modular_state[active_indices], operand_values,
                                self.modulus),
                            modular_multiply_state(
                                modular_state[active_indices], operand_values,
                                self.modulus),
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
            macro_selected_steps.append(macro_selected_step)
            step_state = accumulator
            if self.numeric_state_dim:
                step_state = step_state + self.numeric_state_scale * self.numeric_state_projection(
                    numeric_state
                )
            if self.modular_prior_enabled:
                if self.modular_prior_mode == "fixed":
                    step_features = nn.functional.one_hot(
                        modular_accumulator, self.modulus
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
            "macro_selected_ids": torch.stack(macro_selected_steps, dim=1),
            "macro_selected_weights": macro_selected_weights,
            "step_logits": torch.stack(step_logits, dim=1),
            "executed_steps": executed_mask.sum(dim=1),
            "executed_mask": executed_mask,
            "macro_router_entropy": macro_step_entropies.sum()
            / executed_mask.sum().clamp_min(1),
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
            + (
                count_parameters(self.register_writer)
                if self.state_layout == "flat"
                else count_parameters(self.slot_writers)
            )
            + count_parameters(self.route_context)
            + count_parameters(self.output)
        )
        if self.numeric_state_dim:
            shared += (
                count_parameters(self.numeric_value_encoder)
                + count_parameters(self.numeric_operation_embedding)
                + count_parameters(self.numeric_transition)
                + count_parameters(self.numeric_state_projection)
            )
        if self.operation_adapter_rank:
            shared += (
                self.operation_adapter_down.numel()
                + self.operation_adapter_up.numel()
                + self.operation_adapter_bias.numel()
            )
            if self.operation_adapter_gate_enabled:
                shared += self.operation_adapter_gate.numel()
        if self.predecessor_operation_context:
            shared += count_parameters(self.predecessor_operation_embedding)
        if self.operation_read_adapter_rank:
            shared += (
                self.operation_read_adapter_down.numel()
                + self.operation_read_adapter_up.numel()
                + self.operation_read_adapter_bias.numel()
            )
        if self.operation_write_adapter_rank:
            shared += (
                self.operation_write_adapter_down.numel()
                + self.operation_write_adapter_up.numel()
                + self.operation_write_adapter_bias.numel()
            )
        if self.operation_transition_rank:
            shared += (
                self.operation_transition_down.numel()
                + self.operation_transition_up.numel()
                + self.operation_transition_bias.numel()
            )
        if self.write_gate_enabled:
            shared += count_parameters(self.write_gate)
        if self.circuit_input_norm is not None:
            shared += count_parameters(self.circuit_input_norm)
        if self.modular_prior_enabled:
            shared += count_parameters(self.modular_projection)
            if self.modular_prior_mode == "templates":
                shared += self.modular_template_logits.numel()
        circuit_bank = self.circuits[0] if self.operation_circuit_bank else self.circuits
        if self.circuit_bank_mode == "factorized":
            factor_row = (
                circuit_bank.down_factors[0].numel()
                + circuit_bank.up_factors[0].numel()
                + circuit_bank.bias_factors[0].numel()
                + circuit_bank.factor_mix[0].numel()
            )
            bank_count = min(self.max_ops, 3) if self.operation_circuit_bank else 1
            active_circuit_params = factor_row * self.router.active_circuits * 2 * bank_count
            candidate_params = self.router.keys[0].numel() * self.router.factor_candidate_pool * 2
        else:
            one_circuit = (
                circuit_bank.down[0].numel()
                + circuit_bank.up[0].numel()
                + circuit_bank.bias[0].numel()
            )
            bank_count = min(self.max_ops, 3) if self.operation_circuit_bank else 1
            active_circuit_params = one_circuit * self.router.active_circuits * bank_count
            candidate_params = self.router.keys[0].numel() * self.router.candidate_pool
        operation_router_key_params = 0
        operation_router_key_active = 0
        if self.operation_router_keys:
            operation_router_key_params = self.router.operation_key_deltas.numel()
            operation_router_key_active = (
                self.router.keys[0].numel()
                * self.router.factor_candidate_pool
                * 2
                * bank_count
            )
            candidate_params += operation_router_key_active
        macro_total = 0
        macro_active = 0
        if self.macro_cell_count:
            macro_total = count_parameters(self.macro_router) + count_parameters(
                self.macro_cell_bank
            )
            macro_router_shared = (
                self.macro_router.level_projections.numel()
                + self.macro_router.level_bias.numel()
            )
            macro_router_candidates = (
                self.macro_router.keys.shape[1] * self.macro_candidate_pool
            )
            macro_active = (
                macro_router_shared
                + macro_router_candidates
                + self.macro_cell_bank.parameters_per_cell * self.active_macro_cells
            )
        return {
            "total_params": total,
            "active_params_estimate": shared + candidate_params + active_circuit_params,
            "active_fraction": (shared + candidate_params + active_circuit_params) / total,
            "active_circuit_params": active_circuit_params,
            "max_ops": self.max_ops,
            "modulus": self.modulus,
            "circuit_bank_mode": self.circuit_bank_mode,
            "routing_mode": "factorized" if self.circuit_bank_mode == "factorized" else "hierarchical",
            "input_reinjection_scale": self.input_reinjection_scale,
            "write_gate": self.write_gate_enabled,
            "value_encoder_mode": self.value_encoder_mode,
            "factor_mix_mode": self.factor_mix_mode,
            "factor_count": self.router.factor_count if self.circuit_bank_mode == "factorized" else None,
            "factor_candidate_pool": (
                self.router.factor_candidate_pool
                if self.circuit_bank_mode == "factorized" else None
            ),
            "factor_capacity": (
                self.router.factor_capacity
                if self.circuit_bank_mode == "factorized" else None
            ),
            "route_context_mode": self.route_context_mode,
            "state_layout": self.state_layout,
            "predecessor_operation_context": self.predecessor_operation_context,
            "operation_adapter_rank": self.operation_adapter_rank,
            "operation_adapter_scale": self.operation_adapter_scale,
            "operation_adapter_gate": self.operation_adapter_gate_enabled,
            "operation_adapter_gate_value": (
                float(torch.tanh(self.operation_adapter_gate).detach().cpu())
                if self.operation_adapter_gate_enabled else None
            ),
            "operation_read_adapter_rank": self.operation_read_adapter_rank,
            "operation_read_adapter_scale": self.operation_read_adapter_scale,
            "operation_write_adapter_rank": self.operation_write_adapter_rank,
            "operation_write_adapter_scale": self.operation_write_adapter_scale,
            "operation_circuit_bank": self.operation_circuit_bank,
            "operation_router_keys": self.operation_router_keys,
            "operation_router_key_params": operation_router_key_params,
            "operation_router_key_active_estimate": operation_router_key_active,
            "operation_transition_rank": self.operation_transition_rank,
            "operation_transition_scale": self.operation_transition_scale,
            "numeric_state_dim": self.numeric_state_dim,
            "numeric_state_scale": self.numeric_state_scale,
            "modular_prior": self.modular_prior_enabled,
            "modular_prior_mode": self.modular_prior_mode,
            "modular_template_init": self.modular_template_init,
            "circuit_residual_scale": self.circuit_residual_scale,
            "circuit_input_norm": self.circuit_input_norm_enabled,
            "output_mode": self.output_mode,
            "output_temperature": self.output_temperature,
            "output_scalar_bias": self.output_scalar_bias,
            "macro_cell_count": self.macro_cell_count,
            "macro_cell_rank": self.macro_cell_rank,
            "macro_cell_depth": self.macro_cell_depth,
            "macro_router_branch": self.macro_router_branch,
            "macro_router_depth": self.macro_router_depth,
            "macro_candidate_pool": self.macro_candidate_pool,
            "active_macro_cells": self.active_macro_cells if self.macro_cell_count else 0,
            "macro_cell_scale": self.macro_cell_scale,
            "macro_total_params": macro_total,
            "macro_active_params_estimate": macro_active,
        }
