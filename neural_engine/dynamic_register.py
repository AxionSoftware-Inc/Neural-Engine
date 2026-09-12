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
from .operator_valued import OperatorValuedLinear
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


class FactorizedDigitOutput(nn.Module):
    """Factor a large class index into two additive digit heads.

    ``digit_count`` optionally splits the class index into more than two
    digits. ``projection_rank`` shares a low-rank state bottleneck between all
    digit classifiers. Both are opt-in output-codec controls; the circuit bank
    and recurrent state are unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        digit_base: int,
        projection_rank: int = 0,
        digit_count: int = 2,
        interaction_rank: int = 0,
        context_mode: str = "soft",
    ):
        super().__init__()
        if digit_base < 2:
            raise ValueError("digit_base must be at least two")
        if digit_count < 2:
            raise ValueError("digit_count must be at least two")
        factor = digit_base ** (digit_count - 1)
        if num_classes % factor:
            raise ValueError(
                "num_classes must be divisible by digit_base^(digit_count-1)"
            )
        if projection_rank < 0:
            raise ValueError("projection_rank must be non-negative")
        if interaction_rank < 0:
            raise ValueError("interaction_rank must be non-negative")
        if context_mode not in {"soft", "straight_through_hard"}:
            raise ValueError(
                "context_mode must be soft or straight_through_hard"
            )
        self.num_classes = int(num_classes)
        self.digit_base = int(digit_base)
        self.digit_count = int(digit_count)
        self.high_classes = num_classes // factor
        self.projection_rank = int(projection_rank)
        self.interaction_rank = int(interaction_rank)
        self.context_mode = context_mode
        if self.projection_rank:
            self.shared_projection = nn.Linear(input_dim, self.projection_rank)
            classifier_dim = self.projection_rank
        else:
            self.shared_projection = None
            classifier_dim = input_dim
        digit_sizes = [self.high_classes] + [self.digit_base] * (self.digit_count - 1)
        self.digit_heads = nn.ModuleList(
            nn.Linear(classifier_dim, size) for size in digit_sizes
        )
        if self.interaction_rank:
            self.digit_context_embeddings = nn.ModuleList(
                nn.Embedding(size, self.interaction_rank)
                for size in digit_sizes[:-1]
            )
            self.digit_context_projections = nn.ModuleList(
                nn.Linear(self.interaction_rank, classifier_dim, bias=False)
                for _ in digit_sizes[:-1]
            )
        else:
            self.digit_context_embeddings = None
            self.digit_context_projections = None
        # Keep the two-head names for callers and checkpoints using the
        # original compact interface.
        self.high = self.digit_heads[0]
        self.low = self.digit_heads[-1]
        self.out_features = self.num_classes

    def digit_logits(self, states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.shared_projection is not None:
            states = self.shared_projection(states)
        if self.interaction_rank == 0:
            return tuple(head(states) for head in self.digit_heads)
        base_states = states
        logits = []
        for index, head in enumerate(self.digit_heads):
            current = head(states)
            logits.append(current)
            if index < len(self.digit_heads) - 1:
                probabilities = current.softmax(dim=-1)
                if self.context_mode == "straight_through_hard":
                    hard = torch.nn.functional.one_hot(
                        current.argmax(dim=-1), num_classes=current.shape[-1]
                    ).to(probabilities.dtype)
                    probabilities = probabilities + (hard - probabilities).detach()
                context = probabilities @ self.digit_context_embeddings[index].weight
                states = base_states + self.digit_context_projections[index](context)
        return tuple(logits)

    def combine(
        self, *digit_logits: torch.Tensor
    ) -> torch.Tensor:
        if len(digit_logits) != self.digit_count:
            raise ValueError("digit logits do not match configured digit count")
        combined = digit_logits[0]
        for logits in digit_logits[1:]:
            combined = (combined.unsqueeze(-1) + logits.unsqueeze(-2)).reshape(
                combined.shape[0], -1
            )
        return combined

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.combine(*self.digit_logits(states))


class GeometricFactorizedDigitOutput(nn.Module):
    """Factorized digit output with ordered scalar coordinates.

    Each digit head predicts one continuous coordinate and scores digit classes
    by distance to their ordered positions.  This is an opt-in alternative to
    independent categorical logits; it is intended to test whether unseen
    digit values can transfer through an ordered readout.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        digit_base: int,
        projection_rank: int = 0,
        digit_count: int = 2,
        temperature: float = 1.0,
    ):
        super().__init__()
        if digit_base < 2:
            raise ValueError("digit_base must be at least two")
        if digit_count < 2:
            raise ValueError("digit_count must be at least two")
        factor = digit_base ** (digit_count - 1)
        if num_classes % factor:
            raise ValueError(
                "num_classes must be divisible by digit_base^(digit_count-1)"
            )
        if projection_rank < 0:
            raise ValueError("projection_rank must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.num_classes = int(num_classes)
        self.digit_base = int(digit_base)
        self.digit_count = int(digit_count)
        self.high_classes = num_classes // factor
        self.projection_rank = int(projection_rank)
        self.temperature = float(temperature)
        if self.projection_rank:
            self.shared_projection = nn.Linear(input_dim, self.projection_rank)
            classifier_dim = self.projection_rank
        else:
            self.shared_projection = None
            classifier_dim = input_dim
        digit_sizes = [self.high_classes] + [self.digit_base] * (self.digit_count - 1)
        self.coordinate_heads = nn.ModuleList(
            nn.Linear(classifier_dim, 1) for _ in digit_sizes
        )
        for index, size in enumerate(digit_sizes):
            self.register_buffer(
                f"positions_{index}",
                torch.arange(size, dtype=torch.float32),
                persistent=False,
            )
        self.out_features = self.num_classes
        self.high = self.coordinate_heads[0]
        self.low = self.coordinate_heads[-1]

    def digit_logits(self, states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.shared_projection is not None:
            states = self.shared_projection(states)
        scale = self.temperature ** 2
        logits = []
        for index, head in enumerate(self.coordinate_heads):
            coordinate = head(states)
            positions = getattr(self, f"positions_{index}").to(states.dtype)
            logits.append(-0.5 * (coordinate - positions).square() / scale)
        return tuple(logits)

    def combine(self, *digit_logits: torch.Tensor) -> torch.Tensor:
        if len(digit_logits) != self.digit_count:
            raise ValueError("digit logits do not match configured digit count")
        combined = digit_logits[0]
        for logits in digit_logits[1:]:
            combined = (combined.unsqueeze(-1) + logits.unsqueeze(-2)).reshape(
                combined.shape[0], -1
            )
        return combined

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.combine(*self.digit_logits(states))


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
        modulus: int | None = VALUE_MODULUS,
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
        ordered_factor_slots: bool = False,
        query_factor_mix_scale: float = 0.0,
        factor_pair_rank: int = 0,
        factor_pair_scale: float = 1.0,
        circuit_mode: str = "serial",
        route_exploration_prob: float = 0.05,
        input_reinjection_scale: float = 0.0,
        write_gate: bool = False,
        value_encoder_mode: str = "learned",
        value_encoder_modulus: int | None = None,
        factor_mix_mode: str = "per_address",
        route_context_mode: str = "full",
        state_layout: str = "flat",
        state_update_mode: str = "overwrite",
        state_residual_scale: float = 0.25,
        predecessor_operation_context: bool = False,
        operation_adapter_rank: int = 0,
        operation_adapter_scale: float = 1.0,
        operation_adapter_gate: bool = False,
        operation_read_adapter_rank: int = 0,
        operation_read_adapter_scale: float = 1.0,
        operation_write_adapter_rank: int = 0,
        operation_write_adapter_scale: float = 1.0,
        operation_write_adapter_mode: str = "post_state",
        operation_output_adapter_rank: int = 0,
        operation_output_adapter_scale: float = 1.0,
        operation_circuit_bank: bool = False,
        operation_router_keys: bool = False,
        operation_transition_rank: int = 0,
        operation_transition_scale: float = 1.0,
        operation_bilinear_transition_rank: int = 0,
        operation_bilinear_transition_scale: float = 1.0,
        structured_scalar_state: bool = False,
        structured_scalar_scale: float = 1.0,
        structured_scalar_read_scale: float = 0.0,
        structured_scalar_authoritative: bool = False,
        algebraic_state_mode: str = "none",
        algebraic_state_scale: float = 1.0,
        algebraic_output_bridge_scale: float = 0.0,
        algebraic_state_write_scale: float = 0.0,
        algebraic_state_authoritative_read: bool = False,
        algebraic_output_decoder: bool = False,
        algebraic_integer_output_decoder: bool = False,
        algebraic_integer_digit_dim: int = 16,
        algebraic_integer_output_decoder_mode: str = "all",
        algebraic_integer_output_head: bool = False,
        algebraic_integer_output_factor_rank: int | None = None,
        algebraic_integer_output_digit_interaction_rank: int | None = None,
        algebraic_integer_state_read_scale: float = 0.0,
        algebraic_state_value_scale: float = 4096.0,
        algebraic_state_fourier_base: int = 128,
        operator_valued_product_encoder: bool = False,
        operator_valued_packet_width: int = 16,
        operator_valued_basis_count: int = 8,
        numeric_state_dim: int = 0,
        numeric_state_scale: float = 1.0,
        numeric_state_value_scale: float = 128.0,
        typed_digit_state: bool = False,
        typed_digit_dim: int = 16,
        typed_digit_base: int = 512,
        typed_digit_count: int = 4,
        typed_digit_scale: float = 1.0,
        typed_digit_value_offset: int = 0,
        typed_digit_operand_offset: int = 0,
        typed_digit_carry_chain: bool = False,
        typed_digit_multiply_convolution: bool = False,
        modular_prior: bool = False,
        modular_prior_mode: str = "fixed",
        modular_template_init: str = "identity",
        circuit_residual_scale: float = 1.0,
        circuit_input_norm: bool = False,
        output_mode: str = "learned",
        output_temperature: float = 16.0,
        output_scalar_bias: float = 0.0,
        output_digit_base: int = 128,
        output_factor_rank: int = 0,
        output_digit_count: int = 2,
        output_digit_interaction_rank: int = 0,
        output_digit_context_mode: str = "soft",
        output_digit_geometry: bool = False,
        output_digit_temperature: float = 1.0,
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
        if modulus is not None and modulus < 2:
            raise ValueError("modulus must be at least two")
        if modular_prior and modulus is None:
            raise ValueError("modular_prior requires a finite modulus")
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
        if value_encoder_modulus is not None and value_encoder_modulus < 2:
            raise ValueError("value_encoder_modulus must be at least two")
        if route_context_mode not in {"full", "operation_step", "hybrid"}:
            raise ValueError(
                "route_context_mode must be full, operation_step, or hybrid"
            )
        if state_layout not in {"flat", "dual_slot"}:
            raise ValueError("state_layout must be flat or dual_slot")
        if state_layout == "dual_slot" and state_dim % 2:
            raise ValueError("state_dim must be even for dual_slot state layout")
        if state_update_mode not in {"overwrite", "residual"}:
            raise ValueError("state_update_mode must be overwrite or residual")
        if state_residual_scale < 0.0:
            raise ValueError("state_residual_scale must be non-negative")
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
        if operation_output_adapter_rank < 0:
            raise ValueError("operation_output_adapter_rank must be non-negative")
        if operation_output_adapter_scale < 0.0:
            raise ValueError("operation_output_adapter_scale must be non-negative")
        if operation_write_adapter_mode not in {
            "post_state", "pre_writer", "terminal_only"
        }:
            raise ValueError(
                "operation_write_adapter_mode must be post_state, pre_writer, "
                "or terminal_only"
            )
        if operation_transition_rank < 0:
            raise ValueError("operation_transition_rank must be non-negative")
        if operation_transition_scale < 0.0:
            raise ValueError("operation_transition_scale must be non-negative")
        if operation_bilinear_transition_rank < 0:
            raise ValueError(
                "operation_bilinear_transition_rank must be non-negative"
            )
        if operation_bilinear_transition_scale < 0.0:
            raise ValueError(
                "operation_bilinear_transition_scale must be non-negative"
            )
        if structured_scalar_scale < 0.0:
            raise ValueError("structured_scalar_scale must be non-negative")
        if structured_scalar_read_scale < 0.0:
            raise ValueError("structured_scalar_read_scale must be non-negative")
        if structured_scalar_authoritative and not structured_scalar_state:
            raise ValueError("structured_scalar_authoritative requires structured_scalar_state")
        if algebraic_state_mode not in {"none", "polynomial2", "polynomial2_fourier"}:
            raise ValueError(
                "algebraic_state_mode must be none, polynomial2, or polynomial2_fourier"
            )
        if algebraic_state_mode != "none" and modulus is not None:
            raise ValueError("algebraic_state_mode currently requires modulus=None")
        if algebraic_state_scale < 0.0:
            raise ValueError("algebraic_state_scale must be non-negative")
        if algebraic_output_bridge_scale < 0.0:
            raise ValueError("algebraic_output_bridge_scale must be non-negative")
        if algebraic_output_bridge_scale and algebraic_state_mode == "none":
            raise ValueError(
                "algebraic_output_bridge_scale requires algebraic_state_mode"
            )
        if algebraic_state_write_scale < 0.0:
            raise ValueError("algebraic_state_write_scale must be non-negative")
        if algebraic_state_write_scale and algebraic_state_mode == "none":
            raise ValueError(
                "algebraic_state_write_scale requires algebraic_state_mode"
            )
        if algebraic_state_authoritative_read and algebraic_state_mode == "none":
            raise ValueError(
                "algebraic_state_authoritative_read requires algebraic_state_mode"
            )
        if algebraic_output_decoder and algebraic_state_mode == "none":
            raise ValueError(
                "algebraic_output_decoder requires algebraic_state_mode"
            )
        if algebraic_integer_output_decoder and algebraic_state_mode == "none":
            raise ValueError(
                "algebraic_integer_output_decoder requires algebraic_state_mode"
            )
        if algebraic_integer_digit_dim < 1:
            raise ValueError("algebraic_integer_digit_dim must be positive")
        if algebraic_integer_output_decoder_mode not in {"all", "multiply_only"}:
            raise ValueError(
                "algebraic_integer_output_decoder_mode must be all or multiply_only"
            )
        if algebraic_state_value_scale <= 0.0:
            raise ValueError("algebraic_state_value_scale must be positive")
        if algebraic_state_fourier_base < 2:
            raise ValueError("algebraic_state_fourier_base must be at least two")
        if operator_valued_packet_width < 1:
            raise ValueError("operator_valued_packet_width must be positive")
        if operator_valued_basis_count < 1:
            raise ValueError("operator_valued_basis_count must be positive")
        if operator_valued_product_encoder and state_dim % operator_valued_packet_width:
            raise ValueError(
                "state_dim must be divisible by operator_valued_packet_width"
            )
        if numeric_state_dim < 0:
            raise ValueError("numeric_state_dim must be non-negative")
        if numeric_state_scale < 0.0:
            raise ValueError("numeric_state_scale must be non-negative")
        if numeric_state_value_scale <= 0.0:
            raise ValueError("numeric_state_value_scale must be positive")
        if typed_digit_dim < 1:
            raise ValueError("typed_digit_dim must be positive")
        if typed_digit_base < 2:
            raise ValueError("typed_digit_base must be at least two")
        if typed_digit_count < 2:
            raise ValueError("typed_digit_count must be at least two")
        if typed_digit_scale < 0.0:
            raise ValueError("typed_digit_scale must be non-negative")
        if modular_prior_mode not in {"fixed", "templates"}:
            raise ValueError("modular_prior_mode must be fixed or templates")
        if modular_template_init not in {"identity", "random"}:
            raise ValueError("modular_template_init must be identity or random")
        if circuit_residual_scale < 0.0:
            raise ValueError("circuit_residual_scale must be non-negative")
        if output_mode not in {"learned", "scalar_gaussian", "factorized_digits"}:
            raise ValueError(
                "output_mode must be learned, scalar_gaussian, or factorized_digits"
            )
        if algebraic_integer_output_head and not algebraic_integer_output_decoder:
            raise ValueError(
                "algebraic_integer_output_head requires algebraic_integer_output_decoder"
            )
        if algebraic_integer_output_head and output_mode != "factorized_digits":
            raise ValueError(
                "algebraic_integer_output_head requires factorized_digits output"
            )
        if algebraic_integer_output_factor_rank is None:
            algebraic_integer_output_factor_rank = output_factor_rank
        if algebraic_integer_output_digit_interaction_rank is None:
            algebraic_integer_output_digit_interaction_rank = (
                output_digit_interaction_rank
            )
        if algebraic_integer_output_factor_rank < 0:
            raise ValueError(
                "algebraic_integer_output_factor_rank must be non-negative"
            )
        if algebraic_integer_output_digit_interaction_rank < 0:
            raise ValueError(
                "algebraic_integer_output_digit_interaction_rank must be non-negative"
            )
        if algebraic_integer_state_read_scale < 0.0:
            raise ValueError("algebraic_integer_state_read_scale must be non-negative")
        if algebraic_integer_state_read_scale and not algebraic_integer_output_decoder:
            raise ValueError(
                "algebraic_integer_state_read_scale requires algebraic_integer_output_decoder"
            )
        if output_temperature <= 0.0:
            raise ValueError("output_temperature must be positive")
        if output_digit_base < 2:
            raise ValueError("output_digit_base must be at least two")
        if output_factor_rank < 0:
            raise ValueError("output_factor_rank must be non-negative")
        if output_digit_interaction_rank < 0:
            raise ValueError("output_digit_interaction_rank must be non-negative")
        if output_digit_context_mode not in {"soft", "straight_through_hard"}:
            raise ValueError(
                "output_digit_context_mode must be soft or straight_through_hard"
            )
        if output_digit_temperature <= 0.0:
            raise ValueError("output_digit_temperature must be positive")
        if output_digit_geometry and output_mode != "factorized_digits":
            raise ValueError("output_digit_geometry requires factorized_digits output")
        if output_digit_geometry and output_digit_interaction_rank:
            raise ValueError(
                "output_digit_geometry is incompatible with digit interaction"
            )
        if output_digit_interaction_rank and output_mode != "factorized_digits":
            raise ValueError(
                "output_digit_interaction_rank requires factorized_digits output"
            )
        if output_factor_rank and output_mode != "factorized_digits":
            raise ValueError("output_factor_rank requires factorized_digits output")
        if output_digit_count < 2:
            raise ValueError("output_digit_count must be at least two")
        if output_mode == "factorized_digits":
            factor = output_digit_base ** (output_digit_count - 1)
            if num_classes % factor:
                raise ValueError(
                    "num_classes must be divisible by output_digit_base^(output_digit_count-1)"
                )
        if typed_digit_state and output_mode != "factorized_digits":
            raise ValueError("typed digit state requires factorized_digits output")
        if typed_digit_state and (
            typed_digit_base != output_digit_base
            or typed_digit_count != output_digit_count
        ):
            raise ValueError(
                "typed digit state must use the configured factorized output layout"
            )
        if typed_digit_multiply_convolution and not (
            typed_digit_state and typed_digit_carry_chain
        ):
            raise ValueError(
                "typed_digit_multiply_convolution requires typed carry state"
            )
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
        self.modulus = None if modulus is None else int(modulus)
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.internal_steps = max_ops
        self.circuit_mode = circuit_mode
        self.circuit_bank_mode = circuit_bank_mode
        self.route_exploration_prob = route_exploration_prob
        self.input_reinjection_scale = float(input_reinjection_scale)
        self.write_gate_enabled = bool(write_gate)
        self.value_encoder_mode = value_encoder_mode
        self.value_encoder_modulus = int(
            self.modulus
            if self.modulus is not None
            else (value_encoder_modulus or VALUE_MODULUS)
        )
        self.factor_mix_mode = factor_mix_mode
        self.ordered_factor_slots = bool(ordered_factor_slots)
        self.query_factor_mix_scale = float(query_factor_mix_scale)
        if self.query_factor_mix_scale < 0.0:
            raise ValueError("query_factor_mix_scale must be non-negative")
        self.factor_pair_rank = int(factor_pair_rank)
        self.factor_pair_scale = float(factor_pair_scale)
        if self.factor_pair_rank < 0:
            raise ValueError("factor_pair_rank must be non-negative")
        if self.factor_pair_scale < 0.0:
            raise ValueError("factor_pair_scale must be non-negative")
        self.route_context_mode = route_context_mode
        self.state_layout = state_layout
        self.state_update_mode = state_update_mode
        self.state_residual_scale = float(state_residual_scale)
        self.predecessor_operation_context = bool(predecessor_operation_context)
        self.operation_adapter_rank = int(operation_adapter_rank)
        self.operation_adapter_scale = float(operation_adapter_scale)
        self.operation_adapter_gate_enabled = bool(operation_adapter_gate)
        self.operation_read_adapter_rank = int(operation_read_adapter_rank)
        self.operation_read_adapter_scale = float(operation_read_adapter_scale)
        self.operation_write_adapter_rank = int(operation_write_adapter_rank)
        self.operation_write_adapter_scale = float(operation_write_adapter_scale)
        self.operation_write_adapter_mode = operation_write_adapter_mode
        self.operation_output_adapter_rank = int(operation_output_adapter_rank)
        self.operation_output_adapter_scale = float(operation_output_adapter_scale)
        self.operation_circuit_bank = bool(operation_circuit_bank)
        self.operation_router_keys = bool(operation_router_keys)
        self.operation_transition_rank = int(operation_transition_rank)
        self.operation_transition_scale = float(operation_transition_scale)
        self.operation_bilinear_transition_rank = int(
            operation_bilinear_transition_rank
        )
        self.operation_bilinear_transition_scale = float(
            operation_bilinear_transition_scale
        )
        self.structured_scalar_state = bool(structured_scalar_state)
        self.structured_scalar_scale = float(structured_scalar_scale)
        self.structured_scalar_read_scale = float(structured_scalar_read_scale)
        self.structured_scalar_authoritative = bool(structured_scalar_authoritative)
        self.algebraic_state_mode = algebraic_state_mode
        self.algebraic_state_scale = float(algebraic_state_scale)
        self.algebraic_output_bridge_scale = float(algebraic_output_bridge_scale)
        self.algebraic_state_write_scale = float(algebraic_state_write_scale)
        self.algebraic_state_authoritative_read = bool(
            algebraic_state_authoritative_read
        )
        self.algebraic_output_decoder_enabled = bool(algebraic_output_decoder)
        self.algebraic_integer_output_decoder_enabled = bool(
            algebraic_integer_output_decoder
        )
        self.algebraic_integer_digit_dim = int(algebraic_integer_digit_dim)
        self.algebraic_integer_output_decoder_mode = (
            algebraic_integer_output_decoder_mode
        )
        self.algebraic_integer_output_head_enabled = bool(
            algebraic_integer_output_head
        )
        self.algebraic_integer_output_factor_rank = int(
            algebraic_integer_output_factor_rank
        )
        self.algebraic_integer_output_digit_interaction_rank = int(
            algebraic_integer_output_digit_interaction_rank
        )
        self.algebraic_integer_state_read_scale = float(
            algebraic_integer_state_read_scale
        )
        self.algebraic_state_value_scale = float(algebraic_state_value_scale)
        self.algebraic_state_fourier_base = int(algebraic_state_fourier_base)
        self.operator_valued_product_encoder = bool(operator_valued_product_encoder)
        self.operator_valued_packet_width = int(operator_valued_packet_width)
        self.operator_valued_basis_count = int(operator_valued_basis_count)
        self.numeric_state_dim = int(numeric_state_dim)
        self.numeric_state_scale = float(numeric_state_scale)
        self.numeric_state_value_scale = float(numeric_state_value_scale)
        self.typed_digit_state = bool(typed_digit_state)
        self.typed_digit_dim = int(typed_digit_dim)
        self.typed_digit_base = int(typed_digit_base)
        self.typed_digit_count = int(typed_digit_count)
        self.typed_digit_scale = float(typed_digit_scale)
        self.typed_digit_value_offset = int(typed_digit_value_offset)
        self.typed_digit_operand_offset = int(typed_digit_operand_offset)
        self.typed_digit_carry_chain = bool(typed_digit_carry_chain)
        self.typed_digit_multiply_convolution = bool(
            typed_digit_multiply_convolution
        )
        self.modular_prior_enabled = bool(modular_prior)
        self.modular_prior_mode = modular_prior_mode
        self.modular_template_init = modular_template_init
        self.circuit_residual_scale = float(circuit_residual_scale)
        self.circuit_input_norm_enabled = bool(circuit_input_norm)
        self.output_mode = output_mode
        self.output_temperature = float(output_temperature)
        self.output_scalar_bias = float(output_scalar_bias)
        self.output_digit_base = int(output_digit_base)
        self.output_factor_rank = int(output_factor_rank)
        self.output_digit_count = int(output_digit_count)
        self.output_digit_interaction_rank = int(output_digit_interaction_rank)
        self.output_digit_context_mode = output_digit_context_mode
        self.output_digit_geometry = bool(output_digit_geometry)
        self.output_digit_temperature = float(output_digit_temperature)
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
        product_transform: nn.Module = (
            OperatorValuedLinear(
                state_dim,
                state_dim,
                packet_width=self.operator_valued_packet_width,
                basis_count=self.operator_valued_basis_count,
            )
            if self.operator_valued_product_encoder else nn.Linear(state_dim, state_dim)
        )
        self.product_encoder = nn.Sequential(
            nn.LayerNorm(state_dim), product_transform, nn.GELU()
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
        if self.typed_digit_state:
            self.typed_digit_state_dim = self.typed_digit_dim * self.typed_digit_count
            self.typed_digit_sizes = [
                num_classes // (
                    self.typed_digit_base ** (self.typed_digit_count - 1)
                )
            ] + [self.typed_digit_base] * (self.typed_digit_count - 1)
            self.typed_digit_embeddings = nn.ModuleList(
                nn.Embedding(size, self.typed_digit_dim)
                for size in self.typed_digit_sizes
            )
            operation_embedding_dim = (
                self.typed_digit_dim
                if self.typed_digit_carry_chain
                else self.typed_digit_state_dim
            )
            self.typed_digit_operation_embedding = nn.Embedding(
                3, operation_embedding_dim
            )
            if self.typed_digit_carry_chain:
                # Process the least-significant slot first, then pass a
                # learned carry packet toward more-significant slots.  This
                # is an inductive bias for composition, not an exact
                # arithmetic oracle.
                self.typed_digit_carry_position = nn.Embedding(
                    self.typed_digit_count, self.typed_digit_dim
                )
                carry_transition_dim = 4 * self.typed_digit_dim
                self.typed_digit_carry_transition = nn.Sequential(
                    nn.LayerNorm(carry_transition_dim),
                    nn.Linear(carry_transition_dim, 2 * self.typed_digit_dim),
                    nn.GELU(),
                    nn.Linear(2 * self.typed_digit_dim, 2 * self.typed_digit_dim),
                    nn.Tanh(),
                )
                if self.typed_digit_multiply_convolution:
                    multiply_transition_dim = 5 * self.typed_digit_dim
                    self.typed_digit_multiply_transition = nn.Sequential(
                        nn.LayerNorm(multiply_transition_dim),
                        nn.Linear(multiply_transition_dim, 2 * self.typed_digit_dim),
                        nn.GELU(),
                        nn.Linear(2 * self.typed_digit_dim, 2 * self.typed_digit_dim),
                        nn.Tanh(),
                    )
                else:
                    self.typed_digit_multiply_transition = None
                self.typed_digit_transition = None
            else:
                typed_transition_dim = 3 * self.typed_digit_state_dim
                self.typed_digit_transition = nn.Sequential(
                    nn.LayerNorm(typed_transition_dim),
                    nn.Linear(typed_transition_dim, 2 * self.typed_digit_state_dim),
                    nn.GELU(),
                    nn.Linear(2 * self.typed_digit_state_dim, self.typed_digit_state_dim),
                    nn.Tanh(),
                )
                self.typed_digit_carry_position = None
                self.typed_digit_carry_transition = None
                self.typed_digit_multiply_transition = None
            self.typed_digit_projection = nn.Sequential(
                nn.LayerNorm(self.typed_digit_state_dim),
                nn.Linear(self.typed_digit_state_dim, state_dim),
                nn.Tanh(),
            )
            self.typed_digit_heads = nn.ModuleList(
                nn.Linear(self.typed_digit_dim, size)
                for size in self.typed_digit_sizes
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
        if self.operation_output_adapter_rank:
            self.operation_output_adapter_down = nn.Parameter(torch.empty(
                3, state_dim, self.operation_output_adapter_rank
            ))
            self.operation_output_adapter_up = nn.Parameter(torch.empty(
                3, self.operation_output_adapter_rank, state_dim
            ))
            self.operation_output_adapter_bias = nn.Parameter(torch.zeros(3, state_dim))
            nn.init.normal_(self.operation_output_adapter_down, std=0.02)
            nn.init.normal_(self.operation_output_adapter_up, std=0.02)
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
        if self.operation_bilinear_transition_rank:
            bilinear_rank = self.operation_bilinear_transition_rank
            self.operation_bilinear_acc_down = nn.Parameter(torch.empty(
                3, state_dim, bilinear_rank
            ))
            self.operation_bilinear_operand_down = nn.Parameter(torch.empty(
                3, state_dim, bilinear_rank
            ))
            self.operation_bilinear_up = nn.Parameter(torch.empty(
                3, bilinear_rank, state_dim
            ))
            self.operation_bilinear_bias = nn.Parameter(
                torch.zeros(3, state_dim)
            )
            nn.init.normal_(self.operation_bilinear_acc_down, std=0.02)
            nn.init.normal_(self.operation_bilinear_operand_down, std=0.02)
            nn.init.normal_(self.operation_bilinear_up, std=0.02)
        if self.structured_scalar_state:
            # A shared scalar value lane.  The four learned coefficients are
            # intentionally operation-specific but the state format is not:
            # every primitive receives (old, operand, product, bias).
            self.structured_scalar_transition = nn.Parameter(torch.empty(3, 4))
            with torch.no_grad():
                self.structured_scalar_transition.zero_()
                self.structured_scalar_transition[:, 0].fill_(1.0)
                self.structured_scalar_transition[:, 1:].normal_(std=0.02)
            self.structured_scalar_projection = nn.Sequential(
                nn.Linear(1, state_dim), nn.GELU()
            )
        if self.algebraic_state_mode in {"polynomial2", "polynomial2_fourier"}:
            # Keep a compact, reusable algebraic packet alongside the learned
            # state.  The transition is exact for ordinary integer
            # add/subtract/multiply; only its dense projection is learned.
            algebraic_input_dim = 2
            if self.algebraic_state_mode == "polynomial2_fourier":
                algebraic_input_dim += 3 * 2 * 7
            self.algebraic_state_projection = nn.Sequential(
                nn.Linear(algebraic_input_dim, state_dim), nn.Tanh()
            )
            if self.algebraic_output_decoder_enabled:
                # Separate output codec: do not force the recurrent learned
                # state to serve as the value-to-digit representation.
                self.algebraic_output_decoder = nn.Sequential(
                    nn.LayerNorm(algebraic_input_dim),
                    nn.Linear(algebraic_input_dim, state_dim),
                    nn.GELU(),
                )
            if self.algebraic_integer_output_decoder_enabled:
                self.algebraic_integer_digit_embeddings = nn.ModuleList(
                    nn.Embedding(self.output_digit_base, self.algebraic_integer_digit_dim)
                    for _ in range(self.output_digit_count)
                )
                integer_feature_dim = (
                    self.output_digit_count * self.algebraic_integer_digit_dim + 1
                )
                self.algebraic_integer_output_decoder = nn.Sequential(
                    nn.LayerNorm(integer_feature_dim),
                    nn.Linear(integer_feature_dim, state_dim),
                    nn.GELU(),
                )
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
                    num_circuits, state_dim, circuit_rank, factor_count, factor_mix_mode,
                    ordered_factor_slots, query_factor_mix_scale,
                    factor_pair_rank, factor_pair_scale
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
        elif output_mode == "factorized_digits":
            output_head = (
                GeometricFactorizedDigitOutput(
                    state_dim,
                    num_classes,
                    output_digit_base,
                    output_factor_rank,
                    output_digit_count,
                    output_digit_temperature,
                )
                if output_digit_geometry else
                FactorizedDigitOutput(
                    state_dim,
                    num_classes,
                    output_digit_base,
                    output_factor_rank,
                    output_digit_count,
                    output_digit_interaction_rank,
                    output_digit_context_mode,
                )
            )
            self.output = nn.Sequential(
                nn.LayerNorm(state_dim),
                output_head,
            )
        else:
            self.output = nn.Sequential(
                nn.LayerNorm(state_dim), nn.Linear(state_dim, num_classes)
            )
        if self.algebraic_integer_output_head_enabled:
            self.algebraic_integer_output_head = FactorizedDigitOutput(
                state_dim,
                num_classes,
                output_digit_base,
                self.algebraic_integer_output_factor_rank,
                output_digit_count,
                self.algebraic_integer_output_digit_interaction_rank,
                output_digit_context_mode,
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
            value_modulus=self.value_encoder_modulus,
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

    def _operation_bilinear_transition(
        self,
        state: torch.Tensor,
        operand: torch.Tensor,
        operation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a learned operation-specific low-rank state/operand product."""
        acc_projection = torch.einsum(
            "bd,bdr->br", state,
            self.operation_bilinear_acc_down[operation_ids],
        )
        operand_projection = torch.einsum(
            "bd,bdr->br", operand,
            self.operation_bilinear_operand_down[operation_ids],
        )
        fused = acc_projection * operand_projection
        adapted = torch.einsum(
            "br,brd->bd", fused, self.operation_bilinear_up[operation_ids]
        ) + self.operation_bilinear_bias[operation_ids]
        return nn.functional.gelu(adapted)

    def _typed_digit_indices(self, values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Convert non-negative shifted values into the typed digit slots."""
        base = self.typed_digit_base
        digits = [
            (values // (base ** (self.typed_digit_count - 1))).clamp(
                0, self.typed_digit_sizes[0] - 1
            )
        ]
        for index in range(1, self.typed_digit_count):
            power = base ** (self.typed_digit_count - 1 - index)
            digits.append((values // power).remainder(base))
        return tuple(digits)

    def _typed_digit_state_from_values(
        self, values: torch.Tensor, offset: int
    ) -> torch.Tensor:
        shifted = (values + int(offset)).clamp_min(0).to(dtype=torch.long)
        digits = self._typed_digit_indices(shifted)
        return torch.cat(
            [embedding(digit) for embedding, digit in zip(
                self.typed_digit_embeddings, digits
            )],
            dim=-1,
        )

    def _typed_digit_transition(
        self,
        state: torch.Tensor,
        operand: torch.Tensor,
        operation_ids: torch.Tensor,
    ) -> torch.Tensor:
        operation = self.typed_digit_operation_embedding(operation_ids)
        if not self.typed_digit_carry_chain:
            update = self.typed_digit_transition(
                torch.cat((state, operand, operation), dim=-1)
            )
            return state + self.typed_digit_scale * update
        state_slots = state.reshape(
            *state.shape[:-1], self.typed_digit_count, self.typed_digit_dim
        )
        operand_slots = operand.reshape(
            *operand.shape[:-1], self.typed_digit_count, self.typed_digit_dim
        )
        multiply_mask = operation_ids.eq(2)
        multiply_products = None
        if (
            self.typed_digit_multiply_transition is not None
            and multiply_mask.any()
        ):
            # Multiplication is a cross-digit convolution.  In
            # least-significant-first order, output position k depends on all
            # pairs (i, k-i), whereas the ordinary carry chain only sees
            # matching slots.
            state_lsd = state_slots.flip(-2)
            operand_lsd = operand_slots.flip(-2)
            product_lsd = torch.zeros_like(state_lsd)
            for output_index in range(self.typed_digit_count):
                product_lsd[..., output_index, :] = torch.stack(
                    [
                        state_lsd[..., left_index, :]
                        * operand_lsd[..., output_index - left_index, :]
                        for left_index in range(output_index + 1)
                    ],
                    dim=-2,
                ).sum(dim=-2)
            multiply_products = product_lsd.flip(-2)
        carry = operation
        updated_slots = [None] * self.typed_digit_count
        for index in reversed(range(self.typed_digit_count)):
            position = self.typed_digit_carry_position.weight[index]
            position = position.reshape(
                *((1,) * (state_slots.ndim - 2)), self.typed_digit_dim
            ).expand_as(state_slots[..., index, :])
            transition_input = torch.cat(
                (state_slots[..., index, :], operand_slots[..., index, :], carry, position),
                dim=-1,
            )
            transition = self.typed_digit_carry_transition(transition_input)
            if multiply_products is not None:
                multiply_transition = self.typed_digit_multiply_transition(
                    torch.cat((transition_input, multiply_products[..., index, :]), dim=-1)
                )
                transition = torch.where(
                    multiply_mask.unsqueeze(-1),
                    multiply_transition,
                    transition,
                )
            slot_update, carry_update = transition.chunk(2, dim=-1)
            updated_slot = state_slots[..., index, :] + self.typed_digit_scale * slot_update
            carry = torch.tanh(carry + self.typed_digit_scale * carry_update)
            updated_slots[index] = updated_slot
        return torch.stack(updated_slots, dim=-2).reshape(
            *state.shape[:-1], self.typed_digit_state_dim
        )

    def _typed_digit_logits(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        slots = state.reshape(
            *state.shape[:-1], self.typed_digit_count, self.typed_digit_dim
        )
        return tuple(
            head(slots[..., index, :])
            for index, head in enumerate(self.typed_digit_heads)
        )

    def _operation_output_adapter(
        self, state: torch.Tensor, operation_ids: torch.Tensor
    ) -> torch.Tensor:
        down = torch.einsum(
            "bd,bdr->br", state,
            self.operation_output_adapter_down[operation_ids]
        )
        adapted = torch.einsum(
            "br,brd->bd", down,
            self.operation_output_adapter_up[operation_ids]
        ) + self.operation_output_adapter_bias[operation_ids]
        return nn.functional.gelu(adapted)

    def _algebraic_state_update(
        self,
        state: torch.Tensor,
        operand: torch.Tensor,
        operation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the exact degree-2 polynomial coordinate transition.

        The packet stores normalized ``x`` and ``x**2``.  This is deliberately
        a small fixed semantic primitive, not a learned arithmetic oracle:
        the model still has to learn how to project and use the packet.
        """
        value_scale = self.algebraic_state_value_scale
        square_scale = value_scale * value_scale
        value = state[:, 0] * value_scale
        square = state[:, 1] * square_scale
        operand = operand.to(value.dtype)
        add_value = value + operand
        subtract_value = value - operand
        multiply_value = value * operand
        add = torch.stack((add_value, add_value.square()), dim=-1)
        subtract = torch.stack((subtract_value, subtract_value.square()), dim=-1)
        multiply = torch.stack((multiply_value, multiply_value.square()), dim=-1)
        updated = torch.where(
            operation_ids.unsqueeze(-1).eq(0), add,
            torch.where(operation_ids.unsqueeze(-1).eq(1), subtract, multiply),
        )
        return torch.stack(
            (updated[:, 0] / value_scale, updated[:, 1] / square_scale), dim=-1
        )

    def _algebraic_state_features(self, state: torch.Tensor) -> torch.Tensor:
        """Encode the algebraic packet for the learned query/output bridge."""
        if self.algebraic_state_mode != "polynomial2_fourier":
            return state
        value = state[:, 0] * self.algebraic_state_value_scale
        features = [state]
        for period in (
            float(self.algebraic_state_fourier_base),
            float(self.algebraic_state_fourier_base ** 2),
            self.algebraic_state_value_scale,
        ):
            angle = value.unsqueeze(-1) * (2.0 * math.pi / period)
            for harmonic in (1, 2, 4, 8, 16, 32, 64):
                features.append(torch.sin(angle * harmonic))
                features.append(torch.cos(angle * harmonic))
        return torch.cat(features, dim=-1)

    def _algebraic_integer_output_features(
        self, values: torch.Tensor
    ) -> torch.Tensor:
        """Encode an exact integer value as learned base-digit features.

        The recurrent algebraic packet remains float for the normal query
        path.  This diagnostic output path keeps a separate int64 register so
        large products do not lose low-order bits before digit decoding.
        """
        magnitude = values.abs()
        digits = []
        for _ in range(self.output_digit_count):
            digits.append(torch.remainder(magnitude, self.output_digit_base))
            magnitude = torch.div(
                magnitude, self.output_digit_base, rounding_mode="floor"
            )
        embedded = [embedding(digit) for embedding, digit in zip(
            self.algebraic_integer_digit_embeddings, digits
        )]
        sign = values.lt(0).to(embedded[0].dtype).unsqueeze(-1)
        return torch.cat((*embedded, sign), dim=-1)

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
        collect_state_stats: bool = False,
        return_full_logits: bool = True,
        teacher_stage_targets: torch.Tensor | None = None,
        teacher_forcing_probability: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del adaptive
        batch_size = inputs.shape[0]
        device = inputs.device
        if not 0.0 <= teacher_forcing_probability <= 1.0:
            raise ValueError("teacher_forcing_probability must be in [0, 1]")
        if teacher_stage_targets is not None:
            if not self.typed_digit_state:
                raise ValueError(
                    "teacher_stage_targets require typed_digit_state"
                )
            if tuple(teacher_stage_targets.shape) != (
                batch_size, self.max_ops
            ):
                raise ValueError(
                    "teacher_stage_targets must have shape [batch, max_ops]"
                )
        encoded = self.encode_program(inputs)
        operation_tokens = inputs[:, 1:1 + self.max_ops]
        operation_ids = (operation_tokens - 2).clamp(0, 2)
        operation_mask = operation_tokens.ge(2)
        operands = encoded[:, self.value_start:self.value_start + self.max_ops + 1]
        operand_states = self.operand_encoder(operands)
        operand_mask = inputs[:, self.value_start:self.value_start + self.max_ops + 1].ne(0)
        if self.structured_scalar_state:
            scalar_operands = (
                inputs[:, self.value_start:self.value_start + self.max_ops + 1]
                - VALUE_TOKEN_OFFSET
            ).to(operand_states.dtype)
            scalar_state = scalar_operands[:, 0]
        if self.algebraic_state_mode in {"polynomial2", "polynomial2_fourier"}:
            algebraic_operands = (
                inputs[:, self.value_start:self.value_start + self.max_ops + 1]
                - VALUE_TOKEN_OFFSET
            ).to(operand_states.dtype)
            value_scale = self.algebraic_state_value_scale
            algebraic_state = torch.stack((
                algebraic_operands[:, 0] / value_scale,
                algebraic_operands[:, 0].square() / (value_scale * value_scale),
            ), dim=-1)
        if self.algebraic_integer_output_decoder_enabled:
            algebraic_integer_operands = (
                inputs[:, self.value_start:self.value_start + self.max_ops + 1]
                - VALUE_TOKEN_OFFSET
            ).to(torch.long)
            algebraic_integer_state = algebraic_integer_operands[:, 0]
        if self.numeric_state_dim:
            numeric_values = (
                inputs[:, self.value_start:self.value_start + self.max_ops + 1]
                - VALUE_TOKEN_OFFSET
            ).clamp_min(0).to(operand_states.dtype)
            if self.modulus is not None:
                numeric_values = numeric_values.clamp_max(self.modulus - 1)
                numeric_value_scale = float(max(self.modulus - 1, 1))
            else:
                numeric_value_scale = self.numeric_state_value_scale
            numeric_operands = self.numeric_value_encoder(
                numeric_values.unsqueeze(-1) / numeric_value_scale
            )
            numeric_state = numeric_operands[:, 0]
        if self.typed_digit_state:
            typed_values = (
                inputs[:, self.value_start:self.value_start + self.max_ops + 1]
                - VALUE_TOKEN_OFFSET
            )
            typed_state = self._typed_digit_state_from_values(
                typed_values[:, 0], self.typed_digit_value_offset
            )
            typed_operands = self._typed_digit_state_from_values(
                typed_values, self.typed_digit_operand_offset
            )
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
        digit_steps = []
        scalar_step_states = []
        pre_state_steps = []
        query_state_steps = []
        post_state_steps = []
        step_state_steps = []
        typed_digit_steps = []
        algebraic_state_steps = []
        # Padded steps keep the last real operation so the terminal readout
        # remains operation-conditioned for shorter programs.
        last_operation_ids = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )

        for step in range(self.max_ops):
            if collect_state_stats:
                pre_state_steps.append(accumulator.clone())
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
            query_snapshot = torch.zeros_like(accumulator) if collect_state_stats else None
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
                if self.algebraic_state_authoritative_read:
                    # The exact packet is already maintained by the forward
                    # pass.  This opt-in path changes only which representation
                    # the pair/router reads; the learned accumulator still
                    # receives the normal sparse writer update below.
                    read_accumulator = self.algebraic_state_projection(
                        self._algebraic_state_features(
                            algebraic_state[active_indices]
                        )
                    )
                elif (
                    self.structured_scalar_state
                    and self.structured_scalar_authoritative
                ):
                    read_accumulator = self.structured_scalar_projection(
                        scalar_state[active_indices].unsqueeze(-1)
                    )
                elif self.structured_scalar_state and self.structured_scalar_read_scale:
                    scalar_read = self.structured_scalar_projection(
                        scalar_state[active_indices].unsqueeze(-1)
                    )
                    read_accumulator = read_accumulator + (
                        self.structured_scalar_read_scale * scalar_read
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
                if self.typed_digit_state:
                    typed_candidate = self._typed_digit_transition(
                        typed_state[active_indices],
                        typed_operands[active_indices, step + 1],
                        current_operation_ids,
                    )
                    query = query + self.typed_digit_scale * (
                        self.typed_digit_projection(typed_candidate)
                    )
                if self.structured_scalar_state:
                    scalar_operand = scalar_operands[active_indices, step + 1]
                    scalar_features = torch.stack((
                        scalar_state[active_indices],
                        scalar_operand,
                        scalar_state[active_indices] * scalar_operand,
                        torch.ones_like(scalar_operand),
                    ), dim=-1)
                    scalar_candidate = (
                        scalar_features
                        * self.structured_scalar_transition[current_operation_ids]
                    ).sum(dim=-1)
                    query = query + self.structured_scalar_scale * (
                        self.structured_scalar_projection(
                            scalar_candidate.unsqueeze(-1)
                        )
                    )
                if self.algebraic_state_mode in {"polynomial2", "polynomial2_fourier"}:
                    query = query + self.algebraic_state_scale * (
                        self.algebraic_state_projection(
                            self._algebraic_state_features(algebraic_state[active_indices])
                        )
                    )
                if self.algebraic_integer_state_read_scale:
                    query = query + self.algebraic_integer_state_read_scale * (
                        self.algebraic_integer_output_decoder(
                            self._algebraic_integer_output_features(
                                algebraic_integer_state[active_indices]
                            )
                        )
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
                if collect_state_stats:
                    query_snapshot[active_indices] = query
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
                if (
                    self.operation_write_adapter_rank
                    and self.operation_write_adapter_mode == "pre_writer"
                ):
                    write_input = write_input + self.operation_write_adapter_scale * (
                        self._operation_write_adapter(
                            write_input, current_operation_ids
                        )
                    )
                if self.operation_transition_rank:
                    write_input = write_input + self.operation_transition_scale * (
                        self._operation_transition(
                            write_input, current_operation_ids
                        )
                    )
                if self.operation_bilinear_transition_rank:
                    write_input = write_input + (
                        self.operation_bilinear_transition_scale
                        * self._operation_bilinear_transition(
                            active_accumulator,
                            active_operand,
                            current_operation_ids,
                        )
                    )
                if self.algebraic_state_write_scale:
                    # Reuse the compact semantic packet at the write boundary.
                    # Query-side exposure alone can be erased by the sparse
                    # circuit correction and recurrent writer; this optional
                    # bridge tests whether repeated state writes need the same
                    # persistent value signal.
                    write_input = write_input + self.algebraic_state_write_scale * (
                        self.algebraic_state_projection(
                            self._algebraic_state_features(
                                algebraic_state[active_indices]
                            )
                        )
                    )
                candidate = self._write_state(active_accumulator, write_input)
                if self.state_update_mode == "residual":
                    candidate = active_accumulator + self.state_residual_scale * candidate
                if self.operation_write_adapter_rank:
                    if self.operation_write_adapter_mode == "post_state":
                        candidate = candidate + self.operation_write_adapter_scale * (
                            self._operation_write_adapter(
                                candidate, current_operation_ids
                            )
                        )
                    elif self.operation_write_adapter_mode == "terminal_only":
                        terminal_mask = ~operation_mask[active_indices, step + 1:].any(dim=1)
                        if terminal_mask.any():
                            terminal_candidate = candidate[terminal_mask]
                            candidate = candidate.clone()
                            candidate[terminal_mask] = terminal_candidate + (
                                self.operation_write_adapter_scale
                                * self._operation_write_adapter(
                                    terminal_candidate,
                                    current_operation_ids[terminal_mask],
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
                if self.typed_digit_state:
                    next_typed_state = typed_state.clone()
                    next_typed_state[active_indices] = typed_candidate
                    typed_state = next_typed_state
                    if (
                        teacher_stage_targets is not None
                        and teacher_forcing_probability > 0.0
                    ):
                        teacher_values = teacher_stage_targets[
                            active_indices, step
                        ].to(dtype=torch.long)
                        teacher_state = self._typed_digit_state_from_values(
                            teacher_values, self.typed_digit_value_offset
                        )
                        if teacher_forcing_probability >= 1.0:
                            typed_state[active_indices] = teacher_state
                        else:
                            use_teacher = torch.rand(
                                teacher_state.shape[0],
                                device=device,
                            ).lt(teacher_forcing_probability)
                            typed_state[active_indices] = torch.where(
                                use_teacher.unsqueeze(-1),
                                teacher_state,
                                typed_state[active_indices],
                            )
                if self.structured_scalar_state:
                    next_scalar_state = scalar_state.clone()
                    next_scalar_state[active_indices] = scalar_candidate
                    scalar_state = next_scalar_state
                if self.algebraic_state_mode in {"polynomial2", "polynomial2_fourier"}:
                    next_algebraic_state = algebraic_state.clone()
                    next_algebraic_state[active_indices] = self._algebraic_state_update(
                        algebraic_state[active_indices],
                        algebraic_operands[active_indices, step + 1],
                        current_operation_ids,
                    )
                    algebraic_state = next_algebraic_state
                if self.algebraic_integer_output_decoder_enabled:
                    integer_operand = algebraic_integer_operands[
                        active_indices, step + 1
                    ]
                    integer_state = algebraic_integer_state[active_indices]
                    integer_updated = torch.where(
                        current_operation_ids.eq(0),
                        integer_state + integer_operand,
                        torch.where(
                            current_operation_ids.eq(1),
                            integer_state - integer_operand,
                            integer_state * integer_operand,
                        ),
                    )
                    next_algebraic_integer_state = algebraic_integer_state.clone()
                    next_algebraic_integer_state[active_indices] = integer_updated
                    algebraic_integer_state = next_algebraic_integer_state
                next_last_operation_ids = last_operation_ids.clone()
                next_last_operation_ids[active_indices] = current_operation_ids
                last_operation_ids = next_last_operation_ids
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
            if collect_state_stats:
                query_state_steps.append(query_snapshot)
                post_state_steps.append(accumulator.clone())
            selected_steps.append(selected_step)
            macro_selected_steps.append(macro_selected_step)
            step_state = accumulator
            if self.numeric_state_dim:
                step_state = step_state + self.numeric_state_scale * self.numeric_state_projection(
                    numeric_state
                )
            if self.typed_digit_state:
                step_state = step_state + self.typed_digit_scale * (
                    self.typed_digit_projection(typed_state)
                )
            if self.structured_scalar_state:
                scalar_projection = self.structured_scalar_projection(
                    scalar_state.unsqueeze(-1)
                )
                if self.structured_scalar_authoritative:
                    step_state = scalar_projection
                else:
                    step_state = step_state + self.structured_scalar_scale * (
                        scalar_projection
                    )
            if self.algebraic_state_mode in {"polynomial2", "polynomial2_fourier"}:
                step_state = step_state + self.algebraic_state_scale * (
                    self.algebraic_state_projection(
                        self._algebraic_state_features(algebraic_state)
                    )
                )
            if self.modular_prior_enabled:
                if self.modular_prior_mode == "fixed":
                    step_features = nn.functional.one_hot(
                        modular_accumulator, self.modulus
                    ).to(accumulator.dtype)
                else:
                    step_features = modular_state
                step_state = step_state + self.modular_projection(step_features)
            if self.operation_output_adapter_rank:
                step_state = step_state + self.operation_output_adapter_scale * (
                    self._operation_output_adapter(step_state, last_operation_ids)
                )
            if collect_state_stats:
                step_state_steps.append(step_state.clone())
                if self.algebraic_state_mode in {"polynomial2", "polynomial2_fourier"}:
                    algebraic_state_steps.append(algebraic_state.clone())
            if self.output_mode == "factorized_digits":
                digits = None
                if self.algebraic_integer_output_decoder_enabled:
                    integer_output_state = self.algebraic_integer_output_decoder(
                        self._algebraic_integer_output_features(
                            algebraic_integer_state
                        )
                    )
                    if self.algebraic_integer_output_decoder_mode == "multiply_only":
                        learned_output_state = self.output[0](step_state)
                        multiply_mask = last_operation_ids.eq(2).unsqueeze(-1)
                        if self.algebraic_integer_output_head_enabled:
                            learned_digits = self.output[1].digit_logits(
                                learned_output_state
                            )
                            integer_digits = self.algebraic_integer_output_head.digit_logits(
                                integer_output_state
                            )
                            digits = tuple(
                                torch.where(multiply_mask, integer, learned)
                                for integer, learned in zip(
                                    integer_digits, learned_digits
                                )
                            )
                        else:
                            output_state = torch.where(
                                multiply_mask,
                                integer_output_state,
                                learned_output_state,
                            )
                    else:
                        if self.algebraic_integer_output_head_enabled:
                            digits = self.algebraic_integer_output_head.digit_logits(
                                integer_output_state
                            )
                        else:
                            output_state = integer_output_state
                elif self.algebraic_output_decoder_enabled:
                    output_state = self.algebraic_output_decoder(
                        self._algebraic_state_features(algebraic_state)
                    )
                else:
                    output_state = self.output[0](step_state)
                    if self.algebraic_output_bridge_scale:
                        output_state = output_state + (
                            self.algebraic_output_bridge_scale
                            * self.algebraic_state_projection(
                                self._algebraic_state_features(algebraic_state)
                            )
                        )
                if digits is None:
                    digits = self.output[1].digit_logits(output_state)
                if return_full_logits:
                    step_logits.append(self.output[1].combine(*digits))
                digit_steps.append(digits)
            else:
                step_logits.append(self.output(step_state))
            if self.typed_digit_state:
                typed_digit_steps.append(self._typed_digit_logits(typed_state))
            if self.structured_scalar_state:
                scalar_step_states.append(scalar_state)

        full_step_logits = (
            torch.stack(step_logits, dim=1) if return_full_logits else None
        )
        stats = {
            "active_circuits": torch.tensor(self.router.active_circuits, device=device),
            "internal_steps": torch.tensor(self.max_ops, device=device),
            "router_entropy": step_entropies.sum() / executed_mask.sum().clamp_min(1),
            "selected_ids": torch.stack(selected_steps, dim=1),
            "selected_weights": selected_weights,
            "macro_selected_ids": torch.stack(macro_selected_steps, dim=1),
            "macro_selected_weights": macro_selected_weights,
            "step_logits": full_step_logits,
            "executed_steps": executed_mask.sum(dim=1),
            "executed_mask": executed_mask,
            "macro_router_entropy": macro_step_entropies.sum()
            / executed_mask.sum().clamp_min(1),
        }
        if self.structured_scalar_state:
            stats["structured_scalar_states"] = torch.stack(
                scalar_step_states, dim=1
            )
        if self.output_mode == "factorized_digits":
            digit_logits = tuple(
                torch.stack([step[index] for step in digit_steps], dim=1)
                for index in range(self.output_digit_count)
            )
            stats["digit_logits"] = digit_logits
            if self.output_digit_count == 2:
                stats["digit_high_logits"] = digit_logits[0]
                stats["digit_low_logits"] = digit_logits[1]
        if self.typed_digit_state:
            stats["typed_digit_logits"] = tuple(
                torch.stack([step[index] for step in typed_digit_steps], dim=1)
                for index in range(self.typed_digit_count)
            )
        if collect_state_stats:
            stats["pre_accumulator_states"] = torch.stack(pre_state_steps, dim=1)
            stats["query_states"] = torch.stack(query_state_steps, dim=1)
            stats["post_accumulator_states"] = torch.stack(post_state_steps, dim=1)
            stats["step_states"] = torch.stack(step_state_steps, dim=1)
            if self.algebraic_state_mode in {"polynomial2", "polynomial2_fourier"}:
                stats["algebraic_state_features"] = torch.stack(
                    algebraic_state_steps, dim=1
                )
        self._last_route = stats
        if return_full_logits:
            final_logits = stats["step_logits"][:, -1]
        else:
            # Training with factorized digits only consumes the compact digit
            # logits; avoid materializing a potentially enormous Cartesian
            # class matrix. Evaluation keeps the default full-logit path.
            final_logits = digit_steps[-1][0]
        return final_logits, stats

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
        if self.typed_digit_state:
            shared += (
                count_parameters(self.typed_digit_embeddings)
                + count_parameters(self.typed_digit_operation_embedding)
                + (
                    count_parameters(self.typed_digit_carry_position)
                    + count_parameters(self.typed_digit_carry_transition)
                    + (
                        count_parameters(self.typed_digit_multiply_transition)
                        if self.typed_digit_multiply_transition is not None
                        else 0
                    )
                    if self.typed_digit_carry_chain
                    else count_parameters(self.typed_digit_transition)
                )
                + count_parameters(self.typed_digit_projection)
                + count_parameters(self.typed_digit_heads)
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
        if self.operation_output_adapter_rank:
            shared += (
                self.operation_output_adapter_down.numel()
                + self.operation_output_adapter_up.numel()
                + self.operation_output_adapter_bias.numel()
            )
        if self.operation_transition_rank:
            shared += (
                self.operation_transition_down.numel()
                + self.operation_transition_up.numel()
                + self.operation_transition_bias.numel()
            )
        if self.operation_bilinear_transition_rank:
            shared += (
                self.operation_bilinear_acc_down.numel()
                + self.operation_bilinear_operand_down.numel()
                + self.operation_bilinear_up.numel()
                + self.operation_bilinear_bias.numel()
            )
        if self.structured_scalar_state:
            shared += (
                self.structured_scalar_transition.numel()
                + count_parameters(self.structured_scalar_projection)
            )
        if self.algebraic_state_mode in {"polynomial2", "polynomial2_fourier"}:
            shared += count_parameters(self.algebraic_state_projection)
            if self.algebraic_output_decoder_enabled:
                shared += count_parameters(self.algebraic_output_decoder)
            if self.algebraic_integer_output_decoder_enabled:
                shared += count_parameters(self.algebraic_integer_digit_embeddings)
                shared += count_parameters(self.algebraic_integer_output_decoder)
                if self.algebraic_integer_output_head_enabled:
                    shared += count_parameters(self.algebraic_integer_output_head)
        if self.write_gate_enabled:
            shared += count_parameters(self.write_gate)
        if self.circuit_input_norm is not None:
            shared += count_parameters(self.circuit_input_norm)
        if self.modular_prior_enabled:
            shared += count_parameters(self.modular_projection)
            if self.modular_prior_mode == "templates":
                shared += self.modular_template_logits.numel()
        circuit_bank = self.circuits[0] if self.operation_circuit_bank else self.circuits
        pair_basis_active = 0
        if self.circuit_bank_mode == "factorized":
            if self.ordered_factor_slots:
                down_row = circuit_bank.down_factors[0, 0]
                up_row = circuit_bank.up_factors[0, 0]
                bias_row = circuit_bank.bias_factors[0, 0]
            else:
                down_row = circuit_bank.down_factors[0]
                up_row = circuit_bank.up_factors[0]
                bias_row = circuit_bank.bias_factors[0]
            factor_row = (
                down_row.numel() + up_row.numel() + bias_row.numel()
                + circuit_bank.factor_mix[0].numel()
            )
            if self.query_factor_mix_scale:
                gate_row = (
                    circuit_bank.factor_gate_keys[0, 0]
                    if self.ordered_factor_slots else circuit_bank.factor_gate_keys[0]
                )
                factor_row += gate_row.numel()
            if self.factor_pair_rank:
                factor_row += circuit_bank.pair_codes[0].numel() * 2
            bank_count = min(self.max_ops, 3) if self.operation_circuit_bank else 1
            active_circuit_params = factor_row * self.router.active_circuits * 2 * bank_count
            candidate_params = self.router.keys[0].numel() * self.router.factor_candidate_pool * 2
            if self.factor_pair_rank:
                pair_basis_active = (
                    circuit_bank.pair_down_basis.numel()
                    + circuit_bank.pair_up_basis.numel()
                    + circuit_bank.pair_bias_basis.numel()
                ) * bank_count
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
        active_total = shared + pair_basis_active + candidate_params + active_circuit_params
        return {
            "total_params": total,
            "active_params_estimate": active_total,
            "active_fraction": active_total / total,
            "active_circuit_params": active_circuit_params,
            "max_ops": self.max_ops,
            "modulus": self.modulus,
            "circuit_bank_mode": self.circuit_bank_mode,
            "routing_mode": "factorized" if self.circuit_bank_mode == "factorized" else "hierarchical",
            "input_reinjection_scale": self.input_reinjection_scale,
            "write_gate": self.write_gate_enabled,
            "value_encoder_mode": self.value_encoder_mode,
            "value_encoder_modulus": self.value_encoder_modulus,
            "factor_mix_mode": self.factor_mix_mode,
            "ordered_factor_slots": self.ordered_factor_slots,
            "query_factor_mix_scale": self.query_factor_mix_scale,
            "factor_pair_rank": self.factor_pair_rank,
            "factor_pair_scale": self.factor_pair_scale,
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
            "state_update_mode": self.state_update_mode,
            "state_residual_scale": self.state_residual_scale,
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
            "operation_write_adapter_mode": self.operation_write_adapter_mode,
            "operation_output_adapter_rank": self.operation_output_adapter_rank,
            "operation_output_adapter_scale": self.operation_output_adapter_scale,
            "operation_circuit_bank": self.operation_circuit_bank,
            "operation_router_keys": self.operation_router_keys,
            "operation_router_key_params": operation_router_key_params,
            "operation_router_key_active_estimate": operation_router_key_active,
            "operation_transition_rank": self.operation_transition_rank,
            "operation_transition_scale": self.operation_transition_scale,
            "operation_bilinear_transition_rank": (
                self.operation_bilinear_transition_rank
            ),
            "operation_bilinear_transition_scale": (
                self.operation_bilinear_transition_scale
            ),
            "structured_scalar_state": self.structured_scalar_state,
            "structured_scalar_scale": self.structured_scalar_scale,
            "structured_scalar_read_scale": self.structured_scalar_read_scale,
            "structured_scalar_authoritative": self.structured_scalar_authoritative,
            "algebraic_state_mode": self.algebraic_state_mode,
            "algebraic_state_scale": self.algebraic_state_scale,
            "algebraic_output_bridge_scale": self.algebraic_output_bridge_scale,
            "algebraic_state_write_scale": self.algebraic_state_write_scale,
            "algebraic_state_authoritative_read": self.algebraic_state_authoritative_read,
            "algebraic_output_decoder": self.algebraic_output_decoder_enabled,
            "algebraic_integer_output_decoder": self.algebraic_integer_output_decoder_enabled,
            "algebraic_integer_digit_dim": self.algebraic_integer_digit_dim,
            "algebraic_integer_output_decoder_mode": self.algebraic_integer_output_decoder_mode,
            "algebraic_integer_output_head": self.algebraic_integer_output_head_enabled,
            "algebraic_integer_output_factor_rank": self.algebraic_integer_output_factor_rank,
            "algebraic_integer_output_digit_interaction_rank": self.algebraic_integer_output_digit_interaction_rank,
            "algebraic_integer_state_read_scale": self.algebraic_integer_state_read_scale,
            "algebraic_state_value_scale": self.algebraic_state_value_scale,
            "algebraic_state_fourier_base": self.algebraic_state_fourier_base,
            "operator_valued_product_encoder": self.operator_valued_product_encoder,
            "operator_valued_packet_width": self.operator_valued_packet_width,
            "operator_valued_basis_count": self.operator_valued_basis_count,
            "operator_valued_product_scalar_dof": (
                count_parameters(self.product_encoder[1])
                if self.operator_valued_product_encoder else 0
            ),
            "numeric_state_dim": self.numeric_state_dim,
            "numeric_state_scale": self.numeric_state_scale,
            "numeric_state_value_scale": self.numeric_state_value_scale,
            "typed_digit_state": self.typed_digit_state,
            "typed_digit_dim": self.typed_digit_dim,
            "typed_digit_base": self.typed_digit_base,
            "typed_digit_count": self.typed_digit_count,
            "typed_digit_scale": self.typed_digit_scale,
            "typed_digit_value_offset": self.typed_digit_value_offset,
            "typed_digit_operand_offset": self.typed_digit_operand_offset,
            "typed_digit_carry_chain": self.typed_digit_carry_chain,
            "typed_digit_multiply_convolution": (
                self.typed_digit_multiply_convolution
            ),
            "modular_prior": self.modular_prior_enabled,
            "modular_prior_mode": self.modular_prior_mode,
            "modular_template_init": self.modular_template_init,
            "circuit_residual_scale": self.circuit_residual_scale,
            "circuit_input_norm": self.circuit_input_norm_enabled,
            "output_mode": self.output_mode,
            "output_temperature": self.output_temperature,
            "output_scalar_bias": self.output_scalar_bias,
            "output_digit_base": self.output_digit_base,
            "output_factor_rank": self.output_factor_rank,
            "output_digit_count": self.output_digit_count,
            "output_digit_interaction_rank": self.output_digit_interaction_rank,
            "output_digit_context_mode": self.output_digit_context_mode,
            "output_digit_geometry": self.output_digit_geometry,
            "output_digit_temperature": self.output_digit_temperature,
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
