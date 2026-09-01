from __future__ import annotations

import torch
from torch import nn

from .circuits import MicroCircuitBank
from .encoding import VALUE_HARMONICS, encode_tokens
from .instrumentation import count_parameters
from .router import HierarchicalRouter, StableFamilyRouter


class TypedRegisterNeuralEngine(nn.Module):
    """Sparse recurrent composition core with explicit typed registers.

    This model is intentionally narrow: it is a research control for the
    fixed-format composition benchmark.  It does not use attention.  The
    input contains two operation tokens followed by three numeric operands.
    The execution graph is explicit:

        (a, b, op1) -> partial -> (partial, c, op2) -> final -> readout

    Each stage has its own register write and conditions the hierarchical
    router with an operator embedding.  The circuit bank remains sparse and
    large, but the model no longer asks one pooled recurrent state to infer
    both dataflow and operation order at the same time.
    """

    def __init__(self, vocab_size: int = 128, num_classes: int = 64,
                 seq_len: int = 8, d_model: int = 384, state_dim: int = 384,
                 num_circuits: int = 1408, circuit_rank: int = 16,
                 router_branch: int = 8, router_depth: int = 4,
                 candidate_pool: int = 32, active_circuits: int = 8,
                 internal_steps: int = 3, router_addresses: int = 1,
                 slot_count: int = 6, numeric_value_encoding: bool = True,
                 route_exploration_prob: float = 0.0,
                 routing_capacity: int | None = None,
                 routing_depth: int | None = None,
                 circuit_mode: str = "serial",
                 readout_mode: str = "routed",
                 route_query_mode: str = "value_and_type",
                 route_context_dim: int = 32,
                 routing_mode: str = "global",
                 family_count: int = 9,
                 shared_fraction: float = 0.125,
                 typed_route_partitions: bool = False,
                 typed_route_shared: bool = False,
                 operator_partition_count: int = 4):
        super().__init__()
        if slot_count < 6:
            raise ValueError("TypedRegisterNeuralEngine requires six input slots")
        if internal_steps != 3:
            raise ValueError("TypedRegisterNeuralEngine currently requires internal_steps=3")
        if circuit_mode not in {"parallel", "serial"}:
            raise ValueError("circuit_mode must be 'parallel' or 'serial'")
        if readout_mode not in {"routed", "direct"}:
            raise ValueError("readout_mode must be 'routed' or 'direct'")
        if route_query_mode not in {"value_and_type", "typed", "compressed"}:
            raise ValueError("route_query_mode must be 'value_and_type', 'typed', or 'compressed'")
        if routing_mode not in {"global", "family_local"}:
            raise ValueError("routing_mode must be 'global' or 'family_local'")
        if routing_mode == "family_local" and (typed_route_partitions or typed_route_shared):
            raise ValueError("family_local routing cannot combine with typed route partitions")
        if route_context_dim < 1 or route_context_dim > state_dim:
            raise ValueError("route_context_dim must be between 1 and state_dim")
        if typed_route_partitions and (
                operator_partition_count < 4 or num_circuits % operator_partition_count != 0):
            raise ValueError("typed route partitions require at least four equal bank partitions")
        if not 0.0 <= route_exploration_prob <= 1.0:
            raise ValueError("route_exploration_prob must be between 0 and 1")

        self.state_dim = state_dim
        self.internal_steps = internal_steps
        self.slot_count = slot_count
        self.circuit_mode = circuit_mode
        self.readout_mode = readout_mode
        self.route_query_mode = route_query_mode
        self.route_context_dim = route_context_dim
        self.routing_mode = routing_mode
        self.family_count = family_count
        self.shared_fraction = shared_fraction
        self.typed_route_partitions = typed_route_partitions
        self.typed_route_shared = typed_route_shared
        self.operator_partition_count = operator_partition_count
        self.operator_partition_size = num_circuits // operator_partition_count
        self.numeric_value_encoding = numeric_value_encoding
        self.route_exploration_prob = route_exploration_prob
        # The register graph is deliberately fixed-depth; adaptive halting
        # would allow the model to skip a required write in this control.
        self.adaptive_halting = False
        self.adaptive_inference = False

        embedding_vocab = 16 if numeric_value_encoding else vocab_size
        self.token_embedding = nn.Embedding(embedding_vocab, d_model, padding_idx=0)
        self.value_encoder = (nn.Linear(1 + 2 * len(VALUE_HARMONICS), d_model)
                              if numeric_value_encoding else None)
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, d_model))
        self.position_scale = nn.Parameter(torch.zeros(seq_len, d_model))
        self.position_bias = nn.Parameter(torch.zeros(seq_len, d_model))

        self.operand_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, state_dim),
            nn.GELU(),
        )
        # Pair encoding keeps left/right operand order explicit.  A sum would
        # make the first primitive operation unnecessarily commutative.
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(2 * state_dim),
            nn.Linear(2 * state_dim, state_dim),
            nn.GELU(),
        )
        self.operation_embedding = nn.Embedding(3, state_dim)
        self.stage_embedding = nn.Parameter(torch.zeros(3, state_dim))
        self.route_value_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, route_context_dim),
            nn.GELU(),
            nn.Linear(route_context_dim, state_dim),
        )
        self.route_context_scale = nn.Parameter(torch.tensor(0.25))
        self.partial_writer = nn.Sequential(
            nn.LayerNorm(2 * state_dim),
            nn.Linear(2 * state_dim, state_dim),
            nn.Tanh(),
        )
        self.final_writer = nn.Sequential(
            nn.LayerNorm(2 * state_dim),
            nn.Linear(2 * state_dim, state_dim),
            nn.Tanh(),
        )
        self.readout_writer = nn.Sequential(
            nn.LayerNorm(2 * state_dim),
            nn.Linear(2 * state_dim, state_dim),
            nn.Tanh(),
        )

        if routing_mode == "family_local":
            self.router = StableFamilyRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, router_addresses,
                routing_capacity=routing_capacity, routing_depth=routing_depth,
                family_count=family_count, shared_fraction=shared_fraction,
            )
        else:
            self.router = HierarchicalRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, router_addresses,
                routing_capacity=routing_capacity, routing_depth=routing_depth,
            )
        self.circuits = MicroCircuitBank(num_circuits, state_dim, circuit_rank)
        self.output = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, num_classes))

        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.position_scale, std=0.01)
        nn.init.normal_(self.position_bias, std=0.01)
        nn.init.normal_(self.stage_embedding, std=0.02)
        nn.init.normal_(self.operation_embedding.weight, std=0.02)
        self._last_route: dict[str, torch.Tensor] = {}

    def encode_slots(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[1] < self.slot_count:
            raise ValueError("inputs are shorter than configured slot_count")
        tokens = encode_tokens(inputs, self.token_embedding, self.value_encoder)
        length = inputs.shape[1]
        positions = self.position_embedding[:length]
        scale = self.position_scale[:length]
        bias = self.position_bias[:length]
        tokens = tokens * (1.0 + scale) + positions + bias
        mask = inputs.ne(0).unsqueeze(-1)
        return tokens * mask

    def _apply_circuits(self, query: torch.Tensor, selected: torch.Tensor,
                        weights: torch.Tensor) -> torch.Tensor:
        if self.circuit_mode == "serial":
            return self.circuits.forward_serial(query, selected, weights)
        return self.circuits(query, selected, weights)

    def _route_stage(self, query: torch.Tensor, stage: int,
                     operator_ids: torch.Tensor,
                     selected_steps: list[torch.Tensor],
                     selected_weights: torch.Tensor,
                     route_gains: torch.Tensor,
                     step_entropies: torch.Tensor,
                     family_steps: list[torch.Tensor],
                     shared_fallback_fractions: list[torch.Tensor]) -> torch.Tensor:
        # Operator and stage are part of the router query, so equal primitive
        # operations can reuse a circuit family across different compositions.
        routed_query = query + self.stage_embedding[stage]
        if self.route_query_mode == "typed":
            # Keep numeric/value information in the circuit input, but make
            # routing decisions depend only on the role and primitive op.
            # This prevents the large bank from fragmenting into a different
            # route for every operand tuple.
            router_query = self.operation_embedding(operator_ids) + self.stage_embedding[stage]
        elif self.route_query_mode == "compressed":
            # Preserve a small value-dependent routing context without giving
            # the router the full high-dimensional numeric state.  The bank
            # circuit still receives the complete query below.
            compressed_value = self.route_value_encoder(query)
            router_query = (self.operation_embedding(operator_ids)
                            + self.stage_embedding[stage]
                            + self.route_context_scale * compressed_value)
        else:
            router_query = routed_query
        if self.typed_route_partitions:
            if self.typed_route_shared:
                if stage < 2:
                    # Partition 0 is shared; partitions 1..3 are private
                    # add/subtract/multiply banks.  Readout uses shared only.
                    private_ids = operator_ids + 1
                    private_offsets = private_ids * self.operator_partition_size
                    routing_windows = torch.stack([
                        torch.zeros_like(private_offsets), private_offsets
                    ], dim=1)
                else:
                    routing_windows = torch.zeros(
                        query.shape[0], 1, dtype=torch.long, device=query.device)
                routing_offset: int | torch.Tensor = 0
            else:
                partition_ids = (operator_ids if stage < 2 else
                                 torch.full_like(operator_ids, self.operator_partition_count - 1))
                routing_offset = partition_ids * self.operator_partition_size
                routing_windows = None
            local_capacity: int | None = self.operator_partition_size
        else:
            routing_offset = 0
            routing_windows = None
            local_capacity = None
        exploration_prob = (self.route_exploration_prob if self.training else 0.0)
        if self.routing_mode == "family_local":
            # The register graph supplies a stable semantic family.  For the
            # two primitive stages this is op x stage; readout gets a separate
            # family for each final operator as well.
            family_ids = operator_ids + (3 * stage)
            selected, weights, route_stats = self.router(
                router_query, family_ids,
                exploration_prob=exploration_prob,
            )
        else:
            selected, weights, route_stats = self.router(
                router_query,
                exploration_prob=exploration_prob,
                routing_offset=routing_offset,
                routing_capacity=local_capacity,
                routing_windows=routing_windows,
            )
        delta = self._apply_circuits(routed_query, selected, weights)
        selected_steps.append(selected)
        selected_weights[:, stage] = weights
        route_gains[:, stage] = route_stats["route_gain"]
        step_entropies[:, stage] = route_stats["router_entropy"]
        if self.routing_mode == "family_local":
            family_steps.append(route_stats["family_ids"])
            shared_fallback_fractions.append(route_stats["shared_selected_fraction"])
        return delta * route_stats["route_gain"].unsqueeze(-1)

    def forward(self, inputs: torch.Tensor, adaptive: bool | None = None,
                coverage: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del adaptive  # The explicit three-stage graph is always executed.
        if coverage:
            raise NotImplementedError("coverage regularization is not implemented for typed registers")
        batch_size = inputs.shape[0]
        device = inputs.device
        slots = self.encode_slots(inputs)
        operands = self.operand_encoder(slots[:, 3:6])
        op_ids = (inputs[:, 1:3] - 2).clamp(0, 2)
        op1 = self.operation_embedding(op_ids[:, 0])
        op2 = self.operation_embedding(op_ids[:, 1])

        selected_steps: list[torch.Tensor] = []
        family_steps: list[torch.Tensor] = []
        shared_fallback_fractions: list[torch.Tensor] = []
        selected_weights = torch.zeros(
            batch_size, self.internal_steps, self.router.active_circuits, device=device)
        route_gains = torch.ones(batch_size, self.internal_steps, device=device)
        step_entropies = torch.zeros(batch_size, self.internal_steps, device=device)

        first_pair = self.pair_encoder(torch.cat([operands[:, 0], operands[:, 1]], dim=-1))
        first_query = first_pair + op1
        first_delta = self._route_stage(
            first_query, 0, op_ids[:, 0], selected_steps, selected_weights,
            route_gains, step_entropies, family_steps, shared_fallback_fractions)
        partial = self.partial_writer(torch.cat([first_query, first_delta], dim=-1))

        second_pair = self.pair_encoder(torch.cat([partial, operands[:, 2]], dim=-1))
        second_query = second_pair + op2
        second_delta = self._route_stage(
            second_query, 1, op_ids[:, 1], selected_steps, selected_weights,
            route_gains, step_entropies, family_steps, shared_fallback_fractions)
        final = self.final_writer(torch.cat([second_query, second_delta], dim=-1))

        readout_query = final
        if self.readout_mode == "routed":
            readout_delta = self._route_stage(
                readout_query, 2, op_ids[:, 1], selected_steps, selected_weights,
                route_gains, step_entropies, family_steps, shared_fallback_fractions)
            readout = self.readout_writer(torch.cat([readout_query, readout_delta], dim=-1))
        else:
            # The final register is already the typed result of op2.  A third
            # bank lookup adds a value-conditioned lookup table at readout and
            # was observed to use nearly the entire bank with little reuse.
            # Keep the route tensor shape stable for analysis/optimizers while
            # marking this deterministic stage as non-routed.
            selected_steps.append(torch.full(
                (batch_size, self.router.active_circuits), -1,
                dtype=torch.long, device=device))
            readout = final
        stage_states = torch.stack([partial, final, readout], dim=1)
        step_logits = self.output(stage_states.reshape(-1, self.state_dim))
        step_logits = step_logits.reshape(batch_size, self.internal_steps, -1)

        stats = {
            "active_circuits": torch.tensor(self.router.active_circuits, device=device),
            "internal_steps": torch.tensor(self.internal_steps, device=device),
            "router_entropy": (step_entropies.mean() if self.readout_mode == "routed"
                                else step_entropies[:, :2].mean()),
            "selected_ids": torch.stack(selected_steps, dim=1),
            "selected_weights": selected_weights,
            "route_gains": route_gains,
            "step_logits": step_logits,
            "executed_steps": torch.full((batch_size,), self.internal_steps,
                                          dtype=torch.long, device=device),
            "executed_mask": torch.ones(batch_size, self.internal_steps,
                                         dtype=torch.bool, device=device),
            "register_norms": torch.stack([
                partial.norm(dim=-1), final.norm(dim=-1), readout.norm(dim=-1)
            ], dim=1),
        }
        if self.routing_mode == "family_local":
            stats.update({
                "family_ids": torch.stack(family_steps, dim=1),
                "shared_selected_fraction": torch.stack(
                    shared_fallback_fractions).mean(),
            })
        self._last_route = stats
        return step_logits[:, -1], stats

    def parameter_report(self) -> dict[str, int | float]:
        total = count_parameters(self)
        shared = (
            count_parameters(self.token_embedding)
            + self.position_embedding.numel()
            + self.position_scale.numel()
            + self.position_bias.numel()
            + (count_parameters(self.value_encoder) if self.value_encoder is not None else 0)
            + count_parameters(self.operand_encoder)
            + count_parameters(self.pair_encoder)
            + count_parameters(self.operation_embedding)
            + self.stage_embedding.numel()
            + count_parameters(self.route_value_encoder)
            + self.route_context_scale.numel()
            + count_parameters(self.partial_writer)
            + count_parameters(self.final_writer)
            + count_parameters(self.readout_writer)
            + self.router.level_projections.numel()
            + self.router.level_bias.numel()
            + count_parameters(self.output)
        )
        one_circuit = (
            self.circuits.down[0].numel()
            + self.circuits.up[0].numel()
            + self.circuits.bias[0].numel()
        )
        candidate_key_params = self.router.keys[0].numel() * self.router.candidate_pool
        active = shared + candidate_key_params + one_circuit * self.router.active_circuits
        return {
            "total_params": total,
            "active_params_estimate": active,
            "active_fraction": active / total,
            "active_circuit_params": one_circuit * self.router.active_circuits,
            "route_exploration_prob": self.route_exploration_prob,
            "register_graph": "operands->partial->final->readout",
            "readout_mode": self.readout_mode,
            "route_query_mode": self.route_query_mode,
            "route_context_dim": self.route_context_dim,
            "routing_mode": self.routing_mode,
            "family_count": self.family_count,
            "shared_fraction": self.shared_fraction,
            "typed_route_partitions": self.typed_route_partitions,
            "typed_route_shared": self.typed_route_shared,
        }
