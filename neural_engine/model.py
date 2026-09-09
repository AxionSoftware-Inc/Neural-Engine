from __future__ import annotations

import torch
from torch import nn

from .circuits import (FactorizedMicroCircuitBank, MicroCircuitBank,
                        SharedResidualMicroCircuitBank)
from .encoding import (VALUE_HARMONICS, VALUE_MODULUS, VALUE_TOKEN_OFFSET,
                       encode_tokens)
from .instrumentation import count_parameters
from .router import (FactorizedRouter, FlatRouter, HierarchicalRouter,
                      ProbeRouteRouter, StableFamilyRouter)
from .state import PersistentState


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
                 input_reinjection: float = 1.0, circuit_delta_scale: float = 1.0,
                 input_reinjection_schedule: list[float] | tuple[float, ...] | None = None,
                 correction_gate_mode: str = "none", memory_write_mode: str = "none",
                 post_correction_residual_scale: float = 0.0,
                 circuit_bank_mode: str = "independent", shared_rank: int = 8,
                 factor_count: int | None = None,
                 factor_candidate_pool: int | None = None,
                 factor_pair_rank: int = 0, factor_pair_scale: float = 1.0,
                 routing_reuse_weight: float = 0.0, routing_reuse_start_level: int = 0,
                 route_exploration_prob: float = 0.0,
                 routing_capacity: int | None = None, routing_depth: int | None = None,
                 router_variant: str = "global", family_count: int = 2,
                 shared_fraction: float = 0.125,
                 soft_routing_temperature: float = 0.0,
                 route_target_supervision: bool = False):
        super().__init__()
        if circuit_mode not in {"parallel", "serial"}:
            raise ValueError("circuit_mode must be 'parallel' or 'serial'")
        if memory_write_mode not in {"none", "gated"}:
            raise ValueError("memory_write_mode must be 'none' or 'gated'")
        if circuit_bank_mode not in {"independent", "shared_residual", "factorized"}:
            raise ValueError("circuit_bank_mode must be 'independent', 'shared_residual', or 'factorized'")
        if circuit_bank_mode == "factorized" and router_variant != "factorized":
            raise ValueError("factorized circuit banks require router_variant='factorized'")
        if router_variant == "factorized" and circuit_bank_mode != "factorized":
            raise ValueError("router_variant='factorized' requires circuit_bank_mode='factorized'")
        if shared_rank < 1:
            raise ValueError("shared_rank must be positive")
        if not 0.0 < halt_threshold < 1.0:
            raise ValueError("halt_threshold must be between 0 and 1")
        self.state_dim = state_dim
        self.active_circuits = active_circuits
        self.internal_steps = internal_steps
        self.slot_count = slot_count
        self.use_task_context = task_context
        self.task_context_update = task_context_update
        self.circuit_mode = circuit_mode
        self.circuit_bank_mode = circuit_bank_mode
        self.shared_rank = int(shared_rank)
        self.factor_count = factor_count
        self.factor_candidate_pool = factor_candidate_pool
        self.factor_pair_rank = int(factor_pair_rank)
        self.factor_pair_scale = float(factor_pair_scale)
        self.numeric_value_encoding = numeric_value_encoding
        self.adaptive_halting = adaptive_halting
        self.adaptive_inference = adaptive_halting
        self.halt_threshold = halt_threshold
        self.routing_coverage_temperature = routing_coverage_temperature
        if not 0.0 <= route_exploration_prob <= 1.0:
            raise ValueError("route_exploration_prob must be between 0 and 1")
        self.route_exploration_prob = route_exploration_prob
        self.input_reinjection = input_reinjection
        if input_reinjection_schedule is None:
            input_reinjection_schedule = [input_reinjection] * internal_steps
        if len(input_reinjection_schedule) != internal_steps:
            raise ValueError("input_reinjection_schedule must match internal_steps")
        if any(float(value) < 0.0 for value in input_reinjection_schedule):
            raise ValueError("input_reinjection_schedule values must be non-negative")
        self.input_reinjection_schedule = tuple(float(value) for value in input_reinjection_schedule)
        if circuit_delta_scale <= 0.0:
            raise ValueError("circuit_delta_scale must be positive")
        self.circuit_delta_scale = circuit_delta_scale
        if routing_reuse_weight < 0.0:
            raise ValueError("routing_reuse_weight must be non-negative")
        self.routing_reuse_weight = routing_reuse_weight
        if routing_reuse_start_level < 0:
            raise ValueError("routing_reuse_start_level must be non-negative")
        self.routing_reuse_start_level = routing_reuse_start_level
        if correction_gate_mode not in {"none", "route_bounded"}:
            raise ValueError("correction_gate_mode must be 'none' or 'route_bounded'")
        self.correction_gate_mode = correction_gate_mode
        if post_correction_residual_scale < 0.0:
            raise ValueError("post_correction_residual_scale must be non-negative")
        self.post_correction_residual_scale = float(post_correction_residual_scale)
        self.memory_write_mode = memory_write_mode
        if router_variant not in {"global", "flat", "probe", "family_local", "family_conditioned", "factorized"}:
            raise ValueError("router_variant must be 'global', 'flat', 'probe', 'family_local', 'family_conditioned', or 'factorized'")
        if router_variant in {"family_local", "family_conditioned"} and family_count < 2:
            raise ValueError("NeuralEngineV0 semantic family routing requires at least two families")
        self.router_variant = router_variant
        self.family_count = family_count
        self.shared_fraction = shared_fraction
        self.route_target_supervision = route_target_supervision
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
        self.family_embeddings = None
        if router_variant == "factorized":
            self.router = FactorizedRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, router_addresses,
                routing_capacity=routing_capacity, routing_depth=routing_depth,
                factor_count=factor_count,
                factor_candidate_pool=factor_candidate_pool,
            )
        elif router_variant == "family_local":
            self.router = StableFamilyRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, router_addresses,
                routing_capacity=routing_capacity, routing_depth=routing_depth,
                family_count=family_count, shared_fraction=shared_fraction,
            )
        elif router_variant == "flat":
            self.router = FlatRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, router_addresses,
                routing_capacity=routing_capacity, routing_depth=routing_depth,
                soft_routing_temperature=soft_routing_temperature)
        elif router_variant == "probe":
            self.router = ProbeRouteRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, router_addresses,
                routing_capacity=routing_capacity, routing_depth=1)
        else:
            self.router = HierarchicalRouter(
                state_dim, num_circuits, router_branch, router_depth,
                candidate_pool, active_circuits, router_addresses,
                routing_capacity=routing_capacity, routing_depth=routing_depth,
                soft_routing_temperature=soft_routing_temperature)
            if router_variant == "family_conditioned":
                self.family_embeddings = nn.Parameter(torch.empty(family_count, state_dim))
                nn.init.normal_(self.family_embeddings, std=0.02)
        if circuit_bank_mode == "factorized":
            self.circuits = FactorizedMicroCircuitBank(
                num_circuits, state_dim, circuit_rank,
                factor_count=factor_count,
                factor_pair_rank=factor_pair_rank,
                factor_pair_scale=factor_pair_scale,
            )
        elif circuit_bank_mode == "shared_residual":
            self.circuits = SharedResidualMicroCircuitBank(
                num_circuits, state_dim, circuit_rank, shared_rank
            )
        else:
            self.circuits = MicroCircuitBank(num_circuits, state_dim, circuit_rank)
        self.correction_gate = (
            nn.Linear(2 * state_dim, 1)
            if correction_gate_mode == "route_bounded" else None
        )
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
        if self.correction_gate is not None:
            # 2 * sigmoid(0) = 1, so the optional gate starts exactly as the
            # default un-gated correction path.
            nn.init.zeros_(self.correction_gate.weight)
            nn.init.zeros_(self.correction_gate.bias)
        self._last_route: dict[str, torch.Tensor] = {}

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = encode_tokens(inputs, self.token_embedding, self.value_encoder)
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

    def semantic_family_ids(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return stable task-domain families without selecting a circuit ID."""
        task_ids = (inputs[:, 0] - 1).clamp(0, 14)
        if self.family_count == 2:
            return task_ids.ge(9).long()
        if self.family_count == 4:
            return torch.where(task_ids < 3, 0,
                               torch.where(task_ids < 6, 1,
                                           torch.where(task_ids < 9, 2, 3)))
        return task_ids.remainder(self.family_count)

    def forward(self, inputs: torch.Tensor, adaptive: bool | None = None,
                forced_selected_ids: torch.Tensor | None = None,
                forced_selected_weights: torch.Tensor | None = None,
                forced_route_gains: torch.Tensor | None = None,
                coverage: bool = False,
                collect_stats: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
        selected_steps = [] if collect_stats else None
        candidate_steps = [] if collect_stats else None
        query_steps = [] if collect_stats else None
        coverage_losses = []
        routing_target_losses = []
        routing_reuse_losses = []
        soft_route = (self.training and getattr(self.router, "soft_routing_temperature", 0.0) > 0.0
                      and self.router_variant in {"global", "family_conditioned"}
                      and getattr(self, "routing_mode", "learned") != "controlled_task")
        route_width = self.router.candidate_pool if soft_route else self.active_circuits
        selected_weights = (torch.zeros(batch_size, self.internal_steps, route_width,
                                         device=inputs.device) if collect_stats else None)
        route_gains = (torch.ones(batch_size, self.internal_steps, device=inputs.device)
                       if collect_stats else None)
        step_entropies = (torch.zeros(batch_size, self.internal_steps, device=inputs.device)
                          if collect_stats else None)
        executed_mask = (torch.zeros(batch_size, self.internal_steps, dtype=torch.bool,
                                     device=inputs.device) if collect_stats else None)
        step_logits = (torch.zeros(batch_size, self.internal_steps, num_classes,
                                   device=inputs.device) if collect_stats else None)
        halt_logits = (torch.zeros(batch_size, self.internal_steps, device=inputs.device)
                       if collect_stats else None)
        last_logits = torch.zeros(batch_size, num_classes, device=inputs.device)
        active = torch.ones(batch_size, dtype=torch.bool, device=inputs.device)
        if forced_selected_ids is not None:
            expected_shape = (batch_size, self.internal_steps, self.active_circuits)
            if tuple(forced_selected_ids.shape) != expected_shape:
                raise ValueError(f"forced_selected_ids must have shape {expected_shape}")
            if (forced_selected_ids < -1).any():
                raise ValueError("forced_selected_ids may use -1 only as a no-override sentinel")
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
            # With adaptive execution disabled every sample is active at every
            # stage.  Avoid the dynamic nonzero/index path so fixed-shape
            # serving can be captured by CUDA Graphs and pays fewer launches.
            if not use_adaptive:
                active_indices = torch.arange(batch_size, device=inputs.device)
            else:
                active_indices = active.nonzero(as_tuple=False).squeeze(-1)
            selected_step = (torch.full((batch_size, route_width), -1,
                                        dtype=torch.long, device=inputs.device)
                             if collect_stats else None)
            candidate_step = (torch.full((batch_size, self.router.candidate_pool), -1,
                                         dtype=torch.long, device=inputs.device)
                              if collect_stats else None)
            if active_indices.numel() == 0:
                if collect_stats:
                    selected_steps.append(selected_step)
                    candidate_steps.append(candidate_step)
                    query_steps.append(torch.zeros(batch_size, self.state_dim, device=inputs.device))
                    step_logits[:, step] = last_logits
                continue
            active_state = state[active_indices]
            # A distinct query per recurrent step encourages compositional
            # paths instead of routing every step from the same representation.
            step_query = active_state + self.step_embedding[step]
            if task_context is not None:
                step_query = step_query + task_context[active_indices]
            if collect_stats:
                query_step = torch.zeros(batch_size, self.state_dim, device=inputs.device)
                query_step[active_indices] = step_query
            router_kwargs = {
                "coverage": coverage,
                "coverage_temperature": self.routing_coverage_temperature,
                "exploration_prob": (self.route_exploration_prob if self.training else 0.0),
            }
            if (self.training and self.routing_reuse_weight > 0.0
                    and self.router_variant in {"global", "family_conditioned"}):
                router_kwargs["reuse_task_ids"] = (inputs[:, 0] - 1).clamp(0, 14)[active_indices]
                router_kwargs["reuse_start_level"] = self.routing_reuse_start_level
            if self.route_target_supervision and self.router_variant in {"global", "family_conditioned"}:
                group_count = max(1, self.router.num_circuits // self.active_circuits)
                task_ids = (inputs[:, 0] - 1).clamp(0, 14)
                target_bases = (task_ids.remainder(group_count) * self.active_circuits)[active_indices]
                router_kwargs["target_bases"] = target_bases
            family_ids = None
            if self.router_variant in {"family_local", "family_conditioned"}:
                family_ids = self.semantic_family_ids(inputs)[active_indices]
            if self.router_variant == "family_conditioned":
                router_query = step_query + self.family_embeddings[family_ids]
            else:
                router_query = step_query
            if self.router_variant == "family_local":
                if coverage:
                    raise ValueError("family_local router does not support coverage regularization")
                selected, weights, route_stats = self.router(
                    router_query, family_ids,
                    **router_kwargs)
            else:
                if not collect_stats and isinstance(self.router, HierarchicalRouter):
                    router_kwargs["collect_stats"] = False
                selected, weights, route_stats = self.router(router_query, **router_kwargs)
            route_gain = route_stats["route_gain"]
            route_candidates = route_stats.get("candidate_ids")
            if (collect_stats and route_candidates is not None
                    and route_candidates.shape[-1] == self.router.candidate_pool):
                candidate_step[active_indices] = route_candidates
            if "routing_coverage_loss" in route_stats:
                coverage_losses.append(route_stats["routing_coverage_loss"])
            if "routing_target_loss" in route_stats:
                routing_target_losses.append(route_stats["routing_target_loss"])
            if "routing_reuse_loss" in route_stats:
                routing_reuse_losses.append(route_stats["routing_reuse_loss"])
            if forced_selected_ids is not None:
                forced_ids = forced_selected_ids[active_indices, step].to(device=inputs.device)
                override = forced_ids[:, 0].ge(0)
                if override.any():
                    selected = torch.where(override.unsqueeze(-1), forced_ids, selected)
                    if forced_selected_weights is None:
                        override_weights = torch.full(
                            (active_indices.numel(), self.active_circuits),
                            1.0 / self.active_circuits, device=inputs.device)
                    else:
                        override_weights = forced_selected_weights[active_indices, step].to(
                            device=inputs.device)
                    weights = torch.where(override.unsqueeze(-1), override_weights, weights)
                    if forced_route_gains is None:
                        override_gains = torch.ones_like(route_gain)
                    else:
                        override_gains = forced_route_gains[active_indices, step].to(
                            device=inputs.device)
                    route_gain = torch.where(override, override_gains, route_gain)
                if self.circuit_mode == "serial":
                    circuit_delta = self.circuits.forward_serial(step_query, selected, weights)
                else:
                    circuit_delta = self.circuits(step_query, selected, weights)
            elif self.circuit_mode == "serial":
                circuit_delta = self.circuits.forward_serial(step_query, selected, weights)
            else:
                circuit_delta = self.circuits(step_query, selected, weights)
            if self.correction_gate is not None:
                gate_input = torch.cat((step_query, circuit_delta), dim=-1)
                correction_gate = 2.0 * torch.sigmoid(
                    self.correction_gate(gate_input)).squeeze(-1)
            else:
                correction_gate = torch.ones_like(route_gain)
            delta = (circuit_delta * route_gain.unsqueeze(-1)
                     * self.circuit_delta_scale * correction_gate.unsqueeze(-1))
            update = (delta + self.input_reinjection_schedule[step] * encoded[active_indices]
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
            if self.post_correction_residual_scale:
                # Optional causal bypass: the selected correction reaches the
                # recurrent state after the GRU instead of being fully gated
                # through its update.  The default scale is exactly zero.
                updated_state = updated_state + (
                    self.post_correction_residual_scale * delta
                )
            if active_indices.numel() == batch_size:
                state = updated_state
            else:
                next_state = state.clone()
                next_state[active_indices] = updated_state
                state = next_state
            if collect_stats:
                selected_step[active_indices] = selected
                selected_steps.append(selected_step)
                candidate_steps.append(candidate_step)
                query_steps.append(query_step)
                selected_weights[active_indices, step] = weights
                route_gains[active_indices, step] = route_gain
                executed_mask[active_indices, step] = True
                step_entropies[active_indices, step] = route_stats["router_entropy"]
            updated_logits = self.output(updated_state)
            if active_indices.numel() == batch_size:
                last_logits = updated_logits
            else:
                next_logits = last_logits.clone()
                next_logits[active_indices] = updated_logits
                last_logits = next_logits
            if collect_stats:
                step_logits[:, step] = last_logits
            if self.halt_head is not None:
                updated_halt_logits = self.halt_head(updated_state).squeeze(-1)
                if collect_stats:
                    halt_logits[active_indices, step] = updated_halt_logits
                if use_adaptive:
                    should_halt = torch.sigmoid(updated_halt_logits) >= self.halt_threshold
                    if step == self.internal_steps - 1:
                        should_halt = torch.ones_like(should_halt, dtype=torch.bool)
                    next_active = active.clone()
                    next_active[active_indices[should_halt]] = False
                    active = next_active
        if not collect_stats:
            self._last_route = {}
            return last_logits, {}
        stats = {
            "active_circuits": torch.tensor(self.active_circuits, device=inputs.device),
            "internal_steps": torch.tensor(self.internal_steps, device=inputs.device),
            "router_entropy": step_entropies.sum() / executed_mask.sum().clamp_min(1),
            "selected_ids": torch.stack(selected_steps, dim=1),
            "candidate_ids": torch.stack(candidate_steps, dim=1),
            "query_states": torch.stack(query_steps, dim=1),
            "selected_weights": selected_weights,
            "route_gains": route_gains,
            "step_logits": step_logits,
            "halt_logits": halt_logits,
            "executed_steps": executed_mask.sum(dim=1),
            "executed_mask": executed_mask,
        }
        if coverage_losses:
            stats["routing_coverage_loss"] = torch.stack(coverage_losses).mean()
        if routing_target_losses:
            stats["routing_target_loss"] = torch.stack(routing_target_losses).mean()
        if routing_reuse_losses:
            stats["routing_reuse_loss"] = torch.stack(routing_reuse_losses).mean()
        self._last_route = stats
        return last_logits, stats

    def parameter_report(self) -> dict[str, int | float]:
        total = count_parameters(self)
        shared = count_parameters(self.token_embedding) + self.position_embedding.numel()
        if self.value_encoder is not None:
            shared += count_parameters(self.value_encoder)
        shared += self.position_scale.numel() + self.position_bias.numel() + count_parameters(self.encoder)
        shared += count_parameters(self.state) + self.step_embedding.numel()
        if hasattr(self.router, "level_projections"):
            shared += self.router.level_projections.numel() + self.router.level_bias.numel()
        if self.router_variant == "family_local":
            shared += self.router.family_embeddings.numel()
        elif self.router_variant == "family_conditioned":
            shared += self.family_embeddings.numel()
        shared += count_parameters(self.output)
        if self.task_context_embedding is not None:
            shared += count_parameters(self.task_context_embedding)
        if self.halt_head is not None:
            shared += count_parameters(self.halt_head)
        if self.memory_write is not None:
            shared += count_parameters(self.memory_write)
        if self.correction_gate is not None:
            shared += count_parameters(self.correction_gate)
        if self.circuit_bank_mode == "shared_residual":
            shared += self.circuits.shared_down.numel()
            shared += self.circuits.shared_up.numel()
            shared += self.circuits.shared_bias.numel()
        if self.circuit_bank_mode == "factorized":
            factor_row = (self.circuits.down_factors[0].numel()
                          + self.circuits.up_factors[0].numel()
                          + self.circuits.bias_factors[0].numel())
            active_circuit_params = factor_row * self.active_circuits * 2
            if self.circuits.factor_mix_mode == "per_address":
                active_circuit_params += self.active_circuits * self.circuits.factor_mix[0].numel()
            if self.circuits.factor_pair_rank:
                active_circuit_params += (
                    self.circuits.pair_down_basis.numel()
                    + self.circuits.pair_up_basis.numel()
                    + self.circuits.pair_bias_basis.numel()
                    + 2 * self.active_circuits * self.circuits.factor_pair_rank
                )
            candidate_key_params = (self.router.keys[0].numel()
                                    * self.router.factor_candidate_pool)
        else:
            one_circuit = (self.circuits.down[0].numel()
                           + self.circuits.up[0].numel()
                           + self.circuits.bias[0].numel())
            active_circuit_params = one_circuit * self.active_circuits
            candidate_key_params = self.router.keys[0].numel() * self.router.candidate_pool
        active = shared + candidate_key_params + active_circuit_params
        return {
            "total_params": total,
            "active_params_estimate": active,
            "active_fraction": active / total,
            "active_circuit_params": active_circuit_params,
            "circuit_bank_mode": self.circuit_bank_mode,
            "shared_rank": self.shared_rank,
            "route_exploration_prob": self.route_exploration_prob,
        }
