"""Opt-in circuit credit/compute diagnostics. No router or body modifications.

Counts describe PyTorch-level parameter accesses, not cache/DRAM traffic.
Diagnostic snapshots are deliberately outside inference latency measurements.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F


def final_margin_loss(logits, targets, margin=1.0):
    """Multiclass hinge against the hardest wrong final-output class (not CE)."""
    correct = logits.gather(1, targets[:, None]).squeeze(1)
    wrong = logits.scatter(1, targets[:, None], float("-inf")).max(1).values
    return F.relu(margin + wrong - correct)


class CircuitLedger:
    """Actual row forwards, gradients, optimizer changes and task associations.

    before_step/after_step snapshots ALL circuit rows so Adam momentum/decay
    updates on zero-gradient rows cannot masquerade as sparse credit. This
    optional diagnostic has O(bank parameters) memory/read cost; never enable
    snapshots in inference throughput tests or call it a sparse optimizer.
    """
    def __init__(self, model, tasks=15):
        self.bank = model.circuits
        self.n = self.bank.num_circuits
        self.usage = torch.zeros(self.n, dtype=torch.long)
        self.forwards = {}
        self.task_counts = torch.zeros(self.n, tasks, dtype=torch.long)
        self.grad_sum = torch.zeros(self.n, dtype=torch.float64)
        self.grad_steps = torch.zeros(self.n, dtype=torch.long)
        self.updates = torch.zeros(self.n, dtype=torch.long)
        self.steps = 0
        self._phase, self._tasks = None, None
        self._snapshot = None
        self._hook = self.bank.register_forward_pre_hook(self._observe)

    @contextmanager
    def phase(self, name, task_ids):
        previous = self._phase, self._tasks
        self._phase, self._tasks = name, task_ids.detach().cpu()
        try:
            yield
        finally:
            self._phase, self._tasks = previous

    def _observe(self, module, args):
        if self._phase is None:
            return
        ids = args[1].detach().cpu()
        if ids.shape[0] != self._tasks.numel():
            raise ValueError("ledger requires task IDs for the actual (non-adaptive) batch")
        counts = torch.bincount(ids.flatten(), minlength=self.n)
        self.forwards.setdefault(self._phase, torch.zeros_like(self.usage)).add_(counts)
        if self._phase == "main":
            self.usage += counts
            task_ids = self._tasks[:, None].expand_as(ids)
            index = ids * self.task_counts.shape[1] + task_ids
            self.task_counts += torch.bincount(index.flatten(), minlength=self.task_counts.numel()).reshape_as(self.task_counts)

    def before_step(self):
        if self._snapshot is not None:
            raise RuntimeError("after_step must close the previous optimizer observation")
        self._snapshot = {}
        squared = torch.zeros(self.n, dtype=torch.float64)
        for name in ("down", "up", "bias"):
            p = getattr(self.bank, name)
            self._snapshot[name] = p.detach().clone()
            if p.grad is not None:
                squared += p.grad.detach().double().reshape(self.n, -1).square().sum(1).cpu()
        norms = squared.sqrt()
        self.grad_sum += norms
        self.grad_steps += norms.gt(1e-12)
        self.steps += 1

    def after_step(self):
        if self._snapshot is None:
            raise RuntimeError("before_step must precede after_step")
        changed = torch.zeros(self.n, dtype=torch.bool)
        for name, previous in self._snapshot.items():
            changed |= getattr(self.bank, name).detach().ne(previous).reshape(self.n, -1).any(1).cpu()
        self.updates += changed
        self._snapshot = None

    def report(self):
        probabilities = self.task_counts.double() / self.task_counts.sum(1, keepdim=True).clamp_min(1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
        threshold = max(1, math.ceil(self.steps * 0.05))
        return {
            "usage": self.usage.tolist(),
            "forward_count_by_phase": {k: v.tolist() for k, v in self.forwards.items()},
            "gradient_norm_mean_per_optimizer_step": (self.grad_sum / max(self.steps, 1)).tolist(),
            "nonzero_gradient_steps": self.grad_steps.tolist(),
            "actual_update_count": self.updates.tolist(),
            "dead_in_main_training": self.usage.eq(0).tolist(),
            "undertrained_by_gradient_exposure": self.grad_steps.lt(threshold).tolist(),
            "undertrained_threshold_steps": threshold,
            "task_usage_counts": self.task_counts.tolist(),
            "task_usage_entropy": entropy.tolist(),
            "optimizer_steps_observed": self.steps,
            "note": "Task usage/entropy are associations, not functional specialization proof. Gradient statistics cannot be recovered retrospectively from a checkpoint.",
        }

    def close(self):
        self._hook.remove()


def router_accounting(router, state_dim):
    """Per executed decision MACs and param accesses for known native routers."""
    name = type(router).__name__
    E, M = router.routing_capacity, router.candidate_pool
    D = state_dim
    total = sum(p.numel() for p in router.parameters())
    if name == "HierarchicalRouter":
        projection = router.num_addresses * router.active_depth * D * router.branch
        fixed = router.num_addresses * router.active_depth * (D + 1) * router.branch
        key_rows, reads = M, M * D
        pair_macs = 0
        touched = fixed + reads
    elif name == "FlatRouter":
        projection, fixed, key_rows, reads, pair_macs = 0, 0, E, E * D, 0
        touched = reads
    elif name in {"ProbeRouteRouter", "CoupledProbeRouter"}:
        rank = router.pair_rank
        width = router.keys.shape[1]
        if name == "ProbeRouteRouter":
            projection = 2 * D * D + D * rank
            reads = 2 * E * D
        else:
            # Existing forward calls the shared feature projection twice.
            projection = 2 * D * width + width * rank
            reads = 2 * E * width
        pair_macs = M * (M - 1) // 2 * rank
        fixed = total - (E * width + E * rank)
        if name == "ProbeRouteRouter":
            fixed -= E * D
        key_rows = E
        touched = total - (router.num_circuits - M) * rank
    else:
        raise NotImplementedError(f"No cost formula for {name}; do not silently use hierarchical costs")
    return {"type": name, "projection_macs": projection, "key_score_macs": reads,
            "pair_score_macs": pair_macs, "key_parameter_reads": reads,
            "touched_parameters_per_decision_upper_bound": touched,
            "fixed_parameters": fixed, "key_rows_per_decision": key_rows,
            "total_parameters": total}


def sparse_cost_report(model, inputs, stats):
    """Backward-compatible v2 payload: append under `sparse_audit_v2`.

    Legacy instrumentation functions/JSON keys are not overwritten. MACs are
    multiply-accumulate pairs; estimated arithmetic FLOPs = 2*MACs, excluding
    nonlinear/elementwise/reduction/indexing/top-k operations.
    """
    if type(model).__name__ != "NeuralEngineV0" or type(model.circuits).__name__ != "MicroCircuitBank":
        raise NotImplementedError("v2 formulas support independent NeuralEngineV0 circuit banks only")
    if getattr(model, "family_embeddings", None) is not None:
        raise NotImplementedError("semantic family routing needs a separate access formula")
    D, T = model.state_dim, inputs.shape[1]
    bank = model.circuits
    row_params = sum(getattr(bank, name)[0].numel() for name in ("down", "up", "bias"))
    router = router_accounting(model.router, D)
    total = sum(p.numel() for p in model.parameters())  # frozen weights are still parameters
    bank_total = sum(p.numel() for p in bank.parameters())
    shared_total = total - bank_total - router["total_parameters"]
    recurrent = sum(p.numel() for p in model.state.update.parameters())
    head = sum(p.numel() for p in model.output.parameters())
    extra = sum(p.numel() for p in model.memory_write.parameters()) if model.memory_write is not None else 0
    halt = sum(p.numel() for p in model.halt_head.parameters()) if model.halt_head is not None else 0
    components = {
        "encoder": model.encoder[1].in_features * D,
        "initial_state": D * D,
        # encode_tokens evaluates the value encoder at ALL T positions before torch.where.
        "value_encoder": T * model.value_encoder.in_features * model.value_encoder.out_features if model.value_encoder is not None else 0,
        "router_projection_per_step": router["projection_macs"],
        "router_key_score_per_step": router["key_score_macs"],
        "router_pair_score_per_step": router["pair_score_macs"],
        "circuit_body_per_step": model.active_circuits * 2 * D * bank.rank,
        "recurrent_gru_per_step": 6 * D * D,
        "memory_write_per_step": 2 * D * D if model.memory_write is not None else 0,
        "output_head_per_step": D * model.output[-1].out_features,
        "halt_head_per_step": D if model.halt_head is not None else 0,
    }
    fixed_macs = sum(v for k, v in components.items() if not k.endswith("_per_step"))
    step_macs = sum(v for k, v in components.items() if k.endswith("_per_step"))
    unique, touched, steps_list = [], [], []
    shared_access = []
    for b in range(inputs.shape[0]):
        executed = stats["executed_mask"][b].bool()
        steps = int(executed.sum())
        if steps == 0:
            raise ValueError("v2 expects at least one executed decision per example")
        ids = stats["selected_ids"][b][executed]
        candidates = stats["candidate_ids"][b][executed]
        embedding = inputs[b].clamp_max(model.token_embedding.num_embeddings - 1).unique().numel() * model.token_embedding.embedding_dim
        shared = shared_total - model.token_embedding.weight.numel() + embedding
        shared -= sum(getattr(model, n).numel() for n in ("position_embedding", "position_scale", "position_bias"))
        shared += 3 * T * model.token_embedding.embedding_dim
        shared += (steps - model.internal_steps) * D
        if model.task_context_embedding is not None:
            shared += D - model.task_context_embedding.weight.numel()
        r = model.router
        if router["type"] == "HierarchicalRouter":
            router_unique = router["fixed_parameters"] + candidates.unique().numel() * D
        elif router["type"] == "FlatRouter":
            router_unique = r.routing_capacity * D
        else:
            router_unique = router["total_parameters"] - (r.num_circuits - candidates.unique().numel()) * r.pair_rank
        unique.append(shared + router_unique + ids.unique().numel() * row_params)
        shared_access.append(shared)
        touched.append([recurrent + head + extra + halt + D
                        + router["touched_parameters_per_decision_upper_bound"]
                        + row.unique().numel() * row_params for row in ids])
        steps_list.append(steps)
    return {"schema_version": 2, "total_parameters": total,
            "total_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "unique_active_parameters_per_example": unique,
            "shared_controller_parameters_touched_per_example": shared_access,
            "touched_parameters_per_decision_upper_bound": touched,
            "router": router, "mac_components": components,
            "inference_macs_per_example": [fixed_macs + step_macs * t for t in steps_list],
            "inference_arithmetic_flops_per_example": [2 * (fixed_macs + step_macs * t) for t in steps_list],
            "training_probe_teacher_cost": "record separately from inference using explicit forward/backward counters",
            "exclusions": "Elementwise nonlinearities, norms, biases, top-k, gather, memory/cache effects, optimizer and diagnostic snapshot costs. Parameters != MACs != wall-clock latency."}


def sparse_row_credit(model, inputs, targets, plan, weights, gains, probe_rows, objective="margin"):
    """Extra final-output credit ONLY to the sampled replacement circuit rows.

    Controller/router receive no auxiliary gradients. autograd.grad traverses
    the sparse trajectory, but only sampled row gradients are added to the
    ordinary task gradients. No circuit implementation or routing rule changes.
    """
    if objective not in {"margin", "ce"}:
        raise ValueError("objective must be margin or ce")
    parameters = [getattr(model.circuits, n) for n in ("down", "up", "bias")]
    logits, _ = model(inputs, adaptive=False, forced_selected_ids=plan,
                      forced_selected_weights=weights, forced_route_gains=gains)
    loss = (final_margin_loss(logits, targets).mean() if objective == "margin"
            else F.cross_entropy(logits, targets))
    gradients = torch.autograd.grad(loss, parameters)
    rows = probe_rows.unique()
    return loss.detach(), [(p, rows, g.index_select(0, rows).detach()) for p, g in zip(parameters, gradients)]


def training_phase_budget(ledger, inference_cost, *, active=2, steps=3):
    """Convert measured row-forward counts to separate phase MAC budgets.

    This harness always executes K rows at each of T steps. Backward call
    counts are exact, but backward MACs are deliberately not guessed as 2x
    forward: only the bank trains and the controller propagates input grads.
    """
    if active < 1 or steps < 1:
        raise ValueError("active and steps must be positive")
    per_example = inference_cost["inference_macs_per_example"]
    if len(set(per_example)) != 1:
        raise ValueError("phase conversion requires fixed-length execution")
    phases = {}
    for name, counts in ledger["forward_count_by_phase"].items():
        forwards = sum(counts)
        if forwards % (active * steps):
            raise ValueError("row counts are not complete fixed-length trajectories")
        examples = forwards // (active * steps)
        phases[name] = {"circuit_row_forwards": forwards,
                        "example_trajectories": examples,
                        "forward_macs": examples * per_example[0]}
    return {"phases": phases,
            "main_backward_calls": ledger["optimizer_steps_observed"],
            "probe_backward_calls": ledger["optimizer_steps_observed"],
            "training_teacher_forwards": 0,
            "backward_macs": None,
            "note": "Measured forwards; analytical MACs; separate backward call counts. Excludes optimizer, diagnostic snapshots and evaluation-only oracles."}
