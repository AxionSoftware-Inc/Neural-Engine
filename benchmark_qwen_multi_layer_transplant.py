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
    MixedParentChild,
    NESwiGLUBlock,
    benchmark_forward,
    capture_batches,
    ce,
    evaluate_current,
    make_child,
    token_stream,
    train_child,
)


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(layers) < 2 or len(set(layers)) != len(layers):
        raise ValueError("layers must contain at least two distinct indices")
    return layers


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
        if dispatch_mode not in {"grouped", "token-loop"}:
            raise ValueError("transferred sparse child supports grouped or token-loop")
        self.dispatch_mode = dispatch_mode
        chunk = inner_size // num_experts
        self.experts = torch.nn.ModuleList()
        for expert_id in range(num_experts):
            start = expert_id * chunk
            stop = start + chunk
            self.experts.append(QwenSwiGLUSlice(
                parent.gate_proj.weight[start:stop].detach(),
                parent.up_proj.weight[start:stop].detach(),
                parent.down_proj.weight[:, start:stop].detach(),
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
        return self.num_experts / self.active_experts * flat_output.reshape_as(
            hidden_states,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = self.router(hidden_states)
        if self.training and not self.hard_train:
            outputs = torch.stack([
                expert(hidden_states) for expert in self.experts
            ], dim=-2)
            weights = F.softmax(scores / self.temperature, dim=-1)
            self.last_selected = scores.detach().argmax(dim=-1)
            self.last_active_expert_fraction = 1.0
            # At uniform routing, this exactly reconstructs the sum of slices.
            return self.num_experts * (outputs * weights.unsqueeze(-1)).sum(dim=-2)
        top_values, top_ids = scores.topk(self.active_experts, dim=-1)
        weights = F.softmax(top_values / self.temperature, dim=-1)
        self.last_selected = top_ids.detach()
        if not self.training and self.dispatch_mode == "grouped":
            return self._forward_grouped(hidden_states, top_ids, weights)
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_ids = top_ids.reshape(-1, self.active_experts)
        flat_weights = weights.reshape(-1, self.active_experts)
        flat_output = torch.zeros_like(flat_hidden)
        selected_pairs = 0
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(flat_ids == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_output = expert(flat_hidden[token_ids])
            contribution = expert_output * flat_weights[token_ids, slots].unsqueeze(-1)
            flat_output.index_add_(0, token_ids, contribution)
            selected_pairs += int(token_ids.numel())
        self.last_active_expert_fraction = selected_pairs / max(
            flat_hidden.shape[0] * self.num_experts, 1
        )
        # Top-k weights are normalized over the selected groups; rescale to
        # estimate the full intermediate-neuron sum from the active subset.
        return (
            self.num_experts / self.active_experts
            * flat_output.reshape_as(hidden_states)
        )


def make_transferred_routed_qwen_child(
    parent: torch.nn.Module,
    num_experts: int,
    active_experts: int,
    routing_temperature: float,
    calibration_rank: int,
    dispatch_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    child = TransferredRoutedQwenChild(
        parent, num_experts, active_experts, routing_temperature, dispatch_mode,
    ).to(device=device, dtype=dtype)
    if calibration_rank > 0:
        child = CalibratedChild(
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
                importance = outputs.square().mean(dim=-1)
                target = F.softmax(
                    torch.log(importance + 1e-8), dim=-1,
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
    train_ids = token_stream(
        tokenizer, TRAIN_TEXT, args.batch_size,
        args.sequence_length * args.train_batches, device,
    ).reshape(args.train_batches, args.batch_size, args.sequence_length)
    eval_ids = token_stream(
        tokenizer, EVAL_TEXT, args.batch_size,
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

    layers = [model.model.layers[index] for index in layer_indices]
    parents = [layer.mlp for layer in layers]
    hidden_size = int(model.config.hidden_size)
    children = []
    histories = []
    router_histories = []
    local_eval_mse = []

    for layer_index, layer in zip(layer_indices, layers):
        train_io = capture_batches(
            model, tokenizer, TRAIN_TEXT, args.batch_size,
            args.sequence_length, args.train_batches, device, layer_index,
        )
        eval_io = capture_batches(
            model, tokenizer, EVAL_TEXT, args.batch_size,
            args.sequence_length, args.eval_batches, device, layer_index,
        )
        if args.child_kind == "qwen-transfer":
            child = make_transferred_qwen_child(
                parents[len(children)], args.calibration_rank, device, dtype,
            )
            histories.append([])
            router_histories.append([])
        elif args.child_kind == "qwen-transfer-sparse":
            child = make_transferred_routed_qwen_child(
                parents[len(children)], args.num_experts, args.active_experts,
                args.routing_temperature, args.calibration_rank,
                args.dispatch_mode, device, dtype,
            )
            router_histories.append(train_importance_router(
                child, train_io, device, dtype, args.router_supervision_steps,
                args.learning_rate, args.max_grad_norm, args.log_every,
            ))
            histories.append(train_child(
                child, train_io, device, dtype, args.steps,
                args.learning_rate, args.max_grad_norm, args.log_every,
                args.hard_train_steps,
            ))
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
            model, layers, children, list(eval_ids), teacher_logits, device,
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
    parent_params = [sum(parameter.numel() for parameter in parent.parameters()) for parent in parents]
    if args.child_kind == "qwen-transfer":
        effective_child_inner_size = int(parents[0].gate_proj.out_features)
    elif args.child_kind == "qwen-transfer-sparse":
        effective_child_inner_size = int(
            parents[0].gate_proj.out_features // args.num_experts
        )
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
        "num_experts": args.num_experts,
        "active_experts": args.active_experts,
        "hard_route_expected_expert_fraction": (
            args.active_experts / args.num_experts
            if args.child_kind in {"routed", "qwen-transfer-sparse"}
            else 1.0
        ),
        "hard_route_dispatch": (
            f"selected-token-only:{args.dispatch_mode}"
            if args.child_kind in {"routed", "qwen-transfer-sparse"}
            else "single-child"
        ),
        "dispatch_mode": args.dispatch_mode,
        "child_internal_norm": (
            not args.child_no_norm
            if args.child_kind == "routed" else None
        ),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "train_batches": args.train_batches,
        "eval_batches": args.eval_batches,
        "distillation_steps_per_child": args.steps,
        "hard_train_steps_per_child": args.hard_train_steps,
        "router_supervision_steps_per_child": args.router_supervision_steps,
        "joint_distillation_steps": args.joint_steps,
        "joint_learning_rate": args.joint_learning_rate,
        "joint_temperature": args.joint_temperature,
        "teacher_ce": teacher_ce,
        "child_train_history": histories,
        "router_train_history": router_histories,
        "joint_train_history": joint_history,
        "child_local_eval_mse": local_eval_mse,
        "parent_scalar_params_each": parent_params,
        "child_scalar_params_each": child_params,
        "child_parameter_fraction_each": [
            child_count / max(parent_count, 1)
            for child_count, parent_count in zip(child_params, parent_params)
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
        ),
        default="routed",
    )
    parser.add_argument("--calibration-rank", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument("--routing-temperature", type=float, default=1.0)
    parser.add_argument(
        "--dispatch-mode", choices=("grouped", "packed", "token-loop"),
        default="grouped",
    )
    parser.add_argument("--child-no-norm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--hard-train-steps", type=int, default=0,
        help="final child-training steps using the hard top-k route",
    )
    parser.add_argument(
        "--router-supervision-steps", type=int, default=0,
        help="steps distilling frozen expert contribution importance into the router",
    )
    parser.add_argument("--joint-steps", type=int, default=0)
    parser.add_argument("--joint-learning-rate", type=float, default=1e-4)
    parser.add_argument("--joint-temperature", type=float, default=2.0)
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
