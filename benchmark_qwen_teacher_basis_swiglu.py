"""Teacher-derived functional-basis compression for one Qwen FFN.

This is a focused gate for a representation change, not another router test.
The child keeps Qwen's attention-free SwiGLU algebra but reduces the
intermediate width.  Its initial coordinates are derived from the teacher's
gate/value activation covariance on calibration tokens, rather than from
random weights or a contiguous neuron slice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from benchmark_qwen_parent_transplant import (
    EVAL_TEXT,
    TRAIN_TEXT,
    MixedParentChild,
    capture_mlp_io,
    ce,
    token_stream,
)


class TeacherBasisSwiGLU(nn.Module):
    """Compact Qwen-like SwiGLU initialized in a teacher-derived basis."""

    def __init__(self, hidden_size: int, inner_size: int) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, inner_size, bias=False)
        self.output_projection = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output_projection(
            F.silu(self.gate_projection(hidden_states))
            * self.value_projection(hidden_states)
        )


@torch.no_grad()
def activation_basis(
    parent: nn.Module,
    inputs: torch.Tensor,
    inner_size: int,
) -> torch.Tensor:
    """Return top activation directions in teacher-neuron coordinates.

    The same orthogonal coordinates are used for gate and value channels so
    the projected child still has a valid elementwise gated product.  Scaling
    each channel before the covariance prevents raw gate/value magnitudes
    from selecting the basis by units alone.
    """
    gate = parent.gate_proj(inputs).float().reshape(-1, parent.gate_proj.out_features)
    value = parent.up_proj(inputs).float().reshape(-1, parent.up_proj.out_features)
    gate = gate - gate.mean(dim=0, keepdim=True)
    value = value - value.mean(dim=0, keepdim=True)
    gate = gate / gate.std(dim=0, unbiased=False).clamp_min(1e-5)
    value = value / value.std(dim=0, unbiased=False).clamp_min(1e-5)
    covariance = gate.transpose(0, 1) @ gate
    covariance.add_(value.transpose(0, 1) @ value)
    covariance /= max(gate.shape[0], 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    del eigenvalues
    return eigenvectors[:, -inner_size:]


@torch.no_grad()
def initialize_teacher_basis(
    child: TeacherBasisSwiGLU,
    parent: nn.Module,
    inputs: torch.Tensor,
    basis: torch.Tensor | None = None,
) -> dict[str, float]:
    """Project frozen Qwen weights into the activation-derived coordinates."""
    teacher_inner = int(parent.gate_proj.out_features)
    child_inner = int(child.gate_projection.out_features)
    if child_inner > teacher_inner:
        raise ValueError("child inner size cannot exceed teacher inner size")
    if basis is None:
        basis = activation_basis(parent, inputs, child_inner)
    if basis.shape != (teacher_inner, child_inner):
        raise ValueError("basis has an unexpected shape")
    gate = parent.gate_proj.weight.detach().float()
    value = parent.up_proj.weight.detach().float()
    down = parent.down_proj.weight.detach().float()
    child.gate_projection.weight.copy_((basis.transpose(0, 1) @ gate).to(child.gate_projection.weight))
    child.value_projection.weight.copy_((basis.transpose(0, 1) @ value).to(child.value_projection.weight))
    child.output_projection.weight.copy_((down @ basis).to(child.output_projection.weight))
    retained = float(torch.trace(basis.transpose(0, 1) @ basis).item() / teacher_inner)
    return {
        "mode": "teacher-derived-activation-covariance",
        "teacher_inner_size": float(teacher_inner),
        "child_inner_size": float(child_inner),
        "basis_rank_fraction": float(child_inner / teacher_inner),
        "basis_orthogonality_error": float(
            (basis.transpose(0, 1) @ basis - torch.eye(child_inner, device=basis.device))
            .abs().max().item()
        ),
        "basis_trace_fraction": retained,
    }


def train_child(
    child: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    log_every: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(child.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    child.train()
    for step in range(1, steps + 1):
        prediction = child(inputs)
        loss = F.mse_loss(prediction.float(), targets.float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite child loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(child.parameters(), max_grad_norm)
        optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})
    return history


def load_text_file(path: str | None, default: str, label: str) -> str:
    if path is None:
        return default
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"{label} text file is empty: {path}")
    return text


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install requirements-transfer.txt first") from exc

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
        args.model, dtype=dtype, trust_remote_code=False,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, local_files_only=args.local_files_only,
    )
    calibration_text = load_text_file(
        args.calibration_text_file, TRAIN_TEXT, "calibration",
    )
    eval_text = load_text_file(args.eval_text_file, EVAL_TEXT, "evaluation")
    train_ids = token_stream(
        tokenizer, calibration_text, args.batch_size,
        args.sequence_length * args.train_batches, device,
    ).reshape(-1, args.sequence_length)
    eval_ids = token_stream(
        tokenizer, eval_text, args.batch_size,
        args.sequence_length * args.eval_batches, device,
    ).reshape(-1, args.sequence_length)
    with torch.no_grad():
        teacher_logits = model(input_ids=eval_ids, use_cache=False).logits
    teacher_ce = ce(teacher_logits.float(), eval_ids)
    layer = model.model.layers[args.layer_index]
    parent = layer.mlp
    train_io = capture_mlp_io(model, train_ids, args.layer_index)
    eval_io = capture_mlp_io(model, eval_ids, args.layer_index)
    hidden_size = int(model.config.hidden_size)
    teacher_inner = int(parent.gate_proj.out_features)
    if not 1 <= args.inner_size <= teacher_inner:
        raise ValueError("inner-size must be within 1..teacher intermediate size")
    child = TeacherBasisSwiGLU(hidden_size, args.inner_size).to(device=device, dtype=dtype)
    if args.initialization == "teacher-basis":
        basis_stats = initialize_teacher_basis(
            child, parent, train_io["input"],
        )
    else:
        basis_stats = {
            "mode": "random-control",
            "teacher_inner_size": float(teacher_inner),
            "child_inner_size": float(args.inner_size),
            "basis_rank_fraction": float(args.inner_size / teacher_inner),
        }
    with torch.no_grad():
        initial_eval = child(eval_io["input"])
        initial_mse = float(F.mse_loss(initial_eval.float(), eval_io["output"].float()))
    history = train_child(
        child,
        train_io["input"],
        train_io["output"],
        args.steps,
        args.learning_rate,
        args.max_grad_norm,
        args.log_every,
    )
    with torch.no_grad():
        child_eval = child(eval_io["input"])
        local_mse = float(F.mse_loss(child_eval.float(), eval_io["output"].float()))
    morph = []
    for alpha in args.alphas:
        layer.mlp = MixedParentChild(parent, child, alpha)
        with torch.no_grad():
            logits = model(input_ids=eval_ids, use_cache=False).logits
        difference = (logits - teacher_logits).float()
        morph.append({
            "alpha_parent": float(alpha),
            "alpha_child": float(1.0 - alpha),
            "ce": ce(logits.float(), eval_ids),
            "ce_delta": ce(logits.float(), eval_ids) - teacher_ce,
            "logit_mse": float(F.mse_loss(logits.float(), teacher_logits.float())),
            "max_abs_logit_error": float(difference.abs().max()),
            "top1_agreement": float((
                logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)
            ).to(torch.float32).mean()),
        })
    layer.mlp = parent
    parent_params = sum(parameter.numel() for parameter in parent.parameters())
    child_params = sum(parameter.numel() for parameter in child.parameters())
    alpha_zero = morph[-1]
    result: dict[str, object] = {
        "experiment": "qwen_teacher_derived_activation_basis_swiglu",
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "layer_index": args.layer_index,
        "hidden_size": hidden_size,
        "teacher_inner_size": teacher_inner,
        "child_inner_size": args.inner_size,
        "initialization": args.initialization,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "distillation_steps": args.steps,
        "teacher_ce": teacher_ce,
        "basis_initialization": basis_stats,
        "initial_child_local_eval_mse": initial_mse,
        "child_train_history": history,
        "child_local_eval_mse": local_mse,
        "parent_scalar_params": parent_params,
        "child_scalar_params": child_params,
        "child_parameter_fraction": child_params / max(parent_params, 1),
        "active_compute_fraction_estimate": args.inner_size / teacher_inner,
        "morph": morph,
        "quality_gate": {
            "criterion": "alpha=0 local child CE delta <= 0.05 and all outputs finite",
            "passed": bool(
                alpha_zero["ce_delta"] <= args.max_ce_delta
                and torch.isfinite(torch.tensor(alpha_zero["ce"]))
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
    parser.add_argument("--layer-index", type=int, default=26)
    parser.add_argument("--inner-size", type=int, default=192)
    parser.add_argument(
        "--initialization", choices=("teacher-basis", "random"),
        default="teacher-basis",
        help="teacher-derived activation basis or same-size random control",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--calibration-text-file", default=None)
    parser.add_argument("--eval-text-file", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-ce-delta", type=float, default=0.05)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.5, 0.0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_teacher_basis_swiglu_layer26_r192.json")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
