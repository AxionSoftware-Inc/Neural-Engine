"""Oracle screen for conditional active width in a Qwen FFN.

The experiment asks whether token-dependent capacity is useful before a
deployable difficulty router is built. Several compact attention-free Qwen
SwiGLU children are distilled independently. An oracle chooses the width from
the teacher's per-token reconstruction error plus a width penalty, then the
selected child alone is executed in the hard wrapper. The teacher-derived
route is intentionally an upper bound and is never presented as deployable.
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
from benchmark_qwen_teacher_basis_swiglu import TeacherBasisSwiGLU, load_text_file, train_child


class OracleAdaptiveWidthChild(nn.Module):
    """Execute only the child width selected for each token."""

    def __init__(self, children: list[nn.Module], route: torch.Tensor) -> None:
        super().__init__()
        if not children:
            raise ValueError("at least one child is required")
        self.children_bank = nn.ModuleList(children)
        self.register_buffer("route", route.to(dtype=torch.long), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[:-1] != self.route.shape:
            raise ValueError("oracle route shape does not match hidden states")
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_route = self.route.reshape(-1).to(device=hidden_states.device)
        flat_output = torch.zeros_like(flat_hidden)
        for child_index, child in enumerate(self.children_bank):
            token_ids = torch.where(flat_route == child_index)[0]
            if token_ids.numel() == 0:
                continue
            flat_output.index_copy_(0, token_ids, child(flat_hidden[token_ids]))
        return flat_output.reshape_as(hidden_states)


def parse_widths(value: str) -> list[int]:
    widths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not widths or any(width < 1 for width in widths):
        raise ValueError("widths must contain positive integers")
    if widths != sorted(set(widths)):
        raise ValueError("widths must be strictly increasing and unique")
    return widths


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
    widths = parse_widths(args.widths)
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
    if max(widths) > teacher_inner:
        raise ValueError("a width cannot exceed the teacher intermediate size")
    children = []
    histories = []
    local_mse = []
    for width_index, width in enumerate(widths):
        torch.manual_seed(args.seed + 1009 * (width_index + 1))
        child = TeacherBasisSwiGLU(hidden_size, width).to(device=device, dtype=dtype)
        history = train_child(
            child, train_io["input"], train_io["output"], args.steps,
            args.learning_rate, args.max_grad_norm, args.log_every,
        )
        with torch.no_grad():
            output = child(eval_io["input"])
            error = (output.float() - eval_io["output"].float()).square().mean(dim=-1)
        children.append(child.eval())
        histories.append(history)
        local_mse.append(error)
    errors = torch.stack(local_mse, dim=-1)
    width_fractions = torch.tensor(
        [width / teacher_inner for width in widths],
        device=errors.device, dtype=errors.dtype,
    )
    # Normalize the reconstruction term so lambda has an interpretable
    # range across layers and corpora.
    error_scale = errors.mean().clamp_min(1e-6)
    penalties = [float(item) for item in args.lambdas]
    variants = []
    for penalty in penalties:
        route = (errors / error_scale + penalty * width_fractions).argmin(dim=-1)
        counts = torch.bincount(route.reshape(-1), minlength=len(widths)).float()
        active_fraction = float(
            (counts * width_fractions).sum().item() / max(counts.sum().item(), 1.0)
        )
        adaptive = OracleAdaptiveWidthChild(children, route.cpu()).to(device=device)
        layer.mlp = MixedParentChild(parent, adaptive, 0.0)
        with torch.no_grad():
            logits = model(input_ids=eval_ids, use_cache=False).logits
        variant_ce = ce(logits.float(), eval_ids)
        variants.append({
            "lambda": penalty,
            "active_width_fraction": active_fraction,
            "route_fraction_by_width": {
                str(width): float(counts[index].item() / max(counts.sum().item(), 1.0))
                for index, width in enumerate(widths)
            },
            "ce": variant_ce,
            "ce_delta": variant_ce - teacher_ce,
            "top1_agreement": float((
                logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)
            ).to(torch.float32).mean()),
            "local_oracle_mse": float(errors.gather(-1, route.unsqueeze(-1)).mean().item()),
        })
    layer.mlp = parent
    parent_params = sum(parameter.numel() for parameter in parent.parameters())
    child_params = sum(
        sum(parameter.numel() for parameter in child.parameters())
        for child in children
    )
    result: dict[str, object] = {
        "experiment": "qwen_oracle_adaptive_width_compact_swiglu",
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "layer_index": args.layer_index,
        "hidden_size": hidden_size,
        "teacher_inner_size": teacher_inner,
        "child_widths": widths,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "train_batches": args.train_batches,
        "eval_batches": args.eval_batches,
        "distillation_steps_each": args.steps,
        "teacher_ce": teacher_ce,
        "child_train_history": histories,
        "parent_scalar_params": parent_params,
        "child_bank_scalar_params": child_params,
        "child_bank_storage_fraction": child_params / max(parent_params, 1),
        "variants": variants,
        "oracle_note": "route uses teacher local reconstruction error; not deployable",
        "quality_gate": {
            "criterion": "an oracle variant must reach CE delta <= 0.05 at <= 50% active width",
            "passed": any(
                item["ce_delta"] <= args.max_ce_delta
                and item["active_width_fraction"] <= args.max_active_fraction
                for item in variants
            ),
            "max_ce_delta": args.max_ce_delta,
            "max_active_fraction": args.max_active_fraction,
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
    parser.add_argument("--widths", default="192,768,1536")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--calibration-text-file", default=None)
    parser.add_argument("--eval-text-file", default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-ce-delta", type=float, default=0.05)
    parser.add_argument("--max-active-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_oracle_adaptive_width_layer26.json")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
