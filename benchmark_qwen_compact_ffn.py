"""Structural compact-FFN transfer pilot for a real Qwen checkpoint.

V0.46's adaptive oracle shows that contiguous Qwen neuron chunks are not
independent enough for aggressive omission.  This experiment tests a
different representation: distill each selected SwiGLU FFN into a smaller
nonlinear latent width, rather than dropping contiguous chunks.  Attention and
all non-selected layers remain unchanged.

The local distillation phase uses frozen teacher MLP input/output pairs.  The
full-model evaluation is the actual quality check; local MSE alone is not a
success criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


EVAL_TEXT = "\n".join(
    f"Compact FFN evaluation example {index}: a smaller nonlinear circuit "
    f"should preserve the teacher transformation and next-token behavior."
    for index in range(256)
)
TRAIN_TEXT = "\n".join(
    f"Compact FFN training example {index}: distillation learns a latent "
    f"feed-forward circuit that retains useful context transformations."
    for index in range(512)
)


class CompactSwiGLU(nn.Module):
    """A Qwen-shaped gated FFN with a smaller intermediate latent width."""

    def __init__(self, hidden_size: int, latent_size: int):
        super().__init__()
        if hidden_size < 1 or latent_size < 1:
            raise ValueError("hidden_size and latent_size must be positive")
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.gate_proj = nn.Linear(hidden_size, latent_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, latent_size, bias=False)
        self.down_proj = nn.Linear(latent_size, hidden_size, bias=False)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.normal_(self.down_proj.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


def _token_batches(tokenizer, text, batch_size, sequence_length, count, device):
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    needed = batch_size * sequence_length * count
    if ids.numel() < needed:
        ids = ids.repeat((needed + ids.numel() - 1) // ids.numel())
    ids = ids[:needed].reshape(count, batch_size, sequence_length)
    return [batch.to(device) for batch in ids]


def _capture_mlp_io(model, input_ids, layer_indices):
    captured = {
        index: {"input": None, "output": None} for index in layer_indices
    }
    hooks = []
    for index in layer_indices:
        mlp = model.model.layers[index].mlp

        def before(_module, inputs, index=index):
            captured[index]["input"] = inputs[0].detach()

        def after(_module, _inputs, output, index=index):
            value = output[0] if isinstance(output, tuple) else output
            captured[index]["output"] = value.detach()

        hooks.append(mlp.register_forward_pre_hook(before))
        hooks.append(mlp.register_forward_hook(after))
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    for hook in hooks:
        hook.remove()
    for index in layer_indices:
        if captured[index]["input"] is None or captured[index]["output"] is None:
            raise RuntimeError(f"failed to capture MLP IO for layer {index}")
    return captured


def _ce(logits, input_ids):
    return float(F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ))


def _metrics(logits, teacher_logits, input_ids):
    difference = logits.float() - teacher_logits.float()
    ce = _ce(logits.float(), input_ids)
    teacher_ce = _ce(teacher_logits.float(), input_ids)
    return {
        "ce": ce,
        "ce_delta": ce - teacher_ce,
        "logit_mse": float(F.mse_loss(logits.float(), teacher_logits.float())),
        "max_abs_logit_error": float(difference.abs().max()),
        "top1_agreement": float((
            logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)
        ).to(torch.float32).mean()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_qwen_compact_ffn.py requires optional packages: "
            "pip install -r requirements-transfer.txt"
        ) from exc

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
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
    )
    train_batches = _token_batches(
        tokenizer, TRAIN_TEXT, args.batch_size, args.sequence_length,
        args.train_batches, device,
    )
    eval_batches = _token_batches(
        tokenizer, EVAL_TEXT, args.batch_size, args.sequence_length,
        args.eval_batches, device,
    )
    train_ids = torch.cat(train_batches, dim=0)
    eval_ids = torch.cat(eval_batches, dim=0)
    with torch.no_grad():
        teacher_logits = model(input_ids=eval_ids, use_cache=False).logits
    teacher_ce = _ce(teacher_logits.float(), eval_ids)

    layer_count = len(model.model.layers)
    if args.layer_indices:
        layer_indices = [int(value.strip()) for value in args.layer_indices.split(",") if value.strip()]
    else:
        layer_indices = list(range(layer_count))
    if not layer_indices or any(index < 0 or index >= layer_count for index in layer_indices):
        raise ValueError("layer indices must refer to existing Qwen layers")

    train_io = _capture_mlp_io(model, train_ids, layer_indices)
    eval_io = _capture_mlp_io(model, eval_ids, layer_indices)
    hidden_size = int(model.config.hidden_size)
    students = {
        index: CompactSwiGLU(hidden_size, args.latent_size).to(device=device, dtype=dtype)
        for index in layer_indices
    }
    optimizer = torch.optim.AdamW(
        [parameter for student in students.values() for parameter in student.parameters()],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    model_mse = float("inf")
    for step in range(args.steps):
        losses = []
        for index, student in students.items():
            prediction = student(train_io[index]["input"])
            losses.append(F.mse_loss(prediction.float(), train_io[index]["output"].float()))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite compact FFN loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for student in students.values() for parameter in student.parameters()],
            args.max_grad_norm,
        )
        optimizer.step()
        model_mse = float(loss.detach())
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            history.append({"step": step + 1, "local_mse": model_mse})

    for index, student in students.items():
        model.model.layers[index].mlp = student
    with torch.no_grad():
        student_logits = model(input_ids=eval_ids, use_cache=False).logits
    metrics = _metrics(student_logits, teacher_logits, eval_ids)
    local_eval = {
        str(index): {
            "teacher_output_mse_zero": float(F.mse_loss(
                eval_io[index]["output"].float(),
                torch.zeros_like(eval_io[index]["output"]).float(),
            )),
            "student_output_mse": float(F.mse_loss(
                students[index](eval_io[index]["input"]).float(),
                eval_io[index]["output"].float(),
            )),
        }
        for index in layer_indices
    }
    full_ffn_params = len(layer_indices) * 3 * hidden_size * int(model.config.intermediate_size)
    compact_params = len(layer_indices) * 3 * hidden_size * args.latent_size
    report = {
        "experiment": "qwen_compact_nonlinear_ffn",
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "train_batches": args.train_batches,
        "eval_batches": args.eval_batches,
        "layer_indices": layer_indices,
        "layer_count": layer_count,
        "hidden_size": hidden_size,
        "teacher_intermediate_size": int(model.config.intermediate_size),
        "latent_size": args.latent_size,
        "latent_parameter_fraction": float(compact_params / max(full_ffn_params, 1)),
        "steps": args.steps,
        "teacher_ce": teacher_ce,
        "final_local_train_mse": model_mse,
        "local_eval": local_eval,
        "student": metrics,
        "train_history": history,
        "quality_gate": {
            "criterion": "full-model CE delta <= 0.02; latency is measured in a follow-up kernel audit",
            "passed": bool(metrics["ce_delta"] <= args.max_ce_delta),
            "max_ce_delta": args.max_ce_delta,
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--latent-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-ce-delta", type=float, default=0.02)
    parser.add_argument("--layer-indices", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_compact_ffn_all28_latent768.json")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
