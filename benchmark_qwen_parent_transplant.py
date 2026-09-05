"""Minimal Transformer -> attention-free NE function-block handoff.

This is a local gate for the cross-architecture parent-transplant proposal.
One frozen Qwen FFN is distilled into a small LayerNorm/GELU register-style
function block, then the parent contribution is swept from alpha=1 to alpha=0
without changing any other Transformer layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


TRAIN_TEXT = "\n".join(
    f"Parent transfer training example {index}: an attention-free function "
    f"block should preserve the frozen feed-forward computation."
    for index in range(512)
)
EVAL_TEXT = "\n".join(
    f"Parent transfer evaluation example {index}: the child block must "
    f"retain the parent's local transformation after handoff."
    for index in range(256, 384)
)


class NEFunctionBlock(nn.Module):
    """Small attention-free local function block used as the NE child."""

    def __init__(self, hidden_size: int, inner_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.input_projection = nn.Linear(hidden_size, inner_size)
        self.output_projection = nn.Linear(inner_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(self.norm(hidden_states))
        hidden = F.gelu(hidden)
        return self.output_projection(hidden)


class MixedParentChild(nn.Module):
    def __init__(self, parent: nn.Module, child: nn.Module, alpha: float) -> None:
        super().__init__()
        self.parent = parent
        self.child = child
        self.alpha = float(alpha)

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        parent_output = self.parent(hidden_states)
        child_output = self.child(hidden_states)
        return self.alpha * parent_output + (1.0 - self.alpha) * child_output


def token_stream(tokenizer, text: str, batch_size: int, sequence_length: int,
                 device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    needed = batch_size * sequence_length
    if ids.numel() < needed:
        ids = ids.repeat((needed + ids.numel() - 1) // ids.numel())
    return ids[:needed].reshape(batch_size, sequence_length).to(device)


def capture_mlp_io(model, input_ids: torch.Tensor, layer_index: int):
    captured: dict[str, torch.Tensor] = {}
    mlp = model.model.layers[layer_index].mlp

    def before(_module, inputs):
        captured["input"] = inputs[0].detach()

    def after(_module, _inputs, output):
        captured["output"] = (output[0] if isinstance(output, tuple) else output).detach()

    before_hook = mlp.register_forward_pre_hook(before)
    after_hook = mlp.register_forward_hook(after)
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    before_hook.remove()
    after_hook.remove()
    if set(captured) != {"input", "output"}:
        raise RuntimeError("failed to capture Qwen MLP input/output")
    return captured


def ce(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    return float(F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ))


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
        args.tokenizer or args.model, local_files_only=args.local_files_only
    )
    train_ids = token_stream(tokenizer, TRAIN_TEXT, args.batch_size, args.sequence_length, device)
    eval_ids = token_stream(tokenizer, EVAL_TEXT, args.batch_size, args.sequence_length, device)
    with torch.no_grad():
        teacher_logits = model(input_ids=eval_ids, use_cache=False).logits
    teacher_ce = ce(teacher_logits.float(), eval_ids)

    layer = model.model.layers[args.layer_index]
    parent = layer.mlp
    train_io = capture_mlp_io(model, train_ids, args.layer_index)
    eval_io = capture_mlp_io(model, eval_ids, args.layer_index)
    hidden_size = int(model.config.hidden_size)
    child = NEFunctionBlock(hidden_size, args.inner_size).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(child.parameters(), lr=args.learning_rate)
    history = []
    for step in range(1, args.steps + 1):
        prediction = child(train_io["input"])
        loss = F.mse_loss(prediction.float(), train_io["output"].float())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite child loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(child.parameters(), args.max_grad_norm)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            history.append({"step": step, "loss": float(loss.detach().cpu())})

    with torch.no_grad():
        child_eval = child(eval_io["input"])
        local_mse = float(F.mse_loss(child_eval.float(), eval_io["output"].float()))
        parent_energy = float(F.mse_loss(
            eval_io["output"].float(), torch.zeros_like(eval_io["output"]).float()
        ))

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
    result = {
        "experiment": "qwen_transformer_to_ne_function_block_parent_transplant",
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "layer_index": args.layer_index,
        "hidden_size": hidden_size,
        "child_inner_size": args.inner_size,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "distillation_steps": args.steps,
        "teacher_ce": teacher_ce,
        "child_train_history": history,
        "child_local_eval_mse": local_mse,
        "parent_output_energy": parent_energy,
        "parent_scalar_params": parent_params,
        "child_scalar_params": child_params,
        "child_parameter_fraction": child_params / max(parent_params, 1),
        "morph": morph,
        "quality_gate": {
            "criterion": "alpha=0 local child CE delta <= 0.05 and no non-finite output",
            "passed": bool(
                morph[-1]["ce_delta"] <= args.max_ce_delta
                and torch.isfinite(torch.tensor(morph[-1]["ce"]))
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
    parser.add_argument("--inner-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-ce-delta", type=float, default=0.05)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_parent_transplant_layer26.json")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
