"""Multi-batch two-layer Qwen -> attention-free NE parent transplant gate."""

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
    NEFunctionBlock,
    capture_mlp_io,
    ce,
    token_stream,
)


def capture_batches(
    model: nn.Module,
    tokenizer,
    text: str,
    batch_size: int,
    sequence_length: int,
    num_batches: int,
    device: torch.device,
    layer_index: int,
) -> list[dict[str, torch.Tensor]]:
    input_ids = token_stream(
        tokenizer,
        text,
        batch_size,
        sequence_length * num_batches,
        device,
    ).reshape(num_batches, batch_size, sequence_length)
    batches = []
    for ids in input_ids:
        captured = capture_mlp_io(model, ids, layer_index)
        batches.append({key: value.cpu() for key, value in captured.items()})
    return batches


def train_child(
    child: nn.Module,
    io_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    learning_rate: float,
    max_grad_norm: float,
    log_every: int,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(child.parameters(), lr=learning_rate)
    history = []
    child.train()
    for step in range(1, steps + 1):
        batch = io_batches[(step - 1) % len(io_batches)]
        inputs = batch["input"].to(device=device, dtype=dtype)
        targets = batch["output"].to(device=device, dtype=dtype)
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
    child.eval()
    return history


def evaluate_variants(
    model: nn.Module,
    batches: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    teacher_ce: float,
) -> list[dict[str, float | str]]:
    results = []
    for batch_index, ids in enumerate(batches):
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits.float()
        target = teacher_logits[batch_index].float()
        results.append({
            "batch": batch_index,
            "ce": ce(logits, ids),
            "ce_delta": ce(logits, ids) - teacher_ce,
            "logit_mse": float(F.mse_loss(logits, target)),
            "top1_agreement": float((
                logits.argmax(dim=-1) == target.argmax(dim=-1)
            ).to(torch.float32).mean()),
        })
    return results


def evaluate_current(
    model: nn.Module,
    batches: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    teacher_ce: float,
    name: str,
) -> dict[str, object]:
    per_batch = evaluate_variants(
        model, batches, teacher_logits, teacher_ce,
    )
    ce_mean = sum(float(item["ce"]) for item in per_batch) / len(per_batch)
    return {
        "variant": name,
        "ce": ce_mean,
        "ce_delta": ce_mean - teacher_ce,
        "logit_mse": sum(float(item["logit_mse"]) for item in per_batch) / len(per_batch),
        "top1_agreement": sum(float(item["top1_agreement"]) for item in per_batch) / len(per_batch),
        "per_batch": per_batch,
    }


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
        args.sequence_length * args.eval_batches,
        device,
    ).reshape(args.eval_batches, args.batch_size, args.sequence_length)
    eval_ids = token_stream(
        tokenizer, EVAL_TEXT, args.batch_size,
        args.sequence_length * args.eval_batches,
        device,
    ).reshape(args.eval_batches, args.batch_size, args.sequence_length)
    with torch.no_grad():
        teacher_logits = [model(input_ids=ids, use_cache=False).logits.detach() for ids in eval_ids]
    teacher_ce = sum(ce(logits.float(), ids) for logits, ids in zip(teacher_logits, eval_ids)) / len(eval_ids)

    layer25 = model.model.layers[args.first_layer]
    layer26 = model.model.layers[args.second_layer]
    parent25 = layer25.mlp
    parent26 = layer26.mlp
    hidden_size = int(model.config.hidden_size)

    io25 = capture_batches(
        model, tokenizer, TRAIN_TEXT, args.batch_size, args.sequence_length,
        args.train_batches, device, args.first_layer,
    )
    eval_io25 = capture_batches(
        model, tokenizer, EVAL_TEXT, args.batch_size, args.sequence_length,
        args.eval_batches, device, args.first_layer,
    )
    child25 = NEFunctionBlock(hidden_size, args.inner_size).to(device=device, dtype=dtype)
    history25 = train_child(
        child25, io25, device, dtype, args.steps, args.learning_rate,
        args.max_grad_norm, args.log_every,
    )

    # Train the second child after the first handoff, so the second local
    # function sees the upstream representation it must actually consume.
    layer25.mlp = child25
    io26 = capture_batches(
        model, tokenizer, TRAIN_TEXT, args.batch_size, args.sequence_length,
        args.train_batches, device, args.second_layer,
    )
    child26 = NEFunctionBlock(hidden_size, args.inner_size).to(device=device, dtype=dtype)
    history26 = train_child(
        child26, io26, device, dtype, args.steps, args.learning_rate,
        args.max_grad_norm, args.log_every,
    )

    eval_io26 = capture_batches(
        model, tokenizer, EVAL_TEXT, args.batch_size, args.sequence_length,
        args.eval_batches, device, args.second_layer,
    )
    with torch.no_grad():
        local_mse25 = sum(
            F.mse_loss(
                child25(batch["input"].to(device=device, dtype=dtype)).float(),
                batch["output"].to(device=device).float(),
            ).item()
            for batch in eval_io25
        ) / args.eval_batches
        local_mse26 = sum(
            F.mse_loss(
                child26(batch["input"].to(device=device, dtype=dtype)).float(),
                batch["output"].to(device=device).float(),
            ).item()
            for batch in eval_io26
        ) / args.eval_batches

    # Restore the original parent pair before the controlled final sweeps.
    layer25.mlp = parent25
    layer26.mlp = parent26
    variants = []
    for alpha in args.alphas:
        layer25.mlp = MixedParentChild(parent25, child25, alpha)
        layer26.mlp = MixedParentChild(parent26, child26, alpha)
        name = f"shared_alpha_{alpha:g}"
        variants.append(evaluate_current(model, list(eval_ids), teacher_logits, teacher_ce, name))
    layer25.mlp = parent25
    layer26.mlp = MixedParentChild(parent26, child26, 0.0)
    variants.append(evaluate_current(model, list(eval_ids), teacher_logits, teacher_ce, "second_child_only"))
    layer25.mlp = MixedParentChild(parent25, child25, 0.0)
    layer26.mlp = parent26
    variants.append(evaluate_current(model, list(eval_ids), teacher_logits, teacher_ce, "first_child_only"))
    layer25.mlp = parent25
    layer26.mlp = parent26

    both = next(item for item in variants if item["variant"] == "shared_alpha_0")
    result = {
        "experiment": "qwen_two_layer_attention_free_parent_transplant",
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "first_layer": args.first_layer,
        "second_layer": args.second_layer,
        "hidden_size": hidden_size,
        "child_inner_size": args.inner_size,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "train_batches": args.train_batches,
        "eval_batches": args.eval_batches,
        "distillation_steps_per_child": args.steps,
        "teacher_ce": teacher_ce,
        "child25_train_history": history25,
        "child26_train_history": history26,
        "child25_local_eval_mse": local_mse25,
        "child26_local_eval_mse_after_child25": local_mse26,
        "parent_scalar_params_each": sum(parameter.numel() for parameter in parent25.parameters()),
        "child_scalar_params_each": sum(parameter.numel() for parameter in child25.parameters()),
        "child_parameter_fraction_each": sum(parameter.numel() for parameter in child25.parameters()) / sum(parameter.numel() for parameter in parent25.parameters()),
        "variants": variants,
        "quality_gate": {
            "criterion": "both child-only alpha=0 CE delta <= 0.05 and all outputs finite",
            "passed": bool(
                float(both["ce_delta"]) <= args.max_ce_delta
                and torch.isfinite(torch.tensor(float(both["ce"])))
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
    parser.add_argument("--first-layer", type=int, default=25)
    parser.add_argument("--second-layer", type=int, default=26)
    parser.add_argument("--inner-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-ce-delta", type=float, default=0.05)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_two_layer_transplant.json")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
