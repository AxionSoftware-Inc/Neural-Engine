"""Level-1 progressive attention-transplant pilot on Qwen3-0.6B.

One middle Qwen attention block is distilled into an attention-free recurrent
sequence mixer. The teacher remains frozen and all other Qwen layers are
unchanged. This is deliberately a local-block gate for proposal 12, not a
claim that one small student block replaces Transformer attention generally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


TRAIN_TEXT = "\n".join(
    f"Training sequence {index}: a replacement memory mixer should preserve "
    f"the sequence transformation performed by the original attention block."
    for index in range(192)
)
EVAL_TEXT = "\n".join(
    f"Evaluation sequence {index}: replacing attention must retain context "
    f"information and the teacher's next-token behavior."
    for index in range(96, 160)
)


class RecurrentMixingBlock(nn.Module):
    """Attention-free sequence mixer with a bounded persistent hidden state."""

    def __init__(self, hidden_size: int, memory_size: int):
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, memory_size)
        self.memory = nn.GRU(memory_size, memory_size, batch_first=True)
        self.output_projection = nn.Linear(memory_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values, _ = self.memory(self.input_projection(hidden_states))
        return self.output_projection(values)


class AttentionReplacement(nn.Module):
    def __init__(self, mixer: RecurrentMixingBlock):
        super().__init__()
        self.mixer = mixer

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        return self.mixer(hidden_states), None


class ZeroAttention(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        return torch.zeros_like(hidden_states), None


def _token_stream(tokenizer, text: str, batch_size: int, sequence_length: int, device: str):
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    needed = batch_size * sequence_length
    if ids.numel() < needed:
        ids = ids.repeat((needed + ids.numel() - 1) // ids.numel())
    return ids[:needed].reshape(batch_size, sequence_length).to(device)


def _capture_attention_io(model, input_ids: torch.Tensor, layer_index: int):
    layer = model.model.layers[layer_index]
    captured: dict[str, torch.Tensor] = {}

    def before(_module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            raise RuntimeError("could not capture attention input")
        captured["input"] = hidden_states.detach()

    def after(_module, _args, _kwargs, output):
        captured["output"] = (output[0] if isinstance(output, tuple) else output).detach()

    before_hook = layer.self_attn.register_forward_pre_hook(before, with_kwargs=True)
    after_hook = layer.self_attn.register_forward_hook(after, with_kwargs=True)
    with torch.no_grad():
        model(input_ids=input_ids, use_cache=False)
    before_hook.remove()
    after_hook.remove()
    if set(captured) != {"input", "output"}:
        raise RuntimeError("failed to capture both attention tensors")
    return captured["input"], captured["output"]


def _ce(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    return float(F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ))


def _full_metrics(logits: torch.Tensor, teacher_logits: torch.Tensor, input_ids: torch.Tensor):
    difference = (logits - teacher_logits).float()
    return {
        "ce": _ce(logits.float(), input_ids),
        "ce_delta": _ce(logits.float(), input_ids) - _ce(teacher_logits.float(), input_ids),
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
            "benchmark_qwen_attention_transplant.py requires optional packages: "
            "pip install -r requirements-transfer.txt"
        ) from exc

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
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
    train_ids = _token_stream(tokenizer, TRAIN_TEXT, args.batch_size, args.sequence_length, device)
    eval_ids = _token_stream(tokenizer, EVAL_TEXT, args.batch_size, args.sequence_length, device)
    with torch.no_grad():
        teacher_logits = model(input_ids=eval_ids, use_cache=False).logits
    if args.layer_indices:
        layer_indices = [int(value) for value in args.layer_indices.split(",") if value.strip()]
    else:
        layer_indices = [args.layer_index]
    if not layer_indices or any(index < 0 or index >= len(model.model.layers) for index in layer_indices):
        raise ValueError("layer indices must refer to existing Qwen layers")

    captured = [
        (
            index,
            _capture_attention_io(model, train_ids, index),
            _capture_attention_io(model, eval_ids, index),
        )
        for index in layer_indices
    ]
    mixers = []
    train_losses = []
    local_metrics = []
    for index, train_pair, eval_pair in captured:
        train_input, train_target = train_pair
        eval_input, eval_target = eval_pair
        mixer = RecurrentMixingBlock(int(model.config.hidden_size), args.memory_size).to(
            device=device, dtype=dtype
        )
        optimizer = torch.optim.AdamW(mixer.parameters(), lr=args.learning_rate)
        last_loss = 0.0
        for _ in range(args.steps):
            prediction = mixer(train_input)
            loss = F.mse_loss(prediction.float(), train_target.float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss)
        with torch.no_grad():
            teacher_energy = float(F.mse_loss(
                eval_target.float(), torch.zeros_like(eval_target).float()
            ))
            student_mse = float(F.mse_loss(mixer(eval_input).float(), eval_target.float()))
        mixers.append(mixer)
        train_losses.append(last_loss)
        local_metrics.append({
            "layer_index": index,
            "teacher_output_energy": teacher_energy,
            "student_mse": student_mse,
        })

    for index, mixer in zip(layer_indices, mixers):
        model.model.layers[index].self_attn = AttentionReplacement(mixer)
    with torch.no_grad():
        student_logits = model(input_ids=eval_ids, use_cache=False).logits
    student_metrics = _full_metrics(student_logits, teacher_logits, eval_ids)

    for index in layer_indices:
        model.model.layers[index].self_attn = ZeroAttention()
    with torch.no_grad():
        zero_logits = model(input_ids=eval_ids, use_cache=False).logits
    zero_metrics = _full_metrics(zero_logits, teacher_logits, eval_ids)
    return {
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": device,
        "dtype": args.dtype,
        "seed": args.seed,
        "layer_index": args.layer_index,
        "layer_indices": layer_indices,
        "layer_count": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "memory_size": args.memory_size,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "distillation_steps": args.steps,
        "distillation_train_mse": train_losses[0] if len(train_losses) == 1 else train_losses,
        "local_eval": local_metrics,
        "teacher": {"ce": _ce(teacher_logits.float(), eval_ids)},
        "student_recurrent_replacement": student_metrics,
        "zero_attention_control": zero_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layer-index", type=int, default=14)
    parser.add_argument(
        "--layer-indices",
        default=None,
        help="comma-separated progressive replacement layers; overrides --layer-index",
    )
    parser.add_argument("--memory-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
