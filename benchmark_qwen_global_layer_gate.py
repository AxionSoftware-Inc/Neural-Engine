"""End-to-end token-adaptive layer-gate pilot for a Qwen checkpoint.

V0.47 showed that a fixed layer schedule selected from single-layer
sensitivity does not transfer between short evaluation texts.  This pilot
learns a gate from the layer input and trains it against the frozen teacher's
global next-token distribution.  A gate can keep or skip the complete FFN of
selected late layers for each token, so the active budget is allowed to emerge
from the loss instead of being forced to a fixed circuit count.

The training forward still evaluates the wrapped FFN before multiplying by
the soft gate.  Therefore this script measures the quality/route signal, not
deployable latency.  Hard inference is included to expose the gap between a
soft training result and actual conditional execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


TRAIN_TEXT = "\n".join(
    f"Global gate training example {index}: choose whether each late feed "
    f"forward layer is needed for the current token while preserving the "
    f"teacher distribution and reducing unnecessary computation."
    for index in range(512)
)
EVAL_TEXT = "\n".join(
    f"Global gate evaluation example {index}: a token-adaptive layer route "
    f"should skip harmless transformations and retain the teacher's useful "
    f"next-token prediction."
    for index in range(256, 384)
)
EVAL_TEXT_ALT = "\n".join(
    f"Independent gate audit {index}: decide which feed-forward layers are "
    f"needed by this context and preserve the original model's prediction "
    f"under a compute penalty."
    for index in range(384, 640)
)


def _token_batches(
    tokenizer,
    text: str,
    batch_size: int,
    sequence_length: int,
    batch_count: int,
    device: torch.device,
) -> list[torch.Tensor]:
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    needed = batch_size * sequence_length * batch_count
    if ids.numel() < needed:
        ids = ids.repeat((needed + ids.numel() - 1) // ids.numel())
    ids = ids[:needed].reshape(batch_count, batch_size, sequence_length)
    return [batch.to(device) for batch in ids]


def _ce(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    return float(F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ))


def _topk_kl(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    selected = student_logits.float().gather(-1, teacher_indices)
    student_log_probs = F.log_softmax(selected / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_values / temperature, dim=-1)
    return F.kl_div(
        student_log_probs.reshape(-1, selected.shape[-1]),
        teacher_probs.reshape(-1, selected.shape[-1]),
        reduction="batchmean",
    ) * temperature**2


def _metrics(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> dict[str, float]:
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


class TokenLayerGate(nn.Module):
    """Wrap one frozen FFN with a trainable token-level keep/skip gate."""

    def __init__(
        self,
        base_mlp: nn.Module,
        hidden_size: int,
        gate_hidden: int,
        initial_keep: float,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_keep < 1.0:
            raise ValueError("initial_keep must be between zero and one")
        self.base_mlp = base_mlp
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )
        # Start close to the exact teacher path.  A zero final weight makes
        # the initial decision independent of token content.
        nn.init.normal_(self.gate[0].weight, std=0.01)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.gate[-1].weight)
        initial_logit = torch.logit(torch.tensor(initial_keep))
        nn.init.constant_(self.gate[-1].bias, float(initial_logit))
        self.hard = False
        self.threshold = 0.5
        self.last_keep_fraction = 0.0
        self.last_execution_fraction = 0.0
        self.last_probability: torch.Tensor | None = None
        for parameter in self.base_mlp.parameters():
            parameter.requires_grad_(False)

    def keep_probability(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_input = hidden_states.float()
        return torch.sigmoid(self.gate(gate_input))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        keep_probability = self.keep_probability(hidden_states)
        self.last_probability = keep_probability
        self.last_keep_fraction = float(keep_probability.detach().mean())
        if self.hard:
            keep = (keep_probability >= self.threshold).to(hidden_states.dtype)
        else:
            keep = keep_probability.to(hidden_states.dtype)
        self.last_execution_fraction = float(keep.detach().mean())
        # The base MLP is intentionally evaluated in this research pilot;
        # hard deployment needs a kernel that skips it before computation.
        return self.base_mlp(hidden_states) * keep


def _teacher_cache(
    model: nn.Module,
    batches: list[torch.Tensor],
    topk: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]]:
    targets: list[tuple[torch.Tensor, torch.Tensor]] = []
    logits_cache: list[torch.Tensor] = []
    with torch.no_grad():
        for input_ids in batches:
            logits = model(input_ids=input_ids, use_cache=False).logits.float()
            values, indices = logits.topk(topk, dim=-1)
            targets.append((values.detach(), indices.detach()))
            logits_cache.append(logits.detach())
    return targets, logits_cache


def _set_mode(gates: list[TokenLayerGate], hard: bool, threshold: float) -> None:
    for gate in gates:
        gate.hard = hard
        gate.threshold = threshold


def _route_fraction(gates: list[TokenLayerGate]) -> float:
    if not gates:
        return 0.0
    return sum(gate.last_keep_fraction for gate in gates) / len(gates)


def _layer_stats(layer_indices: list[int], gates: list[TokenLayerGate]) -> dict[str, dict[str, float]]:
    return {
        str(index): {
            "soft_keep_fraction": gate.last_keep_fraction,
            "executed_keep_fraction": gate.last_execution_fraction,
        }
        for index, gate in zip(layer_indices, gates)
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_qwen_global_layer_gate.py requires optional packages: "
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
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

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
    eval_text = EVAL_TEXT if args.eval_variant == 0 else EVAL_TEXT_ALT
    eval_batches = _token_batches(
        tokenizer, eval_text, args.batch_size, args.sequence_length,
        args.eval_batches, device,
    )

    # Cache teacher targets before installing any trainable wrappers.
    teacher_train_targets, _ = _teacher_cache(model, train_batches, args.topk)
    teacher_eval_targets, teacher_eval_logits = _teacher_cache(
        model, eval_batches, args.topk
    )
    eval_ids = torch.cat(eval_batches, dim=0)
    teacher_logits = torch.cat(teacher_eval_logits, dim=0)
    teacher_ce = _ce(teacher_logits, eval_ids)

    layer_count = len(model.model.layers)
    if args.layer_indices:
        layer_indices = [
            int(value.strip()) for value in args.layer_indices.split(",")
            if value.strip()
        ]
    else:
        layer_indices = list(range(layer_count))
    if not layer_indices or any(index < 0 or index >= layer_count for index in layer_indices):
        raise ValueError("layer indices must refer to existing Qwen layers")

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gates: list[TokenLayerGate] = []
    for index in layer_indices:
        layer = model.model.layers[index]
        wrapper = TokenLayerGate(
            layer.mlp,
            hidden_size=int(model.config.hidden_size),
            gate_hidden=args.gate_hidden,
            initial_keep=args.initial_keep,
        ).to(device=device)
        layer.mlp = wrapper
        gates.append(wrapper)
    trainable = [parameter for gate in gates for parameter in gate.gate.parameters()]
    if not trainable:
        raise RuntimeError("no trainable gate parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    _set_mode(gates, hard=False, threshold=args.hard_threshold)

    history = []
    for step in range(args.steps):
        batch_index = step % len(train_batches)
        input_ids = train_batches[batch_index]
        teacher_values, teacher_indices = teacher_train_targets[batch_index]
        logits = model(input_ids=input_ids, use_cache=False).logits
        distill_loss = _topk_kl(
            logits,
            teacher_values,
            teacher_indices,
            args.temperature,
        )
        label_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        # `last_keep_fraction` is detached for reporting only; use the gate
        # probability tensors retained by the wrappers for a differentiable
        # compute penalty.
        keep_fraction = torch.stack([
            gate.last_probability.mean() for gate in gates
            if gate.last_probability is not None
        ]).mean()
        loss = (
            args.distill_weight * distill_loss
            + args.label_weight * label_loss
            + args.compute_penalty * keep_fraction
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite gate loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        history.append({
            "step": step + 1,
            "loss": float(loss.detach()),
            "distill_loss": float(distill_loss.detach()),
            "label_loss": float(label_loss.detach()),
            "soft_keep_fraction": float(keep_fraction.detach()),
        })

    results: dict[str, object] = {
        "experiment": "qwen_global_token_layer_gate",
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "layer_indices": layer_indices,
        "layer_count": layer_count,
        "hidden_size": int(model.config.hidden_size),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "train_batches": args.train_batches,
        "eval_batches": args.eval_batches,
        "eval_variant": args.eval_variant,
        "topk": args.topk,
        "compute_penalty": args.compute_penalty,
        "teacher_ce": teacher_ce,
        "history_tail": history[-10:],
    }

    for mode_name, hard in (("soft", False), ("hard", True)):
        _set_mode(gates, hard=hard, threshold=args.hard_threshold)
        with torch.no_grad():
            logits = model(input_ids=eval_ids, use_cache=False).logits
        row = {
            "mode": mode_name,
            "soft_keep_fraction": _route_fraction(gates),
            "executed_keep_fraction": float(sum(
                gate.last_execution_fraction for gate in gates
            ) / len(gates)),
            "overall_ffn_active_fraction": 1.0 - (
                len(layer_indices) / layer_count
            ) * (1.0 - float(sum(
                gate.last_execution_fraction for gate in gates
            ) / len(gates))),
            "layer_stats": _layer_stats(layer_indices, gates),
            **_metrics(logits, teacher_logits, eval_ids),
        }
        results[mode_name] = row

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layer-indices", default="22,24,26,27")
    parser.add_argument("--gate-hidden", type=int, default=64)
    parser.add_argument("--initial-keep", type=float, default=0.98)
    parser.add_argument("--hard-threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--label-weight", type=float, default=0.1)
    parser.add_argument("--compute-penalty", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--train-batches", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--eval-variant", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_global_layer_gate.json")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
