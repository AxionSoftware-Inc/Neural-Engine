"""Learned redundant circuit-basis pilot for Qwen FFN transfer.

Raw contiguous Qwen chunks are not independently droppable: V0.46's oracle
needed almost every chunk for teacher-level fidelity.  This pilot makes the
circuits trainable and exposes them to random active masks while distilling
the teacher MLP output.  It asks whether a redundant learned basis can make a
25% route useful before a learned router is introduced.

Random and contribution-oracle evaluation are both reported.  The random
route is important: if it works, the representation has become redundant; if
only the oracle works, routing remains the bottleneck.  The training path is
not a deployable runtime benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from neural_engine.pretrained_transfer import (
    SwiGLUCircuitBank,
    top_contribution_circuits,
)


TRAIN_TEXT = "\n".join(
    f"Learned basis training example {index}: redundant circuits should "
    f"reconstruct the teacher feed-forward transformation under random "
    f"active masks."
    for index in range(512)
)
EVAL_TEXT = "\n".join(
    f"Learned basis evaluation example {index}: a sparse circuit basis must "
    f"preserve the teacher output even when only a subset is executed."
    for index in range(256, 384)
)


def _token_batches(tokenizer, text, batch_size, sequence_length, count, device):
    ids = tokenizer(text, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    needed = batch_size * sequence_length * count
    if ids.numel() < needed:
        ids = ids.repeat((needed + ids.numel() - 1) // ids.numel())
    ids = ids[:needed].reshape(count, batch_size, sequence_length)
    return [batch.to(device) for batch in ids]


def _capture_mlp_io(model, input_ids, layer_indices):
    captured = {index: {"input": None, "output": None} for index in layer_indices}
    hooks = []
    for index in layer_indices:
        mlp = model.model.layers[index].mlp

        def before(_module, inputs, index=index):
            captured[index]["input"] = inputs[0].detach()

        def after(_module, _inputs, output, index=index):
            captured[index]["output"] = (
                output[0] if isinstance(output, tuple) else output
            ).detach()

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


class LearnedBasisExecution(nn.Module):
    """Execute a trainable basis bank with a fixed active-count control."""

    def __init__(self, bank, active_circuits, route_mode, seed):
        super().__init__()
        if route_mode not in {"random", "oracle"}:
            raise ValueError("route_mode must be random or oracle")
        self.bank = bank
        self.active_circuits = int(active_circuits)
        self.route_mode = route_mode
        self.seed = int(seed)

    def _random_ids(self, hidden_states):
        generator = torch.Generator(device=hidden_states.device)
        generator.manual_seed(self.seed)
        scores = torch.rand(
            *hidden_states.shape[:-1],
            self.bank.num_circuits,
            device=hidden_states.device,
            generator=generator,
        )
        return scores.topk(self.active_circuits, dim=-1).indices

    def forward(self, hidden_states):
        if self.active_circuits >= self.bank.num_circuits:
            return self.bank(hidden_states)
        if self.route_mode == "oracle":
            ids = top_contribution_circuits(
                self.bank, hidden_states, self.active_circuits
            )
        else:
            ids = self._random_ids(hidden_states)
        weights = torch.full(
            ids.shape,
            self.bank.num_circuits / self.active_circuits,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return self.bank.forward_selected(hidden_states, ids, weights)


def _random_selected(bank, hidden_states, active, generator):
    scores = torch.rand(
        *hidden_states.shape[:-1], bank.num_circuits,
        device=hidden_states.device, generator=generator,
    )
    ids = scores.topk(active, dim=-1).indices
    weights = torch.full(
        ids.shape,
        bank.num_circuits / active,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    return bank.forward_selected(hidden_states, ids, weights)


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_qwen_learned_basis.py requires optional packages: "
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
    banks = {
        index: SwiGLUCircuitBank.from_qwen_mlp(
            model.model.layers[index].mlp, args.chunk_size
        ).to(device=device, dtype=dtype)
        for index in layer_indices
    }
    num_circuits = banks[layer_indices[0]].num_circuits
    active = max(1, round(num_circuits * args.active_fraction))
    if any(bank.num_circuits != num_circuits for bank in banks.values()):
        raise ValueError("all selected layers must have the same circuit count")
    if active >= num_circuits:
        raise ValueError("active fraction must leave at least one inactive circuit")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 7001)
    history = []
    # Train one bank at a time.  Jointly retaining six bank backward graphs
    # pushed the 12-GB pilot GPU into severe memory pressure at 50% active;
    # sequential local distillation has the same objective and is reproducible.
    final_train_loss = {}
    for index, bank in banks.items():
        parameters = list(bank.parameters())
        optimizer = torch.optim.AdamW(
            parameters, lr=args.learning_rate, weight_decay=args.weight_decay
        )
        last_loss = float("inf")
        for step in range(args.steps):
            prediction = _random_selected(
                bank, train_io[index]["input"], active, generator
            )
            sparse_loss = F.mse_loss(
                prediction.float(), train_io[index]["output"].float()
            )
            if args.full_reconstruction_weight:
                full_loss = F.mse_loss(
                    bank(train_io[index]["input"]).float(),
                    train_io[index]["output"].float(),
                )
            else:
                full_loss = sparse_loss.new_zeros(())
            loss = sparse_loss + args.full_reconstruction_weight * full_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite basis loss at layer {index}, step {step + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            last_loss = float(loss.detach())
            if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
                history.append({
                    "layer_index": index,
                    "step": step + 1,
                    "loss": last_loss,
                    "sparse_loss": float(sparse_loss.detach()),
                    "full_loss": float(full_loss.detach()),
                })
        final_train_loss[str(index)] = last_loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    levels = []
    for route in ("random", "oracle"):
        replacements = {
            index: LearnedBasisExecution(
                bank, active, route, args.seed + index * 101
            ).to(device)
            for index, bank in banks.items()
        }
        for index, replacement in replacements.items():
            model.model.layers[index].mlp = replacement
        with torch.no_grad():
            logits = model(input_ids=eval_ids, use_cache=False).logits
        levels.append({
            "route": route,
            "active_circuits": active,
            "active_fraction": float(active / num_circuits),
            **_metrics(logits, teacher_logits, eval_ids),
        })

    local_eval = {}
    for index, bank in banks.items():
        hidden = eval_io[index]["input"]
        target = eval_io[index]["output"]
        generator.manual_seed(args.seed + index * 5003)
        with torch.no_grad():
            random_output = _random_selected(bank, hidden, active, generator)
            oracle_ids = top_contribution_circuits(bank, hidden, active)
            oracle_weights = torch.full(
                oracle_ids.shape,
                num_circuits / active,
                device=device,
                dtype=hidden.dtype,
            )
            oracle_output = bank.forward_selected(hidden, oracle_ids, oracle_weights)
        local_eval[str(index)] = {
            "random_output_mse": float(F.mse_loss(random_output.float(), target.float())),
            "oracle_output_mse": float(F.mse_loss(oracle_output.float(), target.float())),
            "teacher_output_mse_zero": float(F.mse_loss(
                target.float(), torch.zeros_like(target).float()
            )),
        }

    hidden_size = int(model.config.hidden_size)
    total_params = len(layer_indices) * 3 * hidden_size * int(model.config.intermediate_size)
    active_params = len(layer_indices) * active * args.chunk_size * hidden_size * 3
    report = {
        "experiment": "qwen_learned_redundant_circuit_basis",
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
        "chunk_size": args.chunk_size,
        "circuits_per_layer": num_circuits,
        "active_circuits": active,
        "active_fraction": float(active / num_circuits),
        "steps": args.steps,
        "full_reconstruction_weight": args.full_reconstruction_weight,
        "teacher_ce": teacher_ce,
        "final_train_loss": final_train_loss,
        "local_eval": local_eval,
        "levels": levels,
        "parameter_estimate": {
            "target_bank_parameters": total_params,
            "active_bank_parameters": active_params,
            "active_target_fraction": float(active_params / max(total_params, 1)),
        },
        "train_history": history,
        "quality_gate": {
            "criterion": "random hard route full-model CE delta <= 0.02; latency requires a later kernel audit",
            "passed": bool(levels[0]["ce_delta"] <= args.max_ce_delta),
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
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--active-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--full-reconstruction-weight", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-ce-delta", type=float, default=0.02)
    parser.add_argument("--layer-indices", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_learned_basis_late6.json")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
