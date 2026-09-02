"""Teacher-distilled residual sparse FFN pilot on a real Qwen checkpoint.

The pilot addresses the failure mode recorded in V0.45: a local chunk oracle
can identify useful FFN chunks, but independently dropping chunks causes a
large global error.  The teacher stays frozen.  Selected Qwen FFN banks are
trained first with a dense soft route against teacher logits, plus an optional
small low-rank residual, and are then evaluated with hard top-k execution.

This is deliberately an experiment, not a production sparse router.  The
training route evaluates all chunks; latency is measured separately for the
hard route and the report must not count dense training as deployment cost.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from benchmark_qwen_sparse import SparseQwenMlp
from neural_engine.pretrained_transfer import (
    SwiGLUCircuitBank,
    TeacherDistilledSparseSwiGLU,
)


TRAIN_TEXT = "\n".join(
    f"Training example {index}: a distilled sparse feed-forward circuit "
    f"should preserve the teacher's useful transformation while reducing "
    f"unnecessary parameter traffic and retaining context."
    for index in range(512)
)
EVAL_TEXT = "\n".join(
    f"Evaluation example {index}: the selected circuits must preserve the "
    f"teacher next-token distribution when omitted feed-forward computation "
    f"is reconstructed by a small residual path."
    for index in range(256, 384)
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


def _metrics(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> dict[str, float]:
    difference = (logits.float() - teacher_logits.float())
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


def _topk_kl(
    student_logits: torch.Tensor,
    teacher_values: torch.Tensor,
    teacher_indices: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    selected = student_logits.float().gather(-1, teacher_indices)
    student_log_probs = F.log_softmax(selected / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_values / temperature, dim=-1)
    flat_student = student_log_probs.reshape(-1, selected.shape[-1])
    flat_teacher = teacher_probs.reshape(-1, selected.shape[-1])
    return F.kl_div(flat_student, flat_teacher, reduction="batchmean") * temperature**2


def _teacher_cache(
    model: nn.Module,
    batches: list[torch.Tensor],
    topk: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]]:
    targets: list[tuple[torch.Tensor, torch.Tensor]] = []
    logits_cache: list[torch.Tensor] = []
    # ``no_grad`` (rather than inference_mode) keeps the cached targets as
    # ordinary tensors that may safely participate in a later backward call.
    with torch.no_grad():
        for input_ids in batches:
            logits = model(input_ids=input_ids, use_cache=False).logits.float()
            values, indices = logits.topk(topk, dim=-1)
            targets.append((values.detach(), indices.detach()))
            logits_cache.append(logits.detach())
    return targets, logits_cache


def _replace_layers(model: nn.Module, replacements: dict[int, nn.Module]) -> None:
    for index, module in replacements.items():
        model.model.layers[index].mlp = module


def _forward_metrics(
    model: nn.Module,
    input_ids: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> dict[str, float]:
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits
    return _metrics(logits, teacher_logits, input_ids)


def _latency_ms(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup: int,
    repeats: int,
) -> float:
    def run_once() -> None:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False).logits

    for _ in range(warmup):
        run_once()
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    start = time.perf_counter()
    for _ in range(repeats):
        run_once()
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)
    return (time.perf_counter() - start) * 1000.0 / max(repeats, 1)


def _set_frozen_except(modules: list[TeacherDistilledSparseSwiGLU]) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    for module in modules:
        for parameter in module.router.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)
        if module.residual is not None:
            for parameter in module.residual.parameters():
                parameter.requires_grad_(True)
                trainable.append(parameter)
    return trainable


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_qwen_distilled_residual.py requires optional packages: "
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
    eval_batches = _token_batches(
        tokenizer, EVAL_TEXT, args.batch_size, args.sequence_length,
        args.eval_batches, device,
    )

    layer_count = len(model.model.layers)
    if args.layer_indices:
        layer_indices = [
            int(value.strip()) for value in args.layer_indices.split(",")
            if value.strip()
        ]
    else:
        layer_indices = list(range(layer_count))
    if not layer_indices or len(set(layer_indices)) != len(layer_indices):
        raise ValueError("layer indices must be non-empty and unique")
    if any(index < 0 or index >= layer_count for index in layer_indices):
        raise ValueError("layer indices must refer to existing Qwen layers")

    train_targets, _ = _teacher_cache(model, train_batches, args.teacher_topk)
    _, eval_teacher_cache = _teacher_cache(model, eval_batches, args.teacher_topk)
    eval_input_ids = torch.cat(eval_batches, dim=0)
    teacher_logits = torch.cat(eval_teacher_cache, dim=0)
    teacher_ce = _ce(teacher_logits, eval_input_ids)

    banks: dict[int, SwiGLUCircuitBank] = {}
    for index in layer_indices:
        banks[index] = SwiGLUCircuitBank.from_qwen_mlp(
            model.model.layers[index].mlp, args.chunk_size
        ).to(device=device)
        banks[index].eval()
    num_circuits = banks[layer_indices[0]].num_circuits
    if any(bank.num_circuits != num_circuits for bank in banks.values()):
        raise ValueError("all selected layers must have the same circuit count")
    active = max(1, round(num_circuits * args.active_fraction))
    if active >= num_circuits:
        raise ValueError("pilot active fraction must leave at least one circuit inactive")

    # Exact-bank control checks that replacement itself is not the source of
    # any subsequent sparse error.
    _replace_layers(model, {index: bank for index, bank in banks.items()})
    exact_metrics = _forward_metrics(model, eval_input_ids, teacher_logits)

    levels: list[dict[str, object]] = []
    for route in ("oracle", "random"):
        controls = {
            index: SparseQwenMlp(
                bank,
                active_circuits=active,
                route_mode=route,
                seed=args.seed + index * 1009,
            )
            for index, bank in banks.items()
        }
        _replace_layers(model, controls)
        metrics = _forward_metrics(model, eval_input_ids, teacher_logits)
        levels.append({
            "route": route,
            "active_circuits": active,
            "active_fraction": float(active / num_circuits),
            **metrics,
        })

    modules: dict[int, TeacherDistilledSparseSwiGLU] = {
        index: TeacherDistilledSparseSwiGLU(
            bank,
            active,
            router_hidden=args.router_hidden,
            residual_rank=args.residual_rank,
            temperature=args.route_temperature,
        ).to(device=device, dtype=dtype)
        for index, bank in banks.items()
    }
    _replace_layers(model, modules)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module_list = [modules[index] for index in layer_indices]
    trainable = _set_frozen_except(module_list)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    train_history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_step = 0
    model.train()
    for step in range(args.steps):
        input_ids = train_batches[step % len(train_batches)]
        teacher_values, teacher_indices = train_targets[step % len(train_targets)]
        mode = "soft" if step < args.soft_warmup else "straight_through"
        for module in module_list:
            module.set_execution_mode(mode)
        route_inputs: dict[int, torch.Tensor] = {}
        hooks = []
        if args.entropy_weight:
            for index in layer_indices:
                hooks.append(modules[index].register_forward_pre_hook(
                    lambda _module, inputs, index=index: route_inputs.__setitem__(
                        index, inputs[0].detach()
                    )
                ))
        student_logits = model(input_ids=input_ids, use_cache=False).logits
        for hook in hooks:
            hook.remove()
        distill_loss = _topk_kl(
            student_logits, teacher_values, teacher_indices, args.distill_temperature
        )
        label_loss = F.cross_entropy(
            student_logits[:, :-1].float().reshape(-1, student_logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        entropies = []
        # Recompute route probabilities on the actual MLP inputs through small
        # hooks; this avoids adding a second model forward just for a metric.
        # The route entropy regularizer is optional and can be disabled with 0.
        entropy_loss = student_logits.float().new_zeros(())
        if args.entropy_weight:
            for index, module in modules.items():
                distribution = module.route_distribution(route_inputs[index])
                entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(-1).mean()
                entropies.append(entropy)
            mean_entropy = torch.stack(entropies).mean()
            entropy_loss = (mean_entropy - torch.log(
                student_logits.new_tensor(float(active))
            )) ** 2

        loss = (
            args.distill_weight * distill_loss
            + args.label_weight * label_loss
            + args.entropy_weight * entropy_loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite training loss at step {step + 1}; "
                "reduce --learning-rate or use --dtype float32"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        loss_value = float(loss.detach())
        if loss_value < best_loss:
            best_loss = loss_value
            best_step = step + 1
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            train_history.append({
                "step": float(step + 1),
                "loss": loss_value,
                "distill_loss": float(distill_loss.detach()),
                "label_loss": float(label_loss.detach()),
                "entropy_loss": float(entropy_loss.detach()),
            })
    model.eval()

    for module in module_list:
        module.set_execution_mode("soft")
    levels.append({
        "route": "distilled_soft",
        "active_circuits": num_circuits,
        "active_fraction": 1.0,
        **_forward_metrics(model, eval_input_ids, teacher_logits),
    })
    for module in module_list:
        module.set_execution_mode("hard")
    hard_metrics = _forward_metrics(model, eval_input_ids, teacher_logits)
    levels.append({
        "route": "distilled_hard_with_residual",
        "active_circuits": active,
        "active_fraction": float(active / num_circuits),
        **hard_metrics,
    })
    for module in module_list:
        module.use_residual = False
    no_residual_metrics = _forward_metrics(model, eval_input_ids, teacher_logits)
    levels.append({
        "route": "distilled_hard_no_residual",
        "active_circuits": active,
        "active_fraction": float(active / num_circuits),
        **no_residual_metrics,
    })
    # Restore the trained residual modules before reporting latency or saving.
    for module in module_list:
        module.use_residual = True

    latency_input = eval_batches[0]
    # Measure teacher, exact bank, and the deployable hard route.  The current
    # implementation is a correctness pilot; kernels may make sparse slower.
    for index in layer_indices:
        model.model.layers[index].mlp = modules[index]
    for module in module_list:
        module.set_execution_mode("hard")
    hard_latency = _latency_ms(model, latency_input, args.latency_warmup, args.latency_repeats)
    for index, bank in banks.items():
        model.model.layers[index].mlp = bank
    exact_latency = _latency_ms(model, latency_input, args.latency_warmup, args.latency_repeats)
    # Keep the report honest: the residual-disabled module is still the final
    # object in the model, so latency reflects sparse circuit execution only.

    total_bank_params = sum(
        int(bank.parameter_report()["total_parameters"]) for bank in banks.values()
    )
    active_bank_params = len(layer_indices) * active * args.chunk_size * int(model.config.hidden_size) * 3
    residual_params = len(layer_indices) * (
        2 * int(model.config.hidden_size) * args.residual_rank
        if args.residual_rank > 0 else 0
    )
    report: dict[str, object] = {
        "experiment": "teacher_distilled_residual_sparse_ffn",
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
        "hidden_size": int(model.config.hidden_size),
        "chunk_size": args.chunk_size,
        "circuits_per_layer": num_circuits,
        "active_circuits": active,
        "active_fraction": float(active / num_circuits),
        "residual_rank": args.residual_rank,
        "teacher_topk": args.teacher_topk,
        "distill_temperature": args.distill_temperature,
        "route_temperature": args.route_temperature,
        "steps": args.steps,
        "best_step": best_step,
        "best_train_loss": best_loss,
        "teacher_ce": teacher_ce,
        "exact_bank": exact_metrics,
        "levels": levels,
        "latency_ms": {
            "exact_bank": exact_latency,
            "hard_route_object": hard_latency,
            "note": "single-batch forward latency; sparse einsum is not kernel-optimized",
        },
        "parameter_estimate": {
            "target_bank_parameters": total_bank_params,
            "target_active_bank_parameters": active_bank_params,
            "always_active_residual_parameters": residual_params,
            "estimated_target_ffn_parameter_fraction": float(
                (active_bank_params + residual_params) / max(total_bank_params, 1)
            ),
        },
        "train_history": train_history,
        "quality_gate": {
            "criterion": "hard distilled CE delta <= 0.02 and actual latency lower than exact bank",
            "passed": bool(
                hard_metrics["ce_delta"] <= args.max_ce_delta
                and hard_latency < exact_latency
            ),
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
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--active-fraction", type=float, default=0.25)
    parser.add_argument("--residual-rank", type=int, default=64)
    parser.add_argument("--router-hidden", type=int, default=128)
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--teacher-topk", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--soft-warmup", type=int, default=10,
                        help="steps of exact soft training before straight-through hard routing")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--label-weight", type=float, default=0.25)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--max-ce-delta", type=float, default=0.02)
    parser.add_argument("--layer-indices", default=None,
                        help="comma-separated layers; default is all layers")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_distilled_residual_pilot.json")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
