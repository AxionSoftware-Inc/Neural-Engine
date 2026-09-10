"""Smoke-test shape reuse and concurrent CUDA-stream safety for native fused serving."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from benchmark_native_fused_shape_sweep import _build_variants, _device
from data.generator import SyntheticTaskGenerator
from train import seed_everything


def _forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits, _ = model(inputs, adaptive=False, collect_stats=False)
    return logits


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().cpu())


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    if device.type != "cuda":
        raise RuntimeError("serving smoke requires a CUDA device")
    result: dict[str, Any] = {
        "experiment": "native_fused_serving_smoke",
        "device": str(device),
        "examples_per_task": args.examples_per_task,
        "sequence_lengths": args.sequence_lengths,
        "checkpoints": {},
    }
    for checkpoint_name in args.checkpoints:
        checkpoint_path = Path(checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        learned_path = checkpoint_path.with_name(
            f"{checkpoint_path.stem}_learned_width.pt")
        learned_checkpoint = torch.load(learned_path, map_location="cpu", weights_only=True)
        models, _ = _build_variants(checkpoint, learned_checkpoint, device)
        seed = int(checkpoint["config"].get("seed", 17))
        seed_everything(seed + 21000)
        generator = SyntheticTaskGenerator(
            seq_len=int(checkpoint["config"]["seq_len"]), seed=seed + 21000)
        full_batch = generator.balanced_batch(args.examples_per_task, device)
        inputs = {
            f"b{full_batch.inputs.shape[0]}_s{sequence_length}":
            full_batch.inputs[:, :sequence_length].contiguous()
            for sequence_length in args.sequence_lengths
        }
        checkpoint_result: dict[str, Any] = {"shapes": {}, "variants": {}}
        for name, reference_name in (
                ("native_fused_fixed_k16", "fixed_k16"),
                ("native_fused_learned", "learned_k8_k16")):
            model = models[name]
            reference = models[reference_name]
            references = {key: _forward(reference, value) for key, value in inputs.items()}
            variant_result: dict[str, Any] = {"shapes": {}, "interleaved_max_error": 0.0,
                                              "concurrent_max_error": 0.0}
            for key, value in inputs.items():
                output = _forward(model, value)
                variant_result["shapes"][key] = {
                    "max_logit_error_vs_torch": _max_error(output, references[key]),
                }
            for key in list(inputs) * 2:
                output = _forward(model, inputs[key])
                variant_result["interleaved_max_error"] = max(
                    variant_result["interleaved_max_error"],
                    _max_error(output, references[key]),
                )
            keys = list(inputs)
            stream_a = torch.cuda.Stream(device=device)
            stream_b = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream_a):
                output_a = _forward(model, inputs[keys[0]])
            with torch.cuda.stream(stream_b):
                output_b = _forward(model, inputs[keys[-1]])
            torch.cuda.synchronize(device)
            variant_result["concurrent_max_error"] = max(
                _max_error(output_a, references[keys[0]]),
                _max_error(output_b, references[keys[-1]]),
            )
            checkpoint_result["variants"][name] = variant_result
        for key in inputs:
            checkpoint_result["shapes"][key] = {"sequence_length": inputs[key].shape[1]}
        result["checkpoints"][checkpoint_path.name] = checkpoint_result
        del models
        gc.collect()
        torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--examples-per-task", type=int, default=8)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[6, 32])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/diagnostic_native_fused_serving_smoke_20260910.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
