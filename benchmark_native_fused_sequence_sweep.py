"""Check native fused parity and timing across supported sequence lengths."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from benchmark_native_fused_shape_sweep import _build_variants, _device, _timed
from data.generator import SyntheticTaskGenerator
from train import seed_everything


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    result: dict[str, Any] = {
        "experiment": "native_fused_sequence_sweep",
        "device": str(device),
        "warmup": args.warmup,
        "repeats": args.repeats,
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
        seed_everything(seed + 19000)
        generator = SyntheticTaskGenerator(
            seq_len=int(checkpoint["config"]["seq_len"]), seed=seed + 19000)
        batch = generator.balanced_batch(args.examples_per_task, device)
        checkpoint_result: dict[str, Any] = {
            "batch_size": int(batch.inputs.shape[0]), "sequences": {}}
        for sequence_length in args.sequence_lengths:
            if sequence_length < int(checkpoint["config"].get("slot_count", 0)):
                raise ValueError("sequence length is shorter than configured slot_count")
            inputs = batch.inputs[:, :sequence_length].contiguous()
            sequence_result: dict[str, Any] = {
                "sequence_length": int(sequence_length), "variants": {}}
            references: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for name in ("fixed_k16", "learned_k8_k16"):
                    references[name], _ = models[name](
                        inputs, adaptive=False, collect_stats=False)
            for name in ("fixed_k16", "native_fused_fixed_k16",
                         "learned_k8_k16", "native_fused_learned"):
                model = models[name]
                timing = _timed(model, inputs, args.warmup, args.repeats, device)
                variant: dict[str, Any] = {
                    "mean_ms": timing,
                    "samples_per_second": float(inputs.shape[0] * 1000.0 / timing),
                }
                if name == "native_fused_fixed_k16":
                    with torch.no_grad():
                        logits, _ = model(inputs, adaptive=False, collect_stats=False)
                    variant["max_logit_error_vs_torch"] = float(
                        (logits - references["fixed_k16"]).abs().max().cpu())
                elif name == "native_fused_learned":
                    with torch.no_grad():
                        logits, _ = model(inputs, adaptive=False, collect_stats=False)
                    variant["max_logit_error_vs_torch"] = float(
                        (logits - references["learned_k8_k16"]).abs().max().cpu())
                sequence_result["variants"][name] = variant
            checkpoint_result["sequences"][str(sequence_length)] = sequence_result
        result["checkpoints"][checkpoint_path.name] = checkpoint_result
        del models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[6, 8, 16, 32])
    parser.add_argument("--examples-per-task", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/diagnostic_native_fused_sequence_sweep_20260910.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
