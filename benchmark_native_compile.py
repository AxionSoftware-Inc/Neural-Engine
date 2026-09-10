"""Probe torch.compile on fixed and learned native execution paths."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from data.generator import SyntheticTaskGenerator
from train import make_model, seed_everything


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _timed(model: torch.nn.Module, inputs: torch.Tensor, warmup: int,
           repeats: int, device: torch.device) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            model(inputs, adaptive=False, collect_stats=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(inputs, adaptive=False, collect_stats=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    result: dict[str, Any] = {
        "experiment": "native_torch_compile_probe",
        "device": str(device),
        "mode": args.mode,
        "fullgraph": args.fullgraph,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "examples_per_task": args.examples_per_task,
        "checkpoints": {},
    }
    for checkpoint_name in args.checkpoints:
        checkpoint_path = Path(checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        base_config = dict(checkpoint["config"])
        configs = {}
        sources = {}
        for name, width in (("fixed_k8", 8), ("fixed_k16", 16)):
            config = dict(base_config)
            config["active_circuits"] = width
            config["dynamic_width_mode"] = "none"
            configs[name] = config
            sources[name] = checkpoint
        if args.include_learned:
            learned_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}_learned_width.pt")
            learned_checkpoint = torch.load(learned_path, map_location="cpu", weights_only=True)
            configs["learned_grouped"] = dict(learned_checkpoint["config"])
            sources["learned_grouped"] = learned_checkpoint
        seed = int(base_config.get("seed", 17))
        seed_everything(seed + 9100)
        generator = SyntheticTaskGenerator(seq_len=int(base_config["seq_len"]), seed=seed + 9100)
        batch = generator.balanced_batch(args.examples_per_task, device)
        checkpoint_result: dict[str, Any] = {
            "batch_size": int(batch.inputs.shape[0]), "variants": {}}
        for name, config in configs.items():
            model = make_model(config).to(device)
            model.load_state_dict(sources[name]["model_state"])
            model.eval()
            variant: dict[str, Any] = {}
            try:
                eager_ms = _timed(model, batch.inputs, args.warmup, args.repeats, device)
                variant["eager_ms"] = eager_ms
                compiled = torch.compile(model, mode=args.mode,
                                         fullgraph=args.fullgraph, dynamic=False)
                compiled_ms = _timed(compiled, batch.inputs, args.warmup, args.repeats, device)
                variant["compiled_ms"] = compiled_ms
                variant["compiled_over_eager"] = compiled_ms / eager_ms
                variant["status"] = "ok"
            except Exception as exc:  # Keep one backend failure from hiding other controls.
                variant["status"] = "error"
                variant["error_type"] = type(exc).__name__
                variant["error"] = str(exc)[-1000:]
            checkpoint_result["variants"][name] = variant
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result["checkpoints"][checkpoint_path.name] = checkpoint_result
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--mode", choices=("default", "reduce-overhead", "max-autotune"),
                        default="reduce-overhead")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--include-learned", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--examples-per-task", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/diagnostic_native_compile_20260910.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
