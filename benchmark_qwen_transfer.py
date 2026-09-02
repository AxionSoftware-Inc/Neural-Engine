"""Optional real-checkpoint audit for Qwen-style FFN conversion.

This script is intentionally optional: the core repository does not require
Transformers or a downloaded checkpoint. When those dependencies and a
local/cache model are available, it converts every Qwen MLP in place and
compares the original and converted full-model logits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neural_engine.pretrained_transfer import SwiGLUCircuitBank


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_qwen_transfer.py requires optional packages: "
            "pip install transformers safetensors"
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
    if args.random_from_config:
        config = AutoConfig.from_pretrained(
            args.model,
            trust_remote_code=False,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_config(config, dtype=dtype).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=False,
            local_files_only=args.local_files_only,
        ).to(device)
    model.eval()
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("could not find model.model.layers in the loaded checkpoint")
    hidden_size = int(model.config.hidden_size)
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.randint(
        vocab_size,
        (args.batch_size, args.sequence_length),
        device=device,
    )
    with torch.no_grad():
        original_logits = model(input_ids=input_ids, use_cache=False).logits

    banks: list[SwiGLUCircuitBank] = []
    layer_errors: list[float] = []
    for layer in layers:
        probe = torch.randn(
            args.batch_size,
            args.sequence_length,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            source_output = layer.mlp(probe)
        bank = SwiGLUCircuitBank.from_qwen_mlp(layer.mlp, args.chunk_size).to(device=device)
        bank.eval()
        with torch.no_grad():
            layer_errors.append(float((bank(probe) - source_output).float().abs().max()))
        banks.append(bank)
        layer.mlp = bank

    with torch.no_grad():
        converted_logits = model(input_ids=input_ids, use_cache=False).logits
    difference = (converted_logits - original_logits).float()
    report = {
        "model": args.model,
        "random_from_config": args.random_from_config,
        "model_path_exists": Path(args.model).exists(),
        "device": device,
        "dtype": args.dtype,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": hidden_size,
        "layer_count": len(layers),
        "chunk_size": args.chunk_size,
        "total_converted_parameters": sum(
            int(bank.parameter_report()["total_parameters"]) for bank in banks
        ),
        "max_layer_mlp_error": max(layer_errors),
        "mean_layer_mlp_error": sum(layer_errors) / len(layer_errors),
        "max_abs_logit_error": float(difference.abs().max()),
        "mean_abs_logit_error": float(difference.abs().mean()),
        "logit_allclose_float32_1e-5": bool(torch.allclose(
            converted_logits.float(), original_logits.float(), atol=1e-5, rtol=1e-5
        )),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--random-from-config",
        action="store_true",
        help="instantiate the architecture from cached config without loading weights",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
