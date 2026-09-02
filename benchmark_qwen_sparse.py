"""Measure pretrained Qwen FFN chunk sparsity against full-model controls.

The oracle route computes every chunk contribution before retaining the
largest ones. It is an upper bound on route quality, not a deployable sparse
router. The random route is a matched active-count control. Both routes keep
Qwen attention, embeddings, norms, and the LM head unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from neural_engine.pretrained_transfer import SwiGLUCircuitBank, top_contribution_circuits


DEFAULT_TEXT = (
    "Neural networks learn useful transformations from examples. "
    "A sparse circuit should preserve the important computation while "
    "avoiding unnecessary parameter traffic. In this pilot we compare "
    "the original Qwen feed-forward layers with exact chunk conversion "
    "and several active-circuit budgets."
)


class SparseQwenMlp(nn.Module):
    def __init__(
        self,
        bank: SwiGLUCircuitBank,
        active_circuits: int,
        route_mode: str,
        seed: int,
    ) -> None:
        super().__init__()
        if route_mode not in {"oracle", "random"}:
            raise ValueError("route_mode must be oracle or random")
        self.bank = bank
        self.active_circuits = int(active_circuits)
        self.route_mode = route_mode
        self.random_generator = torch.Generator(device=bank.gate_weight.device)
        self.random_generator.manual_seed(seed)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.active_circuits >= self.bank.num_circuits:
            return self.bank(hidden_states)
        if self.route_mode == "oracle":
            ids = top_contribution_circuits(
                self.bank, hidden_states, self.active_circuits
            )
        else:
            scores = torch.rand(
                *hidden_states.shape[:-1],
                self.bank.num_circuits,
                device=hidden_states.device,
                generator=self.random_generator,
            )
            ids = scores.topk(self.active_circuits, dim=-1).indices
        return self.bank.forward_selected(hidden_states, ids)


def _loss(logits: torch.Tensor, input_ids: torch.Tensor) -> float:
    return float(F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    ))


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "benchmark_qwen_sparse.py requires optional packages: "
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
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("could not find model.model.layers in the loaded checkpoint")

    if args.text:
        from transformers import AutoTokenizer

        tokenizer_source = args.tokenizer or args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=args.local_files_only,
        )
        encoded = tokenizer(
            args.text,
            add_special_tokens=True,
            return_tensors="pt",
        )["input_ids"]
        if encoded.shape[1] < 2:
            raise ValueError("--text must produce at least two tokens")
        input_ids = encoded[:, :args.sequence_length].to(device)
        input_ids = input_ids.repeat(args.batch_size, 1)
        input_mode = "Qwen tokenizer text; language perplexity control"
    else:
        vocab_size = int(model.config.vocab_size)
        positions = torch.arange(args.sequence_length, device=device).unsqueeze(0)
        input_ids = (positions * 7919 + torch.arange(args.batch_size, device=device).unsqueeze(1) * 104729)
        input_ids = (input_ids.remainder(vocab_size)).to(torch.long)
        input_mode = "deterministic token IDs; logit fidelity control, not language-quality evaluation"
    with torch.no_grad():
        full_logits = model(input_ids=input_ids, use_cache=False).logits
    full_loss = _loss(full_logits.float(), input_ids)

    banks: list[SwiGLUCircuitBank] = []
    for layer in layers:
        bank = SwiGLUCircuitBank.from_qwen_mlp(layer.mlp, args.chunk_size).to(device=device)
        bank.eval()
        banks.append(bank)

    levels = sorted({
        banks[0].num_circuits,
        max(1, round(banks[0].num_circuits * 0.75)),
        max(1, round(banks[0].num_circuits * 0.50)),
        max(1, round(banks[0].num_circuits * 0.25)),
        max(1, round(banks[0].num_circuits * 0.16)),
        max(1, round(banks[0].num_circuits * 0.08)),
    }, reverse=True)
    result: dict[str, object] = {
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": device,
        "dtype": args.dtype,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "hidden_size": int(model.config.hidden_size),
        "layer_count": len(layers),
        "chunk_size": args.chunk_size,
        "circuits_per_layer": banks[0].num_circuits,
        "full_model_ce": full_loss,
        "input_mode": input_mode,
        "levels": [],
    }

    for route_mode in ("oracle", "random"):
        for active in levels:
            for layer, bank in zip(layers, banks):
                layer.mlp = SparseQwenMlp(
                    bank,
                    active_circuits=active,
                    route_mode=route_mode,
                    seed=args.seed + active * 1009 + len(route_mode),
                )
            with torch.no_grad():
                sparse_logits = model(input_ids=input_ids, use_cache=False).logits
            difference = (sparse_logits - full_logits).float()
            result["levels"].append({
                "route": route_mode,
                "active_circuits": active,
                "active_fraction": float(active / banks[0].num_circuits),
                "estimated_ffn_mac_fraction": float(active / banks[0].num_circuits),
                "ce": _loss(sparse_logits.float(), input_ids),
                "ce_delta": _loss(sparse_logits.float(), input_ids) - full_loss,
                "logit_mse": float(F.mse_loss(sparse_logits.float(), full_logits.float())),
                "max_abs_logit_error": float(difference.abs().max()),
                "top1_agreement": float((
                    sparse_logits.argmax(dim=-1) == full_logits.argmax(dim=-1)
                ).to(torch.float32).mean()),
            })
    return result


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
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
