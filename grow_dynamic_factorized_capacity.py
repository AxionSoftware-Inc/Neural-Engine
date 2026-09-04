from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from data.composition import CompositionalProgramGenerator
from train_dynamic_composition import make_model


@torch.no_grad()
def rank_parent_factors(
    checkpoint: str,
    device: torch.device,
    batches: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    parent_config = dict(payload["config"])
    if parent_config.get("architecture") != "dynamic_register":
        raise ValueError("parent checkpoint must use the dynamic_register architecture")
    model = make_model(parent_config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    heldout_pairs = tuple(tuple(pair) for pair in parent_config.get("heldout_pairs", []))
    modulus_config = parent_config.get("generator_modulus", parent_config.get("modulus", 64))
    generator_modulus = None if modulus_config is None else int(modulus_config)
    generator = CompositionalProgramGenerator(
        seq_len=int(parent_config["seq_len"]),
        seed=int(parent_config.get("seed", 17)) + 171,
        value_min=int(parent_config.get("train_value_min", 0)),
        value_max=int(parent_config.get("train_value_max", 63)),
        split="all",
        heldout_pairs=heldout_pairs,
        combination_split="all",
        modulus=generator_modulus,
        target_offset=int(parent_config.get("target_offset", 0)),
    )
    counts = torch.zeros(
        int(model.router.factor_count), dtype=torch.long, device=device
    )
    for _ in range(batches):
        batch = generator.task_balanced_batch(batch_size, device)
        _, stats = model(batch.inputs)
        selected = stats["selected_ids"].reshape(-1)
        first, second = model.router._factor_ids(selected)
        counts += torch.bincount(
            torch.cat((first, second)), minlength=counts.numel()
        )
    ranked = torch.argsort(counts, descending=True)
    ranked = ranked[counts[ranked] > 0].cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ranked, counts.cpu()


@torch.no_grad()
def clone_rows(
    target: torch.Tensor,
    source: torch.Tensor,
    parent_ids: torch.Tensor,
    noise_scale: float,
) -> None:
    parent_rows = source.index_select(0, parent_ids)
    if noise_scale:
        parent_rows = parent_rows + torch.randn_like(parent_rows) * (
            float(source.std().item()) * noise_scale
        )
    target.copy_(parent_rows)


def grow(args: argparse.Namespace) -> dict[str, Any]:
    parent_payload = torch.load(
        Path(args.parent_checkpoint), map_location="cpu", weights_only=True
    )
    parent_config = dict(parent_payload["config"])
    with open(args.target_config, "r", encoding="utf-8") as handle:
        target_config = yaml.safe_load(handle)
    if target_config.get("architecture") != "dynamic_register":
        raise ValueError("target config must use the dynamic_register architecture")
    if parent_config.get("circuit_bank_mode") != "factorized":
        raise ValueError("parent growth currently requires factorized banks")
    if target_config.get("circuit_bank_mode") != "factorized":
        raise ValueError("target growth currently requires factorized banks")
    if not parent_config.get("operation_circuit_bank"):
        raise ValueError("parent growth currently requires operation circuit banks")
    if not target_config.get("operation_circuit_bank"):
        raise ValueError("target growth currently requires operation circuit banks")

    parent_factor_count = int(parent_config["factor_count"])
    target_factor_count = int(target_config["factor_count"])
    if target_factor_count <= parent_factor_count:
        raise ValueError("target config must contain more factor rows than the parent")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    ranked, counts = rank_parent_factors(
        args.parent_checkpoint, device, args.count_batches, args.count_batch_size
    )

    target_model = make_model(target_config)
    parent_state = parent_payload["model_state"]
    target_state = target_model.state_dict()
    copied: dict[str, int | str] = {}
    skipped: list[str] = []
    with torch.no_grad():
        for name, target_tensor in target_state.items():
            parent_tensor = parent_state.get(name)
            if parent_tensor is None:
                skipped.append(name)
                continue
            if tuple(parent_tensor.shape) == tuple(target_tensor.shape):
                target_tensor.copy_(parent_tensor)
                copied[name] = int(target_tensor.shape[0]) if target_tensor.ndim else "scalar"
            elif (
                target_tensor.ndim == parent_tensor.ndim
                and target_tensor.ndim >= 1
                and target_tensor.shape[1:] == parent_tensor.shape[1:]
                and target_tensor.shape[0] >= parent_tensor.shape[0]
            ):
                target_tensor[: parent_tensor.shape[0]].copy_(parent_tensor)
                copied[name] = int(parent_tensor.shape[0])
            else:
                raise ValueError(
                    f"unsupported parent/target tensor shape for {name}: "
                    f"{tuple(parent_tensor.shape)} -> {tuple(target_tensor.shape)}"
                )

        extra = target_factor_count - parent_factor_count
        parent_ids = ranked[: min(extra, ranked.numel())]
        if not parent_ids.numel():
            raise ValueError("parent factor census found no selected factor rows")
        parent_ids = parent_ids.repeat(math.ceil(extra / parent_ids.numel()))[:extra]
        for bank_id in range(3):
            for suffix in ("down_factors", "up_factors", "bias_factors"):
                name = f"circuits.{bank_id}.{suffix}"
                clone_rows(
                    target_state[name][parent_factor_count:],
                    parent_state[name],
                    parent_ids,
                    args.clone_noise,
                )
        clone_rows(
            target_state["router.keys"][parent_factor_count:],
            parent_state["router.keys"],
            parent_ids,
            args.clone_noise,
        )

    target_model.load_state_dict(target_state)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "parent_checkpoint": args.parent_checkpoint,
        "target_config": args.target_config,
        "parent_factor_count": parent_factor_count,
        "target_factor_count": target_factor_count,
        "extra_factor_rows": target_factor_count - parent_factor_count,
        "parent_params": int(sum(value.numel() for value in parent_state.values())),
        "target_params": int(sum(value.numel() for value in target_state.values())),
        "census_batches": args.count_batches,
        "census_batch_size": args.count_batch_size,
        "parent_factors_seen": int((counts > 0).sum()),
        "clone_noise": args.clone_noise,
        "copied_tensors": copied,
        "skipped_tensors": skipped,
    }
    torch.save(
        {
            "model_state": target_model.state_dict(),
            "config": target_config,
            "growth_report": report,
        },
        output_path,
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm-start a larger dynamic operation-bank factorized model"
    )
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--count-batches", type=int, default=64)
    parser.add_argument("--count-batch-size", type=int, default=128)
    parser.add_argument("--clone-noise", type=float, default=0.05)
    grow(parser.parse_args())


if __name__ == "__main__":
    main()
