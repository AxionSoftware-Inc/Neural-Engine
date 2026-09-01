from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from data.generator import SyntheticTaskGenerator
from train import load_config, make_model, seed_everything


@torch.no_grad()
def rank_parent_circuits(checkpoint: str, config: dict[str, Any], device: torch.device,
                         batches: int, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    parent_config = dict(payload["config"])
    model = make_model(parent_config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    generator = SyntheticTaskGenerator(
        parent_config["seq_len"], seed=int(parent_config["seed"]) + 171,
        value_min=int(parent_config.get("train_value_min", 0)),
        value_max=int(parent_config.get("train_value_max", 63)),
        split=str(parent_config.get("train_split", "all")),
    )
    counts = torch.zeros(parent_config["num_circuits"], dtype=torch.long, device=device)
    for _ in range(batches):
        batch = generator.task_balanced_batch(batch_size, device)
        _, stats = model(batch.inputs, adaptive=False)
        selected = stats["selected_ids"].reshape(-1)
        counts += torch.bincount(selected, minlength=counts.numel())
    ranked = torch.argsort(counts, descending=True)
    ranked = ranked[counts[ranked] > 0].cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ranked, counts.cpu()


def clone_rows(target: torch.Tensor, source: torch.Tensor, parent_ids: torch.Tensor,
               noise_scale: float) -> None:
    parent_rows = source.index_select(0, parent_ids)
    if noise_scale:
        scale = float(source.std().item())
        parent_rows = parent_rows + torch.randn_like(parent_rows) * scale * noise_scale
    target.copy_(parent_rows)


def grow(args: argparse.Namespace) -> dict[str, Any]:
    target_config = load_config(args.target_config, smoke=False)
    seed_everything(int(target_config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ranked, counts = rank_parent_circuits(
        args.parent_checkpoint, target_config, device, args.count_batches, args.count_batch_size
    )
    parent_payload = torch.load(Path(args.parent_checkpoint), map_location="cpu", weights_only=True)
    parent_state = parent_payload["model_state"]
    parent_config = dict(parent_payload["config"])
    parent_circuits = int(parent_config["num_circuits"])
    target_circuits = int(target_config["num_circuits"])
    if target_circuits <= parent_circuits:
        raise ValueError("target config must contain more circuits than the parent")
    if ranked.numel() == 0:
        raise ValueError("parent route census found no selected circuits")

    target_model = make_model(target_config)
    target_state = target_model.state_dict()
    for name, value in parent_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name].copy_(value)

    parent_depth = int(parent_config["router_depth"])
    if target_state["router.level_projections"].shape[1] >= parent_depth:
        target_state["router.level_projections"][:, :parent_depth].copy_(
            parent_state["router.level_projections"]
        )
        target_state["router.level_bias"][:, :parent_depth].copy_(
            parent_state["router.level_bias"]
        )

    # Preserve the complete parent bank before initializing the new child
    # rows. These tensors have different first dimensions, so the generic
    # same-shape copy above cannot handle their parent prefixes.
    for name in ("circuits.down", "circuits.up", "circuits.bias", "router.keys"):
        target_state[name][:parent_circuits].copy_(parent_state[name])

    extra = target_circuits - parent_circuits
    parent_ids = ranked[:min(extra, ranked.numel())]
    parent_ids = parent_ids.repeat(math.ceil(extra / parent_ids.numel()))[:extra]
    clone_rows(target_state["circuits.down"][parent_circuits:],
               parent_state["circuits.down"], parent_ids, args.clone_noise)
    clone_rows(target_state["circuits.up"][parent_circuits:],
               parent_state["circuits.up"], parent_ids, args.clone_noise)
    clone_rows(target_state["circuits.bias"][parent_circuits:],
               parent_state["circuits.bias"], parent_ids, args.clone_noise)
    clone_rows(target_state["router.keys"][parent_circuits:],
               parent_state["router.keys"], parent_ids, args.clone_noise)
    target_model.load_state_dict(target_state)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "parent_checkpoint": args.parent_checkpoint,
        "parent_params": int(sum(value.numel() for value in parent_state.values())),
        "target_params": int(sum(value.numel() for value in target_state.values())),
        "parent_circuits": parent_circuits,
        "target_circuits": target_circuits,
        "extra_circuits": extra,
        "census_batches": args.count_batches,
        "census_batch_size": args.count_batch_size,
        "parent_circuits_seen": int((counts > 0).sum()),
        "clone_noise": args.clone_noise,
        "routing_warmup_capacity": target_config.get("routing_capacity"),
        "routing_warmup_depth": target_config.get("routing_depth"),
    }
    torch.save({"model_state": target_model.state_dict(), "config": target_config,
               "report": report}, output)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Grow a trained Neural Engine circuit bank from a parent checkpoint")
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--count-batches", type=int, default=64)
    parser.add_argument("--count-batch-size", type=int, default=128)
    parser.add_argument("--clone-noise", type=float, default=0.05)
    args = parser.parse_args()
    grow(args)


if __name__ == "__main__":
    main()
