from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from train import make_model


def copy_prefix(parent: torch.Tensor, target: torch.Tensor) -> int:
    """Copy a parent tensor into the leading rows of a larger target tensor."""
    if parent.ndim != target.ndim or parent.shape[1:] != target.shape[1:]:
        raise ValueError(f"incompatible growth shapes: {tuple(parent.shape)} -> {tuple(target.shape)}")
    if parent.shape[0] > target.shape[0]:
        raise ValueError(f"target is smaller than parent: {tuple(parent.shape)} -> {tuple(target.shape)}")
    target[:parent.shape[0]].copy_(parent)
    return int(parent.shape[0])


def grow(parent_checkpoint: str, target_config: str, output: str) -> dict[str, object]:
    parent_payload = torch.load(Path(parent_checkpoint), map_location="cpu", weights_only=True)
    with open(target_config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    target_model = make_model(config)
    parent_state = parent_payload.get("model_state", parent_payload)
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
            elif (target_tensor.ndim == parent_tensor.ndim
                  and target_tensor.ndim >= 1
                  and target_tensor.shape[1:] == parent_tensor.shape[1:]
                  and target_tensor.shape[0] >= parent_tensor.shape[0]):
                copied[name] = copy_prefix(parent_tensor, target_tensor)
            else:
                raise ValueError(
                    f"unsupported parent/target tensor shape for {name}: "
                    f"{tuple(parent_tensor.shape)} -> {tuple(target_tensor.shape)}")
    target_model.load_state_dict(target_state)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "parent_checkpoint": parent_checkpoint,
        "target_config": target_config,
        "copied_tensors": copied,
        "skipped_tensors": skipped,
        "target_params": sum(parameter.numel() for parameter in target_model.parameters()),
    }
    torch.save({"model_state": target_model.state_dict(),
                "config": config, "growth_report": report}, output_path)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start a larger factorized typed-register bank")
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    grow(args.parent_checkpoint, args.target_config, args.output)


if __name__ == "__main__":
    main()
