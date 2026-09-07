from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from train import make_model


def expand_state(parent_state: dict[str, torch.Tensor],
                 grown_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy a smaller Neural Engine bank into the prefix of a larger bank."""
    expanded = {}
    for name, target in grown_state.items():
        if name not in parent_state:
            if name in {"correction_gate.weight", "correction_gate.bias"}:
                expanded[name] = torch.zeros_like(target)
                continue
            raise KeyError(f"parent checkpoint is missing {name}")
        source = parent_state[name]
        if source.shape == target.shape:
            expanded[name] = source.clone()
            continue
        if name in {"router.keys", "circuits.down", "circuits.up", "circuits.bias"}:
            if source.ndim != target.ndim or source.shape[1:] != target.shape[1:]:
                raise ValueError(f"incompatible bank shape for {name}")
            if source.shape[0] > target.shape[0]:
                raise ValueError(f"parent bank is larger for {name}")
            value = target.clone()
            value[:source.shape[0]] = source
            expanded[name] = value
            continue
        if name in {"router.level_projections", "router.level_bias"}:
            if source.ndim != target.ndim or source.shape[0] != target.shape[0]:
                raise ValueError(f"incompatible router shape for {name}")
            if source.shape[1] > target.shape[1] or source.shape[2:] != target.shape[2:]:
                raise ValueError(f"parent router is incompatible for {name}")
            value = target.clone()
            value[:, :source.shape[1]] = source
            value[:, source.shape[1]:] = 0
            expanded[name] = value
            continue
        raise ValueError(f"unexpected shape change for {name}: {source.shape} -> {target.shape}")
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start a larger Neural Engine bank from a smaller checkpoint")
    parser.add_argument("--parent", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parent_payload = torch.load(Path(args.parent), map_location="cpu", weights_only=True)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    grown = make_model(config)
    parent_state = parent_payload.get("model_state", parent_payload)
    grown.load_state_dict(expand_state(parent_state, grown.state_dict()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": config,
        "model_state": grown.state_dict(),
        "report": {"expanded_from": str(args.parent)},
    }, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
