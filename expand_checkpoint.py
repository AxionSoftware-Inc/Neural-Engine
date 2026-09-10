from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from train import make_model


def expand_state(parent_state: dict[str, torch.Tensor],
                 grown_state: dict[str, torch.Tensor],
                 parent_config: dict | None = None,
                 grown_config: dict | None = None) -> dict[str, torch.Tensor]:
    """Copy a smaller Neural Engine bank into a larger bank.

    Independent banks use a simple row prefix.  Factorized banks need one
    extra detail: changing ``factor_count`` changes the mapping from a virtual
    circuit id to ``(first_factor, second_factor)``.  The reusable factor rows
    still copy by factor id, while the per-address mix is copied by its
    two-dimensional factor-pair identity instead of by raw address.
    """
    expanded = {}
    parent_factor_count = None
    grown_factor_count = None
    if "router.factor_keys" in parent_state and "router.factor_keys" in grown_state:
        parent_factor_count = int(parent_state["router.factor_keys"].shape[1])
        grown_factor_count = int(grown_state["router.factor_keys"].shape[1])
    stable_prefix = bool(
        grown_config and grown_config.get("factor_address_layout") == "stable_prefix"
    )

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
        if name in {
            "router.factor_keys",
            "circuits.down_factors",
            "circuits.up_factors",
            "circuits.bias_factors",
            "circuits.factor_hidden_gates",
        }:
            if source.ndim != target.ndim or source.shape[0] != target.shape[0]:
                raise ValueError(f"incompatible factor-slot shape for {name}")
            if source.shape[1] > target.shape[1] or source.shape[2:] != target.shape[2:]:
                raise ValueError(f"incompatible factor-row shape for {name}")
            value = target.clone()
            value[:, :source.shape[1]] = source
            expanded[name] = value
            continue
        if name == "circuits.factor_mix":
            if (parent_factor_count is None or grown_factor_count is None
                    or source.ndim != 2 or target.ndim != 2
                    or source.shape[1] != target.shape[1] or source.shape[1] != 2):
                raise ValueError("incompatible factor-mix shape")
            value = target.clone()
            if stable_prefix:
                value[:source.shape[0]] = source
                expanded[name] = value
                continue
            # Preserve learned pair-specific mixing for every pair that exists
            # in both grids.  Newly introduced factor rows keep the target
            # module's initialization.
            common = min(parent_factor_count, grown_factor_count)
            for second in range(common):
                parent_base = second * parent_factor_count
                grown_base = second * grown_factor_count
                width = min(common, source.shape[0] - parent_base,
                            target.shape[0] - grown_base)
                if width > 0:
                    value[grown_base:grown_base + width] = source[parent_base:parent_base + width]
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
    parent_config = parent_payload.get("config")
    grown.load_state_dict(expand_state(
        parent_state, grown.state_dict(), parent_config=parent_config,
        grown_config=config,
    ))
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
