from __future__ import annotations

import torch


def build_factor_address_map(num_circuits: int, factor_count: int,
                             layout: str = "standard",
                             legacy_factor_count: int | None = None) -> torch.Tensor | None:
    """Return a stable virtual-address map for an expanded factor grid.

    ``standard`` packs addresses with ``first + factor_count * second``.
    ``stable_prefix`` puts the old grid first, then appends previously unused
    pairs from the larger grid.  This lets a grown bank keep the parent's
    virtual circuit semantics while still exposing new pair capacity.
    """
    if num_circuits < 1 or factor_count < 1:
        raise ValueError("num_circuits and factor_count must be positive")
    if factor_count * factor_count < num_circuits:
        raise ValueError("factor_count must provide every virtual circuit ID")
    if layout == "standard":
        return None
    if layout != "stable_prefix":
        raise ValueError("factor address layout must be standard or stable_prefix")
    if legacy_factor_count is None:
        raise ValueError("stable_prefix requires legacy_factor_count")
    legacy_factor_count = int(legacy_factor_count)
    if not 1 <= legacy_factor_count <= factor_count:
        raise ValueError("legacy_factor_count must be within the grown factor grid")

    legacy_count = min(num_circuits, legacy_factor_count * legacy_factor_count)
    pairs: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for address in range(legacy_count):
        first = address % legacy_factor_count
        second = address // legacy_factor_count
        pair = (first, second)
        pairs.append(pair)
        used.add(pair)
    for second in range(factor_count):
        for first in range(factor_count):
            if len(pairs) >= num_circuits:
                break
            pair = (first, second)
            if pair not in used:
                pairs.append(pair)
                used.add(pair)
        if len(pairs) >= num_circuits:
            break
    return torch.tensor(pairs, dtype=torch.long)
