from __future__ import annotations

import torch
from torch import nn


_MACRO_ROW_PARAMETERS = {
    "macro_router.keys",
    "macro_cell_bank.down",
    "macro_cell_bank.up",
    "macro_cell_bank.bias",
}
_MACRO_LEVEL_PARAMETERS = {
    "macro_router.level_projections",
    "macro_router.level_bias",
}


def expand_macro_model(
    parent: nn.Module,
    grown: nn.Module,
    *,
    preserve_parent_route: bool = True,
) -> nn.Module:
    """Warm-start a larger macro bank from a trained smaller model.

    Shared engine weights and the parent's macro rows are copied exactly.
    Newly allocated macro rows keep the grown model's initialization.  Router
    levels added by the larger tree are zeroed so the initial hard route stays
    inside the copied parent prefix; exploration can then open the new leaves.
    """
    parent_count = int(getattr(parent, "macro_cell_count", 0))
    grown_count = int(getattr(grown, "macro_cell_count", 0))
    if parent_count < 1 or grown_count <= parent_count:
        raise ValueError("grown must have a larger enabled macro bank")

    parent_state = parent.state_dict()
    grown_state = grown.state_dict()
    for name, target in grown_state.items():
        if name not in parent_state:
            continue
        source = parent_state[name]
        if name in _MACRO_ROW_PARAMETERS:
            if source.ndim != target.ndim or source.shape[1:] != target.shape[1:]:
                raise ValueError(f"incompatible macro row shape for {name}")
            if source.shape[0] != parent_count or target.shape[0] != grown_count:
                raise ValueError(f"unexpected macro row count for {name}")
            target[:parent_count].copy_(source)
        elif name in _MACRO_LEVEL_PARAMETERS:
            if source.ndim != target.ndim or source.shape[0] != target.shape[0]:
                raise ValueError(f"incompatible macro router shape for {name}")
            if source.shape[2:] != target.shape[2:]:
                raise ValueError(f"incompatible macro router level shape for {name}")
            if source.shape[1] > target.shape[1]:
                raise ValueError(f"grown router is shallower for {name}")
            target[:, :source.shape[1]].copy_(source)
            if preserve_parent_route:
                target[:, source.shape[1]:].zero_()
            elif name == "macro_router.level_bias":
                # Keep the newly exposed branches unbiased, while retaining
                # the target model's small random projections so different
                # inputs can open different leaves during warm-up.
                target[:, source.shape[1]:].zero_()
        else:
            if source.shape != target.shape:
                raise ValueError(f"non-macro parameter changed shape: {name}")
            target.copy_(source)

    grown.load_state_dict(grown_state)
    return grown
