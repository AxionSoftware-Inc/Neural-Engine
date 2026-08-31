from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@dataclass
class StepStats:
    active_parameters: int
    active_circuits: int
    router_decisions: int
    router_entropy: float


def scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean().cpu())
    return float(value)
