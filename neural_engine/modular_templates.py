from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .encoding import VALUE_MODULUS, VALUE_TOKEN_OFFSET
from .instrumentation import count_parameters


def modular_add_state(state: torch.Tensor, operand: torch.Tensor,
                      modulus: int) -> torch.Tensor:
    indices = (
        torch.arange(modulus, device=state.device).unsqueeze(0)
        - operand.unsqueeze(1)
    ).remainder(modulus)
    return state.gather(1, indices)


def modular_subtract_state(state: torch.Tensor, operand: torch.Tensor,
                           modulus: int) -> torch.Tensor:
    indices = (
        torch.arange(modulus, device=state.device).unsqueeze(0)
        + operand.unsqueeze(1)
    ).remainder(modulus)
    return state.gather(1, indices)


def modular_multiply_state(state: torch.Tensor, operand: torch.Tensor,
                           modulus: int) -> torch.Tensor:
    left = torch.arange(modulus, device=state.device).unsqueeze(0)
    destinations = (left * operand.unsqueeze(1)).remainder(modulus)
    return torch.zeros_like(state).scatter_add(1, destinations, state)


class TrainableModularTemplateRegister(nn.Module):
    """Attention-free modular register with trainable primitive selection.

    The three primitive actions use equivariant modular wiring: circular
    translation for addition/subtraction and a modular multiplication action.
    The model does not store a dense transition table. Only the mapping from
    input operation token to primitive action is learned.
    """

    def __init__(self, max_ops: int = 6, num_classes: int = VALUE_MODULUS,
                 modulus: int = VALUE_MODULUS,
                 value_token_offset: int = VALUE_TOKEN_OFFSET,
                 template_init: str = "identity"):
        super().__init__()
        if modulus < 2 or num_classes != modulus:
            raise ValueError("num_classes must equal a modulus of at least two")
        if template_init not in {"identity", "random"}:
            raise ValueError("template_init must be identity or random")
        self.max_ops = max_ops
        self.value_start = 1 + max_ops
        self.modulus = modulus
        self.value_token_offset = value_token_offset
        self.template_init = template_init
        # Rows are input operation tokens; columns are add, subtract, multiply.
        self.template_logits = nn.Parameter(torch.empty(3, 3))
        if template_init == "identity":
            with torch.no_grad():
                self.template_logits.copy_(4.0 * torch.eye(3))
        else:
            nn.init.normal_(self.template_logits, std=0.02)
        self.output = nn.Linear(modulus, num_classes)

    def _add_state(self, state: torch.Tensor, operand: torch.Tensor) -> torch.Tensor:
        return modular_add_state(state, operand, self.modulus)

    def _subtract_state(self, state: torch.Tensor, operand: torch.Tensor) -> torch.Tensor:
        return modular_subtract_state(state, operand, self.modulus)

    def _multiply_state(self, state: torch.Tensor, operand: torch.Tensor) -> torch.Tensor:
        return modular_multiply_state(state, operand, self.modulus)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if inputs.shape[1] < self.value_start + self.max_ops + 1:
            raise ValueError("inputs are shorter than the configured program layout")
        values = (inputs[:, self.value_start:self.value_start + self.max_ops + 1]
                  - self.value_token_offset).clamp(0, self.modulus - 1)
        operations = (inputs[:, 1:1 + self.max_ops] - 2).clamp(0, 2)
        active = inputs[:, 1:1 + self.max_ops].ge(2)
        state = F.one_hot(values[:, 0], self.modulus).to(dtype=torch.float32)
        step_logits = []

        for step in range(self.max_ops):
            operand = values[:, step + 1]
            primitives = torch.stack((
                self._add_state(state, operand),
                self._subtract_state(state, operand),
                self._multiply_state(state, operand),
            ), dim=1)
            weights = F.softmax(self.template_logits[operations[:, step]], dim=-1)
            updated = torch.einsum("bt,btv->bv", weights, primitives)
            state = torch.where(active[:, step].unsqueeze(-1), updated, state)
            step_logits.append(self.output(state))

        stats = {
            "step_logits": torch.stack(step_logits, dim=1),
            "template_weights": F.softmax(self.template_logits, dim=-1),
            "active_params": torch.tensor(self.parameter_report()["total_params"]),
        }
        return stats["step_logits"][:, -1], stats

    def parameter_report(self) -> dict[str, int | float | bool]:
        total = count_parameters(self)
        return {
            "total_params": total,
            "active_params_estimate": total,
            "attention": False,
            "dense_transition_table": False,
            "modulus": self.modulus,
            "template_init": self.template_init,
        }
