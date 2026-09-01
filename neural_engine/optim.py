from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


class LazyAdamW(torch.optim.Optimizer):
    """AdamW that keeps optimizer state only for rows with non-zero gradients.

    This is a correctness-first prototype for row-structured parameters such
    as the circuit bank and router key table. Dense controller parameters still
    use ordinary AdamW updates. For lazy parameters, decoupled weight decay is
    applied only to rows touched by the current trajectory; inactive rows are
    left untouched instead of forcing a full-bank sweep.
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], *, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0,
                 lazy_parameters: Iterable[torch.nn.Parameter] = ()):
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight decay: {weight_decay}")
        self._lazy_parameter_ids = {id(parameter) for parameter in lazy_parameters}
        self._active_rows: dict[int, torch.Tensor] = {}
        self._global_step = 0
        self._last_stats: dict[str, int | float] = {}
        super().__init__(params, {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        })

    @torch.no_grad()
    def _step_dense(self, parameter: torch.nn.Parameter, group: dict[str, Any],
                    step: int) -> None:
        gradient = parameter.grad
        if gradient is None:
            return
        if gradient.is_sparse:
            raise RuntimeError("LazyAdamW dense parameters do not accept sparse gradients")
        state = self.state[parameter]
        if not state:
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        if group["weight_decay"]:
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
        denominator = exp_avg_sq.sqrt().add_(group["eps"])
        parameter.addcdiv_(exp_avg, denominator, value=-step_size)

    def set_active_rows(self, rows_by_parameter: dict[torch.nn.Parameter, torch.Tensor]) -> None:
        """Provide route IDs so lazy updates avoid scanning a full bank gradient."""
        self._active_rows = {
            id(parameter): rows.detach().reshape(-1).unique()
            for parameter, rows in rows_by_parameter.items()
        }

    @torch.no_grad()
    def _step_lazy(self, parameter: torch.nn.Parameter, group: dict[str, Any],
                   step: int) -> int:
        gradient = parameter.grad
        if gradient is None:
            return 0
        explicit_rows = self._active_rows.get(id(parameter))
        if explicit_rows is not None:
            row_ids_tensor = explicit_rows.to(device=parameter.device)
            if gradient.is_sparse:
                row_gradients = gradient.to_dense().index_select(0, row_ids_tensor)
            else:
                row_gradients = gradient.index_select(0, row_ids_tensor)
        elif gradient.is_sparse:
            gradient = gradient.coalesce()
            row_ids_tensor = gradient.indices()[0].unique()
            row_gradients = gradient.values()
        else:
            if parameter.ndim == 0:
                raise ValueError("lazy parameters must have a row dimension")
            flattened = gradient.reshape(gradient.shape[0], -1)
            row_ids_tensor = flattened.abs().sum(dim=1).nonzero(as_tuple=False).flatten()
            row_gradients = gradient.index_select(0, row_ids_tensor)
        if row_ids_tensor.numel() == 0:
            return 0
        row_ids = [int(row) for row in row_ids_tensor.tolist()]
        state = self.state[parameter]
        row_map: dict[int, int] = state.setdefault("row_map", {})
        row_count = int(state.get("row_count", 0))
        new_rows = [row for row in row_ids if row not in row_map]
        if new_rows:
            capacity = int(state.get("capacity", 0))
            required = row_count + len(new_rows)
            if required > capacity:
                new_capacity = max(required, max(16, capacity * 2))
                shape = (new_capacity, *parameter.shape[1:])
                exp_avg = torch.zeros(shape, dtype=parameter.dtype, device=parameter.device)
                exp_avg_sq = torch.zeros_like(exp_avg)
                if row_count:
                    exp_avg[:row_count].copy_(state["exp_avg"][:row_count])
                    exp_avg_sq[:row_count].copy_(state["exp_avg_sq"][:row_count])
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
                state["capacity"] = new_capacity
            for offset, row in enumerate(new_rows):
                row_map[row] = row_count + offset
            row_count = required
            state["row_count"] = row_count
        positions = torch.tensor([row_map[row] for row in row_ids], dtype=torch.long,
                                 device=parameter.device)
        exp_avg = state["exp_avg"].index_select(0, positions)
        exp_avg_sq = state["exp_avg_sq"].index_select(0, positions)
        beta1, beta2 = group["betas"]
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
        exp_avg.mul_(beta1).add_(row_gradients, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(row_gradients, row_gradients, value=1.0 - beta2)
        state["exp_avg"].index_copy_(0, positions, exp_avg)
        state["exp_avg_sq"].index_copy_(0, positions, exp_avg_sq)
        parameter_rows = parameter.index_select(0, row_ids_tensor)
        if group["weight_decay"]:
            parameter_rows.mul_(1.0 - group["lr"] * group["weight_decay"])
        denominator = exp_avg_sq.sqrt().add_(group["eps"])
        parameter_rows.addcdiv_(exp_avg, denominator, value=-step_size)
        parameter.index_copy_(0, row_ids_tensor, parameter_rows)
        return int(row_ids_tensor.numel())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._global_step += 1
        lazy_rows = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                if id(parameter) in self._lazy_parameter_ids:
                    lazy_rows += self._step_lazy(parameter, group, self._global_step)
                else:
                    self._step_dense(parameter, group, self._global_step)
        self._last_stats = {
            "lazy_rows_last_step": lazy_rows,
            "lazy_state_rows": sum(
                int(self.state[parameter].get("row_count", 0))
                for group in self.param_groups
                for parameter in group["params"]
                if id(parameter) in self._lazy_parameter_ids
            ),
            "optimizer_step": self._global_step,
        }
        self._active_rows = {}
        return loss

    def report(self) -> dict[str, int | float | str]:
        return {
            "optimizer": "lazy_adamw",
            "lazy_parameter_count": len(self._lazy_parameter_ids),
            "lazy_rows_last_step": int(self._last_stats.get("lazy_rows_last_step", 0)),
            "lazy_state_rows": int(self._last_stats.get("lazy_state_rows", 0)),
            "lazy_weight_decay": "touched_rows_only",
        }
