from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .tasks import MODULUS, TASKS, TaskSpec


TASK_TOKEN_OFFSET = 1
VALUE_TOKEN_OFFSET = 32
PAD_TOKEN = 0


@dataclass
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor
    task_ids: torch.Tensor
    depths: torch.Tensor


class SyntheticTaskGenerator:
    """Generate fresh exact-label algorithmic examples."""

    def __init__(self, seq_len: int = 32, seed: int = 17):
        if seq_len < 6:
            raise ValueError("seq_len must leave room for task and operand tokens")
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def _one(self, task: TaskSpec) -> tuple[list[int], int]:
        values = self.rng.integers(0, MODULUS, size=task.arity).tolist()
        tokens = [TASK_TOKEN_OFFSET + task.task_id]
        tokens += [VALUE_TOKEN_OFFSET + value for value in values]
        tokens += [PAD_TOKEN] * (self.seq_len - len(tokens))
        return tokens, int(task.fn(values))

    def batch(self, batch_size: int, device: str | torch.device = "cpu") -> Batch:
        inputs: list[list[int]] = []
        targets: list[int] = []
        task_ids: list[int] = []
        depths: list[int] = []
        for _ in range(batch_size):
            task = TASKS[int(self.rng.integers(0, len(TASKS)))]
            tokens, target = self._one(task)
            inputs.append(tokens)
            targets.append(target)
            task_ids.append(task.task_id)
            depths.append(task.depth)
        return Batch(
            inputs=torch.tensor(inputs, dtype=torch.long, device=device),
            targets=torch.tensor(targets, dtype=torch.long, device=device),
            task_ids=torch.tensor(task_ids, dtype=torch.long, device=device),
            depths=torch.tensor(depths, dtype=torch.long, device=device),
        )

    def fixed_dataset(self, examples: int, device: str | torch.device = "cpu") -> Batch:
        return self.batch(examples, device=device)


def accuracy_by_task(predictions: torch.Tensor, batch: Batch) -> dict[str, float]:
    from .tasks import TASK_BY_ID

    result: dict[str, float] = {}
    correct = predictions.eq(batch.targets)
    for task_id in torch.unique(batch.task_ids).tolist():
        mask = batch.task_ids.eq(task_id)
        result[TASK_BY_ID[int(task_id)].name] = float(correct[mask].float().mean().cpu())
    return result


def accuracy_by_depth(predictions: torch.Tensor, batch: Batch) -> dict[str, float]:
    correct = predictions.eq(batch.targets)
    result: dict[str, float] = {}
    for depth in torch.unique(batch.depths).tolist():
        mask = batch.depths.eq(depth)
        result[str(int(depth))] = float(correct[mask].float().mean().cpu())
    return result
