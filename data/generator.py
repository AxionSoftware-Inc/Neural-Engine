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
    stage_targets: torch.Tensor | None = None
    stage_mask: torch.Tensor | None = None


class SyntheticTaskGenerator:
    """Generate fresh exact-label algorithmic examples."""

    def __init__(self, seq_len: int = 32, seed: int = 17,
                 value_min: int = 0, value_max: int = MODULUS - 1,
                 split: str = "all"):
        if seq_len < 6:
            raise ValueError("seq_len must leave room for task and operand tokens")
        if not 0 <= value_min <= value_max < MODULUS:
            raise ValueError(f"value range must be within [0, {MODULUS - 1}]")
        if split not in {"all", "train", "heldout"}:
            raise ValueError("split must be 'all', 'train', or 'heldout'")
        self.seq_len = seq_len
        self.value_min = value_min
        self.value_max = value_max
        self.split = split
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _combination_bucket(task: TaskSpec, values: list[int]) -> int:
        """Stable task-aware bucket used to hold out operand combinations."""
        accumulator = 0x811C9DC5 ^ (task.task_id + 1)
        for index, value in enumerate(values, start=1):
            accumulator ^= (index * (value + 1)) & 0xFFFFFFFF
            accumulator = (accumulator * 0x01000193) & 0xFFFFFFFF
        return accumulator % 4

    def _accept_values(self, task: TaskSpec, values: list[int]) -> bool:
        bucket = self._combination_bucket(task, values)
        return self.split == "all" or (self.split == "train" and bucket < 3) or (self.split == "heldout" and bucket == 3)

    def _stage_targets(self, task: TaskSpec, values: list[int], target: int) -> tuple[list[int], list[bool]]:
        """Return deterministic supervision for the recurrent reasoning steps.

        The final target remains the only required label. These optional targets
        expose natural partial results of composed tasks so later recurrent
        steps can learn to consume a useful state instead of rediscovering the
        whole computation from scratch.
        """
        stages = [int(target), int(target), int(target)]
        mask = [False, False, False]
        if task.depth == 1:
            mask[0] = True
        elif task.name == "reverse_sum":
            stages[:2] = [(values[2] * 4 + values[1] * 2) % MODULUS, int(target)]
            mask[:2] = [True, True]
        elif task.name == "lookup":
            selected = values[1] if values[2] % 2 == 0 else values[3]
            stages[:2] = [int(selected), int(target)]
            mask[:2] = [True, True]
        elif task.name == "chain3":
            partial = (values[0] + values[1] + values[2]) % MODULUS
            stages[:2] = [int(partial), int(target)]
            mask[:2] = [True, True]
        elif task.name == "compose_add_mul":
            partial = (values[0] + values[1]) % MODULUS
            stages[:3] = [int(partial), int(target), int(target)]
            mask[:3] = [True, True, True]
        elif task.name == "compose_if":
            condition = int(values[0] > values[1])
            selected = (values[2] - values[3]) if condition else (values[2] + values[3])
            stages[:3] = [condition, int(selected % MODULUS), int(target)]
            mask[:3] = [True, True, True]
        elif task.name == "state_machine":
            left = (values[0] + values[1]) % MODULUS
            right = (values[2] * 3 + values[3]) % MODULUS
            stages[:3] = [int(left), int(right), int(target)]
            mask[:3] = [True, True, True]
        else:
            raise ValueError(f"missing stage target definition for depth-{task.depth} task {task.name}")
        return stages, mask

    def _one(self, task: TaskSpec) -> tuple[list[int], int, list[int], list[bool]]:
        while True:
            values = self.rng.integers(self.value_min, self.value_max + 1, size=task.arity).tolist()
            if self._accept_values(task, values):
                break
        tokens = [TASK_TOKEN_OFFSET + task.task_id]
        tokens += [VALUE_TOKEN_OFFSET + value for value in values]
        tokens += [PAD_TOKEN] * (self.seq_len - len(tokens))
        target = int(task.fn(values))
        stage_targets, stage_mask = self._stage_targets(task, values, target)
        return tokens, target, stage_targets, stage_mask

    @staticmethod
    def _make_batch(rows: list[tuple[list[int], int, int, int, list[int], list[bool]]],
                    device: str | torch.device) -> Batch:
        return Batch(
            inputs=torch.tensor([row[0] for row in rows], dtype=torch.long, device=device),
            targets=torch.tensor([row[1] for row in rows], dtype=torch.long, device=device),
            task_ids=torch.tensor([row[2] for row in rows], dtype=torch.long, device=device),
            depths=torch.tensor([row[3] for row in rows], dtype=torch.long, device=device),
            stage_targets=torch.tensor([row[4] for row in rows], dtype=torch.long, device=device),
            stage_mask=torch.tensor([row[5] for row in rows], dtype=torch.bool, device=device),
        )

    def batch(self, batch_size: int, device: str | torch.device = "cpu") -> Batch:
        rows: list[tuple[list[int], int, int, int, list[int], list[bool]]] = []
        for _ in range(batch_size):
            task = TASKS[int(self.rng.integers(0, len(TASKS)))]
            tokens, target, stage_targets, stage_mask = self._one(task)
            rows.append((tokens, target, task.task_id, task.depth, stage_targets, stage_mask))
        return self._make_batch(rows, device)

    def fixed_dataset(self, examples: int, device: str | torch.device = "cpu") -> Batch:
        return self.batch(examples, device=device)

    def balanced_batch(self, examples_per_task: int = 32,
                       device: str | torch.device = "cpu") -> Batch:
        """Return equal task counts so easy task families cannot dominate accuracy."""
        rows: list[tuple[list[int], int, int, int, list[int], list[bool]]] = []
        for task in TASKS:
            for _ in range(examples_per_task):
                tokens, target, stage_targets, stage_mask = self._one(task)
                rows.append((tokens, target, task.task_id, task.depth, stage_targets, stage_mask))
        order = self.rng.permutation(len(rows))
        rows = [rows[int(index)] for index in order]
        return self._make_batch(rows, device)

    def task_balanced_batch(self, batch_size: int,
                            device: str | torch.device = "cpu") -> Batch:
        """Sample an exactly near-uniform task mix for training."""
        task_indices = np.arange(batch_size) % len(TASKS)
        self.rng.shuffle(task_indices)
        rows: list[tuple[list[int], int, int, int, list[int], list[bool]]] = []
        for task_index in task_indices:
            task = TASKS[int(task_index)]
            tokens, target, stage_targets, stage_mask = self._one(task)
            rows.append((tokens, target, task.task_id, task.depth, stage_targets, stage_mask))
        return self._make_batch(rows, device)

    def composition_batch(self, batch_size: int,
                          device: str | torch.device = "cpu",
                          strength: float = 1.0) -> Batch:
        """Oversample multi-step tasks while retaining every task family.

        Depth-1, depth-2, and depth-3 tasks receive weights 1, 1+strength, and
        1+2*strength. This is a training-only curriculum/mixture experiment;
        evaluation remains uniformly balanced across all task IDs.
        """
        if strength < 0:
            raise ValueError("composition sampling strength must be non-negative")
        weights = np.array([
            1.0 + strength * (task.depth - 1)
            for task in TASKS
        ])
        probabilities = weights / weights.sum()
        task_indices = self.rng.choice(len(TASKS), size=batch_size, p=probabilities)
        rows: list[tuple[list[int], int, int, int, list[int], list[bool]]] = []
        for task_index in task_indices:
            task = TASKS[int(task_index)]
            tokens, target, stage_targets, stage_mask = self._one(task)
            rows.append((tokens, target, task.task_id, task.depth, stage_targets, stage_mask))
        return self._make_batch(rows, device)


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
