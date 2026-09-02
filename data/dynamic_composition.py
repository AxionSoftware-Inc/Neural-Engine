from __future__ import annotations

from itertools import product

import numpy as np
import torch

from .composition import MODULUS, OPERATION_TOKENS, apply_operation
from .generator import Batch, VALUE_TOKEN_OFFSET


PROGRAM_TOKEN = 1


class DynamicCompositionGenerator:
    """Generate variable-length left-fold programs for a recurrent register core.

    The input has a fixed two-region layout so the model can scan a program
    without attention:

        [program, op_1 ... op_max, value_0 ... value_max]

    Unused operation/value positions are padded with zero.  A train split can
    expose depths 1..N while a held-out split exposes deeper programs N+1..M;
    the primitive operator vocabulary remains unchanged.
    """

    def __init__(
        self,
        max_ops: int = 6,
        train_max_ops: int | None = None,
        seed: int = 17,
        value_min: int = 0,
        value_max: int = MODULUS - 1,
        split: str = "all",
    ) -> None:
        if max_ops < 1:
            raise ValueError("max_ops must be positive")
        if train_max_ops is None:
            train_max_ops = max_ops
        if not 1 <= train_max_ops <= max_ops:
            raise ValueError("train_max_ops must be within max_ops")
        if not 0 <= value_min <= value_max < MODULUS:
            raise ValueError(f"value range must be within [0, {MODULUS - 1}]")
        if split not in {"all", "train", "heldout"}:
            raise ValueError("split must be all, train, or heldout")
        self.max_ops = max_ops
        self.train_max_ops = train_max_ops
        self.seq_len = 1 + max_ops + (max_ops + 1)
        self.value_min = value_min
        self.value_max = value_max
        self.split = split
        self.rng = np.random.default_rng(seed)
        self.operation_names = tuple(OPERATION_TOKENS)

    @property
    def allowed_depths(self) -> tuple[int, ...]:
        if self.split == "train":
            return tuple(range(1, self.train_max_ops + 1))
        if self.split == "heldout":
            return tuple(range(self.train_max_ops + 1, self.max_ops + 1))
        return tuple(range(1, self.max_ops + 1))

    @property
    def operation_sequences(self) -> tuple[tuple[str, ...], ...]:
        return tuple(product(self.operation_names, repeat=self.max_ops))

    def _sample_depth(self) -> int:
        depths = self.allowed_depths
        if not depths:
            raise ValueError("split has no allowed program depths")
        return int(depths[int(self.rng.integers(0, len(depths)))])

    def _one(self, depth: int | None = None) -> tuple[list[int], int, int, list[int], list[bool]]:
        if depth is None:
            depth = self._sample_depth()
        if depth not in self.allowed_depths:
            raise ValueError(f"depth {depth} is not allowed for split {self.split}")
        operations = [
            self.operation_names[int(self.rng.integers(0, len(self.operation_names)))]
            for _ in range(depth)
        ]
        values = self.rng.integers(
            self.value_min, self.value_max + 1, size=depth + 1
        ).tolist()
        accumulator = int(values[0])
        stage_targets: list[int] = []
        for operation, value in zip(operations, values[1:]):
            accumulator = apply_operation(operation, accumulator, int(value))
            stage_targets.append(int(accumulator))
        stage_targets.extend([int(accumulator)] * (self.max_ops - depth))
        stage_mask = [True] * depth + [False] * (self.max_ops - depth)

        op_tokens = [OPERATION_TOKENS[name] for name in operations]
        op_tokens.extend([0] * (self.max_ops - depth))
        value_tokens = [VALUE_TOKEN_OFFSET + int(value) for value in values]
        value_tokens.extend([0] * (self.max_ops - depth))
        tokens = [PROGRAM_TOKEN] + op_tokens + value_tokens
        sequence_id = sum(
            self.operation_names.index(operation) * (len(self.operation_names) ** index)
            for index, operation in enumerate(operations)
        )
        return tokens, int(accumulator), sequence_id, stage_targets, stage_mask

    @staticmethod
    def _make_batch(
        rows: list[tuple[list[int], int, int, int, list[int], list[bool]]],
        device: str | torch.device,
    ) -> Batch:
        return Batch(
            inputs=torch.tensor([row[0] for row in rows], dtype=torch.long, device=device),
            targets=torch.tensor([row[1] for row in rows], dtype=torch.long, device=device),
            task_ids=torch.tensor([row[2] for row in rows], dtype=torch.long, device=device),
            depths=torch.tensor([row[3] for row in rows], dtype=torch.long, device=device),
            stage_targets=torch.tensor([row[4] for row in rows], dtype=torch.long, device=device),
            stage_mask=torch.tensor([row[5] for row in rows], dtype=torch.bool, device=device),
        )

    def batch(self, batch_size: int, device: str | torch.device = "cpu") -> Batch:
        rows = []
        for _ in range(batch_size):
            depth = self._sample_depth()
            tokens, target, sequence_id, stage_targets, stage_mask = self._one(depth)
            rows.append((tokens, target, sequence_id, depth, stage_targets, stage_mask))
        return self._make_batch(rows, device)

    def balanced_batch(
        self,
        examples_per_depth: int = 32,
        device: str | torch.device = "cpu",
    ) -> Batch:
        rows = []
        for depth in self.allowed_depths:
            for _ in range(examples_per_depth):
                tokens, target, sequence_id, stage_targets, stage_mask = self._one(depth)
                rows.append((tokens, target, sequence_id, depth, stage_targets, stage_mask))
        order = self.rng.permutation(len(rows))
        return self._make_batch([rows[int(index)] for index in order], device)

    def task_balanced_batch(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> Batch:
        depths = self.allowed_depths
        rows = []
        for index in range(batch_size):
            depth = depths[index % len(depths)]
            tokens, target, sequence_id, stage_targets, stage_mask = self._one(depth)
            rows.append((tokens, target, sequence_id, depth, stage_targets, stage_mask))
        order = self.rng.permutation(len(rows))
        return self._make_batch([rows[int(index)] for index in order], device)
