from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .generator import Batch, VALUE_TOKEN_OFFSET


MODULUS = 64
PROGRAM_TOKEN = 1
OPERATION_TOKENS = {"add": 2, "subtract": 3, "multiply": 4}
OPERATIONS = tuple(OPERATION_TOKENS)
COMPOSITION_PAIRS = tuple((first, second) for first in OPERATIONS for second in OPERATIONS)
DEFAULT_HELDOUT_PAIRS = (("add", "multiply"), ("multiply", "add"))


def apply_operation(name: str, left: int, right: int, modulus: int = MODULUS) -> int:
    if name == "add":
        return (left + right) % modulus
    if name == "subtract":
        return (left - right) % modulus
    if name == "multiply":
        return (left * right) % modulus
    raise ValueError(f"unknown operation: {name}")


def composition_combination_bucket(task_id: int, values: list[int] | tuple[int, ...]) -> int:
    """Return a stable bucket for holding out operand combinations."""
    accumulator = 0x811C9DC5 ^ (task_id + 1)
    for index, value in enumerate(values, start=1):
        accumulator ^= (index * (int(value) + 1)) & 0xFFFFFFFF
        accumulator = (accumulator * 0x01000193) & 0xFFFFFFFF
    return accumulator % 4


@dataclass(frozen=True)
class CompositionSpec:
    name: str
    task_id: int
    first_operation: str
    second_operation: str


COMPOSITION_SPECS = tuple(
    CompositionSpec(f"{first}_then_{second}", index, first, second)
    for index, (first, second) in enumerate(COMPOSITION_PAIRS)
)


class CompositionalProgramGenerator:
    """Generate fixed-format programs with held-out operation compositions.

    The input exposes primitive operation tokens rather than a task token that
    names the final composition.  A model must therefore learn the primitive
    operations and apply them in order. The benchmark trains on most ordered
    operation pairs and evaluates on pairs never shown during training.
    """

    def __init__(self, seq_len: int = 8, seed: int = 17,
                 value_min: int = 0, value_max: int = MODULUS - 1,
                 split: str = "train",
                 heldout_pairs: tuple[tuple[str, str], ...] = DEFAULT_HELDOUT_PAIRS,
                 combination_split: str = "all"):
        if seq_len < 6:
            raise ValueError("seq_len must leave room for program and operands")
        if not 0 <= value_min <= value_max < MODULUS:
            raise ValueError(f"value range must be within [0, {MODULUS - 1}]")
        if split not in {"all", "train", "heldout"}:
            raise ValueError("split must be 'all', 'train', or 'heldout'")
        if combination_split not in {"all", "train", "heldout"}:
            raise ValueError("combination_split must be 'all', 'train', or 'heldout'")
        unknown = set(heldout_pairs) - set(COMPOSITION_PAIRS)
        if unknown:
            raise ValueError(f"unknown held-out composition pairs: {sorted(unknown)}")
        self.seq_len = seq_len
        self.value_min = value_min
        self.value_max = value_max
        self.split = split
        self.combination_split = combination_split
        self.heldout_pairs = frozenset(heldout_pairs)
        self.rng = np.random.default_rng(seed)

    @property
    def allowed_specs(self) -> tuple[CompositionSpec, ...]:
        return tuple(spec for spec in COMPOSITION_SPECS
                     if self.split == "all"
                     or ((spec.first_operation, spec.second_operation) in self.heldout_pairs)
                     == (self.split == "heldout"))

    def _one(self, spec: CompositionSpec) -> tuple[list[int], int, list[int], list[bool]]:
        while True:
            values = self.rng.integers(self.value_min, self.value_max + 1, size=3).tolist()
            bucket = composition_combination_bucket(spec.task_id, values)
            if (self.combination_split == "all"
                    or (self.combination_split == "train" and bucket < 3)
                    or (self.combination_split == "heldout" and bucket == 3)):
                break
        partial = apply_operation(spec.first_operation, values[0], values[1])
        target = apply_operation(spec.second_operation, partial, values[2])
        tokens = [PROGRAM_TOKEN, OPERATION_TOKENS[spec.first_operation],
                  OPERATION_TOKENS[spec.second_operation]]
        tokens += [VALUE_TOKEN_OFFSET + value for value in values]
        tokens += [0] * (self.seq_len - len(tokens))
        return tokens, target, [partial, target, target], [True, True, True]

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
        specs = self.allowed_specs
        rows = []
        for _ in range(batch_size):
            spec = specs[int(self.rng.integers(0, len(specs)))]
            tokens, target, stage_targets, stage_mask = self._one(spec)
            rows.append((tokens, target, spec.task_id, 3, stage_targets, stage_mask))
        return self._make_batch(rows, device)

    def balanced_batch(self, examples_per_task: int = 32,
                       device: str | torch.device = "cpu") -> Batch:
        rows = []
        for spec in self.allowed_specs:
            for _ in range(examples_per_task):
                tokens, target, stage_targets, stage_mask = self._one(spec)
                rows.append((tokens, target, spec.task_id, 3, stage_targets, stage_mask))
        order = self.rng.permutation(len(rows))
        return self._make_batch([rows[int(index)] for index in order], device)

    def task_balanced_batch(self, batch_size: int,
                            device: str | torch.device = "cpu") -> Batch:
        specs = self.allowed_specs
        rows = []
        for index in range(batch_size):
            spec = specs[index % len(specs)]
            tokens, target, stage_targets, stage_mask = self._one(spec)
            rows.append((tokens, target, spec.task_id, 3, stage_targets, stage_mask))
        order = self.rng.permutation(len(rows))
        return self._make_batch([rows[int(index)] for index in order], device)
