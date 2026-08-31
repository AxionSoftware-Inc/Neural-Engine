from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


MODULUS = 64


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_id: int
    arity: int
    depth: int
    fn: Callable[[list[int]], int]


def _mod(value: int) -> int:
    return value % MODULUS


TASKS = [
    TaskSpec("add", 0, 2, 1, lambda x: _mod(x[0] + x[1])),
    TaskSpec("subtract", 1, 2, 1, lambda x: _mod(x[0] - x[1])),
    TaskSpec("multiply", 2, 2, 1, lambda x: _mod(x[0] * x[1])),
    TaskSpec("greater_than", 3, 2, 1, lambda x: int(x[0] > x[1])),
    TaskSpec("less_equal", 4, 2, 1, lambda x: int(x[0] <= x[1])),
    TaskSpec("xor_parity", 5, 2, 1, lambda x: (x[0] ^ x[1]) & 1),
    TaskSpec("max3", 6, 3, 1, lambda x: max(x)),
    TaskSpec("median3", 7, 3, 1, lambda x: sorted(x)[1]),
    TaskSpec("min3", 8, 3, 1, lambda x: min(x)),
    TaskSpec("reverse_sum", 9, 3, 2, lambda x: _mod(x[2] * 4 + x[1] * 2 + x[0])),
    TaskSpec("lookup", 10, 4, 2, lambda x: x[1] if x[2] % 2 == 0 else x[3]),
    TaskSpec("chain3", 11, 4, 2, lambda x: _mod(x[0] + x[1] + x[2] + x[3])),
    TaskSpec("compose_add_mul", 12, 3, 3, lambda x: _mod((x[0] + x[1]) * x[2])),
    TaskSpec("compose_if", 13, 4, 3, lambda x: _mod(x[2] - x[3] if x[0] > x[1] else x[2] + x[3])),
    TaskSpec("state_machine", 14, 4, 3, lambda x: _mod((x[0] + x[1]) ^ ((x[2] * 3) + x[3]))),
]

TASK_BY_ID = {task.task_id: task for task in TASKS}
