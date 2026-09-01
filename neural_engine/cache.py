from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch


class CircuitRowCache:
    """Inference cache that pages selected CPU circuit rows to a target device.

    The cache stores complete `(down, up, bias)` rows and transfers cache misses
    in one indexed batch. It is intentionally a small systems prototype: it
    measures the memory-placement problem without changing routing semantics.
    """

    def __init__(self, bank: Any, capacity: int, device: torch.device | str):
        if capacity < 0:
            raise ValueError("cache capacity must be non-negative")
        self.bank = bank
        self.capacity = min(int(capacity), int(bank.num_circuits))
        self.device = torch.device(device)
        self._entries: OrderedDict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()
        self.reset_metrics(clear_cache=True)

    def reset_metrics(self, *, clear_cache: bool = False) -> None:
        if clear_cache:
            self._entries.clear()
        self.requests = 0
        self.requested_rows = 0
        self.hit_rows = 0
        self.miss_rows = 0
        self.evictions = 0
        self.h2d_bytes = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_rows + self.miss_rows
        return self.hit_rows / total if total else 0.0

    @property
    def resident_rows(self) -> int:
        return len(self._entries)

    def _fetch(self, rows: list[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not rows:
            return {}
        source_device = self.bank.down.device
        row_ids = torch.tensor(rows, dtype=torch.long, device=source_device)
        down = self.bank.down.detach().index_select(0, row_ids).to(self.device, non_blocking=True)
        up = self.bank.up.detach().index_select(0, row_ids).to(self.device, non_blocking=True)
        bias = self.bank.bias.detach().index_select(0, row_ids).to(self.device, non_blocking=True)
        if source_device != self.device:
            self.h2d_bytes += (down.numel() * down.element_size()
                               + up.numel() * up.element_size()
                               + bias.numel() * bias.element_size())
        return {
            row: (down[index], up[index], bias[index])
            for index, row in enumerate(rows)
        }

    def gather(self, circuit_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.bank.down.device.type != "cpu":
            raise ValueError("CircuitRowCache requires the circuit bank to reside on CPU")
        rows = [int(row) for row in circuit_ids.detach().reshape(-1).cpu().tolist()]
        unique_rows = list(dict.fromkeys(rows))
        self.requests += 1
        self.requested_rows += len(unique_rows)
        missing = [row for row in unique_rows if row not in self._entries]
        self.hit_rows += len(unique_rows) - len(missing)
        self.miss_rows += len(missing)
        fetched = self._fetch(missing)
        local_entries: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {
            row: self._entries[row] for row in unique_rows if row in self._entries
        }
        for row in missing:
            entry = fetched[row]
            local_entries[row] = entry
            if self.capacity:
                self._entries[row] = entry
                self._entries.move_to_end(row)
                while len(self._entries) > self.capacity:
                    self._entries.popitem(last=False)
                    self.evictions += 1
        entries = []
        for row in rows:
            entry = local_entries.get(row)
            if entry is None:
                entry = self._entries[row]
                self._entries.move_to_end(row)
            entries.append(entry)
        shape = tuple(circuit_ids.shape)
        down = torch.stack([entry[0] for entry in entries]).reshape(*shape, *self.bank.down.shape[1:])
        up = torch.stack([entry[1] for entry in entries]).reshape(*shape, *self.bank.up.shape[1:])
        bias = torch.stack([entry[2] for entry in entries]).reshape(*shape, *self.bank.bias.shape[1:])
        return down, up, bias
