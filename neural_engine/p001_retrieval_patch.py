from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.hooks import RemovableHandle

from .router import HierarchicalRouter


@dataclass(frozen=True, slots=True)
class StridedCandidateSchedule:
    """Opt-in P-001 candidate schedule that keeps M and active execution fixed.

    The native hierarchical router exposes one contiguous M-sized window.  This
    experimental schedule keeps the same candidate budget but splits it into
    equally sized windows spread across the reachable circuit bank.  It changes
    retrieval only: the existing key-score top-k selector, circuit body,
    correction path, and recurrent state update are untouched.

    For the canonical P-001 setup (E=32, M=8, windows=4), one route decision
    exposes four 2-circuit windows at offsets 0, 8, 16, and 24 using the same
    local leaf coordinate inside each bank quarter.
    """

    windows: int = 4

    def validate(self, router: HierarchicalRouter) -> tuple[int, int]:
        if not isinstance(router, HierarchicalRouter):
            raise TypeError("P-001 strided schedule requires HierarchicalRouter")
        if self.windows < 2:
            raise ValueError("windows must be >= 2")
        if router.candidate_pool % self.windows != 0:
            raise ValueError("candidate_pool must be divisible by windows")
        if router.routing_capacity % self.windows != 0:
            raise ValueError("routing_capacity must be divisible by windows")
        local_capacity = router.routing_capacity // self.windows
        candidates_per_window = router.candidate_pool // self.windows
        if local_capacity < candidates_per_window:
            raise ValueError("each bank window must fit its candidate budget")
        return local_capacity, candidates_per_window

    def install(self, router: HierarchicalRouter) -> RemovableHandle:
        """Install the schedule as a removable forward pre-hook.

        Installation is deliberately runtime-only so existing checkpoints and
        default model construction remain bit-for-bit on the native path unless
        the experiment explicitly opts in.
        """

        self.validate(router)

        def _inject_schedule(module, args, kwargs):
            if not args:
                raise ValueError("router forward requires the state tensor")
            state = args[0]
            if not isinstance(state, torch.Tensor) or state.ndim < 2:
                raise ValueError("router state must be a batched tensor")
            if kwargs.get("routing_windows") is not None:
                raise ValueError("P-001 strided schedule cannot override routing_windows")
            routing_offset = kwargs.get("routing_offset", 0)
            if not isinstance(routing_offset, int) or routing_offset != 0:
                raise ValueError("P-001 strided schedule requires routing_offset=0")
            if kwargs.get("target_bases") is not None:
                raise ValueError("P-001 strided schedule is not mixed with route-target supervision")

            local_capacity, _ = self.validate(module)
            starts = torch.arange(
                self.windows,
                device=state.device,
                dtype=torch.long,
            ) * local_capacity
            patched = dict(kwargs)
            patched["routing_capacity"] = local_capacity
            patched["routing_windows"] = starts.view(1, -1).expand(state.shape[0], -1)
            return args, patched

        return router.register_forward_pre_hook(_inject_schedule, with_kwargs=True)


@contextmanager
def strided_candidate_schedule(
    router: HierarchicalRouter,
    *,
    windows: int = 4,
) -> Iterator[None]:
    """Temporarily enable the P-001 strided candidate schedule."""

    handle = StridedCandidateSchedule(windows=windows).install(router)
    try:
        yield
    finally:
        handle.remove()


__all__ = ["StridedCandidateSchedule", "strided_candidate_schedule"]
