import torch

from neural_engine.p001_retrieval_patch import (
    StridedCandidateSchedule,
    strided_candidate_schedule,
)
from neural_engine.router import HierarchicalRouter


def _router() -> HierarchicalRouter:
    torch.manual_seed(7)
    router = HierarchicalRouter(
        32,
        num_circuits=32,
        branch=4,
        depth=2,
        candidate_pool=8,
        active_circuits=2,
    )
    router.eval()
    return router


def test_strided_schedule_keeps_m8_and_spreads_candidates_across_bank() -> None:
    router = _router()
    state = torch.randn(5, 32)

    with strided_candidate_schedule(router, windows=4):
        selected, weights, stats = router(state)

    candidates = stats["candidate_ids"].reshape(5, 4, 2)
    assert candidates.shape == (5, 4, 2)
    assert selected.shape == (5, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-6)
    for quarter in range(4):
        lo = quarter * 8
        hi = lo + 8
        assert (candidates[:, quarter] >= lo).all()
        assert (candidates[:, quarter] < hi).all()
    for row in range(state.shape[0]):
        assert all(int(value) in stats["candidate_ids"][row].tolist() for value in selected[row])


def test_strided_schedule_is_removed_and_native_forward_window_returns() -> None:
    router = _router()
    state = torch.randn(4, 32)

    with strided_candidate_schedule(router, windows=4):
        _, _, patched = router(state)
    _, _, native = router(state)

    native_ids = native["candidate_ids"]
    expected_next = (native_ids[:, :-1] + 1).remainder(router.routing_capacity)
    assert torch.equal(native_ids[:, 1:], expected_next)
    assert not torch.equal(patched["candidate_ids"], native_ids)


def test_strided_schedule_rejects_incompatible_candidate_budget() -> None:
    router = HierarchicalRouter(
        32,
        num_circuits=32,
        branch=4,
        depth=2,
        candidate_pool=6,
        active_circuits=2,
    )
    schedule = StridedCandidateSchedule(windows=4)
    try:
        schedule.validate(router)
    except ValueError as exc:
        assert "candidate_pool" in str(exc)
    else:
        raise AssertionError("expected incompatible candidate budget to be rejected")
