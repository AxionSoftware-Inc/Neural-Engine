import torch

from expand_checkpoint import expand_state
from neural_engine.factor_layout import build_factor_address_map


def test_expand_state_copies_bank_prefix_and_zeroes_new_router_levels():
    parent = {
        "router.keys": torch.ones(2, 3),
        "circuits.down": torch.ones(2, 3, 4),
        "circuits.up": torch.ones(2, 4, 3),
        "circuits.bias": torch.ones(2, 3),
        "router.level_projections": torch.ones(1, 2, 3, 4),
        "router.level_bias": torch.ones(1, 2, 4),
        "shared": torch.tensor([7.0]),
    }
    grown = {
        "router.keys": torch.zeros(4, 3),
        "circuits.down": torch.zeros(4, 3, 4),
        "circuits.up": torch.zeros(4, 4, 3),
        "circuits.bias": torch.zeros(4, 3),
        "router.level_projections": torch.zeros(1, 3, 3, 4),
        "router.level_bias": torch.zeros(1, 3, 4),
        "shared": torch.tensor([0.0]),
    }

    expanded = expand_state(parent, grown)

    for name in ("router.keys", "circuits.down", "circuits.up", "circuits.bias"):
        assert torch.equal(expanded[name][:2], parent[name])
        assert torch.equal(expanded[name][2:], torch.zeros_like(expanded[name][2:]))
    assert torch.equal(expanded["router.level_projections"][:, :2], parent["router.level_projections"])
    assert torch.equal(expanded["router.level_projections"][:, 2:], torch.zeros_like(expanded["router.level_projections"][:, 2:]))
    assert torch.equal(expanded["router.level_bias"][:, :2], parent["router.level_bias"])
    assert torch.equal(expanded["router.level_bias"][:, 2:], torch.zeros_like(expanded["router.level_bias"][:, 2:]))
    assert torch.equal(expanded["shared"], parent["shared"])


def test_expand_state_preserves_factor_rows_and_remaps_factor_mix_pairs():
    parent = {
        "router.factor_keys": torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2),
        "circuits.down_factors": torch.ones(2, 3, 2, 1),
        "circuits.up_factors": torch.ones(2, 3, 1, 2),
        "circuits.bias_factors": torch.ones(2, 3, 2),
        "circuits.factor_hidden_gates": torch.ones(2, 3, 1),
        "circuits.factor_mix": torch.arange(9 * 2, dtype=torch.float32).reshape(9, 2),
    }
    grown = {
        "router.factor_keys": torch.full((2, 4, 2), -1.0),
        "circuits.down_factors": torch.full((2, 4, 2, 1), -1.0),
        "circuits.up_factors": torch.full((2, 4, 1, 2), -1.0),
        "circuits.bias_factors": torch.full((2, 4, 2), -1.0),
        "circuits.factor_hidden_gates": torch.full((2, 4, 1), -1.0),
        "circuits.factor_mix": torch.full((16, 2), -1.0),
    }

    expanded = expand_state(parent, grown)

    for name in ("router.factor_keys", "circuits.down_factors", "circuits.up_factors",
                 "circuits.bias_factors", "circuits.factor_hidden_gates"):
        assert torch.equal(expanded[name][:, :3], parent[name])
        assert torch.equal(expanded[name][:, 3:], torch.full_like(expanded[name][:, 3:], -1.0))

    # Pair (first=1, second=2) is address 1 + 3*2 = 7 in the parent and
    # address 1 + 4*2 = 9 in the grown grid.
    assert torch.equal(expanded["circuits.factor_mix"][9], parent["circuits.factor_mix"][7])
    # Address 3 is (first=3, second=0) in the grown grid; it uses the new
    # factor row and therefore intentionally keeps the grown initialization.
    assert torch.equal(expanded["circuits.factor_mix"][3], torch.full((2,), -1.0))
    assert torch.equal(expanded["circuits.factor_mix"][15], torch.full((2,), -1.0))


def test_stable_prefix_factor_layout_keeps_parent_addresses_intact():
    address_map = build_factor_address_map(20, 5, "stable_prefix", 3)

    assert torch.equal(address_map[:9], torch.tensor([
        [0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2],
    ]))
    assert len({tuple(pair.tolist()) for pair in address_map}) == 20
    assert (address_map[:, 0] < 5).all()
    assert (address_map[:, 1] < 5).all()
