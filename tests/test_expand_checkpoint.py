import torch

from expand_checkpoint import expand_state


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
