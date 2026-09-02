import pytest

from evaluate_composition import grid_rows, parse_pair


def test_parse_pair_accepts_explicit_composition_label():
    assert parse_pair("add_then_multiply") == ("add", "multiply")


def test_parse_pair_rejects_ambiguous_or_unknown_label():
    with pytest.raises(ValueError):
        parse_pair("add")
    with pytest.raises(ValueError):
        parse_pair("add_then_divide")


def test_grid_rows_explicit_pairs_override_checkpoint_split():
    config = {"seq_len": 8, "heldout_pairs": [["add", "multiply"]]}
    _, targets, labels = grid_rows(
        config, grid_size=2, pairs_override=(("multiply", "add"),))
    assert len(targets) == 8
    assert set(labels) == {"multiply_then_add"}


def test_grid_rows_supports_disjoint_operand_range():
    config = {"seq_len": 8, "heldout_pairs": []}
    inputs, targets, _ = grid_rows(config, grid_size=2, value_min=62, value_max=63)
    assert inputs[:, 3:6].ge(94).all()
    assert inputs[:, 3:6].le(95).all()
    assert len(targets) == 8 * 9


def test_grid_rows_supports_operand_combination_split():
    config = {"seq_len": 8, "heldout_pairs": []}
    _, train_targets, _ = grid_rows(config, grid_size=4, combination_split="train")
    _, heldout_targets, _ = grid_rows(config, grid_size=4, combination_split="heldout")
    assert len(train_targets) + len(heldout_targets) == 4 ** 3 * 9
    assert len(heldout_targets) > 0
