from types import SimpleNamespace

from probe_prior_ablation import _set_mode


def test_prior_ablation_modes_restore_the_baseline_state_path():
    model = SimpleNamespace(
        algebraic_state_mode="polynomial2_fourier",
        algebraic_integer_output_decoder_enabled=True,
        algebraic_integer_output_head_enabled=True,
    )

    _set_mode(
        model,
        "learned_no_prior",
        base_state_mode="polynomial2_fourier",
        base_integer_decoder=True,
        base_integer_head=True,
    )
    assert model.algebraic_state_mode == "none"
    assert not model.algebraic_integer_output_decoder_enabled
    assert not model.algebraic_integer_output_head_enabled

    _set_mode(
        model,
        "learned_readout",
        base_state_mode="polynomial2_fourier",
        base_integer_decoder=True,
        base_integer_head=True,
    )
    assert model.algebraic_state_mode == "polynomial2_fourier"
    assert not model.algebraic_integer_output_decoder_enabled

    _set_mode(
        model,
        "full",
        base_state_mode="polynomial2_fourier",
        base_integer_decoder=True,
        base_integer_head=True,
    )
    assert model.algebraic_state_mode == "polynomial2_fourier"
    assert model.algebraic_integer_output_decoder_enabled
    assert model.algebraic_integer_output_head_enabled
