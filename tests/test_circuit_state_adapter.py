import torch

from neural_engine.instrumentation import estimate_neural_engine_macs
from neural_engine.model import NeuralEngineV0


def _model(adapter_rank: int = 0) -> NeuralEngineV0:
    return NeuralEngineV0(
        vocab_size=32,
        num_classes=5,
        seq_len=4,
        d_model=8,
        state_dim=8,
        num_circuits=16,
        circuit_rank=2,
        router_branch=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        internal_steps=2,
        circuit_state_adapter_rank=adapter_rank,
    ).eval()


def test_route_state_adapter_starts_at_exact_forward_control():
    torch.manual_seed(17)
    control = _model()
    torch.manual_seed(17)
    treatment = _model(adapter_rank=2)
    treatment.load_state_dict(control.state_dict(), strict=False)
    inputs = torch.randint(0, 32, (5, 4))

    control_logits, _ = control(inputs, adaptive=False, collect_stats=True)
    treatment_logits, _ = treatment(inputs, adaptive=False, collect_stats=True)

    assert torch.allclose(control_logits, treatment_logits, atol=1e-6)
    assert treatment.parameter_report()["circuit_state_adapter_rank"] == 2
    assert treatment.parameter_report()["active_circuit_params"] > control.parameter_report()["active_circuit_params"]


def test_route_state_adapter_adds_only_selected_path_macs():
    control = _model()
    treatment = _model(adapter_rank=2)
    control_macs = estimate_neural_engine_macs(control, executed_steps=2)
    treatment_macs = estimate_neural_engine_macs(treatment, executed_steps=2)

    assert treatment_macs["estimated_active_macs_per_sample"] > control_macs["estimated_active_macs_per_sample"]
    assert treatment_macs["estimated_total_parameter_bytes"] > control_macs["estimated_total_parameter_bytes"]
