from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from neural_engine.operator_valued import OperatorValuedLinear


def make_structured_target(output_packets: int, input_packets: int,
                           packet_width: int, rank: int,
                           device: torch.device) -> torch.Tensor:
    basis = torch.randn(rank, packet_width, packet_width, device=device)
    coeff = torch.randn(output_packets, input_packets, rank, device=device)
    return torch.einsum("oia,agh->oigh", coeff, basis)


def make_random_target(output_packets: int, input_packets: int,
                       packet_width: int, device: torch.device) -> torch.Tensor:
    return torch.randn(output_packets, input_packets, packet_width, packet_width, device=device)


def svd_oracle(target_blocks: torch.Tensor, ranks: list[int]) -> list[dict[str, Any]]:
    matrix = target_blocks.reshape(-1, target_blocks.shape[-2] * target_blocks.shape[-1])
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    denominator = torch.linalg.norm(matrix).clamp_min(1e-12)
    rows = []
    for rank in ranks:
        rank = min(rank, singular.numel())
        approximation = (left[:, :rank] * singular[:rank]) @ right[:rank]
        rows.append({
            "rank": rank,
            "relative_frobenius_error": float(
                torch.linalg.norm(matrix - approximation).div(denominator).cpu()
            ),
        })
    return rows


def fit_student(target_blocks: torch.Tensor, packet_width: int, basis_count: int,
                device: torch.device, steps: int, fixed_basis: torch.Tensor | None = None,
                seed: int = 17) -> dict[str, Any]:
    torch.manual_seed(seed)
    output_packets, input_packets = target_blocks.shape[:2]
    layer = OperatorValuedLinear(
        input_packets * packet_width, output_packets * packet_width,
        packet_width=packet_width, basis_count=basis_count, bias=False,
    ).to(device)
    if fixed_basis is not None:
        with torch.no_grad():
            layer.basis.copy_(fixed_basis)
        layer.basis.requires_grad_(False)
        parameters = [layer.coeff]
    else:
        parameters = list(layer.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.05)
    target_weight = target_blocks.permute(0, 3, 1, 2).reshape(
        layer.out_features, layer.in_features
    )
    start = time.perf_counter()
    first_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = (layer.effective_weight() - target_weight).square().mean()
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    relative_error = torch.linalg.norm(layer.effective_weight() - target_weight).div(
        torch.linalg.norm(target_weight).clamp_min(1e-12)
    )
    return {
        "basis_mode": "fixed" if fixed_basis is not None else "learned",
        "basis_count": basis_count,
        "steps": steps,
        "first_loss": first_loss,
        "final_relative_frobenius_error": float(relative_error.detach().cpu()),
        "seconds": elapsed,
        "stored_scalar_values": sum(parameter.numel() for parameter in layer.parameters()),
        "trainable_scalar_dof": sum(
            parameter.numel() for parameter in layer.parameters() if parameter.requires_grad
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    torch.manual_seed(args.seed)
    input_packets = args.features // args.packet_width
    output_packets = args.features // args.packet_width
    structured = make_structured_target(
        output_packets, input_packets, args.packet_width, args.basis_count, device
    )
    random_target = make_random_target(output_packets, input_packets, args.packet_width, device)
    fixed_random_basis = torch.randn(
        args.basis_count, args.packet_width, args.packet_width, device=device
    ) / math.sqrt(args.packet_width)
    report = {
        "device": str(device),
        "features": args.features,
        "packet_width": args.packet_width,
        "basis_count": args.basis_count,
        "structured_target_rank": args.basis_count,
        "structured_target_svd": svd_oracle(structured, args.ranks),
        "random_target_svd": svd_oracle(random_target, args.ranks),
        "structured_target_fit": [
            fit_student(structured, args.packet_width, args.basis_count, device,
                        args.steps, seed=args.seed + 10),
            fit_student(structured, args.packet_width, args.basis_count, device,
                        args.steps, fixed_basis=fixed_random_basis, seed=args.seed + 11),
        ],
        "random_target_fit": [
            fit_student(random_target, args.packet_width, args.basis_count, device,
                        args.steps, seed=args.seed + 12),
            fit_student(random_target, args.packet_width, args.basis_count, device,
                        args.steps, fixed_basis=fixed_random_basis, seed=args.seed + 13),
        ],
        "parameter_accounting": OperatorValuedLinear(
            args.features, args.features, args.packet_width, args.basis_count,
            bias=False,
        ).parameter_report(),
        "controls": {
            "full_dense_scalar_dof": args.features * args.features,
            "operator_scalar_dof_with_bias": args.basis_count * args.packet_width**2
            + input_packets * output_packets * args.basis_count + args.features,
            "fixed_random_basis_trainable_dof": input_packets * output_packets * args.basis_count,
        },
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic operator-valued parameter gate")
    parser.add_argument("--features", type=int, default=384)
    parser.add_argument("--packet-width", type=int, default=16)
    parser.add_argument("--basis-count", type=int, default=8)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/runs/operator_valued_synthetic.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
