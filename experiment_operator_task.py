from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from neural_engine.operator_valued import OperatorValuedLinear


class GlobalLowRankLinear(nn.Module):
    def __init__(self, features: int, rank: int) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.empty(features, rank))
        self.right = nn.Parameter(torch.empty(rank, features))
        nn.init.normal_(self.left, std=0.02)
        nn.init.normal_(self.right, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs @ self.right.t()) @ self.left.t()


class BlockDiagonalLinear(nn.Module):
    def __init__(self, features: int, packet_width: int) -> None:
        super().__init__()
        self.packets = features // packet_width
        self.packet_width = packet_width
        self.blocks = nn.Parameter(torch.empty(self.packets, packet_width, packet_width))
        nn.init.normal_(self.blocks, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        packets = inputs.reshape(-1, self.packets, self.packet_width)
        outputs = torch.einsum("nig,igh->nih", packets, self.blocks)
        return outputs.reshape(*inputs.shape[:-1], -1)


def train_student(name: str, student: nn.Module, x_train: torch.Tensor,
                  y_train: torch.Tensor, x_eval: torch.Tensor,
                  y_eval: torch.Tensor, steps: int, batch_size: int,
                  seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    student = student.to(x_train.device)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.03)
    generator = torch.Generator(device=x_train.device)
    generator.manual_seed(seed + 100)
    first_loss = None
    start = time.perf_counter()
    student.train()
    for _ in range(steps):
        indices = torch.randint(
            x_train.shape[0], (batch_size,), generator=generator, device=x_train.device
        )
        prediction = student(x_train[indices])
        loss = (prediction - y_train[indices]).square().mean()
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if x_train.device.type == "cuda":
        torch.cuda.synchronize(x_train.device)
    elapsed = time.perf_counter() - start
    student.eval()
    with torch.no_grad():
        train_error = (student(x_train) - y_train).square().mean()
        eval_error = (student(x_eval) - y_eval).square().mean()
    target_scale = y_eval.square().mean().clamp_min(1e-12)
    trainable = sum(parameter.numel() for parameter in student.parameters()
                    if parameter.requires_grad)
    stored = sum(parameter.numel() for parameter in student.parameters())
    return {
        "variant": name,
        "first_mse": first_loss,
        "train_relative_mse": float((train_error / target_scale).cpu()),
        "eval_relative_mse": float((eval_error / target_scale).cpu()),
        "trainable_scalar_dof": trainable,
        "stored_scalar_values": stored,
        "seconds": elapsed,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    torch.manual_seed(args.seed)
    teacher = OperatorValuedLinear(
        args.features, args.features, args.packet_width, args.basis_count, bias=False
    ).to(device).eval()
    x_train = torch.randn(args.train_examples, args.features, device=device)
    x_eval = torch.randn(args.eval_examples, args.features, device=device)
    with torch.no_grad():
        y_train = teacher(x_train)
        y_eval = teacher(x_eval)
    fixed_random = torch.randn_like(teacher.basis) / args.packet_width**0.5
    operator = OperatorValuedLinear(
        args.features, args.features, args.packet_width, args.basis_count, bias=False
    )
    fixed = OperatorValuedLinear(
        args.features, args.features, args.packet_width, args.basis_count, bias=False
    )
    with torch.no_grad():
        fixed.basis.copy_(fixed_random)
    fixed.basis.requires_grad_(False)
    variants = [
        ("learned_operator_basis", operator),
        ("fixed_random_basis", fixed),
        ("global_low_rank_equal_dof", GlobalLowRankLinear(args.features, args.basis_count)),
        ("block_diagonal_equal_dof", BlockDiagonalLinear(args.features, args.packet_width)),
        ("full_dense", nn.Linear(args.features, args.features, bias=False)),
    ]
    results = [
        train_student(name, student, x_train, y_train, x_eval, y_eval,
                      args.steps, args.batch_size, args.seed + index + 1)
        for index, (name, student) in enumerate(variants)
    ]
    report = {
        "device": str(device),
        "features": args.features,
        "packet_width": args.packet_width,
        "basis_count": args.basis_count,
        "train_examples": args.train_examples,
        "eval_examples": args.eval_examples,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "teacher": "known shared operator-valued map",
        "results": results,
        "note": "Relative MSE is normalized by evaluation target energy;"
                 " all students see the same teacher-generated task.",
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched operator-valued teacher task")
    parser.add_argument("--features", type=int, default=384)
    parser.add_argument("--packet-width", type=int, default=16)
    parser.add_argument("--basis-count", type=int, default=8)
    parser.add_argument("--train-examples", type=int, default=8192)
    parser.add_argument("--eval-examples", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/runs/operator_valued_task.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
