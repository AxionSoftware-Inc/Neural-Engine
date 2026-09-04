from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch

from neural_engine.circuits import FactorizedMicroCircuitBank


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_routes(batch_size: int, slots: int, num_circuits: int,
                route_pool: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    pool = min(int(route_pool), int(num_circuits))
    route_values = torch.randperm(num_circuits, generator=generator, device=device)[:pool]
    choices = torch.randint(pool, (batch_size, slots), generator=generator, device=device)
    return route_values[choices]


def grouped_forward(bank: FactorizedMicroCircuitBank, state: torch.Tensor,
                    circuit_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compose each slot after factoring duplicate virtual IDs once per slot.

    This keeps the exact serial semantics of ``forward_serial``. It only
    changes the hardware representation: factor rows and composed matrices are
    materialized for unique route IDs, then indexed back to the batch.
    """
    current = state
    for slot in range(circuit_ids.shape[1]):
        slot_ids = circuit_ids[:, slot]
        unique_ids, inverse = torch.unique(slot_ids, sorted=True, return_inverse=True)
        down, up, bias = bank._gather(unique_ids)
        down = down[inverse]
        up = up[inverse]
        bias = bias[inverse]
        hidden = torch.bmm(current.unsqueeze(1), down).squeeze(1)
        hidden = torch.nn.functional.gelu(hidden)
        output = torch.bmm(hidden.unsqueeze(1), up).squeeze(1) + bias
        current = current + weights[:, slot].unsqueeze(-1) * output
    return current - state


def materialize_page(bank: FactorizedMicroCircuitBank) -> tuple[torch.Tensor, ...]:
    ids = torch.arange(bank.num_circuits, device=bank.factor_mix.device)
    with torch.no_grad():
        return bank._gather(ids)


def page_forward(page: tuple[torch.Tensor, ...], state: torch.Tensor,
                 circuit_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    down_page, up_page, bias_page = page
    current = state
    for slot in range(circuit_ids.shape[1]):
        ids = circuit_ids[:, slot]
        hidden = torch.bmm(current.unsqueeze(1), down_page[ids]).squeeze(1)
        hidden = torch.nn.functional.gelu(hidden)
        output = torch.bmm(hidden.unsqueeze(1), up_page[ids]).squeeze(1) + bias_page[ids]
        current = current + weights[:, slot].unsqueeze(-1) * output
    return current - state


@torch.inference_mode()
def benchmark_variant(name: str, forward: Callable[[], torch.Tensor],
                      device: torch.device, warmup: int, iterations: int,
                      reference: torch.Tensor | None = None) -> dict[str, Any]:
    for _ in range(warmup):
        forward()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    output = None
    for _ in range(iterations):
        start = time.perf_counter()
        output = forward()
        synchronize(device)
        durations.append((time.perf_counter() - start) * 1000.0)
    result: dict[str, Any] = {
        "variant": name,
        "mean_ms": sum(durations) / len(durations),
        "p50_ms": sorted(durations)[len(durations) // 2],
        "p95_ms": sorted(durations)[max(0, math.ceil(len(durations) * 0.95) - 1)],
    }
    if reference is not None and output is not None:
        result["max_abs_error_vs_reference"] = float((output - reference).abs().max().cpu())
    if device.type == "cuda":
        result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return result


@torch.inference_mode()
def run_case(args: argparse.Namespace, route_pool: int) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    torch.manual_seed(args.seed)
    bank = FactorizedMicroCircuitBank(
        num_circuits=args.num_circuits,
        state_dim=args.state_dim,
        rank=args.rank,
        factor_count=math.ceil(math.sqrt(args.num_circuits)),
        factor_mix_mode="shared",
    ).to(device).eval()
    state = torch.randn(args.batch_size, args.state_dim, device=device)
    circuit_ids = make_routes(
        args.batch_size, args.slots, args.num_circuits, route_pool, device, args.seed + 1
    )
    weights = torch.rand(args.batch_size, args.slots, device=device)
    unique_per_slot = [int(torch.unique(circuit_ids[:, slot]).numel())
                       for slot in range(args.slots)]
    reference = bank.forward_serial(state, circuit_ids, weights)
    page = materialize_page(bank)
    page_mb = sum(t.numel() * t.element_size() for t in page) / 2**20
    results = [
        benchmark_variant(
            "reference_factorized_gather",
            lambda: bank.forward_serial(state, circuit_ids, weights),
            device, args.warmup, args.iterations, reference,
        ),
        benchmark_variant(
            "grouped_unique_route",
            lambda: grouped_forward(bank, state, circuit_ids, weights),
            device, args.warmup, args.iterations, reference,
        ),
        benchmark_variant(
            "materialized_contiguous_page",
            lambda: page_forward(page, state, circuit_ids, weights),
            device, args.warmup, args.iterations, reference,
        ),
    ]
    for result in results:
        if result["max_abs_error_vs_reference"] > 1e-5:
            raise RuntimeError(
                f"{result['variant']} failed correctness: "
                f"max error={result['max_abs_error_vs_reference']}"
            )
    return {
        "route_pool": route_pool,
        "batch_size": args.batch_size,
        "slots": args.slots,
        "num_circuits": args.num_circuits,
        "factor_count": math.ceil(math.sqrt(args.num_circuits)),
        "state_dim": args.state_dim,
        "rank": args.rank,
        "mean_unique_ids_per_slot": sum(unique_per_slot) / len(unique_per_slot),
        "unique_ids_per_slot": unique_per_slot,
        "materialized_page_mb": page_mb,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark grouped sparse circuit execution")
    parser.add_argument("--num-circuits", type=int, default=1408)
    parser.add_argument("--state-dim", type=int, default=384)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--route-pools", type=int, nargs="+", default=[32, 128, 1408])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/runs/benchmark_grouped_sparse.json")
    args = parser.parse_args()
    report = {
        "device": str(
            torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                         else ("cpu" if args.device == "auto" else args.device))
        ),
        "cases": [run_case(args, pool) for pool in args.route_pools],
        "note": "Correctness is checked against the serial factorized reference;"
                 " custom fused CUDA kernels are not assumed.",
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
