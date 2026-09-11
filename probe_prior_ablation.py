"""Paired ablation of the algebraic prior and exact integer codec.

This probe keeps the learned body and checkpoint fixed while changing only the
readout/state paths at inference.  It separates three effects that otherwise
look like one result:

* ``full``: polynomial state plus the exact integer codec readout;
* ``learned_readout``: polynomial state, but the learned digit readout;
* ``learned_no_prior``: learned digit readout with the algebraic state path
  disabled as well.

The experiment is diagnostic, not a training recipe.  The model was trained
with its configured paths; disabling them at inference measures dependence on
those paths rather than a fair retrained baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from data.composition import OPERATIONS
from diagnose_dynamic_generalization import make_generator
from train_dynamic_composition import evaluate, make_model


def _set_mode(
    model: torch.nn.Module,
    mode: str,
    *,
    base_state_mode: str,
    base_integer_decoder: bool,
    base_integer_head: bool,
) -> None:
    if mode not in {"full", "learned_readout", "learned_no_prior"}:
        raise ValueError(f"unknown ablation mode: {mode}")
    if mode == "full":
        model.algebraic_state_mode = base_state_mode
        model.algebraic_integer_output_decoder_enabled = base_integer_decoder
        model.algebraic_integer_output_head_enabled = base_integer_head
        return

    model.algebraic_state_mode = base_state_mode
    model.algebraic_integer_output_decoder_enabled = False
    model.algebraic_integer_output_head_enabled = False
    if mode == "learned_no_prior":
        model.algebraic_state_mode = "none"


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    modes = ("full", "learned_readout", "learned_no_prior")
    results: dict[str, Any] = {}

    for checkpoint_path in args.checkpoint:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = payload["config"]
        model = make_model(config).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        base_state_mode = model.algebraic_state_mode
        base_integer_decoder = model.algebraic_integer_output_decoder_enabled
        base_integer_head = model.algebraic_integer_output_head_enabled
        checkpoint_results: dict[str, Any] = {}

        for range_index, (value_min, value_max) in enumerate(args.value_range):
            range_name = f"range_{value_min}_{value_max}"
            for operation_index, operation in enumerate((None, *OPERATIONS)):
                operation_name = "all_operations" if operation is None else operation
                generator = make_generator(
                    config,
                    seed=args.seed + range_index * 100 + operation_index,
                    split="heldout",
                    value_min=value_min,
                    value_max=value_max,
                    operation=operation,
                )
                rng_state = copy.deepcopy(generator.rng.bit_generator.state)
                mode_reports: dict[str, Any] = {}
                for mode in modes:
                    generator.rng.bit_generator.state = copy.deepcopy(rng_state)
                    _set_mode(
                        model,
                        mode,
                        base_state_mode=base_state_mode,
                        base_integer_decoder=base_integer_decoder,
                        base_integer_head=base_integer_head,
                    )
                    report = evaluate(
                        model,
                        generator,
                        device,
                        args.examples_per_depth,
                        compact_factorized=True,
                    )
                    report.pop("route_audit", None)
                    mode_reports[mode] = {
                        "accuracy": report["accuracy"],
                        "accuracy_by_depth": report["accuracy_by_depth"],
                        "examples": report.get("examples"),
                    }
                key = f"{range_name}/{operation_name}"
                full = mode_reports["full"]["accuracy"]
                learned = mode_reports["learned_readout"]["accuracy"]
                no_prior = mode_reports["learned_no_prior"]["accuracy"]
                mode_reports["full_minus_learned_readout_pp"] = 100.0 * (
                    full - learned
                )
                mode_reports["learned_readout_minus_no_prior_pp"] = 100.0 * (
                    learned - no_prior
                )
                mode_reports["full_minus_no_prior_pp"] = 100.0 * (full - no_prior)
                checkpoint_results[key] = mode_reports

        results[str(checkpoint_path)] = checkpoint_results

    output = {
        "experiment": "v0.263_prior_and_integer_codec_inference_ablation",
        "device": str(device),
        "checkpoint": args.checkpoint,
        "value_ranges": [list(pair) for pair in args.value_range],
        "examples_per_depth": args.examples_per_depth,
        "seed": args.seed,
        "modes": list(modes),
        "interpretation": (
            "Inference-only ablation. The learned body was trained with the "
            "configured prior paths; no-prior is a dependency stress test, "
            "not a retrained control."
        ),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure dependence on the algebraic prior and exact codec"
    )
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=26300)
    parser.add_argument("--examples-per-depth", type=int, default=256)
    parser.add_argument(
        "--value-range",
        dest="value_range",
        action="append",
        nargs=2,
        type=int,
        metavar=("MIN", "MAX"),
        default=None,
        help="inclusive operand range; repeat for multiple paired evaluations",
    )
    args = parser.parse_args()
    if args.value_range is None:
        args.value_range = [(0, 95), (96, 96), (96, 127)]
    run(args)


if __name__ == "__main__":
    main()
