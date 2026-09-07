"""Profile Native Engine CUDA stages without changing model behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.generator import SyntheticTaskGenerator
from train import make_model, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument("--no-stats", action="store_true",
                        help="Profile serving-style forward without diagnostic tensors")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this profile requires CUDA")

    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    seed_everything(int(config["seed"]))
    device = torch.device("cuda")
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    generator = SyntheticTaskGenerator(
        config["seq_len"],
        seed=int(config["seed"]) + 9,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    batch = generator.task_balanced_batch(args.batch_size, device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(batch.inputs, collect_stats=not args.no_stats)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as profile:
            model(batch.inputs, collect_stats=not args.no_stats)
            torch.cuda.synchronize()
    print(f"checkpoint={args.checkpoint}")
    print(f"batch_size={args.batch_size}")
    print(f"stats_collected={not args.no_stats}")
    print(profile.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=args.row_limit,
    ))


if __name__ == "__main__":
    main()
