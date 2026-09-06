"""Frozen-bank ProbeRoute-2 experiments.

Only the new router is trainable.  Probe branches are evaluated without
gradient, and their final task-loss differences supervise the retriever and
pair selector.  This intentionally keeps the first experiment separate from
circuit co-adaptation.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from train import BatchSource, make_model


def load_models(checkpoint: Path, device: torch.device) -> tuple[NeuralEngineV0, NeuralEngineV0, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    base_config = dict(payload["config"])
    teacher = make_model(base_config)
    if not isinstance(teacher, NeuralEngineV0):
        raise ValueError("ProbeRoute-2 requires a NeuralEngineV0 checkpoint")
    teacher.load_state_dict(payload["model_state"])
    probe_config = dict(base_config)
    probe_config["router_variant"] = "probe"
    # The probe-only parameters are absent from the frozen hierarchical
    # checkpoint.  Fix their initialization so controls differ only by the
    # trained objective, not by an unrelated random start.
    init_seed = int(base_config.get("seed", 0)) + 1907
    random.seed(init_seed)
    torch.manual_seed(init_seed)
    probe = make_model(probe_config)
    if not isinstance(probe, NeuralEngineV0):
        raise ValueError("ProbeRoute-2 model construction failed")
    probe.load_state_dict(payload["model_state"], strict=False)
    teacher.to(device).eval()
    probe.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return teacher, probe, base_config


def heldout_batches(config: dict, device: torch.device, count: int,
                    examples_per_task: int) -> list:
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), int(config["seed"]) + 3,
        value_min=int(config.get("heldout_value_min", config.get("eval_value_min", 0))),
        value_max=int(config.get("heldout_value_max", config.get("eval_value_max", 63))),
        split=str(config.get("heldout_split", "all")),
    )
    source = BatchSource(generator, 256, device)
    return [source.balanced(examples_per_task) for _ in range(count)]


def train_batches(config: dict, device: torch.device, seed: int,
                  count: int, examples_per_task: int) -> list:
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), seed + 1,
        value_min=int(config.get("train_value_min", 0)),
        value_max=int(config.get("train_value_max", 63)),
        split=str(config.get("train_split", "train")),
    )
    source = BatchSource(generator, 128, device)
    return [source.balanced(examples_per_task) for _ in range(count)]


def _target_pair_positions(router, candidate_ids: torch.Tensor,
                           target_pairs: torch.Tensor) -> torch.Tensor:
    left = (candidate_ids == target_pairs[:, 0:1]).long().argmax(dim=-1)
    right = (candidate_ids == target_pairs[:, 1:2]).long().argmax(dim=-1)
    positions = torch.stack([left, right], dim=-1)
    positions, _ = positions.sort(dim=-1)
    matches = router.pair_positions.to(candidate_ids.device).unsqueeze(0).eq(
        positions.unsqueeze(1)
    ).all(dim=-1)
    return matches.float().argmax(dim=-1)


def _retrieval_inclusion_loss(router, query: torch.Tensor,
                              target_pairs: torch.Tensor,
                              weight: torch.Tensor) -> torch.Tensor:
    scores = router.retriever_scores(query)
    target_mask = torch.zeros_like(scores, dtype=torch.bool)
    target_mask.scatter_(1, target_pairs, True)
    other = scores.masked_fill(target_mask, torch.finfo(scores.dtype).min)
    threshold = other.topk(min(7, scores.shape[-1] - 2), dim=-1).values[:, -1]
    target_scores = scores.gather(1, target_pairs)
    margin = F.relu(0.2 + threshold.unsqueeze(-1) - target_scores).mean(dim=-1)
    return (margin * weight).mean()


def _selection_loss(router, query: torch.Tensor, current_pairs: torch.Tensor,
    alternative_pairs: torch.Tensor, delta: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return router.utility_query.weight.sum() * 0.0
    current_score = router.selection_scores(query, current_pairs)
    alternative_score = router.selection_scores(query, alternative_pairs)
    predicted_delta = alternative_score - current_score
    selected_delta = delta[mask]
    selected_prediction = predicted_delta[mask]
    regression = F.smooth_l1_loss(
        selected_prediction, selected_delta.detach(), beta=0.1, reduction="mean",
    )
    ranking_mask = selected_delta.abs().ge(0.02)
    if not bool(ranking_mask.any()):
        return regression
    rank_weight = (selected_delta.abs() / 0.1).clamp_max(2.0)
    ranking = F.softplus(
        -selected_delta.sign() * selected_prediction / 0.1,
    ) * rank_weight
    return regression + 0.1 * ranking[ranking_mask].mean()


def _imitation_loss(router, query: torch.Tensor, teacher_candidates: torch.Tensor,
                    teacher_pairs: torch.Tensor, mode: str) -> torch.Tensor:
    target_pairs, _ = teacher_pairs.sort(dim=-1)
    pair_scores = router.candidate_pair_scores(query, teacher_candidates)
    target_index = _target_pair_positions(router, teacher_candidates, target_pairs)
    target_score = pair_scores.gather(1, target_index.unsqueeze(-1)).squeeze(-1)
    negatives = pair_scores.scatter(1, target_index.unsqueeze(-1),
                                    torch.finfo(pair_scores.dtype).min)
    selection = F.relu(0.2 + negatives.max(dim=-1).values - target_score).mean()
    retrieval = _retrieval_inclusion_loss(
        router, query, target_pairs,
        torch.ones(query.shape[0], device=query.device),
    )
    if mode == "selection-only":
        return selection
    if mode == "retrieval-only":
        return retrieval
    return selection + 0.1 * retrieval


def _sample_probe_pairs(router, selected: torch.Tensor, candidates: torch.Tensor,
                        generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _ = selected.shape
    alternative = selected.clone()
    kinds = torch.zeros(batch, dtype=torch.long, device=selected.device)
    random_values = torch.rand(batch, generator=generator, device=selected.device)
    kinds[random_values.ge(0.75) & random_values.lt(0.85)] = 1  # inside
    kinds[random_values.ge(0.85) & random_values.lt(0.95)] = 2  # outside
    kinds[random_values.ge(0.95)] = 3  # double
    bank = router.num_circuits
    for index in torch.nonzero(kinds, as_tuple=False).flatten().tolist():
        current = set(int(value) for value in selected[index].tolist())
        pool = set(int(value) for value in candidates[index].tolist())
        if kinds[index].item() == 1:
            options = sorted(pool - current)
            if not options:
                kinds[index] = 0
                continue
            replacement = options[int(torch.randint(len(options), (), generator=generator,
                                                    device=selected.device).item())]
            side = int(torch.randint(2, (), generator=generator, device=selected.device).item())
            alternative[index, side] = replacement
        elif kinds[index].item() == 2:
            options = sorted(set(range(bank)) - pool)
            replacement = options[int(torch.randint(len(options), (), generator=generator,
                                                    device=selected.device).item())]
            side = int(torch.randint(2, (), generator=generator, device=selected.device).item())
            alternative[index, side] = replacement
        else:
            options = sorted(set(range(bank)) - current)
            first = options[int(torch.randint(len(options), (), generator=generator,
                                              device=selected.device).item())]
            options.remove(first)
            second = options[int(torch.randint(len(options), (), generator=generator,
                                               device=selected.device).item())]
            alternative[index] = torch.tensor([first, second], device=selected.device)
    return alternative, kinds, kinds.ne(0)


def _set_trainable(router: torch.nn.Module, mode: str) -> list[torch.nn.Parameter]:
    retriever_names = {"retriever_query", "keys"}
    selector_names = {"utility_query", "utility_keys", "pair_embeddings", "interaction_query"}
    parameters = []
    for name, parameter in router.named_parameters():
        root = name.split(".", 1)[0]
        trainable = mode == "full" or (mode == "retrieval-only" and root in retriever_names) \
            or (mode == "selection-only" and root in selector_names)
        parameter.requires_grad_(trainable)
        if trainable:
            parameters.append(parameter)
    return parameters


@torch.inference_mode()
def evaluate(model: NeuralEngineV0, batches: list) -> dict[str, float]:
    losses = []
    correct = 0
    examples = 0
    num_circuits = model.circuits.num_circuits
    selected_counts = torch.zeros(num_circuits, dtype=torch.long,
                                   device=next(model.parameters()).device)
    model.eval()
    for batch in batches:
        logits, stats = model(batch.inputs, adaptive=False)
        losses.append(F.cross_entropy(logits, batch.targets).item())
        correct += int(logits.argmax(-1).eq(batch.targets).sum())
        examples += batch.targets.numel()
        selected_counts.scatter_add_(
            0, stats["selected_ids"].reshape(-1),
            torch.ones(stats["selected_ids"].numel(), dtype=torch.long,
                       device=selected_counts.device),
        )
    used = int(selected_counts.gt(0).sum())
    return {
        "loss": sum(losses) / len(losses),
        "accuracy": correct / examples,
        "examples": examples,
        "dead_circuits": int(num_circuits - used),
        "used_circuits": used,
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    teacher, probe, config = load_models(Path(args.checkpoint), device)
    if not hasattr(probe.router, "selection_scores"):
        raise ValueError("checkpoint probe router was not constructed")
    train_data = train_batches(config, device, int(config["seed"]), args.batch_count,
                               args.examples_per_task)
    heldout_data = heldout_batches(config, device, args.eval_batches, args.examples_per_task)
    teacher_eval = evaluate(teacher, heldout_data)
    trainable = _set_trainable(probe.router, args.mode)
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4) \
        if trainable else None
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 991)
    kind_counts = {"inside": 0, "outside": 0, "double": 0}
    positive_probes = 0
    losses = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = train_data[(step - 1) % len(train_data)]
        if optimizer is None:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if step <= args.imitation_steps:
                _, teacher_stats = teacher(batch.inputs, adaptive=False)
                step_index = int(torch.randint(teacher.internal_steps, (), generator=generator,
                                               device=device).item())
                query = teacher_stats["query_states"][:, step_index]
                teacher_candidates = teacher_stats["candidate_ids"][:, step_index]
                teacher_pairs = teacher_stats["selected_ids"][:, step_index]
            else:
                _, current_stats = probe(batch.inputs, adaptive=False)
                current_pairs = current_stats["selected_ids"]
                candidates = current_stats["candidate_ids"]
                step_index = int(torch.randint(probe.internal_steps, (), generator=generator,
                                               device=device).item())
                query = current_stats["query_states"][:, step_index].detach()
                current_step_pairs = current_pairs[:, step_index]
                candidate_step_ids = candidates[:, step_index]
                alternative, kinds, probed = _sample_probe_pairs(
                    probe.router, current_step_pairs, candidate_step_ids, generator,
                )
                plan = torch.full_like(current_pairs, -1)
                plan[:, step_index] = current_step_pairs
                plan[probed, step_index] = alternative[probed]
                current_loss = F.cross_entropy(
                    probe(batch.inputs, adaptive=False)[0], batch.targets, reduction="none",
                )
                alternative_loss = F.cross_entropy(
                    probe(batch.inputs, adaptive=False, forced_selected_ids=plan)[0],
                    batch.targets, reduction="none",
                )
                delta = current_loss - alternative_loss
                probe_mask = probed & delta.abs().isfinite()
                retrieval_mask = probe_mask & kinds.ge(2) & delta.gt(0.02)
                if bool(retrieval_mask.any()):
                    retrieval_weight = (
                        (delta.abs() / 0.1).clamp_max(2.0).detach()
                        * retrieval_mask.float()
                    )
                else:
                    retrieval_weight = None
                for kind, name in ((1, "inside"), (2, "outside"), (3, "double")):
                    kind_counts[name] += int((kinds == kind).sum())
                positive_probes += int(retrieval_mask.sum())
            
        with torch.enable_grad():
            if step <= args.imitation_steps:
                loss = _imitation_loss(probe.router, query, teacher_candidates,
                                       teacher_pairs, args.mode)
            else:
                selection = _selection_loss(
                    probe.router, query, current_step_pairs, alternative, delta, probe_mask,
                )
                retrieval = (
                    _retrieval_inclusion_loss(
                        probe.router, query, alternative, retrieval_weight,
                    ) if retrieval_weight is not None
                    else probe.router.retriever_query.weight.sum() * 0.0
                )
                if args.mode == "selection-only":
                    loss = selection
                elif args.mode == "retrieval-only":
                    loss = 0.1 * retrieval
                else:
                    loss = selection + 0.1 * retrieval
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    probe_eval = evaluate(probe, heldout_data)
    elapsed = time.perf_counter() - start
    return {
        "experiment": "probe_route_frozen",
        "checkpoint": str(args.checkpoint),
        "seed": int(config["seed"]),
        "mode": args.mode,
        "steps": args.steps,
        "imitation_steps": args.imitation_steps,
        "batch_count": args.batch_count,
        "examples_per_task": args.examples_per_task,
        "trainable_router_params": sum(parameter.numel() for parameter in trainable),
        "training_seconds": elapsed,
        "mean_train_router_loss": sum(losses) / len(losses) if losses else 0.0,
        "probe_kind_counts": kind_counts,
        "positive_retrieval_probes": positive_probes,
        "teacher_heldout": teacher_eval,
        "probe_heldout": probe_eval,
        "heldout_loss_delta_vs_teacher": probe_eval["loss"] - teacher_eval["loss"],
        "heldout_accuracy_delta_vs_teacher": probe_eval["accuracy"] - teacher_eval["accuracy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-bank ProbeRoute-2 trainer")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("initialization-only", "selection-only",
                                            "retrieval-only", "full"), default="full")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--imitation-steps", type=int, default=200)
    parser.add_argument("--batch-count", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--examples-per-task", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
