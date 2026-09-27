"""RLCD training: proper-scoring rewards with GRPO-style group baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .decision import (
    MaskCollator, load_decision_checkpoint, make_decision_model, save_decision_checkpoint,
)
from .model import select_device
from .multitask import MultiQuestionDataset
from .rewards import proper_reward


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def reported_distribution(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(scores, dim=-1) * mask


def _mean_by_type(
    values: torch.Tensor, qtype: torch.Tensor, balance: bool,
) -> torch.Tensor:
    """Mean over rows, or the unweighted mean of per-type means (Choice/Score/Noul)."""
    if not balance:
        return values.mean()
    parts = [
        values[qtype == kind].mean()
        for kind in (0, 1, 2)
        if bool((qtype == kind).any())
    ]
    return torch.stack(parts).mean() if parts else values.mean()


def _reward_by_type(reward: torch.Tensor, qtype: torch.Tensor) -> dict[str, torch.Tensor]:
    names = {0: "choice", 1: "score", 2: "noul"}
    stats: dict[str, torch.Tensor] = {}
    for kind, name in names.items():
        selected = reward[qtype == kind]
        stats[f"reward_{name}"] = selected.mean() if selected.numel() else reward.new_zeros(())
    return stats


def grpo_loss(
    logits: torch.Tensor, batch: dict[str, torch.Tensor],
    spherical_weight: float, ranked_weight: float, policy_weight: float,
    type_balance: bool = False,
) -> dict[str, torch.Tensor]:
    mask = batch["marker_mask"]
    qtype = batch["qtype"]
    reported = reported_distribution(logits, mask)
    reward = proper_reward(
        reported, batch["target"], qtype, mask,
        spherical_weight=spherical_weight, ranked_weight=ranked_weight,
    )
    group_size = int(batch["group_size"].reshape(-1)[0])
    grouped = reward.view(-1, group_size)
    advantage = grouped - grouped.mean(dim=1, keepdim=True)
    log_prob = (batch["target"] * torch.log(reported.clamp_min(1e-12))).sum(-1)
    policy_terms = -(advantage.detach().reshape(-1) * log_prob)
    policy = _mean_by_type(policy_terms, qtype, type_balance)
    supervised = _mean_by_type(-reward, qtype, type_balance)
    return {
        "supervised": supervised,
        "policy": policy,
        "reward": reward.mean(),
        "total": supervised + policy_weight * policy,
        **_reward_by_type(reward, qtype),
    }


def run_epoch(model, loader, device, optimiser, args) -> dict[str, float]:
    training = optimiser is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for host_batch in loader:
        batch = move(host_batch, device)
        with torch.set_grad_enabled(training):
            logits = model(
                batch["input_ids"], batch["attention_mask"],
                batch["marker_pos"], batch["marker_mask"], batch["qtype"],
            )
            losses = grpo_loss(
                logits, batch, args.spherical_weight, args.ranked_weight, args.policy_weight,
                type_balance=args.type_balance,
            )
        if training:
            optimiser.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        count += 1
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {name: value / max(count, 1) for name, value in totals.items()}


def _build_optimiser(model, args):
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    encoder = [parameter for parameter in model.parameters()
               if parameter.requires_grad and id(parameter) in encoder_ids]
    head = [parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in encoder_ids]
    return torch.optim.AdamW(
        [
            {"params": encoder, "lr": args.encoder_learning_rate},
            {"params": head, "lr": args.learning_rate},
        ],
        weight_decay=1e-4,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="runs/akasha-rlcd.pt")
    parser.add_argument("--init", help="optional MASK checkpoint to continue")
    parser.add_argument("--encoder", choices=("tiny", "hf"), default="tiny")
    parser.add_argument("--hf-model", default="bert-base-uncased")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=768)
    parser.add_argument("--head-max-len", type=int, default=384)
    parser.add_argument("--option-max-len", type=int, default=48)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--spherical-weight", type=float, default=0.5)
    parser.add_argument("--ranked-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-mode", choices=("provided", "one_hot"), default="provided",
        help="use supplied target distributions or one-hot labels",
    )
    parser.add_argument(
        "--type-balance", action="store_true",
        help="average Choice/Score/Noul losses equally instead of by question count",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.group_size < 1:
        parser.error("--group-size must be at least 1")
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    config = {
        "kind": "mask",
        "encoder": args.encoder,
        "hf_model": args.hf_model,
        "width": args.width,
        "encoder_layers": args.encoder_layers,
        "encoder_heads": args.encoder_heads,
        "head_layers": args.head_layers,
        "max_len": args.max_len,
        "head_max_len": args.head_max_len,
        "option_max_len": args.option_max_len,
        "dropout": args.dropout,
        "policy_weight": args.policy_weight,
        "type_balance": args.type_balance,
        "temperature": [1.0, 1.0, 1.0],
    }
    if args.init:
        model, tokenizer, config = load_decision_checkpoint(args.init, device)
        config = dict(config)
        config.update(policy_weight=args.policy_weight, type_balance=args.type_balance,
                      target_mode=args.target_mode)
    else:
        model, tokenizer = make_decision_model(config, device)
        config["target_mode"] = args.target_mode
    collator = MaskCollator(
        tokenizer, args.max_len, args.head_max_len, args.option_max_len,
        args.group_size, args.seed, args.target_mode,
    )
    train_loader = DataLoader(
        MultiQuestionDataset(args.train), batch_size=args.batch_size,
        shuffle=True, collate_fn=collator,
    )
    validation_loader = DataLoader(
        MultiQuestionDataset(args.validation), batch_size=args.batch_size,
        collate_fn=MaskCollator(
            tokenizer, args.max_len, args.head_max_len, args.option_max_len,
            1, args.seed, args.target_mode,
        ),
    )
    optimiser = _build_optimiser(model, args)
    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimiser, args)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, device, None, args)
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        print(json.dumps({
            "epoch": epoch + 1, "train": train_metrics,
            "validation": validation_metrics, "device": str(device),
        }))
    if best_state is not None:
        model.load_state_dict(best_state)
    output = Path(args.output)
    save_decision_checkpoint(output, model, config, {"best_validation_loss": best})
    print(json.dumps({"checkpoint": str(output), "best_validation_loss": best}))


if __name__ == "__main__":
    main()
