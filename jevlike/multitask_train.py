"""Train the shared-context Choice/Score/Noul scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .model import select_device
from .multitask import (
    MultiQuestionCollator, MultiQuestionDataset, MultiQuestionTinyScorer,
    multitask_loss,
)


def move(batch: dict, device: torch.device) -> dict:
    return {
        kind: {name: value.to(device) for name, value in values.items()}
        for kind, values in batch.items()
    }


def run_epoch(model, loader, device, optimiser=None) -> dict[str, float]:
    training = optimiser is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    for host_batch in loader:
        batch = move(host_batch, device)
        with torch.set_grad_enabled(training):
            losses = multitask_loss(model(batch), batch)
            loss = losses["total"]
        if training:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        count += 1
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {name: value / max(count, 1) for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="runs/multitask.pt")
    parser.add_argument("--init", help="optional checkpoint whose weights should be continued")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=768)
    parser.add_argument("--question-tokens", type=int, default=512)
    parser.add_argument("--option-tokens", type=int, default=384)
    parser.add_argument("--max-score-levels", type=int, default=10)
    parser.add_argument("--encoder", choices=("mean", "local", "lexical"), default="mean")
    parser.add_argument("--lexical-multiplier", type=float, default=64.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    collator = MultiQuestionCollator(
        args.context_tokens, args.question_tokens, args.option_tokens,
        args.max_score_levels,
    )
    train_loader = DataLoader(
        MultiQuestionDataset(args.train), batch_size=args.batch_size,
        shuffle=True, collate_fn=collator,
    )
    validation_loader = DataLoader(
        MultiQuestionDataset(args.validation), batch_size=args.batch_size,
        collate_fn=collator,
    )
    model = MultiQuestionTinyScorer(
        args.width, args.rank, args.context_tokens, args.question_tokens,
        args.max_score_levels, args.encoder,
        args.lexical_multiplier,
    ).to(device)
    if args.init:
        payload = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["state_dict"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, device, optimiser)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, device)
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            best_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        print(json.dumps({
            "epoch": epoch + 1, "train": train_metrics,
            "validation": validation_metrics, "device": str(device),
        }))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {
            "width": args.width, "rank": args.rank,
            "context_tokens": args.context_tokens,
            "question_tokens": args.question_tokens,
            "option_tokens": args.option_tokens,
            "max_score_levels": args.max_score_levels,
            "encoder": args.encoder,
            "lexical_multiplier": args.lexical_multiplier,
        },
        "state_dict": best_state,
    }, output)
    print(json.dumps({"checkpoint": str(output), "best_validation_loss": best}))


if __name__ == "__main__":
    main()
