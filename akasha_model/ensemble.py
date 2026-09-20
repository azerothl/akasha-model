"""Ensemble evaluation for agreement-based uncertainty and abstention."""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import ChoiceExample, JsonlDataset
from .model import load_checkpoint, select_device
from .train import move


@torch.no_grad()
def metrics(models, loader, device):
    for model in models:
        model.eval()
    correct = total = unanimous = unanimous_correct = 0
    nll = brier = 0.0
    confidences, predictions, labels = [], [], []
    for host_batch in loader:
        batch = move(host_batch, device)
        probabilities = torch.stack([
            model(batch).softmax(-1) for model in models
        ]).mean(0).cpu()
        votes = torch.stack([
            model(batch).argmax(-1).cpu() for model in models
        ])
        batch_labels = batch["labels"].cpu()
        prediction = probabilities.argmax(-1)
        confidence = probabilities.max(-1).values
        is_correct = prediction.eq(batch_labels)
        is_unanimous = votes.eq(votes[0:1]).all(0)
        correct += int(is_correct.sum())
        total += batch_labels.numel()
        unanimous += int(is_unanimous.sum())
        unanimous_correct += int((is_unanimous & is_correct).sum())
        nll += float(-probabilities[torch.arange(len(batch_labels)), batch_labels].clamp_min(1e-12).log().sum())
        targets = torch.nn.functional.one_hot(
            batch_labels, num_classes=probabilities.shape[-1],
        ).to(probabilities.dtype)
        brier += float((probabilities - targets).square().sum(-1).sum())
        confidences.append(confidence)
        predictions.append(prediction)
        labels.append(batch_labels)

    confidence = torch.cat(confidences)
    prediction = torch.cat(predictions)
    labels = torch.cat(labels)
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            gap = prediction[selected].eq(labels[selected]).float().mean()
            gap -= confidence[selected].mean()
            ece += float(selected.float().mean() * gap.abs())
    return {
        "top1": correct / total,
        "ece": ece,
        "nll": nll / total,
        "brier": brier / total,
        "mean_confidence": float(confidence.mean()),
        "unanimous_coverage": unanimous / total,
        "unanimous_accuracy": unanimous_correct / max(1, unanimous),
        "examples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    loaded = [load_checkpoint(path, device) for path in args.checkpoints]
    models = [item[0] for item in loaded]
    collator = loaded[0][1]
    loader = DataLoader(
        JsonlDataset(args.data), batch_size=args.batch_size, collate_fn=collator,
    )
    print(json.dumps({
        "models": args.checkpoints,
        "metrics": metrics(models, loader, device),
        "device": str(device),
    }, indent=2))


def predict_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--context", required=True)
    parser.add_argument("--option", action="append", required=True, dest="options")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if len(args.options) < 2:
        parser.error("at least two --option values are required")
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be between 0 and 1")

    device = select_device(args.device)
    loaded = [load_checkpoint(path, device) for path in args.checkpoints]
    models = [item[0] for item in loaded]
    collator = loaded[0][1]
    example = ChoiceExample(args.context, tuple(args.options), 0)
    batch = move(collator([example]), device)
    with torch.no_grad():
        probabilities = torch.stack([
            model(batch).softmax(-1)[0] for model in models
        ])
    average = probabilities.mean(0)
    predictions = probabilities.argmax(-1)
    confidence, choice = average.max(-1)
    unanimous = bool(predictions.eq(predictions[0]).all())
    abstain = (not unanimous) or float(confidence) < args.confidence_threshold
    print(json.dumps({
        "choice": None if abstain else args.options[int(choice)],
        "abstain": abstain,
        "confidence": float(confidence),
        "unanimous": unanimous,
        "probabilities": {
            option: float(value) for option, value in zip(args.options, average)
        },
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
