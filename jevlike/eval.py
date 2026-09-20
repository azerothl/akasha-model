"""Evaluate accuracy, calibration and a shuffled-context control."""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import JsonlDataset
from .model import load_checkpoint, select_device
from .train import move


def _pad_logits(items: list[torch.Tensor]) -> torch.Tensor:
    width = max(item.shape[1] for item in items)
    fill = -1e4
    return torch.cat([
        F.pad(item.clamp_min(fill), (0, width - item.shape[1]), value=fill)
        for item in items
    ])


@torch.no_grad()
def metrics(model, loader, device, shuffle_context=False):
    model.eval()
    top1 = top3 = total = 0
    confidences, predictions, labels, all_logits = [], [], [], []
    for host_batch in loader:
        batch = move(host_batch, device)
        logits = model(batch, shuffle_context=shuffle_context).cpu()
        probabilities = logits.softmax(-1)
        all_logits.append(logits)
        ranking = logits.topk(min(3, logits.shape[1]), dim=-1).indices
        batch_labels = batch["labels"].cpu()
        top1 += int(ranking[:, 0].eq(batch_labels).sum())
        top3 += int(ranking.eq(batch_labels[:, None]).any(1).sum())
        total += batch_labels.numel()
        confidence, prediction = probabilities.max(-1)
        confidences.append(confidence)
        predictions.append(prediction)
        labels.append(batch["labels"].cpu())
    confidence, prediction, labels = map(torch.cat, (confidences, predictions, labels))
    logits = _pad_logits(all_logits)
    ece = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            gap = prediction[selected].eq(labels[selected]).float().mean()
            gap -= confidence[selected].mean()
            ece += float(selected.float().mean() * gap.abs())
    return {
        "top1": top1 / total,
        "top3": top3 / total,
        "ece": ece,
        "nll": float(F.cross_entropy(logits, labels)),
        "brier": float((logits.softmax(-1) - F.one_hot(labels, logits.shape[1])).square().sum(-1).mean()),
        "examples": labels.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, collator, _ = load_checkpoint(args.checkpoint, device)
    loader = DataLoader(
        JsonlDataset(args.data), batch_size=args.batch_size, collate_fn=collator,
    )
    print(json.dumps({
        "model": metrics(model, loader, device),
        "shuffled_context": metrics(model, loader, device, True),
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
