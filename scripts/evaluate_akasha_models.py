"""Evaluate single checkpoints and a probability-averaged ensemble."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from jevlike.data import JsonlDataset
from jevlike.model import load_checkpoint, select_device
from jevlike.train import move


def ece(confidence, correct):
    result = 0.0
    for lower in torch.linspace(0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            result += float(selected.float().mean() * (correct[selected].float().mean() - confidence[selected].mean()).abs())
    return result


def summarize(probabilities, labels, options, metadata):
    predictions = probabilities.argmax(-1)
    confidence = probabilities.max(-1).values
    correct = predictions.eq(labels)
    gold = [options[i][int(label)] for i, label in enumerate(labels)]
    predicted = [options[i][int(label)] for i, label in enumerate(predictions)]
    names = sorted(set(gold) | set(predicted))
    f1_values = []
    for name in names:
        tp = sum(a == name and b == name for a, b in zip(gold, predicted))
        fp = sum(a != name and b == name for a, b in zip(gold, predicted))
        fn = sum(a == name and b != name for a, b in zip(gold, predicted))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_values.append(2 * precision * recall / max(1e-12, precision + recall))
    abstain = [name == "__abstain__" for name in predicted]
    covered = [not item for item in abstain]
    hard = [m.get("kind") in {"abstain", "ambiguous", "contradictory", "dangerous", "out_of_distribution"} for m in metadata]
    dangerous_errors = sum(h and p != "__abstain__" for h, p in zip(hard, predicted))
    return {
        "accuracy": float(correct.float().mean()),
        "macro_f1": sum(f1_values) / max(1, len(f1_values)),
        "nll": float(F.nll_loss(probabilities.clamp_min(1e-12).log(), labels)),
        "brier": float((probabilities - F.one_hot(labels, probabilities.shape[-1]).to(probabilities.dtype)).square().sum(-1).mean()),
        "ece": ece(confidence, correct),
        "abstention_rate": sum(abstain) / max(1, len(abstain)),
        "coverage": sum(covered) / max(1, len(covered)),
        "conditional_accuracy_after_abstention": sum(c and p for c, p in zip(correct.tolist(), covered)) / max(1, sum(covered)),
        "dangerous_error_rate": dangerous_errors / max(1, sum(hard)),
        "examples": len(labels),
    }


@torch.no_grad()
def evaluate(checkpoints, data_path, metadata_path, device, batch_size):
    loaded = [load_checkpoint(path, device) for path in checkpoints]
    models = [item[0].eval() for item in loaded]
    dataset = JsonlDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=loaded[0][1])
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    all_probs = [[] for _ in models]
    all_options, all_labels, all_meta = [], [], []
    offset = 0
    for host_batch in loader:
        batch = move(host_batch, device)
        probs = [model(batch).softmax(-1).cpu() for model in models]
        for index, value in enumerate(probs):
            all_probs[index].append(value)
        labels = host_batch["labels"].cpu()
        for row in range(len(labels)):
            count = int(host_batch["option_mask"][row].sum())
            all_options.append([dataset[offset + row].options[i] for i in range(count)])
        all_labels.append(labels)
        all_meta.extend(metadata[offset:offset + len(labels)])
        offset += len(labels)
    labels = torch.cat(all_labels)
    probabilities = [torch.cat(items) for items in all_probs]
    width = max(len(item) for item in all_options)
    return {
        Path(path).stem: summarize(value, labels, all_options, all_meta)
        for path, value in zip(checkpoints, probabilities)
    } | {"ensemble": summarize(torch.stack(probabilities).mean(0), labels, all_options, all_meta)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--input", type=Path, default=Path("data/akasha_os_v2"))
    parser.add_argument("--output", type=Path, default=Path("reports/akasha_model_metrics.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    device = select_device(args.device)
    report = {split: evaluate(args.checkpoints, args.input / f"{split}.jsonl", args.input / f"{split}.metadata.jsonl", device, args.batch_size) for split in ("test", "stress")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
