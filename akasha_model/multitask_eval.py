"""Benchmark the shared-context Choice/Score/Noul model."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import select_device
from .multitask import (
    MultiQuestionCollator, MultiQuestionDataset, MultiQuestionTinyScorer,
    apply_multitask_calibration,
)


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def load_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = MultiQuestionTinyScorer(
        width=config["width"],
        rank=config["rank"],
        context_tokens=config["context_tokens"],
        question_tokens=config["question_tokens"],
        max_score_levels=config.get("max_score_levels", 10),
        encoder=config.get("encoder", "mean"),
        lexical_multiplier=config.get("lexical_multiplier", 64.0),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device), config


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ece(confidence: list[float], correct: list[float], bins: int = 10) -> float:
    if not confidence:
        return 0.0
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [i for i, value in enumerate(confidence)
                    if lower <= value < upper or (index == bins - 1 and value == 1)]
        if selected:
            accuracy = _mean([correct[i] for i in selected])
            average_confidence = _mean([confidence[i] for i in selected])
            error += len(selected) / total * abs(accuracy - average_confidence)
    return error


def _coverage_risk(confidence: list[float], correct: list[float]) -> dict[str, dict[str, float]]:
    result = {}
    for threshold in THRESHOLDS:
        selected = [i for i, value in enumerate(confidence) if value >= threshold]
        coverage = len(selected) / max(len(confidence), 1)
        accuracy = _mean([correct[i] for i in selected]) if selected else 0.0
        result[str(threshold)] = {
            "coverage": coverage,
            "accuracy": accuracy,
            "risk": 1.0 - accuracy if selected else 0.0,
        }
    return result


def _confidence(probabilities: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    count = (mask.sum(1) if mask is not None else
             torch.full((probabilities.shape[0],), probabilities.shape[1], device=probabilities.device))
    entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(1)
    return (1 - entropy / count.to(probabilities.dtype).clamp_min(2).log()).clamp(0, 1)


def _store(store: dict[str, dict[str, list]], category: str, values: dict[str, Any]) -> None:
    target = store[category]
    for name, value in values.items():
        target.setdefault(name, []).extend(value if isinstance(value, list) else [value])


def _choice_metrics(values: dict[str, list]) -> dict[str, Any]:
    return {
        "examples": len(values.get("correct", [])),
        "top1": _mean(values.get("correct", [])),
        "top3": _mean(values.get("top3", [])),
        "nll": _mean(values.get("nll", [])),
        "brier": _mean(values.get("brier", [])),
        "ece": _ece(values.get("confidence", []), values.get("correct", [])),
        "mean_confidence": _mean(values.get("confidence", [])),
        "coverage_risk": _coverage_risk(values.get("confidence", []), values.get("correct", [])),
    }


def _score_metrics(values: dict[str, list]) -> dict[str, Any]:
    return {
        "examples": len(values.get("exact", [])),
        "exact_accuracy": _mean(values.get("exact", [])),
        "within_one": _mean(values.get("within_one", [])),
        "mae": _mean(values.get("mae", [])),
        "nll": _mean(values.get("nll", [])),
        "brier": _mean(values.get("brier", [])),
        "ece_within_one": _ece(values.get("confidence", []), values.get("within_one", [])),
        "mean_confidence": _mean(values.get("confidence", [])),
        "coverage_risk_within_one": _coverage_risk(
            values.get("confidence", []), values.get("within_one", []),
        ),
    }


def _binary_auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    positive_scores = [score for score, label in zip(scores, labels) if label]
    negative_scores = [score for score, label in zip(scores, labels) if not label]
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positive_scores for negative in negative_scores
    )
    return wins / (positives * negatives)


def _binary_metrics(values: dict[str, list]) -> dict[str, Any]:
    correct = values.get("correct", [])
    labels = values.get("labels", [])
    scores = values.get("scores", [])
    return {
        "examples": len(correct),
        "accuracy": _mean(correct),
        "brier": _mean(values.get("brier", [])),
        "log_loss": _mean(values.get("log_loss", [])),
        "auroc": _binary_auc(scores, labels),
        "ece": _ece(values.get("confidence", []), correct),
        "mean_confidence": _mean(values.get("confidence", [])),
        "coverage_risk": _coverage_risk(values.get("confidence", []), correct),
    }


def _summarise(store: dict[str, dict[str, list]], kind: str) -> dict[str, Any]:
    summariser = {"choice": _choice_metrics, "score": _score_metrics,
                  "noul": _binary_metrics}[kind]
    return {category: summariser(values) for category, values in sorted(store.items())}


@torch.no_grad()
def evaluate(model, path: str | Path, device: torch.device,
             batch_size: int = 64, calibration: dict[str, float] | None = None) -> dict[str, Any]:
    dataset = MultiQuestionDataset(path)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=MultiQuestionCollator(
            model.context_position.num_embeddings,
            model.question_position.num_embeddings,
            384,
            model.max_score_levels,
        ),
    )
    raw_metadata = []
    metadata_path = Path(path).with_name(Path(path).stem + ".metadata.jsonl")
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    raw_metadata.append(item.get("stress_type") or item.get("variant", "default"))
    else:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    raw_metadata.append(item.get("stress_type", "default"))
    stores = {kind: defaultdict(dict) for kind in ("choice", "score", "noul")}
    offset = 0
    model.eval()
    for batch in loader:
        batch = {
            kind: {name: value.to(device) for name, value in values.items()}
            for kind, values in batch.items()
        }
        outputs = apply_multitask_calibration(model(batch), calibration)
        batch_size_states = batch["states"]["context_ids"].shape[0]
        categories = [raw_metadata[offset + index] for index in range(batch_size_states)]
        offset += batch_size_states

        if "choice" in outputs:
            data, logits = batch["choice"], outputs["choice"]
            probabilities = logits.softmax(-1)
            labels = data["labels"]
            mask = data["option_mask"]
            confidence = _confidence(probabilities, mask)
            prediction = probabilities.argmax(-1)
            top3 = logits.topk(min(3, logits.shape[1]), dim=-1).indices
            correct = prediction.eq(labels).float()
            top3_correct = top3.eq(labels[:, None]).any(1).float()
            nll = -probabilities.gather(1, labels[:, None]).clamp_min(1e-12).log()
            targets = F.one_hot(labels, num_classes=logits.shape[1]).to(probabilities.dtype)
            brier = (probabilities - targets).square().sum(1)
            for index, state_index in enumerate(data["state_indices"].tolist()):
                _store(stores["choice"], categories[state_index], {
                    "correct": float(correct[index]), "top3": float(top3_correct[index]),
                    "nll": float(nll[index]), "brier": float(brier[index]),
                    "confidence": float(confidence[index]),
                })

        if "score" in outputs:
            data, logits = batch["score"], outputs["score"]
            probabilities = logits.softmax(-1)
            labels = data["labels"]
            confidence = _confidence(probabilities, data["level_mask"])
            levels = torch.arange(logits.shape[1], device=device, dtype=probabilities.dtype)
            expected = (probabilities * levels).sum(1)
            prediction = probabilities.argmax(-1)
            exact = prediction.eq(labels).float()
            within_one = (expected - labels).abs().le(1).float()
            mae = (expected - labels).abs()
            nll = -probabilities.gather(1, labels[:, None]).clamp_min(1e-12).log()
            targets = F.one_hot(labels, num_classes=logits.shape[1]).to(probabilities.dtype)
            brier = (probabilities - targets).square().sum(1)
            for index, state_index in enumerate(data["state_indices"].tolist()):
                _store(stores["score"], categories[state_index], {
                    "exact": float(exact[index]), "within_one": float(within_one[index]),
                    "mae": float(mae[index]), "nll": float(nll[index]),
                    "brier": float(brier[index]), "confidence": float(confidence[index]),
                })

        if "noul" in outputs:
            data, logits = batch["noul"], outputs["noul"]
            probabilities = logits.sigmoid()
            labels = data["labels"].long()
            prediction = probabilities.ge(0.5).long()
            correct = prediction.eq(labels).float()
            confidence = (2 * (probabilities - 0.5).abs()).clamp(0, 1)
            brier = (probabilities - labels.float()).square()
            log_loss = -(
                labels.float() * probabilities.clamp_min(1e-12).log() +
                (1 - labels.float()) * (1 - probabilities).clamp_min(1e-12).log()
            )
            for index, state_index in enumerate(data["state_indices"].tolist()):
                _store(stores["noul"], categories[state_index], {
                    "correct": float(correct[index]), "brier": float(brier[index]),
                    "log_loss": float(log_loss[index]), "confidence": float(confidence[index]),
                    "scores": float(probabilities[index]), "labels": int(labels[index]),
                })

    return {kind: _summarise(store, kind) for kind, store in stores.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data")
    parser.add_argument("--stress", help="optional stress JSONL evaluated separately")
    parser.add_argument("--output", help="write the report as JSON")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, config = load_checkpoint(args.checkpoint, device)
    report = {
        "checkpoint": str(args.checkpoint),
        "config": config,
        "device": str(device),
        "test": evaluate(model, args.data, device, args.batch_size, config.get("calibration")),
    }
    if args.stress:
        report["stress"] = evaluate(model, args.stress, device, args.batch_size, config.get("calibration"))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
