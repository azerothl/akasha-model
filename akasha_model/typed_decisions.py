"""Convert and score LocalLLaMA/typed-decisions without mixing official splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from .decision import MaskCollator, load_decision_checkpoint
from .model import select_device
from .multitask import validate_multi
from .multitask_eval import _ece, _mean
from .rewards import confidence_from_probs

PUBLISHED = {
    "jev": {
        "source": "TypeSafe Jev 1.13.0, third-party published, not measured here",
        "accuracy": 0.727,
        "soft_accuracy": 0.580,
        "brier": 0.148,
        "ece": 0.144,
        "score_mae": 0.391,
    },
    "laya_typed_decisions": {
        "source": "convaiinnovations/laya-typed-decisions, published by Laya",
        "accuracy": 0.766,
        "soft_accuracy": 0.471,
        "brier": 0.062,
        "ece": 0.213,
        "score_mae": 0.242,
    },
}

def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def convert_typed_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map one Hub row onto the Akasha multitask JSONL schema."""
    state = _parse_json(row["state"])
    questions = _parse_json(row["questions"])
    gold = _parse_json(row["gold"])
    converted = []
    for question_id, definition in questions.items():
        kind = definition["type"]
        answer = gold[question_id]
        item: dict[str, Any] = {
            "id": question_id,
            "type": kind,
            "instructions": definition["instructions"],
        }
        if kind == "choice":
            criteria = definition["criteria"]
            names = list(criteria.keys())
            item["options"] = [
                {"name": name, "description": criteria[name] or ""} for name in names
            ]
            label = answer["label"]
            item["label"] = names.index(label) if isinstance(label, str) else int(label)
            probabilities = answer.get("probabilities") or {}
            item["target"] = [float(probabilities.get(name, 0.0)) for name in names]
        elif kind == "score":
            levels = list(definition["criteria"])
            item["levels"] = levels
            item["label"] = int(answer["label"])
            probabilities = answer.get("probabilities") or {}
            item["target"] = [
                float(probabilities.get(str(index), probabilities.get(index, 0.0)))
                for index in range(len(levels))
            ]
        else:
            criteria = definition.get("criteria") or {}
            if isinstance(criteria, dict) and criteria.get("true") and criteria.get("false"):
                item["criteria"] = criteria
            probability = answer.get("noul")
            if probability is None:
                probability = answer.get("probabilities", {}).get("true", 0.5)
            probability = float(probability)
            item["label"] = int(probability >= 0.5)
            item["target"] = [1.0 - probability, probability]
        converted.append(item)
    return {
        "context": state,
        "questions": converted,
        "workflow": row.get("workflow"),
        "id": row.get("id"),
    }


def write_typed_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _hub_split(config: str, split: str):
    if load_dataset is None:
        raise ImportError("Hub access requires akasha-model[eval] (datasets)")
    return load_dataset("LocalLLaMA/typed-decisions", config, split=split)


def prepare_typed_decisions(output: Path, config: str = "all") -> dict[str, int]:
    """Download official Hub splits and write JSONL without reshuffling families."""
    counts = {}
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        dataset = _hub_split(config, split)
        rows = [convert_typed_row(item) for item in dataset]
        write_typed_jsonl(rows, output / f"{split}.jsonl")
        counts[split] = len(rows)
    return counts


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarise_typed_records(records: list[dict[str, Any]], permutations: int = 1) -> dict[str, Any]:
    """Aggregate question records while retaining the historical entropy ECE."""
    result = {}
    for name in ("all", "choice", "score", "noul"):
        selected = [item for item in records if name == "all" or item["kind"] == name]
        values = lambda key: [float(item[key]) for item in selected if key in item]
        summary = {
            "examples": len(selected),
            "accuracy": _mean(values("correct")),
            "label_accuracy": _mean(values("label_correct")),
            "soft_accuracy": _mean(values("soft")),
            "brier": _mean(values("brier")),
            "ece": _ece(values("confidence"), values("correct")),
            "nll": _mean(values("nll")),
            "ece_maxprob": _ece(values("max_probability"), values("label_correct"), bins=15),
            "score_mae": _mean(values("score_mae")) if values("score_mae") else None,
        }
        if permutations > 1:
            summary["permutation_stability"] = {
                "permutations": permutations,
                "top1_flip_rate": _mean(values("top1_flip_rate")),
                "mean_js_divergence": _mean(values("mean_js_divergence")),
            }
        result[name] = summary
    return result


@torch.no_grad()
def evaluate_typed_records(
    model, tokenizer, rows: list[dict[str, Any]], device: torch.device,
    max_len: int, head_max_len: int, option_max_len: int, batch_size: int = 8,
    permutations: int = 1, seed: int = 7,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if permutations < 1:
        raise ValueError("permutations must be at least one")
    collator = MaskCollator(
        tokenizer, max_len, head_max_len, option_max_len,
        group_size=permutations, seed=seed,
    )
    examples = [validate_multi(row) for row in rows]
    records: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        batch = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in collator(chunk).items()
        }
        logits = model(
            batch["input_ids"], batch["attention_mask"],
            batch["marker_pos"], batch["marker_mask"], batch["qtype"],
        )
        reported = torch.softmax(
            logits.masked_fill(~batch["marker_mask"], torch.finfo(logits.dtype).min), -1,
        )
        index = 0
        for state_offset, example in enumerate(chunk):
            for question in example.questions:
                count = int(batch["marker_mask"][index].sum())
                orderings = collator._orders(count, state_offset)
                aligned = []
                for order in orderings:
                    probabilities = reported[index, :count].cpu().numpy()
                    canonical = np.empty(count, dtype=np.float64)
                    canonical[order] = probabilities
                    aligned.append(canonical)
                    index += 1
                predicted = aligned[0]
                target = batch["target"][index - permutations, :count].cpu().numpy()
                qtype = int(batch["qtype"][index - permutations])
                kind = {0: "choice", 1: "score", 2: "noul"}[qtype]
                record = {
                    "row_index": start + state_offset,
                    "question_id": question["id"],
                    "kind": kind,
                    "correct": int(predicted.argmax() == target.argmax()),
                    "label_correct": int(predicted.argmax() == int(batch["label"][index - permutations])),
                    "soft": float((predicted * target).sum()),
                    "brier": float(((predicted - target) ** 2).sum()),
                    "nll": float(-(target * np.log(np.clip(predicted, 1e-12, 1))).sum()),
                    "confidence": confidence_from_probs(predicted, count),
                    "max_probability": float(predicted.max()),
                }
                if kind == "score":
                    levels = np.arange(count)
                    record["score_mae"] = abs(
                        float(np.dot(predicted, levels) - np.dot(target, levels))
                    )
                if permutations > 1:
                    distributions = np.stack(aligned)
                    mean = distributions.mean(axis=0)
                    record["top1_flip_rate"] = float(np.mean(
                        distributions[1:].argmax(axis=1) != predicted.argmax()
                    ))
                    record["mean_js_divergence"] = float(np.mean(np.sum(
                        distributions * (
                            np.log(np.clip(distributions, 1e-12, 1))
                            - np.log(np.clip(mean, 1e-12, 1))
                        ), axis=1,
                    )))
                records.append(record)
        if index != reported.shape[0]:
            raise ValueError("permutation alignment did not consume the full batch")
    return summarise_typed_records(records, permutations), records


def evaluate_typed_rows(
    model, tokenizer, rows: list[dict[str, Any]], device: torch.device,
    max_len: int, head_max_len: int, option_max_len: int, batch_size: int = 8,
    permutations: int = 1,
) -> dict[str, Any]:
    return evaluate_typed_records(
        model, tokenizer, rows, device, max_len, head_max_len, option_max_len,
        batch_size, permutations,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", help="MASK-format RLCD checkpoint")
    parser.add_argument("--prepare", type=Path, help="download Hub splits into this directory")
    parser.add_argument("--config", default="all", help="Hub config name, default all")
    parser.add_argument("--data", help="local JSONL to evaluate instead of downloading")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--output", help="write the JSON report")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=1,
                        help="measure prediction stability over option orderings")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.prepare:
        counts = prepare_typed_decisions(args.prepare, args.config)
        print(json.dumps({"prepared": str(args.prepare), "counts": counts}))
        if args.checkpoint is None:
            return
    if args.checkpoint is None:
        parser.error("pass a checkpoint, or use --prepare on its own")
    device = select_device(args.device)
    model, tokenizer, config = load_decision_checkpoint(args.checkpoint, device)
    if args.data:
        rows = _load_jsonl(Path(args.data))
        source = args.data
    else:
        dataset = _hub_split(args.config, args.split)
        rows = [convert_typed_row(item) for item in dataset]
        source = f"LocalLLaMA/typed-decisions:{args.config}/{args.split}"
    report = {
        "checkpoint": args.checkpoint,
        "source": source,
        "device": str(device),
        "metrics": evaluate_typed_rows(
            model, tokenizer, rows, device,
            config.get("max_len", 768), config.get("head_max_len", 384),
            config.get("option_max_len", 48), args.batch_size, args.permutations,
        ),
        "published_comparison": PUBLISHED,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
