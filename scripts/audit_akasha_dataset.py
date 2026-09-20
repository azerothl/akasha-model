"""Audit an abstention-aware Akasha OS JevLike dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SPLITS = ("train", "validation", "test", "stress")


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\W+", " ", value, flags=re.UNICODE).strip()


def read(path: Path, metadata_path: Path) -> tuple[list[dict], list[dict]]:
    rows, metadata = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("context"), str) or not isinstance(row.get("options"), list):
                raise ValueError(f"invalid row in {path}")
            if len(row["options"]) < 2 or not isinstance(row.get("label"), int):
                raise ValueError(f"invalid options/label in {path}")
            if not 0 <= row["label"] < len(row["options"]):
                raise ValueError(f"label out of range in {path}")
            if any(not isinstance(option, str) or not option for option in row["options"]):
                raise ValueError(f"invalid option in {path}")
            rows.append(row)
    if metadata_path.exists():
        metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(metadata) != len(rows):
            raise ValueError(f"metadata count mismatch for {path}")
    return rows, metadata


def audit(root: Path) -> dict:
    data: dict[str, list[dict]] = {}
    meta: dict[str, list[dict]] = {}
    exact = Counter()
    norm = Counter()
    for split in SPLITS:
        data[split], meta[split] = read(root / f"{split}.jsonl", root / f"{split}.metadata.jsonl")
        exact.update(row["context"] for row in data[split])
        norm.update(normalized(row["context"]) for row in data[split])
    split_sets = {split: set(row["context"] for row in rows) for split, rows in data.items()}
    normalized_sets = {split: set(normalized(row["context"]) for row in rows) for split, rows in data.items()}
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[f"{left}_vs_{right}"] = {
            "exact": len(split_sets[left] & split_sets[right]),
            "normalized": len(normalized_sets[left] & normalized_sets[right]),
        }
    label_counts = {split: Counter(row["options"][row["label"]] for row in rows) for split, rows in data.items()}
    families = {split: set(item.get("family") for item in meta[split]) for split in SPLITS}
    actions = {split: set(label_counts[split]) for split in SPLITS}
    report = {
        "counts": {split: len(data[split]) for split in SPLITS},
        "families": {split: len(families[split]) for split in SPLITS},
        "distinct_actions": len(set().union(*actions.values())),
        "label_distribution": {split: dict(sorted(label_counts[split].items())) for split in SPLITS},
        "abstention_rate": {split: label_counts[split].get("__abstain__", 0) / max(1, len(data[split])) for split in SPLITS},
        "context_length": {split: {"mean": sum(len(x["context"]) for x in data[split]) / max(1, len(data[split])), "max": max((len(x["context"]) for x in data[split]), default=0)} for split in SPLITS},
        "options": {split: sum(len(x["options"]) for x in data[split]) / max(1, len(data[split])) for split in SPLITS},
        "duplicate_exact_contexts": sum(value - 1 for value in exact.values() if value > 1),
        "duplicate_normalized_contexts": sum(value - 1 for value in norm.values() if value > 1),
        "overlaps": overlaps,
        "family_overlap_train_test": len(families["train"] & families["test"]),
        "families_missing_from_test": sorted(families["train"] - families["test"]),
        "underrepresented_actions": sorted(action for action, count in label_counts["train"].items() if count < max(10, len(data["train"]) // max(1, len(actions["train"])) // 2)),
        "schema": "context/options/label; __abstain__ is a valid action",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/akasha_os_v2"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.input)
    output = args.output or args.input / "audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
