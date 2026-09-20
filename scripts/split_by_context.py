"""Create leakage-resistant JSONL splits by keeping each context in one split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def bucket(context: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{context}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", default="context-split-v1")
    args = parser.parse_args()
    if args.validation_ratio < 0 or args.test_ratio < 0:
        parser.error("split ratios must be non-negative")
    if args.validation_ratio + args.test_ratio >= 1:
        parser.error("validation and test ratios must sum to less than one")

    rows = []
    for path in sorted(args.input.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise SystemExit(f"no JSONL rows found in {args.input}")

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["context"]].append(row)

    splits = {"train": [], "validation": [], "test": []}
    for context, context_rows in grouped.items():
        value = bucket(context, args.seed)
        if value < args.test_ratio:
            split = "test"
        elif value < args.test_ratio + args.validation_ratio:
            split = "validation"
        else:
            split = "train"
        splits[split].extend(context_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        with (args.output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        split: {"rows": len(items), "contexts": len({row["context"] for row in items})}
        for split, items in splits.items()
    }, sort_keys=True))


if __name__ == "__main__":
    main()
