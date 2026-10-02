#!/usr/bin/env python3
"""Audit a mix JSONL dataset for context leakage and required provenance fields.

Fails (exit 1) when the same ``context_key`` appears in train and a held-out
split, or when ``source`` / ``licence`` fields are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_SPLITS = ("train.jsonl", "validation.jsonl", "test.jsonl")
REQUIRED_FIELDS = ("source", "licence", "label_licence", "descriptor_licence", "context_key")


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    critical: list[str] = []
    missing = [name for name in REQUIRED_SPLITS if not (root / name).is_file()]
    if missing:
        print(json.dumps({"ok": False, "critical": [f"missing:{m}" for m in missing]}))
        return 1

    splits = {name: _load(root / name) for name in REQUIRED_SPLITS}
    for name, rows in splits.items():
        if not rows:
            critical.append(f"empty:{name}")
        for index, row in enumerate(rows):
            for field in REQUIRED_FIELDS:
                if field not in row or row[field] in (None, ""):
                    critical.append(f"missing_field:{name}:{index}:{field}")
                    break
            if "context" not in row:
                critical.append(f"missing_context:{name}:{index}")

    train_keys = {row["context_key"] for row in splits["train.jsonl"] if "context_key" in row}
    for name in ("validation.jsonl", "test.jsonl"):
        overlap = sorted(train_keys & {row.get("context_key") for row in splits[name]})
        if overlap:
            critical.append(f"context_key_leak:{name}:{len(overlap)}")

    ok = not critical
    print(json.dumps({
        "ok": ok,
        "critical": critical,
        "counts": {name: len(rows) for name, rows in splits.items()},
    }, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
