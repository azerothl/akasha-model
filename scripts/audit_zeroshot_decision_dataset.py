#!/usr/bin/env python3
"""Audit a zero-shot decision dataset for leakage / mix failures.

Fails (exit 1) on critical leakage: overlapping split_group across train and
held-out splits, or missing required files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REQUIRED = (
    "train.jsonl",
    "val.jsonl",
    "test_domain.jsonl",
    "test_labels.jsonl",
    "stress.jsonl",
)


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        print(json.dumps({"ok": False, "critical": [f"missing:{m}" for m in missing]}))
        return 1

    splits = {name: _load(root / name) for name in REQUIRED}
    train_groups = {row["split_group"] for row in splits["train.jsonl"]}
    critical: list[str] = []
    collisions = {}
    for name in ("val.jsonl", "test_domain.jsonl", "test_labels.jsonl"):
        overlap = sorted(train_groups & {row["split_group"] for row in splits[name]})
        collisions[name] = len(overlap)
        if overlap:
            critical.append(f"split_group_leak:{name}:{len(overlap)}")

    train_domains = {row["domain"] for row in splits["train.jsonl"]}
    test_domains = {row["domain"] for row in splits["test_domain.jsonl"]}
    domain_holdout_ok = train_domains.isdisjoint(test_domains)
    if not domain_holdout_ok:
        critical.append("domain_holdout_failed")

    surfaces = Counter(
        row.get("meta", {}).get("label_surface", "?") for row in splits["train.jsonl"]
    )
    arbitrary_share = surfaces.get("arbitrary", 0) / max(1, len(splits["train.jsonl"]))

    gates = {
        "unseen_domain_choice_acc": None,
        "unseen_label_choice_acc": None,
        "description_ablation_acc": None,
        "name_only_ablation_acc": None,
        "ece_after_calibration": None,
    }
    report = {
        "ok": not critical,
        "critical": critical,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "collisions_split_group_with_train": collisions,
        "domain_holdout_ok": domain_holdout_ok,
        "train_domains": sorted(train_domains),
        "test_domain_domains": sorted(test_domains),
        "arbitrary_label_share_train": arbitrary_share,
        "label_surface_counts_train": dict(surfaces),
        "gates_template": gates,
        "note": (
            "Optional Path B probe only. Gate identity remains Path A "
            "authorization; not a Jev/System-One cutover gate."
        ),
    }
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "gates.json").write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
