#!/usr/bin/env python3
"""Import Option B (human) mix JSONL. Splits by context_key. Never writes audio.

This does **not** invent labels. Feed annotator JSONL. Schema fixtures with
``annotation_status=format_only`` are for tests only (``--allow-format-fixture``).

Tests that consume the imported files measure **annotator agreement** on the
typed questions — not mix quality, and not Option A written-rule agreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from akasha_model.mix import (
    OPTION_B_LABEL_MODE,
    mix_decision_request_from_row,
    mix_option_b_schema_path,
    row_has_audio_payload,
)

FORMAT_SOURCES = {"akasha-schema-fixture"}
REQUIRED_FIELDS = ("source", "licence", "label_licence", "context_key")


def _bucket(context_key: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{context_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    return rows


def collect_inputs(path: Path) -> list[dict]:
    if path.is_file():
        return _load_jsonl(path)
    if not path.is_dir():
        raise SystemExit(f"input not found: {path}")
    rows: list[dict] = []
    files = sorted(path.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no JSONL files in {path}")
    for file_path in files:
        rows.extend(_load_jsonl(file_path))
    return rows


def validate_row(row: dict, *, allow_format_fixture: bool, index: int) -> list[str]:
    errors: list[str] = []
    if row_has_audio_payload(row):
        errors.append(f"row {index}: audio payload is forbidden")
    for field in REQUIRED_FIELDS:
        if not row.get(field):
            errors.append(f"row {index}: missing {field}")
    if row.get("label_mode") != OPTION_B_LABEL_MODE:
        errors.append(f"row {index}: label_mode must be {OPTION_B_LABEL_MODE}")
    status = row.get("annotation_status", "human")
    source = str(row.get("source") or "")
    is_fixture = status == "format_only" or source in FORMAT_SOURCES
    if is_fixture and not allow_format_fixture:
        errors.append(
            f"row {index}: format fixture refused (pass --allow-format-fixture for tests only)"
        )
    if status == "human" and source in FORMAT_SOURCES:
        errors.append(f"row {index}: fixture source cannot be marked human")
    try:
        mix_decision_request_from_row(row)
    except (ValueError, KeyError, TypeError) as exc:
        errors.append(f"row {index}: {exc}")
    return errors


def split_rows(
    rows: list[dict],
    *,
    seed: str,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        value = _bucket(str(row["context_key"]), seed)
        if value < test_ratio:
            split = "test"
        elif value < test_ratio + validation_ratio:
            split = "validation"
        else:
            split = "train"
        splits[split].append(row)
    return splits


def write_licence_report(rows: list[dict], path: Path) -> dict:
    by_licence = Counter(str(row.get("label_licence") or row.get("licence")) for row in rows)
    by_source = Counter(str(row.get("source")) for row in rows)
    statuses = Counter(str(row.get("annotation_status", "human")) for row in rows)
    report = {
        "rows": len(rows),
        "label_mode": OPTION_B_LABEL_MODE,
        "schema": str(mix_option_b_schema_path().name),
        "measures": (
            "annotator agreement on Choice/Score/Noul — not mix quality, "
            "not Option A written-rule agreement"
        ),
        "proportions_by_licence": dict(by_licence),
        "proportions_by_source": dict(by_source),
        "annotation_status": dict(statuses),
        "audio_committed": False,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", default="mix-context-split-v1")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--allow-format-fixture",
        action="store_true",
        help="accept annotation_status=format_only rows (tests only; never train)",
    )
    args = parser.parse_args()
    rows = collect_inputs(args.input)
    if not rows:
        print(json.dumps({"ok": False, "error": "no rows"}), file=sys.stderr)
        return 1
    errors: list[str] = []
    for index, row in enumerate(rows):
        errors.extend(
            validate_row(row, allow_format_fixture=args.allow_format_fixture, index=index)
        )
    if errors:
        print(json.dumps({"ok": False, "errors": errors[:20], "error_count": len(errors)}))
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "all.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    splits = split_rows(
        rows,
        seed=args.split_seed,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    for name, items in splits.items():
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = write_licence_report(rows, args.output / "licence_report.json")
    print(json.dumps({
        "ok": True,
        "output": str(args.output),
        "rows": len(rows),
        "splits": {name: len(items) for name, items in splits.items()},
        "licence_report": report,
        "human_rows": report["annotation_status"].get("human", 0),
        "format_only_rows": report["annotation_status"].get("format_only", 0),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
