#!/usr/bin/env python3
"""Import host authorize-export JSONL into Path B multitask splits.

Public hook for [#44](https://github.com/azerothl/akasha-model/issues/44).
Does not execute tools. Does not read private OS stores — only a local JSONL
file the host already exported/scrubbed.

Anti-leakage: rows are hashed into train/validation/test by ``split_group``
(or ``situation_id``), never by random shuffle of the same identity.

Torch-free: builds the same Choice/Score/Noul question shape as
``gate_questions_template`` without importing ``gate_multitask``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from akasha_model.tool_calling import ToolSpec

from catalog import tool_catalog


RISK_LEVELS = ("low", "medium", "high")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _bucket(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # Stable 80/10/10 by hex nibble — identities never cross splits.
    nibble = int(digest[0], 16)
    if nibble < 13:  # 0..12 → ~81%
        return "train"
    if nibble < 15:  # 13..14 → ~12.5%
        return "validation"
    return "test"


def _route_label(tools: Mapping[str, ToolSpec], tool_name: str) -> int:
    names = sorted(tools)
    if tool_name not in names:
        raise SystemExit(f"route_tool {tool_name!r} not in catalog: {names}")
    return names.index(tool_name)


def _questions_template(
    tools: Mapping[str, ToolSpec], *, route_label: int = 0,
) -> list[dict[str, Any]]:
    options = [
        {"name": name, "description": tools[name].description}
        for name in sorted(tools)
    ]
    if not 0 <= route_label < len(options):
        raise SystemExit("route_label out of range for the tool catalog")
    return [
        {
            "id": "route",
            "type": "choice",
            "instructions": (
                "Choose which catalog tool should be authorized for execution "
                "given the proposed call and situation."
            ),
            "options": options,
            "label": route_label,
        },
        {
            "id": "risk",
            "type": "score",
            "instructions": (
                "Rate the operational risk of executing the proposed tool call."
            ),
            "levels": list(RISK_LEVELS),
            "label": 0,
        },
        {
            "id": "authorized",
            "type": "noul",
            "instructions": "Is the operation authorized in this situation?",
            "criteria": {
                "true": "Policy and user intent allow this tool call.",
                "false": "Authorization is missing or the call is out of policy.",
            },
            "label": 0,
        },
        {
            "id": "sufficient_context",
            "type": "noul",
            "instructions": "Is there enough context to execute safely?",
            "criteria": {
                "true": "Arguments and situation are clear enough to proceed.",
                "false": "Key details are missing or the request is ambiguous.",
            },
            "label": 0,
        },
        {
            "id": "capability_present",
            "type": "noul",
            "instructions": "Does the host hold the required capability?",
            "criteria": {
                "true": "The required capability is granted for this session.",
                "false": "The required capability is absent or revoked.",
            },
            "label": 0,
        },
        {
            "id": "confirmation_needed",
            "type": "noul",
            "instructions": "Is explicit human confirmation still required?",
            "criteria": {
                "true": "The call is irreversible or marked as needing confirmation.",
                "false": "No further human confirmation is required.",
            },
            "label": 0,
        },
    ]


def _to_multitask(row: dict, tools: Mapping[str, ToolSpec]) -> dict:
    labels = row.get("labels") or {}
    proposal = row.get("proposal") or {}
    tool_name = labels.get("route_tool") or proposal.get("tool_name")
    if not tool_name:
        raise SystemExit(f"row missing route_tool / proposal.tool_name: {row!r}")
    route = _route_label(tools, tool_name)
    questions = _questions_template(tools, route_label=route)
    by_id = {question["id"]: question for question in questions}
    by_id["risk"]["label"] = int(labels.get("risk", 0))
    for noul_id in (
        "authorized",
        "sufficient_context",
        "capability_present",
        "confirmation_needed",
    ):
        by_id[noul_id]["label"] = int(labels.get(noul_id, 0))
    context = {
        "situation_id": row.get("situation_id"),
        "split_group": row.get("split_group") or row.get("situation_id"),
        "kind": "host_export",
        "expected_status": row.get("expected_status"),
        "proposal": proposal.get("tool_name"),
        "arguments": proposal.get("arguments") or {},
        "confirmation_given": bool(row.get("confirmation_given", False)),
        "situation": row.get("situation") or "",
    }
    return {"context": context, "questions": list(by_id.values())}


def write_splits(rows: list[dict], output: Path) -> dict[str, int]:
    tools = tool_catalog()
    buckets: dict[str, list[dict]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row in rows:
        key = str(row.get("split_group") or row.get("situation_id") or "")
        if not key:
            raise SystemExit("each row needs split_group or situation_id")
        buckets[_bucket(key)].append(_to_multitask(row, tools))

    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, items in buckets.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        counts[split] = len(items)
    meta = {
        "source": "host_authorize_export",
        "counts": counts,
        "note": (
            "Imported Path B authorize rows. Demo/prod checkpoint policy: "
            "gate-tiny.pt is demo-only; keep host weights out of git."
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8",
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = _load_jsonl(args.input)
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    counts = write_splits(rows, args.output)
    print(json.dumps({"output": str(args.output), "counts": counts}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
