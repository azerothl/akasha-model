"""Generate data, optionally train, and gate proposals with a tiny scorer.

Nothing is executed. Hosts should call their executor only when
``plan.status == "ready"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from akasha_model import ToolProposal, describe_plan
from akasha_model.gate import default_gate_planner
from akasha_model.gate_multitask import (
    load_gate_multitask,
    plan_scored_proposal,
)
from akasha_model.multitask import MultiQuestionCollator
from catalog import tool_catalog
from generate_data import SCENARIO_SPECS, write_splits
from train_tiny import main as train_tiny_main


def _context_for(spec: dict, variant: int = 200) -> dict:
    return {
        "family": spec["family"],
        "kind": spec["kind"],
        "expected_status": spec["expected_status"],
        "variant": variant,
        "proposal": spec["tool"],
        "arguments": spec["arguments"],
        "confirmation_given": spec["confirmation_given"],
        "situation": f"{spec['situation']} [context-id {spec['family']}:{variant}]",
    }


def run_scored_cases(checkpoint: Path, device: torch.device) -> list[dict]:
    model, config = load_gate_multitask(checkpoint, device)
    collator = MultiQuestionCollator(
        config["context_tokens"], config["question_tokens"],
        config.get("option_tokens", 128), config.get("max_score_levels", 10),
    )
    tools = tool_catalog()
    planner = default_gate_planner()
    rows = []
    for spec in SCENARIO_SPECS:
        proposal = ToolProposal(spec["tool"], spec["arguments"])
        plan = plan_scored_proposal(
            model, collator, tools, proposal,
            context=_context_for(spec),
            device=device,
            confirmation_given=spec["confirmation_given"],
            planner=planner,
        )
        expected = spec["expected_status"]
        if expected == "abstain":
            matched = plan.status in {"abstain", "blocked"}
        else:
            matched = plan.status == expected
        rows.append({
            "family": spec["family"],
            "expected": expected,
            "status": plan.status,
            "reason": plan.reason,
            "summary": describe_plan(plan),
            "match": matched,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/gate"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("examples/gate/checkpoints/gate-tiny.pt"),
    )
    parser.add_argument("--train", action="store_true",
                        help="generate JSONL and train before scoring")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    if args.train or not args.checkpoint.is_file():
        write_splits(args.data)
        sys.argv = [
            "train_tiny.py",
            "--data", str(args.data),
            "--output", str(args.checkpoint),
            "--epochs", str(args.epochs),
            "--device", args.device,
        ]
        train_tiny_main()

    rows = run_scored_cases(args.checkpoint, device)
    print("Akasha scored tool gate (no side effects)\n")
    for row in rows:
        mark = "OK" if row["match"] else "MISS"
        print(f"## {row['family']} [{mark}]")
        print(f"  expected: {row['expected']}")
        print(f"  gate: {row['summary']}")
        print()
    matched = sum(1 for row in rows if row["match"])
    print(json.dumps({
        "matched": matched,
        "total": len(rows),
        "checkpoint": str(args.checkpoint),
    }))
    statuses = {row["status"] for row in rows}
    if "ready" not in statuses or statuses <= {"ready"}:
        raise SystemExit("scored demo did not produce both ready and non-ready plans")


if __name__ == "__main__":
    main()
