"""Log gate outcomes and print threshold calibration suggestions.

Simulates a short host session, appends JSONL under ``data/gate/outcomes.jsonl``
(gitignored), then summarizes. No real side effects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from akasha_model import (
    GateSignals,
    ScoreLevel,
    ScoreQuestion,
    ToolProposal,
    run_gated_call,
)
from akasha_model.outcomes import (
    append_outcome,
    load_outcomes,
    record_from_host_outcome,
    suggest_threshold_updates,
    summarize_outcomes,
)
from akasha_model.primitives import score_result
from catalog import tool_catalog
from recording_host import RecordingHost


def risk_score(level_mass: dict[int, float]):
    question = ScoreQuestion(
        "risk",
        "Rate the operational risk of executing the proposed tool call.",
        (ScoreLevel("low"), ScoreLevel("medium"), ScoreLevel("high")),
    )
    return score_result(question, level_mass)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=Path, default=Path("data/gate/outcomes.jsonl"),
    )
    args = parser.parse_args()

    tools = tool_catalog()
    host = RecordingHost(
        tool_capabilities={
            name: spec.required_capability for name, spec in tools.items()
        },
    )

    session = [
        (
            ToolProposal("fs.read", {"path": "notes.txt"}),
            GateSignals(
                authorized=0.95, sufficient_context=0.92,
                capability_present=0.94, confirmation_needed=0.05,
                risk=risk_score({0: 0.85, 1: 0.10, 2: 0.05}),
            ),
            {"success": True, "user_forced": False, "notes": "read ok"},
        ),
        (
            ToolProposal("fs.delete", {"path": "notes.txt"}),
            GateSignals(
                authorized=0.95, sufficient_context=0.90,
                capability_present=0.93,
                risk=risk_score({0: 0.10, 1: 0.30, 2: 0.60}),
            ),
            {
                "success": True, "user_forced": True,
                "notes": "user overrode block after review",
            },
        ),
        (
            ToolProposal("payments.charge", {"amount_eur": 400.0, "customer_id": "c1"}),
            GateSignals(
                authorized=0.90, sufficient_context=0.88,
                capability_present=0.91, confirmation_needed=0.95,
                risk=risk_score({0: 0.05, 1: 0.15, 2: 0.80}),
                confirmation_given=True,
            ),
            {"success": None, "user_forced": False, "notes": "blocked high risk"},
        ),
        (
            ToolProposal("mail.broadcast", {"list": "team-eng", "body": "hi"}),
            GateSignals(
                authorized=0.93, sufficient_context=0.91,
                capability_present=0.95, confirmation_needed=0.85,
                risk=risk_score({0: 0.55, 1: 0.35, 2: 0.10}),
                confirmation_given=True,
            ),
            {"success": True, "user_forced": False, "notes": "broadcast ok"},
        ),
        (
            ToolProposal("fs.read", {"path": "missing.txt"}),
            GateSignals(
                authorized=0.95, sufficient_context=0.90,
                capability_present=0.94, confirmation_needed=0.05,
                risk=risk_score({0: 0.80, 1: 0.15, 2: 0.05}),
            ),
            {"success": False, "user_forced": False, "notes": "file missing"},
        ),
    ]

    # Fresh demo log each run for deterministic CI.
    if args.log.is_file():
        args.log.unlink()

    print("Akasha gate outcomes demo (fake host)\n")
    for proposal, signals, meta in session:
        outcome = run_gated_call(tools, proposal, signals, host)
        record = record_from_host_outcome(
            outcome,
            success=meta["success"],
            user_forced=meta["user_forced"],
            notes=meta["notes"],
            signals=signals,
        )
        append_outcome(args.log, record)
        print(
            f"- {proposal.tool_name}: gate={outcome.plan.status} "
            f"host={outcome.action} success={record.success} "
            f"forced={record.user_forced}",
        )

    rows = load_outcomes(args.log)
    summary = summarize_outcomes(rows)
    advice = suggest_threshold_updates(rows)
    print()
    print(json.dumps({"log": str(args.log), "summary": summary, "advice": advice}, indent=2))
    if summary["count"] < 5:
        raise SystemExit("expected at least 5 outcome rows")


if __name__ == "__main__":
    main()
