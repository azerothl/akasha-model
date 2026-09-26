"""Minimal host loop: gate → dispatch → fake execute only on ready.

Demonstrates the Akasha OS integration contract without shipping an OS.
Uses scripted GateSignals (Path A). For the scored path see
``run_scored_gated_call`` in ``akasha_model.gate_multitask``.
"""

from __future__ import annotations

import json

from akasha_model import (
    GateSignals,
    ScoreLevel,
    ScoreQuestion,
    ToolProposal,
    describe_plan,
    run_gated_call,
)
from akasha_model.host import describe_outcome
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
    tools = tool_catalog()
    host = RecordingHost(
        tool_capabilities={
            name: spec.required_capability for name, spec in tools.items()
        },
    )
    cases = [
        (
            "Safe read → host should execute",
            ToolProposal("fs.read", {"path": "notes.txt"}),
            GateSignals(
                authorized=0.95, sufficient_context=0.92,
                capability_present=0.94, confirmation_needed=0.05,
                risk=risk_score({0: 0.85, 1: 0.10, 2: 0.05}),
            ),
        ),
        (
            "Delete without confirm → skip (blocked)",
            ToolProposal("fs.delete", {"path": "notes.txt"}),
            GateSignals(
                authorized=0.95, sufficient_context=0.90,
                capability_present=0.93,
                risk=risk_score({0: 0.10, 1: 0.30, 2: 0.60}),
            ),
        ),
        (
            "Shell → gate or host refuses",
            ToolProposal("shell.run", {"command": "echo hi"}),
            GateSignals(
                authorized=0.95, sufficient_context=0.90,
                capability_present=0.95,
                risk=risk_score({0: 0.50, 1: 0.40, 2: 0.10}),
                confirmation_given=True,
            ),
        ),
    ]

    print("Akasha host dispatch demo (fake side effects only)\n")
    for title, proposal, signals in cases:
        outcome = run_gated_call(tools, proposal, signals, host)
        print(f"## {title}")
        print(f"  gate: {describe_plan(outcome.plan)}")
        print(f"  host: {describe_outcome(outcome)}")
        if outcome.did_execute:
            print(f"  result: {json.dumps(outcome.result)}")
        print()

    print(json.dumps({
        "executions": len(host.log),
        "log": host.log,
    }, indent=2))
    if not host.log:
        raise SystemExit("expected at least one fake execution on ready")
    if any(item["tool"] == "fs.delete" for item in host.log):
        raise SystemExit("host must not execute blocked delete")


if __name__ == "__main__":
    main()
