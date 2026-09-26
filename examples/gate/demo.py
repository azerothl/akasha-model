"""Non-executing gate demo: System 2 proposes, Akasha authorizes or blocks.

Run from the repository root after ``uv pip install -e '.[dev]'``:

```sh
python examples/gate/demo.py
```

No tools are executed. The host only prints what it would do on ``ready``.
"""

from __future__ import annotations

from akasha_model import (
    GateSignals,
    ScoreLevel,
    ScoreQuestion,
    ToolProposal,
    describe_plan,
    evaluate_gate,
)
from akasha_model.gate import default_gate_planner
from akasha_model.primitives import score_result
from catalog import tool_catalog


def risk_score(level_mass: dict[int, float]):
    question = ScoreQuestion(
        "risk",
        "Rate the operational risk of executing the proposed tool.",
        (ScoreLevel("low"), ScoreLevel("medium"), ScoreLevel("high")),
    )
    return score_result(question, level_mass)


def host_would_execute(plan) -> str:
    if plan.status != "ready":
        return "host: skip execution"
    return f"host: would execute {plan.tool_name}({plan.arguments})"


def run_scenarios() -> list[tuple[str, str, str]]:
    tools = tool_catalog()
    planner = default_gate_planner()
    scenarios: list[tuple[str, ToolProposal, GateSignals]] = [
        (
            "Safe read",
            ToolProposal("fs.read", {"path": "notes.txt"}),
            GateSignals(
                authorized=0.95,
                sufficient_context=0.92,
                capability_present=0.94,
                confirmation_needed=0.05,
                risk=risk_score({0: 0.85, 1: 0.10, 2: 0.05}),
            ),
        ),
        (
            "Delete without human confirmation",
            ToolProposal("fs.delete", {"path": "notes.txt"}),
            GateSignals(
                authorized=0.95,
                sufficient_context=0.90,
                capability_present=0.93,
                risk=risk_score({0: 0.10, 1: 0.30, 2: 0.60}),
                confirmation_given=False,
            ),
        ),
        (
            "Ambiguous mail action",
            ToolProposal(
                "mail.broadcast",
                {"list": "all-company", "body": "Office closed Friday."},
                choice_probabilities={
                    "fs.read": 0.05,
                    "fs.delete": 0.05,
                    "mail.reply": 0.40,
                    "mail.broadcast": 0.40,
                    "payments.charge": 0.05,
                    "shell.run": 0.05,
                },
            ),
            GateSignals(
                authorized=0.80,
                sufficient_context=0.75,
                capability_present=0.90,
                confirmation_needed=0.80,
                risk=risk_score({0: 0.20, 1: 0.40, 2: 0.40}),
            ),
        ),
        (
            "High-risk payment (risk gate)",
            ToolProposal(
                "payments.charge",
                {"amount_eur": 400.0, "customer_id": "cust_9"},
            ),
            GateSignals(
                authorized=0.90,
                sufficient_context=0.88,
                capability_present=0.91,
                confirmation_needed=0.95,
                risk=risk_score({0: 0.05, 1: 0.15, 2: 0.80}),
                confirmation_given=True,
            ),
        ),
        (
            "Shell without capability",
            ToolProposal("shell.run", {"command": "rm -rf /"}),
            GateSignals(
                authorized=0.95,
                sufficient_context=0.90,
                capability_present=0.15,
                risk=risk_score({0: 0.40, 1: 0.40, 2: 0.20}),
            ),
        ),
        (
            "Broadcast with confirmation",
            ToolProposal(
                "mail.broadcast",
                {"list": "team-eng", "body": "Standup moved to 10:00."},
            ),
            GateSignals(
                authorized=0.93,
                sufficient_context=0.91,
                capability_present=0.95,
                confirmation_needed=0.85,
                risk=risk_score({0: 0.55, 1: 0.35, 2: 0.10}),
                confirmation_given=True,
            ),
        ),
    ]

    rows: list[tuple[str, str, str]] = []
    for title, proposal, signals in scenarios:
        plan = evaluate_gate(tools, proposal, signals, planner=planner)
        rows.append((title, describe_plan(plan), host_would_execute(plan)))
    return rows


def main() -> None:
    print("Akasha tool gate demo (no side effects)\n")
    for title, summary, host in run_scenarios():
        print(f"## {title}")
        print(f"  gate: {summary}")
        print(f"  {host}")
        print()


if __name__ == "__main__":
    main()
