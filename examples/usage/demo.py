"""First-run usage demo: Path A tool gate + ticket-router API shape.

CPU only, no torch, no Hub download, no side effects.

Run from the repository root after ``uv pip install -e '.[dev]'``:

```sh
python examples/usage/demo.py
```

This is the smallest product-wedge walkthrough for a new user. Deeper gate
demos live under ``examples/gate/``; training / Path B need the ``[torch]``
extra.
"""

from __future__ import annotations

from akasha_model import (
    ChoiceQuestion,
    GateSignals,
    OptionSpec,
    ScoreLevel,
    ScoreQuestion,
    ToolProposal,
    ToolSpec,
    describe_plan,
    evaluate_gate,
)
from akasha_model.gate import default_gate_planner
from akasha_model.primitives import choice_result, score_result


def _risk(level_mass: dict[int, float]):
    question = ScoreQuestion(
        "risk",
        "Rate the operational risk of executing the proposed tool.",
        (ScoreLevel("low"), ScoreLevel("medium"), ScoreLevel("high")),
    )
    return score_result(question, level_mass)


def _tools() -> dict[str, ToolSpec]:
    return {
        "fs.read": ToolSpec(
            "fs.read",
            "Read a workspace file.",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec(
            "fs.delete",
            "Delete a workspace file.",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="workspace_access",
            irreversible=True,
        ),
        "mail.broadcast": ToolSpec(
            "mail.broadcast",
            "Send mail to a large distribution list.",
            parameters={
                "type": "object",
                "required": ["list", "body"],
                "properties": {
                    "list": {"type": "string"},
                    "body": {"type": "string"},
                },
                "additionalProperties": False,
            },
            required_capability="mail_send",
            requires_confirmation=True,
        ),
    }


def run_tool_gate() -> list[tuple[str, str]]:
    """System 2 proposes; Akasha authorizes / abstains / blocks. No execution."""
    tools = _tools()
    planner = default_gate_planner()
    scenarios: list[tuple[str, ToolProposal, GateSignals, str]] = [
        (
            "Safe read",
            ToolProposal("fs.read", {"path": "notes.txt"}),
            GateSignals(
                authorized=0.95,
                sufficient_context=0.92,
                capability_present=0.94,
                confirmation_needed=0.05,
                risk=_risk({0: 0.85, 1: 0.10, 2: 0.05}),
            ),
            "ready",
        ),
        (
            "Delete without confirmation",
            ToolProposal("fs.delete", {"path": "notes.txt"}),
            GateSignals(
                authorized=0.95,
                sufficient_context=0.90,
                capability_present=0.93,
                risk=_risk({0: 0.10, 1: 0.30, 2: 0.60}),
                confirmation_given=False,
            ),
            "blocked",
        ),
        (
            "Ambiguous broadcast (split choice mass)",
            ToolProposal(
                "mail.broadcast",
                {"list": "all-company", "body": "Office closed Friday."},
                choice_probabilities={
                    "fs.read": 0.20,
                    "fs.delete": 0.40,
                    "mail.broadcast": 0.40,
                },
            ),
            GateSignals(
                authorized=0.80,
                sufficient_context=0.75,
                capability_present=0.90,
                confirmation_needed=0.80,
                risk=_risk({0: 0.20, 1: 0.40, 2: 0.40}),
            ),
            "abstain",
        ),
    ]

    rows: list[tuple[str, str]] = []
    for title, proposal, signals, expected in scenarios:
        plan = evaluate_gate(tools, proposal, signals, planner=planner)
        if plan.status != expected:
            raise SystemExit(
                f"{title}: expected {expected}, got {plan.status} "
                f"({describe_plan(plan)})"
            )
        host = (
            f"host: would execute {plan.tool_name}({plan.arguments})"
            if plan.status == "ready"
            else "host: skip execution"
        )
        rows.append((title, f"{describe_plan(plan)} | {host}"))
    return rows


def run_ticket_router() -> list[tuple[str, str]]:
    """Bounded menu + abstain threshold (API shape; scripted probabilities).

    A trained scorer would fill these masses; Path A shows the host contract
    without torch. Real train/predict steps are in docs/use-cases.md.
    """
    question = ChoiceQuestion(
        "queue",
        "Choose the support queue.",
        (
            OptionSpec("refund", "Billing correction or money back."),
            OptionSpec("sales", "Pricing, upgrade, or purchase help."),
            OptionSpec("technical", "Product bug or how-to."),
        ),
    )
    min_confidence = 0.75
    tickets: list[tuple[str, dict[str, float]]] = [
        (
            "Double charge on order #4412",
            {"refund": 0.95, "sales": 0.03, "technical": 0.02},
        ),
        (
            "Unclear message: maybe bug, maybe invoice",
            {"refund": 0.34, "sales": 0.33, "technical": 0.33},
        ),
    ]

    rows: list[tuple[str, str]] = []
    for title, masses in tickets:
        result = choice_result(question, masses)
        if result.confidence < min_confidence or result.selected is None:
            action = (
                f"abstain (confidence={result.confidence:.2f} "
                f"< {min_confidence:.2f}) → escalate to human"
            )
        else:
            selected_p = result.probabilities[result.selected]
            action = (
                f"route → {result.selected} "
                f"(p={selected_p:.2f}, confidence={result.confidence:.2f})"
            )
        rows.append((title, action))
    return rows


def main() -> None:
    print("Akasha Model — first-run usage demo")
    print("Path A only: no torch, no Hub, no side effects.\n")

    print("=== 1. Tool gate (product wedge) ===")
    print("System 2 proposes a call; evaluate_gate returns ready/abstain/blocked.\n")
    for title, line in run_tool_gate():
        print(f"## {title}")
        print(f"  {line}")
        print()

    print("=== 2. Ticket router (Choice + abstain) ===")
    print("Same typed-menu API; host policy owns escalation.\n")
    for title, line in run_ticket_router():
        print(f"## {title}")
        print(f"  {line}")
        print()

    print("Next steps")
    print("  python examples/gate/demo.py          # more gate scenarios")
    print("  python examples/gate/host_demo.py     # fake ToolHost dispatch")
    print("  docs/using-the-tool-gate.md           # Path A–D integrator guide")
    print("  docs/use-cases.md                     # triage / multitask / vision")


if __name__ == "__main__":
    main()
