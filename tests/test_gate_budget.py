"""Host-owned budget signal for Path A (#22)."""

from akasha_model.gate import GateSignals, ToolProposal, evaluate_gate
from akasha_model.tool_calling import ToolSpec


def test_budget_exhausted_blocks_with_stable_reason() -> None:
    tools = {
        "fs.read": ToolSpec(
            "fs.read",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec("fs.delete"),
    }
    plan = evaluate_gate(
        tools,
        ToolProposal("fs.read", {"path": "notes.txt"}),
        GateSignals(
            authorized=0.95,
            sufficient_context=0.9,
            capability_present=0.92,
            budget_remaining=0,
        ),
    )
    assert plan.status == "blocked"
    assert plan.reason == "budget_exceeded"
    assert not plan.executable


def test_positive_budget_does_not_force_block() -> None:
    tools = {
        "fs.read": ToolSpec(
            "fs.read",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec("fs.delete"),
    }
    plan = evaluate_gate(
        tools,
        ToolProposal("fs.read", {"path": "notes.txt"}),
        GateSignals(
            authorized=0.95,
            sufficient_context=0.9,
            capability_present=0.92,
            confirmation_needed=0.05,
            budget_remaining=3,
        ),
    )
    assert plan.status == "ready"
