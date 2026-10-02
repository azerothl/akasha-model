"""Path A gate / host / outcomes — must run without importing torch."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

from akasha_model import (
    DEFAULT_MAX_RISK_SCORE,
    DEFAULT_MIN_CHOICE_CONFIDENCE,
    DEFAULT_MIN_CHOICE_PROBABILITY,
    DEFAULT_NOUL_THRESHOLD,
    GateSignals,
    ToolProposal,
    ToolCallPlanner,
    ToolSpec,
    append_outcome,
    default_gate_planner,
    dispatch_plan,
    evaluate_gate,
    load_outcomes,
    record_from_host_outcome,
    run_gated_call,
    suggest_threshold_updates,
    summarize_outcomes,
)
from akasha_model.primitives import ChoiceQuestion, OptionSpec, choice_result


def test_path_a_modules_have_no_torch_import() -> None:
    import akasha_model.gate as gate
    import akasha_model.host as host
    import akasha_model.mix as mix
    import akasha_model.outcomes as outcomes
    import akasha_model.tool_calling as tool_calling
    import akasha_model.primitives as primitives

    for module in (gate, host, mix, outcomes, tool_calling, primitives):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "import torch" not in source
        assert "from torch" not in source


def _tool_choice(selected: str = "fs.read"):
    question = ChoiceQuestion(
        "route", "Choose a tool.",
        (OptionSpec("fs.read"), OptionSpec("fs.delete")),
    )
    return choice_result(
        question, {"fs.read": 0.9 if selected == "fs.read" else 0.1,
                    "fs.delete": 0.1 if selected == "fs.read" else 0.9},
    )


def test_tool_call_planner_returns_a_non_executing_ready_plan():
    spec = ToolSpec(
        "fs.read", "Read a file.",
        parameters={"type": "object", "required": ["path"],
                    "properties": {"path": {"type": "string"}},
                    "additionalProperties": False},
        required_capability="workspace_access",
    )
    plan = ToolCallPlanner().plan(
        _tool_choice(), {spec.name: spec}, arguments={"path": "notes.txt"},
        nouls={"authorized": 0.95, "sufficient_context": 0.90,
               "capability_present": 0.92, "confirmation_needed": 0.05},
    )
    assert plan.status == "ready"
    assert plan.executable
    assert plan.arguments == {"path": "notes.txt"}


def test_tool_call_planner_blocks_missing_capability_and_confirmation():
    spec = ToolSpec("fs.delete", irreversible=True, required_capability="workspace_access")
    plan = ToolCallPlanner().plan(
        _tool_choice("fs.delete"), {spec.name: spec},
        nouls={"authorized": 0.95, "sufficient_context": 0.95,
               "capability_present": 0.20},
        confirmation_given=True,
    )
    assert plan.status == "blocked"
    assert "capability" in plan.reason

    plan = ToolCallPlanner().plan(
        _tool_choice("fs.delete"), {spec.name: spec},
        nouls={"authorized": 0.95, "sufficient_context": 0.95,
               "capability_present": 0.95},
    )
    assert plan.status == "blocked"
    assert "confirmation" in plan.reason


def test_tool_call_planner_abstains_on_weak_choice():
    spec = ToolSpec("fs.read")
    choice = choice_result(
        ChoiceQuestion("route", "Choose.", (OptionSpec("fs.read"), OptionSpec("fs.delete"))),
        {"fs.read": 0.51, "fs.delete": 0.49},
    )
    plan = ToolCallPlanner(min_choice_probability=0.60).plan(
        choice, {spec.name: spec},
        nouls={"authorized": 1.0, "sufficient_context": 1.0},
    )
    assert plan.status == "abstain"


def test_evaluate_gate_ready_abstain_and_blocked():
    tools = {
        "fs.read": ToolSpec(
            "fs.read",
            parameters={"type": "object", "required": ["path"],
                        "properties": {"path": {"type": "string"}},
                        "additionalProperties": False},
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec(
            "fs.delete", irreversible=True, required_capability="workspace_access",
        ),
    }
    ready = evaluate_gate(
        tools,
        ToolProposal("fs.read", {"path": "notes.txt"}),
        GateSignals(
            authorized=0.95, sufficient_context=0.9, capability_present=0.92,
            confirmation_needed=0.05,
        ),
    )
    assert ready.status == "ready"
    assert ready.executable
    assert ready.arguments == {"path": "notes.txt"}

    abstain = evaluate_gate(
        tools,
        ToolProposal(
            "fs.read",
            choice_probabilities={"fs.read": 0.52, "fs.delete": 0.48},
        ),
        GateSignals(authorized=1.0, sufficient_context=1.0, capability_present=1.0),
        planner=ToolCallPlanner(min_choice_probability=0.60),
    )
    assert abstain.status == "abstain"

    blocked = evaluate_gate(
        tools,
        ToolProposal("fs.delete", {"path": "notes.txt"}),
        GateSignals(
            authorized=0.95, sufficient_context=0.95, capability_present=0.95,
        ),
    )
    assert blocked.status == "blocked"
    assert "confirmation" in blocked.reason


def test_default_gate_planner_matches_documented_thresholds():
    planner = default_gate_planner()
    assert planner.min_choice_probability == DEFAULT_MIN_CHOICE_PROBABILITY
    assert planner.min_choice_confidence == DEFAULT_MIN_CHOICE_CONFIDENCE
    assert planner.noul_threshold == DEFAULT_NOUL_THRESHOLD
    assert planner.max_risk_score == DEFAULT_MAX_RISK_SCORE


def test_dispatch_plan_executes_only_on_ready():
    class FakeHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict | None]] = []

        def check_permissions(self, tool_name, arguments):
            return True, "ok"

        def execute(self, tool_name, arguments):
            self.calls.append((tool_name, dict(arguments or {})))
            return {"ok": True}

    tools = {
        "fs.read": ToolSpec(
            "fs.read",
            parameters={"type": "object", "required": ["path"],
                        "properties": {"path": {"type": "string"}},
                        "additionalProperties": False},
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec(
            "fs.delete", irreversible=True, required_capability="workspace_access",
        ),
    }
    host = FakeHost()
    ready_outcome = run_gated_call(
        tools,
        ToolProposal("fs.read", {"path": "notes.txt"}),
        GateSignals(
            authorized=0.95, sufficient_context=0.9, capability_present=0.92,
            confirmation_needed=0.05,
        ),
        host,
    )
    assert ready_outcome.action == "executed"
    assert ready_outcome.did_execute
    assert host.calls == [("fs.read", {"path": "notes.txt"})]

    host.calls.clear()
    blocked_outcome = run_gated_call(
        tools,
        ToolProposal("fs.delete", {"path": "notes.txt"}),
        GateSignals(
            authorized=0.95, sufficient_context=0.95, capability_present=0.95,
        ),
        host,
    )
    assert blocked_outcome.action == "skipped_blocked"
    assert host.calls == []

    abstain_plan = evaluate_gate(
        tools,
        ToolProposal(
            "fs.read",
            choice_probabilities={"fs.read": 0.52, "fs.delete": 0.48},
        ),
        GateSignals(authorized=1.0, sufficient_context=1.0, capability_present=1.0),
        planner=ToolCallPlanner(min_choice_probability=0.60),
    )
    abstain_outcome = dispatch_plan(abstain_plan, host)
    assert abstain_outcome.action == "skipped_abstain"
    assert host.calls == []


def test_dispatch_plan_respects_host_permission_denial():
    class DenyHost:
        def check_permissions(self, tool_name, arguments):
            return False, "denied by ACL"

        def execute(self, tool_name, arguments):
            raise AssertionError("execute must not run after permission denial")

    tools = {
        "fs.read": ToolSpec(
            "fs.read",
            parameters={"type": "object", "required": ["path"],
                        "properties": {"path": {"type": "string"}},
                        "additionalProperties": False},
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec("fs.delete"),
    }
    outcome = run_gated_call(
        tools,
        ToolProposal("fs.read", {"path": "notes.txt"}),
        GateSignals(
            authorized=0.95, sufficient_context=0.9, capability_present=0.92,
            confirmation_needed=0.05,
        ),
        DenyHost(),
    )
    assert outcome.action == "rejected_by_host"
    assert "ACL" in outcome.host_reason


def test_outcome_jsonl_roundtrip_and_summary(tmp_path):
    tools = {
        "fs.read": ToolSpec(
            "fs.read",
            parameters={"type": "object", "required": ["path"],
                        "properties": {"path": {"type": "string"}},
                        "additionalProperties": False},
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec("fs.delete", irreversible=True),
    }

    class OkHost:
        def check_permissions(self, tool_name, arguments):
            return True, "ok"

        def execute(self, tool_name, arguments):
            return {"ok": True}

    outcome = run_gated_call(
        tools,
        ToolProposal("fs.read", {"path": "notes.txt"}),
        GateSignals(
            authorized=0.95, sufficient_context=0.9, capability_present=0.92,
            confirmation_needed=0.05,
        ),
        OkHost(),
    )
    path = tmp_path / "outcomes.jsonl"
    record = record_from_host_outcome(outcome, success=True, notes="ok")
    append_outcome(path, record)
    append_outcome(path, record_from_host_outcome(
        run_gated_call(
            tools,
            ToolProposal("fs.delete", {"path": "x"}),
            GateSignals(authorized=0.9, sufficient_context=0.9, capability_present=0.9),
            OkHost(),
        ),
        user_forced=True, success=True, notes="overrode block",
    ))
    rows = load_outcomes(path)
    assert len(rows) == 2
    summary = summarize_outcomes(rows)
    assert summary["executed"] == 1
    assert summary["user_forced"] == 1
    advice = suggest_threshold_updates(rows)
    assert "suggestions" in advice
    assert advice["summary"]["count"] == 2


def test_path_a_import_works_without_torch_installed():
    """When torch is absent, Path A imports must still resolve."""
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")

    import akasha_model

    assert akasha_model.evaluate_gate is evaluate_gate
    assert akasha_model.run_gated_call is run_gated_call
    assert importlib.util.find_spec("torch") is None


def test_usage_demo_path_a_scenarios():
    """First-run demo must stay torch-free and assert ready/blocked/abstain."""
    usage_dir = Path(__file__).resolve().parents[1] / "examples" / "usage"
    sys.path.insert(0, str(usage_dir))
    try:
        usage_trace = importlib.import_module("decision_trace")
        usage_visual = importlib.import_module("visual_demo")
        usage_trace = importlib.reload(usage_trace)
        usage_visual = importlib.reload(usage_visual)
        trace = usage_trace.build_trace()
        html = usage_visual.render_html(trace)
    finally:
        sys.path.pop(0)
        for name in ("decision_trace", "visual_demo"):
            sys.modules.pop(name, None)

    assert trace["path"] == "A"
    statuses = [step["verdict"]["status"] for step in trace["steps"]]
    assert statuses[:3] == ["ready", "blocked", "abstain"]
    assert [step["title"] for step in trace["steps"][:3]] == [
        "Safe read",
        "Delete without confirmation",
        "Ambiguous broadcast",
    ]
    ticket_steps = [step for step in trace["steps"] if step["kind"] == "ticket"]
    assert ticket_steps[0]["verdict"]["status"] == "ready"
    assert "refund" in ticket_steps[0]["verdict"]["host"]
    assert ticket_steps[1]["verdict"]["status"] == "abstain"
    assert '"status":"ready"' in html
    assert '"status":"blocked"' in html
    assert '"status":"abstain"' in html
    assert "probability" in html
    assert importlib.util.find_spec("torch") is None
