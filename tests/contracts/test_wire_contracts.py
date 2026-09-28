"""Golden wire fixtures for ToolCallPlan / HostOutcome / request envelopes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from akasha_model.gate import GateSignals, ToolProposal, evaluate_gate
from akasha_model.host import dispatch_plan, run_gated_call
from akasha_model.wire import (
    CONTRACT_VERSION,
    outcome_from_dict,
    outcome_to_dict,
    plan_from_dict,
    plan_to_dict,
    request_from_dict,
    request_to_dict,
    schema_path,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_schema_files_are_published() -> None:
    for name in (
        "tool-gate-request.schema.json",
        "tool-call-plan.schema.json",
        "host-outcome.schema.json",
    ):
        path = schema_path(name)
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("$schema")
        assert payload["properties"]["contract_version"]["const"] == CONTRACT_VERSION


def test_request_roundtrip_matches_fixture() -> None:
    raw = _load("path_a_ready_request.json")
    tools, proposal, signals, request_id = request_from_dict(raw)
    assert request_id == "req-ready-001"
    rebuilt = request_to_dict(tools, proposal, signals, request_id=request_id)
    assert rebuilt["contract_version"] == CONTRACT_VERSION
    assert rebuilt["proposal"]["tool_name"] == "fs.read"
    assert rebuilt["signals"]["authorized"] == pytest.approx(0.95)


def test_evaluate_gate_ready_wire_matches_status_fixture() -> None:
    tools, proposal, signals, request_id = request_from_dict(
        _load("path_a_ready_request.json"),
    )
    plan = evaluate_gate(tools, proposal, signals)
    wire = plan_to_dict(plan, request_id=request_id)
    expected = _load("path_a_ready_plan.json")
    assert wire["contract_version"] == expected["contract_version"]
    assert wire["status"] == expected["status"]
    assert wire["tool_name"] == expected["tool_name"]
    assert wire["arguments"] == expected["arguments"]
    assert wire["executable"] is True
    assert wire["request_id"] == expected["request_id"]
    # Confidence is derived from entropy; keep loose bound for golden stability.
    assert wire["choice_probability"] == pytest.approx(expected["choice_probability"])
    assert 0.0 < wire["choice_confidence"] < 1.0
    assert plan_from_dict(wire).status == "ready"


def test_blocked_plan_fixture_roundtrip() -> None:
    expected = _load("path_a_blocked_plan.json")
    plan = plan_from_dict(expected)
    assert plan.status == "blocked"
    assert plan.executable is False
    wire = plan_to_dict(plan)
    assert wire["status"] == "blocked"
    assert wire["reason"] == expected["reason"]


def test_host_outcome_wire_roundtrip() -> None:
    class OkHost:
        def check_permissions(self, tool_name, arguments):
            return True, "ok"

        def execute(self, tool_name, arguments):
            return {"ok": True}

    tools, proposal, signals, request_id = request_from_dict(
        _load("path_a_ready_request.json"),
    )
    outcome = run_gated_call(tools, proposal, signals, OkHost())
    wire = outcome_to_dict(outcome, request_id=request_id)
    expected = _load("path_a_executed_outcome.json")
    assert wire["contract_version"] == expected["contract_version"]
    assert wire["action"] == expected["action"]
    assert wire["did_execute"] is True
    assert wire["result"] == {"ok": True}
    assert wire["request_id"] == request_id
    assert outcome_from_dict(wire).action == "executed"


def test_dispatch_skipped_blocked_action() -> None:
    tools, proposal, signals, _ = request_from_dict(_load("path_a_ready_request.json"))
    blocked_proposal = ToolProposal("fs.delete", {"path": "notes.txt"})
    plan = evaluate_gate(
        tools,
        blocked_proposal,
        GateSignals(
            authorized=0.95, sufficient_context=0.95, capability_present=0.95,
        ),
    )
    assert plan.status == "blocked"

    class BoomHost:
        def check_permissions(self, tool_name, arguments):
            raise AssertionError("must not check")

        def execute(self, tool_name, arguments):
            raise AssertionError("must not execute")

    outcome = dispatch_plan(plan, BoomHost())
    wire = outcome_to_dict(outcome)
    assert wire["action"] == "skipped_blocked"
    assert wire["did_execute"] is False


def test_unknown_contract_version_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported plan contract_version"):
        plan_from_dict({
            "contract_version": 999,
            "status": "ready",
            "tool_name": "fs.read",
            "arguments": {},
            "reason": "x",
            "choice_probability": 1.0,
            "choice_confidence": 1.0,
            "risk_score": None,
            "executable": True,
        })
