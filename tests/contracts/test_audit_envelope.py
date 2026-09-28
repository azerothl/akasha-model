"""Golden audit envelopes + request_id echo (#19, #24)."""

from __future__ import annotations

from akasha_model.audit import (
    build_audit_envelope,
    envelope_from_dict,
    envelope_to_dict,
    verify_envelope,
)
from akasha_model.gate import GateSignals, ToolProposal, evaluate_gate
from akasha_model.host import run_gated_call
from akasha_model.tool_calling import ToolSpec
from akasha_model.wire import CONTRACT_VERSION, plan_to_dict, request_to_dict


def _tools() -> dict[str, ToolSpec]:
    return {
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
        "fs.delete": ToolSpec("fs.delete", irreversible=True),
    }


def test_request_id_echoed_in_plan_wire() -> None:
    tools = _tools()
    proposal = ToolProposal("fs.read", {"path": "notes.txt"})
    signals = GateSignals(
        authorized=0.95, sufficient_context=0.9, capability_present=0.92,
        confirmation_needed=0.05,
    )
    plan = evaluate_gate(tools, proposal, signals, request_id="req-abc")
    wire = plan_to_dict(plan, request_id="req-abc")
    assert wire["request_id"] == "req-abc"
    assert wire["contract_version"] == CONTRACT_VERSION
    # Missing id still works.
    assert "request_id" not in plan_to_dict(plan)


def test_build_and_verify_envelope_roundtrip() -> None:
    tools = _tools()
    proposal = ToolProposal("fs.read", {"path": "notes.txt"})
    signals = GateSignals(
        authorized=0.95, sufficient_context=0.9, capability_present=0.92,
        confirmation_needed=0.05,
    )
    plan = evaluate_gate(tools, proposal, signals, request_id="req-env-1")
    envelope = build_audit_envelope(
        plan=plan,
        proposal=proposal,
        signals=signals,
        tools=tools,
        request_id="req-env-1",
        recorded_at="2026-09-27T00:00:00+00:00",
    )
    assert envelope.request_id == "req-env-1"
    assert verify_envelope(envelope)
    payload = envelope_to_dict(envelope)
    assert envelope_from_dict(payload).content_sha256 == envelope.content_sha256
    assert verify_envelope(payload)
    # Tamper a field that actually changes
    payload["plan"] = {**payload["plan"], "reason": "tampered"}
    assert not verify_envelope(payload)


def test_envelope_with_host_outcome() -> None:
    class OkHost:
        def check_permissions(self, tool_name, arguments):
            return True, "ok"

        def execute(self, tool_name, arguments):
            return {"ok": True}

    tools = _tools()
    proposal = ToolProposal("fs.read", {"path": "notes.txt"})
    signals = GateSignals(
        authorized=0.95, sufficient_context=0.9, capability_present=0.92,
        confirmation_needed=0.05,
    )
    outcome = run_gated_call(
        tools, proposal, signals, OkHost(), request_id="req-host-1",
    )
    envelope = build_audit_envelope(
        plan=outcome.plan,
        proposal=proposal,
        signals=signals,
        outcome=outcome,
        request_id="req-host-1",
        recorded_at="2026-09-27T00:00:00+00:00",
    )
    assert envelope.host_action == "executed"
    assert verify_envelope(envelope)
    # Golden shape freeze (hash depends on plan confidence floats — check fields).
    payload = envelope_to_dict(envelope)
    assert payload["contract_version"] == CONTRACT_VERSION
    assert payload["request_id"] == "req-host-1"
    assert payload["host_action"] == "executed"
    assert len(payload["content_sha256"]) == 64


def test_request_envelope_carries_request_id() -> None:
    tools = _tools()
    proposal = ToolProposal("fs.read", {"path": "notes.txt"})
    signals = GateSignals(authorized=0.9, sufficient_context=0.9, capability_present=0.9)
    wire = request_to_dict(tools, proposal, signals, request_id="req-req-1")
    assert wire["request_id"] == "req-req-1"
