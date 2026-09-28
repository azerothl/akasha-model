"""Tamper-evident (hash-only) audit envelopes for Path A plans / outcomes.

v1 is content-hash integrity without cryptography. Hosts may attach signatures
later. Construction has no side effects and does not execute tools.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .gate import GateSignals, ToolProposal
from .host import HostOutcome
from .tool_calling import ToolCallPlan, ToolSpec
from .wire import (
    CONTRACT_VERSION,
    outcome_to_dict,
    plan_to_dict,
    proposal_to_dict,
    signals_to_dict,
    tool_spec_to_dict,
)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8",
    )


def content_hash(payload: Mapping[str, Any]) -> str:
    """Stable SHA-256 hex digest over canonical JSON (Python/Rust parity target)."""
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class GateAuditEnvelope:
    """Package-owned audit record for one evaluate / dispatch decision."""

    contract_version: int
    request_id: str | None
    recorded_at: str
    proposal: Mapping[str, Any]
    signals: Mapping[str, Any] | None
    plan: Mapping[str, Any]
    host_action: str | None
    host_reason: str | None
    tools: Mapping[str, Any] | None
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return envelope_to_dict(self)


def envelope_to_dict(envelope: GateAuditEnvelope) -> dict[str, Any]:
    return {
        "contract_version": envelope.contract_version,
        "request_id": envelope.request_id,
        "recorded_at": envelope.recorded_at,
        "proposal": dict(envelope.proposal),
        "signals": None if envelope.signals is None else dict(envelope.signals),
        "plan": dict(envelope.plan),
        "host_action": envelope.host_action,
        "host_reason": envelope.host_reason,
        "tools": None if envelope.tools is None else dict(envelope.tools),
        "content_sha256": envelope.content_sha256,
    }


def envelope_from_dict(data: Mapping[str, Any]) -> GateAuditEnvelope:
    version = int(data.get("contract_version", CONTRACT_VERSION))
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"unsupported envelope contract_version {version}; "
            f"this package speaks {CONTRACT_VERSION}"
        )
    return GateAuditEnvelope(
        contract_version=version,
        request_id=data.get("request_id"),
        recorded_at=str(data["recorded_at"]),
        proposal=dict(data["proposal"]),
        signals=None if data.get("signals") is None else dict(data["signals"]),
        plan=dict(data["plan"]),
        host_action=data.get("host_action"),
        host_reason=data.get("host_reason"),
        tools=None if data.get("tools") is None else dict(data["tools"]),
        content_sha256=str(data["content_sha256"]),
    )


def _body_for_hash(
    *,
    contract_version: int,
    request_id: str | None,
    recorded_at: str,
    proposal: Mapping[str, Any],
    signals: Mapping[str, Any] | None,
    plan: Mapping[str, Any],
    host_action: str | None,
    host_reason: str | None,
    tools: Mapping[str, Any] | None,
) -> dict[str, Any]:
    # Hash excludes content_sha256 itself.
    return {
        "contract_version": contract_version,
        "request_id": request_id,
        "recorded_at": recorded_at,
        "proposal": dict(proposal),
        "signals": None if signals is None else dict(signals),
        "plan": dict(plan),
        "host_action": host_action,
        "host_reason": host_reason,
        "tools": None if tools is None else dict(tools),
    }


def build_audit_envelope(
    *,
    plan: ToolCallPlan,
    proposal: ToolProposal,
    signals: GateSignals | None = None,
    tools: Mapping[str, ToolSpec] | None = None,
    outcome: HostOutcome | None = None,
    request_id: str | None = None,
    recorded_at: str | None = None,
) -> GateAuditEnvelope:
    """Build an envelope without I/O. Idempotency / persistence stay host-owned."""
    stamp = recorded_at or datetime.now(timezone.utc).isoformat()
    plan_wire = plan_to_dict(plan, request_id=request_id)
    proposal_wire = proposal_to_dict(proposal)
    signals_wire = None if signals is None else signals_to_dict(signals)
    tools_wire = (
        None
        if tools is None
        else {name: tool_spec_to_dict(spec) for name, spec in tools.items()}
    )
    host_action = outcome.action if outcome is not None else None
    host_reason = outcome.host_reason if outcome is not None else None
    if outcome is not None:
        # Prefer outcome plan wire (same plan, ensures consistency).
        plan_wire = outcome_to_dict(outcome, request_id=request_id)["plan"]
    body = _body_for_hash(
        contract_version=CONTRACT_VERSION,
        request_id=request_id,
        recorded_at=stamp,
        proposal=proposal_wire,
        signals=signals_wire,
        plan=plan_wire,
        host_action=host_action,
        host_reason=host_reason,
        tools=tools_wire,
    )
    digest = content_hash(body)
    return GateAuditEnvelope(
        contract_version=CONTRACT_VERSION,
        request_id=request_id,
        recorded_at=stamp,
        proposal=proposal_wire,
        signals=signals_wire,
        plan=plan_wire,
        host_action=host_action,
        host_reason=host_reason,
        tools=tools_wire,
        content_sha256=digest,
    )


def verify_envelope(envelope: GateAuditEnvelope | Mapping[str, Any]) -> bool:
    data = envelope_to_dict(envelope) if isinstance(envelope, GateAuditEnvelope) else dict(envelope)
    expected = data.get("content_sha256")
    body = {k: v for k, v in data.items() if k != "content_sha256"}
    return expected == content_hash(body)


__all__ = [
    "GateAuditEnvelope",
    "build_audit_envelope",
    "content_hash",
    "envelope_from_dict",
    "envelope_to_dict",
    "verify_envelope",
]
