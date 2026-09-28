"""Versioned JSON wire helpers for Path A gate / host types.

These serializers are the package-owned contract for Akasha OS Rust bridges
and other hosts. They do not execute tools and do not import torch.

Breaking field renames or status enum changes require bumping
``CONTRACT_VERSION`` and a migration note in ``docs/contracts/README.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .gate import GateSignals, ToolProposal
from .host import HostOutcome
from .primitives import ScoreResult
from .tool_calling import ToolCallPlan, ToolSpec

CONTRACT_VERSION = 1

_SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"


def schema_path(name: str) -> Path:
    """Return the on-disk path of a published JSON Schema file."""
    path = _SCHEMA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"unknown wire schema: {name}")
    return path


def tool_spec_to_dict(spec: ToolSpec) -> dict[str, Any]:
    """Serialize a :class:`ToolSpec` to the wire object shape."""
    return {
        "name": spec.name,
        "description": spec.description,
        "parameters": dict(spec.parameters) if spec.parameters is not None else None,
        "required_capability": spec.required_capability,
        "requires_confirmation": spec.requires_confirmation,
        "irreversible": spec.irreversible,
    }


def tool_spec_from_dict(data: Mapping[str, Any]) -> ToolSpec:
    """Parse a wire tool catalog entry into :class:`ToolSpec`."""
    return ToolSpec(
        name=str(data["name"]),
        description=str(data.get("description") or ""),
        parameters=data.get("parameters"),
        required_capability=data.get("required_capability"),
        requires_confirmation=bool(data.get("requires_confirmation", False)),
        irreversible=bool(data.get("irreversible", False)),
    )


def proposal_to_dict(proposal: ToolProposal) -> dict[str, Any]:
    return {
        "tool_name": proposal.tool_name,
        "arguments": (
            dict(proposal.arguments) if proposal.arguments is not None else None
        ),
        "choice_probabilities": (
            dict(proposal.choice_probabilities)
            if proposal.choice_probabilities is not None
            else None
        ),
    }


def proposal_from_dict(data: Mapping[str, Any]) -> ToolProposal:
    return ToolProposal(
        tool_name=str(data["tool_name"]),
        arguments=data.get("arguments"),
        choice_probabilities=data.get("choice_probabilities"),
    )


def _risk_to_dict(risk: ScoreResult | None) -> dict[str, Any] | None:
    if risk is None:
        return None
    return {
        "score": float(risk.score),
        "confidence": float(risk.confidence),
        "probabilities": {str(k): float(v) for k, v in risk.probabilities.items()},
        "legend": {str(k): str(v) for k, v in risk.legend.items()},
        "question_id": risk.question_id,
    }


def _risk_from_dict(data: Mapping[str, Any] | None) -> ScoreResult | None:
    if data is None:
        return None
    probabilities = {int(k): float(v) for k, v in dict(data["probabilities"]).items()}
    legend = {int(k): str(v) for k, v in dict(data.get("legend") or {}).items()}
    return ScoreResult(
        score=float(data["score"]),
        legend=legend,
        probabilities=probabilities,
        confidence=float(data.get("confidence", 0.0)),
        question_id=str(data.get("question_id") or ""),
    )


def signals_to_dict(signals: GateSignals) -> dict[str, Any]:
    return {
        "authorized": float(signals.authorized),
        "sufficient_context": float(signals.sufficient_context),
        "capability_present": (
            None
            if signals.capability_present is None
            else float(signals.capability_present)
        ),
        "confirmation_needed": (
            None
            if signals.confirmation_needed is None
            else float(signals.confirmation_needed)
        ),
        "confirmation_given": bool(signals.confirmation_given),
        "risk": _risk_to_dict(signals.risk),
    }


def signals_from_dict(data: Mapping[str, Any]) -> GateSignals:
    return GateSignals(
        authorized=float(data["authorized"]),
        sufficient_context=float(data["sufficient_context"]),
        capability_present=data.get("capability_present"),
        confirmation_needed=data.get("confirmation_needed"),
        risk=_risk_from_dict(data.get("risk")),
        confirmation_given=bool(data.get("confirmation_given", False)),
    )


def plan_to_dict(
    plan: ToolCallPlan,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Serialize a plan for hosts / Rust bridges (includes ``contract_version``)."""
    payload: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "status": plan.status,
        "tool_name": plan.tool_name,
        "arguments": plan.arguments,
        "reason": plan.reason,
        "choice_probability": float(plan.choice_probability),
        "choice_confidence": float(plan.choice_confidence),
        "risk_score": (
            None if plan.risk_score is None else float(plan.risk_score)
        ),
        "executable": plan.executable,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def plan_from_dict(data: Mapping[str, Any]) -> ToolCallPlan:
    """Parse a wire plan. Rejects unknown ``contract_version`` values."""
    version = int(data.get("contract_version", CONTRACT_VERSION))
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"unsupported plan contract_version {version}; "
            f"this package speaks {CONTRACT_VERSION}"
        )
    return ToolCallPlan(
        status=data["status"],
        tool_name=data.get("tool_name"),
        arguments=data.get("arguments"),
        reason=str(data.get("reason") or ""),
        choice_probability=float(data.get("choice_probability") or 0.0),
        choice_confidence=float(data.get("choice_confidence") or 0.0),
        risk_score=data.get("risk_score"),
    )


def outcome_to_dict(
    outcome: HostOutcome,
    *,
    request_id: str | None = None,
    include_result: bool = True,
) -> dict[str, Any]:
    """Serialize a host outcome. ``result`` is omitted when ``include_result`` is false."""
    payload: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "action": outcome.action,
        "plan": plan_to_dict(outcome.plan, request_id=request_id),
        "host_reason": outcome.host_reason,
        "did_execute": outcome.did_execute,
    }
    if include_result:
        payload["result"] = outcome.result
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def outcome_from_dict(data: Mapping[str, Any]) -> HostOutcome:
    version = int(data.get("contract_version", CONTRACT_VERSION))
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"unsupported outcome contract_version {version}; "
            f"this package speaks {CONTRACT_VERSION}"
        )
    plan_data = data["plan"]
    if "contract_version" not in plan_data:
        plan_data = {**plan_data, "contract_version": version}
    return HostOutcome(
        action=data["action"],
        plan=plan_from_dict(plan_data),
        result=data.get("result"),
        host_reason=str(data.get("host_reason") or ""),
    )


def request_to_dict(
    tools: Mapping[str, ToolSpec],
    proposal: ToolProposal,
    signals: GateSignals,
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Serialize a full Path A evaluate request envelope."""
    payload: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "tools": {name: tool_spec_to_dict(spec) for name, spec in tools.items()},
        "proposal": proposal_to_dict(proposal),
        "signals": signals_to_dict(signals),
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return payload


def request_from_dict(
    data: Mapping[str, Any],
) -> tuple[dict[str, ToolSpec], ToolProposal, GateSignals, str | None]:
    """Parse a Path A request envelope into Python objects."""
    version = int(data.get("contract_version", CONTRACT_VERSION))
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"unsupported request contract_version {version}; "
            f"this package speaks {CONTRACT_VERSION}"
        )
    tools = {
        name: tool_spec_from_dict(spec)
        for name, spec in dict(data["tools"]).items()
    }
    proposal = proposal_from_dict(data["proposal"])
    signals = signals_from_dict(data["signals"])
    request_id = data.get("request_id")
    return tools, proposal, signals, request_id


__all__ = [
    "CONTRACT_VERSION",
    "outcome_from_dict",
    "outcome_to_dict",
    "plan_from_dict",
    "plan_to_dict",
    "proposal_from_dict",
    "proposal_to_dict",
    "request_from_dict",
    "request_to_dict",
    "schema_path",
    "signals_from_dict",
    "signals_to_dict",
    "tool_spec_from_dict",
    "tool_spec_to_dict",
]
