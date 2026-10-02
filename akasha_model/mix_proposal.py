"""Mix proposal batches: confirmable, one-action undo, no package side effects.

A mix plan is a batch of before/after modifications returned to the host with
status ready/abstain. Applying settings stays host-owned. Audit envelopes reuse
``build_audit_envelope`` by framing the batch as a ``mix.apply_batch`` tool plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .audit import GateAuditEnvelope, build_audit_envelope
from .gate import GateSignals, ToolProposal
from .mix_presets import MixSettingsProposal
from .tool_calling import ToolCallPlan, ToolCallPlanner, ToolSpec

MixBatchStatus = Literal["ready", "abstain"]


@dataclass(frozen=True)
class MixChange:
    """One setting modification with before/after values for undo."""

    target: str
    before: Any
    after: Any

    def inverted(self) -> MixChange:
        return MixChange(self.target, before=self.after, after=self.before)

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "before": self.before, "after": self.after}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MixChange:
        return cls(str(data["target"]), data.get("before"), data.get("after"))


@dataclass(frozen=True)
class MixProposalBatch:
    """Non-executing batch of mix modifications for host confirmation."""

    status: MixBatchStatus
    reason: str
    changes: tuple[MixChange, ...]
    sources: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.status not in {"ready", "abstain"}:
            raise ValueError("status must be ready or abstain")
        object.__setattr__(self, "sources", dict(self.sources))

    @property
    def undo(self) -> tuple[MixChange, ...]:
        """Inverse batch for one-action cancellation."""
        return tuple(change.inverted() for change in reversed(self.changes))

    @property
    def executable(self) -> bool:
        return self.status == "ready" and bool(self.changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "changes": [c.to_dict() for c in self.changes],
            "undo": [c.to_dict() for c in self.undo],
            "sources": dict(self.sources),
        }


MIX_APPLY_TOOL = ToolSpec(
    name="mix.apply_batch",
    description=(
        "Apply a confirmed batch of mix setting changes. Host must obtain "
        "explicit user confirmation before calling; package never executes."
    ),
    parameters={
        "type": "object",
        "additionalProperties": False,
        "required": ["changes"],
        "properties": {
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["target", "before", "after"],
                    "properties": {
                        "target": {"type": "string"},
                        "before": {},
                        "after": {},
                    },
                },
            },
            "undo": {"type": "array"},
        },
    },
    requires_confirmation=True,
    irreversible=False,
    placement_tags=("mix", "settings"),
)


def build_mix_proposal_batch(
    proposal: MixSettingsProposal,
    *,
    current_settings: Mapping[str, Any],
) -> MixProposalBatch:
    """Diff proposed settings against current values. Pure: no apply."""
    if proposal.status != "ready":
        return MixProposalBatch(
            status="abstain",
            reason=proposal.reason,
            changes=(),
            sources={},
        )
    changes: list[MixChange] = []
    for target, after in proposal.settings.items():
        before = current_settings.get(target)
        if before == after:
            continue
        changes.append(MixChange(target=target, before=before, after=after))
    if not changes:
        return MixProposalBatch(
            status="abstain",
            reason="proposed settings match current values",
            changes=(),
            sources=dict(proposal.sources),
        )
    return MixProposalBatch(
        status="ready",
        reason=proposal.reason,
        changes=tuple(changes),
        sources=dict(proposal.sources),
    )


def mix_batch_as_tool_plan(
    batch: MixProposalBatch,
    *,
    planner: ToolCallPlanner | None = None,
) -> ToolCallPlan:
    """Frame a mix batch as a ToolCallPlan for audit / host confirmation."""
    _ = planner  # reserved for future threshold wiring; batch already decided
    if batch.status != "ready":
        return ToolCallPlan(
            status="abstain",
            tool_name=None,
            arguments=None,
            reason=batch.reason,
        )
    return ToolCallPlan(
        status="ready",
        tool_name=MIX_APPLY_TOOL.name,
        arguments={
            "changes": [c.to_dict() for c in batch.changes],
            "undo": [c.to_dict() for c in batch.undo],
        },
        reason=batch.reason,
        choice_probability=1.0,
        choice_confidence=1.0,
    )


def build_mix_audit_envelope(
    batch: MixProposalBatch,
    *,
    request_id: str | None = None,
) -> GateAuditEnvelope:
    """Audit envelope via ``build_audit_envelope`` (same hash schema as gate)."""
    plan = mix_batch_as_tool_plan(batch)
    proposal = ToolProposal(
        tool_name=MIX_APPLY_TOOL.name if batch.status == "ready" else "mix.noop",
        arguments={
            "changes": [c.to_dict() for c in batch.changes],
            "undo": [c.to_dict() for c in batch.undo],
            "batch_status": batch.status,
            "reason": batch.reason,
        },
    )
    signals = GateSignals(
        authorized=1.0,
        sufficient_context=1.0,
        capability_present=1.0,
        confirmation_needed=1.0,
        confirmation_given=False,
    )
    return build_audit_envelope(
        plan=plan,
        proposal=proposal,
        signals=signals,
        tools={MIX_APPLY_TOOL.name: MIX_APPLY_TOOL},
        request_id=request_id,
    )


def apply_changes_locally(
    settings: Mapping[str, Any],
    changes: Sequence[MixChange],
) -> dict[str, Any]:
    """Pure helper for hosts/tests: copy settings and apply a batch in memory."""
    result = dict(settings)
    for change in changes:
        result[change.target] = change.after
    return result


__all__ = [
    "MIX_APPLY_TOOL",
    "MixBatchStatus",
    "MixChange",
    "MixProposalBatch",
    "apply_changes_locally",
    "build_mix_audit_envelope",
    "build_mix_proposal_batch",
    "mix_batch_as_tool_plan",
]
