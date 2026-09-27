"""Optional Path A authority profiles (MIDAS-inspired, executor-free).

Profiles tune *how far* authorization goes once a tool is proposed. Catalog
policy (#15) decides *what* may be proposed; OS ``check_permissions`` remains
the final grant check.

Statuses stay ``ready`` / ``abstain`` / ``blocked`` for backward compatibility.
When a profile is supplied, ``plan.reason`` is prefixed with a stable code:

- ``escalate:`` — within authority chain but over profile threshold (human review)
- ``reject:`` — hard policy / risk deny
- ``clarify:`` — insufficient context / missing required context keys
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .tool_calling import ToolCallPlan


@dataclass(frozen=True)
class AuthorityProfile:
    """Optional thresholds bound to a decision surface / tool class.

    All fields are optional; ``None`` means “use the planner defaults / ignore”.
    """

    name: str = "default"
    min_authorized: float | None = None
    min_sufficient_context: float | None = None
    max_risk_score: float | None = None
    max_consequence: float | None = None
    required_context_keys: tuple[str, ...] = ()


def annotate_reason(code: str, detail: str) -> str:
    return f"{code}: {detail}"


def apply_authority_profile(
    plan: ToolCallPlan,
    *,
    profile: AuthorityProfile | None,
    authorized: float,
    sufficient_context: float,
    risk_score: float | None,
    consequence: float | None,
    context: Mapping[str, object] | None,
) -> ToolCallPlan:
    """Re-map or annotate a plan when an authority profile is active.

    Callers that omit ``profile`` keep today's behaviour unchanged.
    """
    if profile is None:
        return plan

    if profile.required_context_keys:
        ctx = context or {}
        missing = [key for key in profile.required_context_keys if key not in ctx]
        if missing:
            return ToolCallPlan(
                "abstain",
                plan.tool_name,
                None,
                annotate_reason(
                    "clarify",
                    "missing required context key(s): " + ", ".join(missing),
                ),
                plan.choice_probability,
                plan.choice_confidence,
                plan.risk_score,
            )

    if (
        profile.min_sufficient_context is not None
        and sufficient_context < profile.min_sufficient_context
    ):
        return ToolCallPlan(
            "abstain",
            plan.tool_name,
            None,
            annotate_reason(
                "clarify",
                "sufficient_context below authority profile minimum",
            ),
            plan.choice_probability,
            plan.choice_confidence,
            plan.risk_score,
        )

    if profile.min_authorized is not None and authorized < profile.min_authorized:
        return ToolCallPlan(
            "blocked",
            plan.tool_name,
            None,
            annotate_reason("reject", "authorized below authority profile minimum"),
            plan.choice_probability,
            plan.choice_confidence,
            plan.risk_score,
        )

    if (
        profile.max_risk_score is not None
        and risk_score is not None
        and risk_score > profile.max_risk_score
    ):
        return ToolCallPlan(
            "blocked",
            plan.tool_name,
            None,
            annotate_reason("reject", "risk score exceeds authority profile maximum"),
            plan.choice_probability,
            plan.choice_confidence,
            risk_score,
        )

    if (
        profile.max_consequence is not None
        and consequence is not None
        and consequence > profile.max_consequence
    ):
        return ToolCallPlan(
            "abstain",
            plan.tool_name,
            None,
            annotate_reason(
                "escalate",
                "consequence exceeds authority profile; human review required",
            ),
            plan.choice_probability,
            plan.choice_confidence,
            plan.risk_score,
        )

    # Already non-ready from planner: prefix stable codes where obvious.
    if plan.status == "blocked" and plan.reason.startswith(
        ("authorization", "Risk score", "required capability", "explicit human"),
    ):
        return ToolCallPlan(
            plan.status,
            plan.tool_name,
            plan.arguments,
            annotate_reason("reject", plan.reason),
            plan.choice_probability,
            plan.choice_confidence,
            plan.risk_score,
        )
    if plan.status == "blocked" and "context sufficiency" in plan.reason:
        return ToolCallPlan(
            "abstain",
            plan.tool_name,
            None,
            annotate_reason("clarify", plan.reason),
            plan.choice_probability,
            plan.choice_confidence,
            plan.risk_score,
        )
    return plan


__all__ = ["AuthorityProfile", "annotate_reason", "apply_authority_profile"]
