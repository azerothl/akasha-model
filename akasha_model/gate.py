"""Host-facing tool gate: typed signals in, ``ToolCallPlan`` out, never execute.

This module packages the existing :class:`~akasha_model.tool_calling.ToolCallPlanner`
for the product direction where a System 2 proposes a tool call and Akasha
decides ``ready`` / ``abstain`` / ``blocked``. Side effects stay in the host
(Akasha OS or another runtime).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .primitives import (
    ChoiceQuestion,
    ChoiceResult,
    OptionSpec,
    ScoreResult,
    choice_result,
)
from .authority import AuthorityProfile, apply_authority_profile
from .catalog import CatalogPolicy, check_catalog_policy
from .tool_calling import ToolCallPlan, ToolCallPlanner, ToolSpec

# Default planner thresholds for the gate wedge. Documented in
# ``docs/using-the-tool-gate.md`` and ``examples/gate/README.md``. Tune on a
# held-out authorize-tool-call split before changing production defaults.
DEFAULT_MIN_CHOICE_PROBABILITY = 0.55
DEFAULT_MIN_CHOICE_CONFIDENCE = 0.50
DEFAULT_NOUL_THRESHOLD = 0.70
DEFAULT_MAX_RISK_SCORE = 1.5


def default_gate_planner() -> ToolCallPlanner:
    """Conservative planner used by ``evaluate_gate`` when none is supplied."""
    return ToolCallPlanner(
        min_choice_probability=DEFAULT_MIN_CHOICE_PROBABILITY,
        min_choice_confidence=DEFAULT_MIN_CHOICE_CONFIDENCE,
        noul_threshold=DEFAULT_NOUL_THRESHOLD,
        max_risk_score=DEFAULT_MAX_RISK_SCORE,
    )


@dataclass(frozen=True)
class ToolProposal:
    """What a System 2 (or policy) wants to run — not yet authorized."""

    tool_name: str
    arguments: Mapping[str, Any] | None = None
    choice_probabilities: Mapping[str, float] | None = None


@dataclass(frozen=True)
class GateSignals:
    """Authorization / risk signals that feed the planner nouls and score.

    Probabilities are in ``[0, 1]``. When a capability is required by the
    selected :class:`ToolSpec`, set ``capability_present`` (AND of all caps)
    and/or ``capability_signals`` (per-token map, preferred for multi-cap tools).

    ``budget_remaining`` is host-owned: the OS increments counters; when it
    reports a remaining budget ≤ 0 the gate returns ``blocked`` with reason
    ``budget_exceeded``. ``None`` means “no budget policy supplied”.
    """

    authorized: float
    sufficient_context: float
    capability_present: float | None = None
    confirmation_needed: float | None = None
    risk: ScoreResult | None = None
    confirmation_given: bool = False
    budget_remaining: float | None = None
    capability_signals: Mapping[str, float] | None = None
    consequence: float | None = None
    context: Mapping[str, object] | None = None


def choice_from_proposal(
    tools: Mapping[str, ToolSpec],
    proposal: ToolProposal,
    *,
    peak: float = 0.90,
) -> ChoiceResult:
    """Build a Choice result peaked on the proposed tool (or an explicit dist)."""
    if len(tools) < 2:
        raise ValueError("gate choice requires at least two tools in the catalog")
    names = sorted(tools)
    question = ChoiceQuestion(
        "route",
        "Choose which tool to authorize for execution.",
        tuple(OptionSpec(name, tools[name].description) for name in names),
    )
    if proposal.choice_probabilities is not None:
        return choice_result(question, dict(proposal.choice_probabilities))
    if proposal.tool_name not in tools:
        raise ValueError(
            f"proposed tool {proposal.tool_name!r} is not in the tool catalog"
        )
    if not 0.5 < peak < 1.0:
        raise ValueError("peak must be in (0.5, 1)")
    rest = (1.0 - peak) / (len(names) - 1)
    probabilities = {
        name: (peak if name == proposal.tool_name else rest) for name in names
    }
    return choice_result(question, probabilities)


def evaluate_gate(
    tools: Mapping[str, ToolSpec],
    proposal: ToolProposal,
    signals: GateSignals,
    *,
    planner: ToolCallPlanner | None = None,
    choice: ChoiceResult | None = None,
    catalog_policy: CatalogPolicy | None = None,
    authority_profile: AuthorityProfile | None = None,
) -> ToolCallPlan:
    """Return a non-executing plan for a proposed tool call.

    Parameters
    ----------
    tools:
        Host catalog keyed by tool name.
    proposal:
        Proposed tool and arguments (and optional soft choice distribution).
    signals:
        Noul / risk / confirmation inputs for the planner.
    planner:
        Optional custom thresholds; defaults are conservative for a demo gate.
    choice:
        Optional precomputed Choice result (e.g. from a trained scorer).
    catalog_policy:
        Optional phase-3 allow/deny/placement constraints (see
        :mod:`akasha_model.catalog`).
    authority_profile:
        Optional MIDAS-inspired profile for escalate / reject / clarify reason
        codes. Omitted → today's ``ready`` / ``abstain`` / ``blocked`` behaviour.
    """
    blocked = check_catalog_policy(tools, proposal.tool_name, catalog_policy)
    if blocked is not None:
        return blocked
    active = planner or default_gate_planner()
    resolved = choice if choice is not None else choice_from_proposal(tools, proposal)
    if signals.budget_remaining is not None and signals.budget_remaining <= 0:
        probability = float(
            resolved.probabilities.get(resolved.selected or "", 0.0)
        ) if resolved.selected else 0.0
        return ToolCallPlan(
            "blocked",
            proposal.tool_name if proposal.tool_name in tools else resolved.selected,
            None,
            "budget_exceeded",
            probability,
            resolved.confidence,
            signals.risk.score if signals.risk else None,
        )
    nouls: dict[str, float] = {
        "authorized": signals.authorized,
        "sufficient_context": signals.sufficient_context,
    }
    if signals.capability_present is not None:
        nouls["capability_present"] = signals.capability_present
    if signals.capability_signals:
        for cap, value in signals.capability_signals.items():
            nouls[f"capability:{cap}"] = float(value)
    if signals.confirmation_needed is not None:
        nouls["confirmation_needed"] = signals.confirmation_needed
    plan = active.plan(
        resolved,
        tools,
        arguments=proposal.arguments,
        nouls=nouls,
        score=signals.risk,
        confirmation_given=signals.confirmation_given,
    )
    return apply_authority_profile(
        plan,
        profile=authority_profile,
        authorized=signals.authorized,
        sufficient_context=signals.sufficient_context,
        risk_score=signals.risk.score if signals.risk else None,
        consequence=signals.consequence,
        context=signals.context,
    )


def describe_plan(plan: ToolCallPlan) -> str:
    """One-line host-facing summary (for CLIs and demos)."""
    tool = plan.tool_name or "—"
    return f"{plan.status.upper()}: {tool} — {plan.reason}"


__all__ = [
    "DEFAULT_MAX_RISK_SCORE",
    "DEFAULT_MIN_CHOICE_CONFIDENCE",
    "DEFAULT_MIN_CHOICE_PROBABILITY",
    "DEFAULT_NOUL_THRESHOLD",
    "AuthorityProfile",
    "CatalogPolicy",
    "GateSignals",
    "ToolProposal",
    "choice_from_proposal",
    "default_gate_planner",
    "describe_plan",
    "evaluate_gate",
]
