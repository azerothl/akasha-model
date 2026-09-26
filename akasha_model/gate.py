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
from .tool_calling import ToolCallPlan, ToolCallPlanner, ToolSpec


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
    selected :class:`ToolSpec`, ``capability_present`` should be set.
    """

    authorized: float
    sufficient_context: float
    capability_present: float | None = None
    confirmation_needed: float | None = None
    risk: ScoreResult | None = None
    confirmation_given: bool = False


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
    """
    active = planner or ToolCallPlanner(
        min_choice_probability=0.55,
        min_choice_confidence=0.50,
        noul_threshold=0.70,
        max_risk_score=1.5,
    )
    resolved = choice if choice is not None else choice_from_proposal(tools, proposal)
    nouls: dict[str, float] = {
        "authorized": signals.authorized,
        "sufficient_context": signals.sufficient_context,
    }
    if signals.capability_present is not None:
        nouls["capability_present"] = signals.capability_present
    if signals.confirmation_needed is not None:
        nouls["confirmation_needed"] = signals.confirmation_needed
    return active.plan(
        resolved,
        tools,
        arguments=proposal.arguments,
        nouls=nouls,
        score=signals.risk,
        confirmation_given=signals.confirmation_given,
    )


def describe_plan(plan: ToolCallPlan) -> str:
    """One-line host-facing summary (for CLIs and demos)."""
    tool = plan.tool_name or "—"
    return f"{plan.status.upper()}: {tool} — {plan.reason}"


__all__ = [
    "GateSignals",
    "ToolProposal",
    "choice_from_proposal",
    "describe_plan",
    "evaluate_gate",
]
