"""Host integration contract for gated tool calls.

Akasha produces a :class:`~akasha_model.tool_calling.ToolCallPlan`. This module
dispatches that plan to an application-supplied host. The package still has
**no** built-in tool executor: the host owns permission re-checks and side
effects (Akasha OS or any other runtime).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from .gate import GateSignals, ToolProposal, default_gate_planner, evaluate_gate
from .tool_calling import ToolCallPlan, ToolCallPlanner, ToolSpec


HostAction = Literal[
    "executed",
    "skipped_abstain",
    "skipped_blocked",
    "rejected_by_host",
]


@dataclass(frozen=True)
class HostOutcome:
    """Result of handing a plan to the host runtime."""

    action: HostAction
    plan: ToolCallPlan
    result: Any | None
    host_reason: str

    @property
    def did_execute(self) -> bool:
        return self.action == "executed"


class ToolHost(Protocol):
    """Application / OS side of the gate.

    Implement this in Akasha OS (or a test double). ``execute`` must perform
    real side effects only when called — ``dispatch_plan`` calls it solely
    after ``plan.status == "ready"`` *and* ``check_permissions`` succeeds.
    """

    def check_permissions(
        self, tool_name: str, arguments: Mapping[str, Any] | None,
    ) -> tuple[bool, str]:
        """Final host-side authorization (capabilities, ACLs, confirmations)."""

    def execute(
        self, tool_name: str, arguments: Mapping[str, Any] | None,
    ) -> Any:
        """Run the tool. Called only after ready + successful permission check."""


def dispatch_plan(plan: ToolCallPlan, host: ToolHost) -> HostOutcome:
    """Route a gate plan to the host without embedding an executor in Akasha.

    - ``abstain`` → skip, no ``execute``
    - ``blocked`` → skip, no ``execute``
    - ``ready`` → ``check_permissions`` then maybe ``execute``
    """
    if plan.status == "abstain":
        return HostOutcome(
            "skipped_abstain", plan, None, plan.reason,
        )
    if plan.status != "ready" or not plan.executable:
        return HostOutcome(
            "skipped_blocked", plan, None, plan.reason,
        )
    if plan.tool_name is None:
        return HostOutcome(
            "skipped_blocked", plan, None, "ready plan missing tool_name",
        )
    allowed, reason = host.check_permissions(plan.tool_name, plan.arguments)
    if not allowed:
        return HostOutcome("rejected_by_host", plan, None, reason)
    result = host.execute(plan.tool_name, plan.arguments)
    return HostOutcome(
        "executed", plan, result, "host executed after ready and permission check",
    )


def run_gated_call(
    tools: Mapping[str, ToolSpec],
    proposal: ToolProposal,
    signals: GateSignals,
    host: ToolHost,
    *,
    planner: ToolCallPlanner | None = None,
) -> HostOutcome:
    """Evaluate the gate then dispatch to the host (scripted / explicit signals)."""
    plan = evaluate_gate(
        tools, proposal, signals, planner=planner or default_gate_planner(),
    )
    return dispatch_plan(plan, host)


def describe_outcome(outcome: HostOutcome) -> str:
    """One-line summary for CLIs and demos."""
    tool = outcome.plan.tool_name or "—"
    return (
        f"{outcome.action.upper()}: {tool} — {outcome.host_reason} "
        f"(gate={outcome.plan.status})"
    )


__all__ = [
    "HostAction",
    "HostOutcome",
    "ToolHost",
    "describe_outcome",
    "dispatch_plan",
    "run_gated_call",
]
