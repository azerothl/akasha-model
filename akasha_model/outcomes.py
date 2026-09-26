"""Record and summarize real gate outcomes for threshold calibration.

Phase 2 of the gate product: learn from what happened after
``ready`` / ``abstain`` / ``blocked``, without putting an executor in the
model. Hosts (Akasha OS) append JSONL records; this module aggregates them
and suggests threshold moves.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .gate import (
    DEFAULT_MAX_RISK_SCORE,
    DEFAULT_MIN_CHOICE_CONFIDENCE,
    DEFAULT_MIN_CHOICE_PROBABILITY,
    DEFAULT_NOUL_THRESHOLD,
    GateSignals,
)
from .host import HostOutcome


@dataclass(frozen=True)
class GateOutcomeRecord:
    """One post-hoc observation after a gated tool attempt."""

    proposal_tool: str
    plan_status: str
    host_action: str
    plan_reason: str = ""
    proposal_arguments: Mapping[str, Any] | None = None
    success: bool | None = None
    user_forced: bool = False
    notes: str = ""
    signals: Mapping[str, float] | None = None
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.proposal_arguments is not None:
            payload["proposal_arguments"] = dict(self.proposal_arguments)
        if self.signals is not None:
            payload["signals"] = dict(self.signals)
        return payload


def signals_snapshot(signals: GateSignals | None) -> dict[str, float] | None:
    """Flatten GateSignals floats for logging (risk as scalar if present)."""
    if signals is None:
        return None
    snapshot: dict[str, float] = {
        "authorized": float(signals.authorized),
        "sufficient_context": float(signals.sufficient_context),
    }
    if signals.capability_present is not None:
        snapshot["capability_present"] = float(signals.capability_present)
    if signals.confirmation_needed is not None:
        snapshot["confirmation_needed"] = float(signals.confirmation_needed)
    if signals.risk is not None:
        snapshot["risk_score"] = float(signals.risk.score)
    return snapshot


def record_from_host_outcome(
    outcome: HostOutcome,
    *,
    success: bool | None = None,
    user_forced: bool = False,
    notes: str = "",
    signals: GateSignals | None = None,
) -> GateOutcomeRecord:
    """Build a record from a ``HostOutcome`` (call after the host returns)."""
    plan = outcome.plan
    resolved_success = success
    if resolved_success is None and outcome.did_execute:
        # Best-effort: treat a dict with ok=False as failure when present.
        result = outcome.result
        if isinstance(result, Mapping) and "ok" in result:
            resolved_success = bool(result["ok"])
    return GateOutcomeRecord(
        proposal_tool=plan.tool_name or "",
        proposal_arguments=plan.arguments,
        plan_status=plan.status,
        plan_reason=plan.reason,
        host_action=outcome.action,
        success=resolved_success,
        user_forced=user_forced,
        notes=notes,
        signals=signals_snapshot(signals),
    )


def append_outcome(path: str | Path, record: GateOutcomeRecord) -> None:
    """Append one JSONL outcome row (creates parent dirs)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")


def load_outcomes(path: str | Path) -> list[GateOutcomeRecord]:
    """Load outcome records from JSONL."""
    target = Path(path)
    if not target.is_file():
        return []
    rows: list[GateOutcomeRecord] = []
    with target.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(GateOutcomeRecord(
                proposal_tool=str(payload.get("proposal_tool", "")),
                proposal_arguments=payload.get("proposal_arguments"),
                plan_status=str(payload.get("plan_status", "")),
                plan_reason=str(payload.get("plan_reason", "")),
                host_action=str(payload.get("host_action", "")),
                success=payload.get("success"),
                user_forced=bool(payload.get("user_forced", False)),
                notes=str(payload.get("notes", "")),
                signals=payload.get("signals"),
                recorded_at=str(payload.get("recorded_at", "")),
            ))
    return rows


def summarize_outcomes(records: Iterable[GateOutcomeRecord]) -> dict[str, Any]:
    """Aggregate counts useful for gate calibration."""
    rows = list(records)
    by_action = Counter(row.host_action for row in rows)
    by_status = Counter(row.plan_status for row in rows)
    executed = [row for row in rows if row.host_action == "executed"]
    successes = [row for row in executed if row.success is True]
    failures = [row for row in executed if row.success is False]
    forced = [row for row in rows if row.user_forced]
    false_blocks = [
        row for row in forced
        if row.plan_status in {"blocked", "abstain"} and row.success is not False
    ]
    return {
        "count": len(rows),
        "by_host_action": dict(by_action),
        "by_plan_status": dict(by_status),
        "executed": len(executed),
        "execution_success": len(successes),
        "execution_failure": len(failures),
        "execution_success_rate": (
            len(successes) / len(executed) if executed else None
        ),
        "user_forced": len(forced),
        "false_positive_blocks": len(false_blocks),
        "false_positive_rate_among_forced": (
            len(false_blocks) / len(forced) if forced else None
        ),
    }


def suggest_threshold_updates(
    records: Iterable[GateOutcomeRecord],
    *,
    min_choice_probability: float = DEFAULT_MIN_CHOICE_PROBABILITY,
    min_choice_confidence: float = DEFAULT_MIN_CHOICE_CONFIDENCE,
    noul_threshold: float = DEFAULT_NOUL_THRESHOLD,
    max_risk_score: float = DEFAULT_MAX_RISK_SCORE,
) -> dict[str, Any]:
    """Heuristic suggestions only — never auto-writes new defaults.

    Rules of thumb (need enough volume before acting):
    - Many user_forced through block/abstain → thresholds may be too strict
      (lower noul / raise max_risk slightly).
    - Many executed failures with high risk_score → lower max_risk_score.
    """
    rows = list(records)
    summary = summarize_outcomes(rows)
    suggestions: list[str] = []
    proposed = {
        "min_choice_probability": min_choice_probability,
        "min_choice_confidence": min_choice_confidence,
        "noul_threshold": noul_threshold,
        "max_risk_score": max_risk_score,
    }
    if summary["count"] < 5:
        suggestions.append(
            "Collect at least ~5 outcome rows before changing defaults.",
        )
        return {
            "current": dict(proposed),
            "proposed": dict(proposed),
            "suggestions": suggestions,
            "summary": summary,
        }

    forced = summary["user_forced"]
    false_blocks = summary["false_positive_blocks"]
    if forced >= 3 and false_blocks / max(forced, 1) >= 0.5:
        proposed["noul_threshold"] = round(max(0.55, noul_threshold - 0.05), 2)
        proposed["max_risk_score"] = round(min(2.0, max_risk_score + 0.1), 2)
        suggestions.append(
            "Frequent user overrides of block/abstain: consider slightly "
            "looser noul_threshold and max_risk_score.",
        )

    failures = [row for row in rows
                if row.host_action == "executed" and row.success is False]
    risky_failures = [
        row for row in failures
        if row.signals and float(row.signals.get("risk_score", 0)) >= max_risk_score - 0.2
    ]
    if len(failures) >= 3 and len(risky_failures) >= 2:
        proposed["max_risk_score"] = round(max(0.8, max_risk_score - 0.1), 2)
        suggestions.append(
            "Several failed executions near the risk ceiling: consider a "
            "stricter max_risk_score.",
        )

    if not suggestions:
        suggestions.append(
            "No strong threshold signal yet; keep current defaults and keep logging.",
        )
    return {
        "current": {
            "min_choice_probability": min_choice_probability,
            "min_choice_confidence": min_choice_confidence,
            "noul_threshold": noul_threshold,
            "max_risk_score": max_risk_score,
        },
        "proposed": proposed,
        "suggestions": suggestions,
        "summary": summary,
    }


__all__ = [
    "GateOutcomeRecord",
    "append_outcome",
    "load_outcomes",
    "record_from_host_outcome",
    "signals_snapshot",
    "suggest_threshold_updates",
    "summarize_outcomes",
]
