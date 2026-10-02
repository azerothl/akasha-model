"""Structured decision traces for Path A usage demos (text + visual).

CPU only: no torch, no Hub. Probabilities are explicit host/scripted signals
so the first-run walkthrough stays install-light; a trained scorer would fill
the same fields under Path B.
"""

from __future__ import annotations

from typing import Any

from akasha_model import (
    ChoiceQuestion,
    GateSignals,
    OptionSpec,
    ScoreLevel,
    ScoreQuestion,
    ToolProposal,
    ToolSpec,
    describe_plan,
    evaluate_gate,
)
from akasha_model.gate import (
    DEFAULT_MAX_RISK_SCORE,
    DEFAULT_MIN_CHOICE_CONFIDENCE,
    DEFAULT_MIN_CHOICE_PROBABILITY,
    DEFAULT_NOUL_THRESHOLD,
    choice_from_proposal,
    default_gate_planner,
)
from akasha_model.primitives import choice_result, score_result


def tools() -> dict[str, ToolSpec]:
    return {
        "fs.read": ToolSpec(
            "fs.read",
            "Read a workspace file.",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="workspace_access",
        ),
        "fs.delete": ToolSpec(
            "fs.delete",
            "Delete a workspace file.",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
            required_capability="workspace_access",
            irreversible=True,
        ),
        "mail.broadcast": ToolSpec(
            "mail.broadcast",
            "Send mail to a large distribution list.",
            parameters={
                "type": "object",
                "required": ["list", "body"],
                "properties": {
                    "list": {"type": "string"},
                    "body": {"type": "string"},
                },
                "additionalProperties": False,
            },
            required_capability="mail_send",
            requires_confirmation=True,
        ),
    }


def _risk(level_mass: dict[int, float]):
    question = ScoreQuestion(
        "risk",
        "Rate the operational risk of executing the proposed tool.",
        (ScoreLevel("low"), ScoreLevel("medium"), ScoreLevel("high")),
    )
    return score_result(question, level_mass)


def _noul_question(
    question_id: str,
    prompt: str,
    probability: float,
    threshold: float,
) -> dict[str, Any]:
    return {
        "type": "noul",
        "id": question_id,
        "prompt": prompt,
        "probability": float(probability),
        "threshold": float(threshold),
        "passes": float(probability) >= float(threshold),
    }


def _choice_question(
    question: ChoiceQuestion,
    probabilities: dict[str, float],
    *,
    min_confidence: float | None = None,
) -> dict[str, Any]:
    result = choice_result(question, probabilities)
    options = []
    for option in question.options:
        options.append(
            {
                "name": option.name,
                "description": option.description or "",
                "probability": float(result.probabilities[option.name]),
            }
        )
    selected = result.selected
    abstained = False
    if min_confidence is not None and result.confidence < min_confidence:
        abstained = True
        selected = None
    return {
        "type": "choice",
        "id": question.question_id,
        "prompt": question.instructions,
        "options": options,
        "selected": selected,
        "confidence": float(result.confidence),
        "min_confidence": min_confidence,
        "abstained": abstained,
    }


def _score_question_payload(result) -> dict[str, Any]:
    levels = []
    for index, name in sorted(result.legend.items()):
        levels.append(
            {
                "index": int(index),
                "name": name,
                "probability": float(result.probabilities[index]),
            }
        )
    return {
        "type": "score",
        "id": result.question_id or "risk",
        "prompt": "Rate the operational risk of executing the proposed tool.",
        "levels": levels,
        "score": float(result.score),
        "confidence": float(result.confidence),
        "max_allowed": float(DEFAULT_MAX_RISK_SCORE),
        "passes": float(result.score) <= float(DEFAULT_MAX_RISK_SCORE),
    }


def _gate_step(
    *,
    step_id: str,
    title: str,
    situation: str,
    proposal: ToolProposal,
    signals: GateSignals,
    expected: str,
) -> dict[str, Any]:
    catalog = tools()
    planner = default_gate_planner()
    choice = choice_from_proposal(catalog, proposal)
    plan = evaluate_gate(catalog, proposal, signals, planner=planner, choice=choice)
    if plan.status != expected:
        raise SystemExit(
            f"{title}: expected {expected}, got {plan.status} ({describe_plan(plan)})"
        )

    noul_threshold = planner.noul_threshold
    questions: list[dict[str, Any]] = [
        {
            "type": "choice",
            "id": "route",
            "prompt": "Choose which tool to authorize for execution.",
            "options": [
                {
                    "name": name,
                    "description": catalog[name].description,
                    "probability": float(choice.probabilities[name]),
                }
                for name in sorted(catalog)
            ],
            "selected": choice.selected,
            "confidence": float(choice.confidence),
            "min_confidence": float(planner.min_choice_confidence),
            "min_probability": float(planner.min_choice_probability),
            "abstained": False,
        },
        _noul_question(
            "authorized",
            "Is the proposed call authorized?",
            signals.authorized,
            noul_threshold,
        ),
        _noul_question(
            "sufficient_context",
            "Is there enough context to run safely?",
            signals.sufficient_context,
            noul_threshold,
        ),
    ]
    if signals.capability_present is not None:
        questions.append(
            _noul_question(
                "capability_present",
                "Does the host hold the required capability?",
                signals.capability_present,
                noul_threshold,
            )
        )
    if signals.confirmation_needed is not None:
        questions.append(
            _noul_question(
                "confirmation_needed",
                "Does this call need human confirmation?",
                signals.confirmation_needed,
                noul_threshold,
            )
        )
    if signals.risk is not None:
        questions.append(_score_question_payload(signals.risk))

    host = (
        f"would execute {plan.tool_name}({dict(plan.arguments or {})})"
        if plan.status == "ready"
        else "skip execution"
    )
    return {
        "id": step_id,
        "section": "Tool gate",
        "title": title,
        "kind": "gate",
        "state": {
            "situation": situation,
            "proposal": {
                "tool": proposal.tool_name,
                "arguments": dict(proposal.arguments or {}),
            },
            "confirmation_given": signals.confirmation_given,
        },
        "questions": questions,
        "verdict": {
            "status": plan.status,
            "reason": plan.reason,
            "summary": describe_plan(plan),
            "choice_probability": float(plan.choice_probability),
            "choice_confidence": float(plan.choice_confidence),
            "risk_score": (
                None if plan.risk_score is None else float(plan.risk_score)
            ),
            "host": host,
        },
    }


def _ticket_step(
    *,
    step_id: str,
    title: str,
    ticket_text: str,
    masses: dict[str, float],
    min_confidence: float,
) -> dict[str, Any]:
    question = ChoiceQuestion(
        "queue",
        "Choose the support queue.",
        (
            OptionSpec("refund", "Billing correction or money back."),
            OptionSpec("sales", "Pricing, upgrade, or purchase help."),
            OptionSpec("technical", "Product bug or how-to."),
        ),
    )
    payload = _choice_question(question, masses, min_confidence=min_confidence)
    if payload["abstained"]:
        host = (
            f"escalate to human "
            f"(confidence={payload['confidence']:.2f} < {min_confidence:.2f})"
        )
        status = "abstain"
        reason = "choice confidence below host threshold"
    else:
        selected = payload["selected"]
        selected_p = next(
            option["probability"]
            for option in payload["options"]
            if option["name"] == selected
        )
        host = f"route → {selected} (p={selected_p:.2f})"
        status = "ready"
        reason = "choice confidence meets host threshold"

    return {
        "id": step_id,
        "section": "Ticket router",
        "title": title,
        "kind": "ticket",
        "state": {
            "situation": ticket_text,
            "proposal": None,
            "confirmation_given": False,
        },
        "questions": [payload],
        "verdict": {
            "status": status,
            "reason": reason,
            "summary": f"{status.upper()}: {reason}",
            "choice_probability": (
                None
                if payload["selected"] is None
                else next(
                    option["probability"]
                    for option in payload["options"]
                    if option["name"] == payload["selected"]
                )
            ),
            "choice_confidence": float(payload["confidence"]),
            "risk_score": None,
            "host": host,
        },
    }


def build_trace() -> dict[str, Any]:
    """Return a JSON-serialisable walkthrough of gate + ticket decisions."""
    steps = [
        _gate_step(
            step_id="gate-safe-read",
            title="Safe read",
            situation="User asks to open notes.txt in the workspace.",
            proposal=ToolProposal("fs.read", {"path": "notes.txt"}),
            signals=GateSignals(
                authorized=0.95,
                sufficient_context=0.92,
                capability_present=0.94,
                confirmation_needed=0.05,
                risk=_risk({0: 0.85, 1: 0.10, 2: 0.05}),
            ),
            expected="ready",
        ),
        _gate_step(
            step_id="gate-delete-blocked",
            title="Delete without confirmation",
            situation="System 2 proposes deleting notes.txt with no human confirm.",
            proposal=ToolProposal("fs.delete", {"path": "notes.txt"}),
            signals=GateSignals(
                authorized=0.95,
                sufficient_context=0.90,
                capability_present=0.93,
                risk=_risk({0: 0.10, 1: 0.30, 2: 0.60}),
                confirmation_given=False,
            ),
            expected="blocked",
        ),
        _gate_step(
            step_id="gate-broadcast-abstain",
            title="Ambiguous broadcast",
            situation="Choice mass is split; mail.broadcast needs confirmation.",
            proposal=ToolProposal(
                "mail.broadcast",
                {"list": "all-company", "body": "Office closed Friday."},
                choice_probabilities={
                    "fs.read": 0.20,
                    "fs.delete": 0.40,
                    "mail.broadcast": 0.40,
                },
            ),
            signals=GateSignals(
                authorized=0.80,
                sufficient_context=0.75,
                capability_present=0.90,
                confirmation_needed=0.80,
                risk=_risk({0: 0.20, 1: 0.40, 2: 0.40}),
            ),
            expected="abstain",
        ),
        _ticket_step(
            step_id="ticket-refund",
            title="Clear refund request",
            ticket_text="Double charge on order #4412",
            masses={"refund": 0.95, "sales": 0.03, "technical": 0.02},
            min_confidence=0.75,
        ),
        _ticket_step(
            step_id="ticket-unclear",
            title="Unclear ticket",
            ticket_text="Unclear message: maybe bug, maybe invoice",
            masses={"refund": 0.34, "sales": 0.33, "technical": 0.33},
            min_confidence=0.75,
        ),
    ]
    return {
        "title": "Akasha Model — visual usage demo",
        "path": "A",
        "note": (
            "Path A signals only: no torch, no Hub download, no side effects. "
            "Bars show the same typed Choice / Score / Noul inputs the planner "
            "reads before returning ready / abstain / blocked."
        ),
        "thresholds": {
            "min_choice_probability": DEFAULT_MIN_CHOICE_PROBABILITY,
            "min_choice_confidence": DEFAULT_MIN_CHOICE_CONFIDENCE,
            "noul_threshold": DEFAULT_NOUL_THRESHOLD,
            "max_risk_score": DEFAULT_MAX_RISK_SCORE,
        },
        "steps": steps,
    }


__all__ = ["build_trace", "tools"]
