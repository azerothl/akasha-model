from akasha_model import (
    AuthorityProfile,
    GateSignals,
    ToolProposal,
    ToolSpec,
    evaluate_gate,
)
from akasha_model.primitives import ScoreLevel, ScoreQuestion, score_result


def _tools():
    return {
        "fs.read": ToolSpec(
            "fs.read",
            required_capability="workspace_access",
            parameters={
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False,
            },
        ),
        "payments.charge": ToolSpec(
            "payments.charge",
            required_capability="billing",
            irreversible=True,
        ),
    }


def test_without_profile_keeps_legacy_behaviour() -> None:
    plan = evaluate_gate(
        _tools(),
        ToolProposal("fs.read", {"path": "a.txt"}),
        GateSignals(
            authorized=0.95,
            sufficient_context=0.9,
            capability_present=0.92,
            confirmation_needed=0.05,
        ),
    )
    assert plan.status == "ready"
    assert not plan.reason.startswith(("escalate:", "reject:", "clarify:"))


def test_high_consequence_escalates() -> None:
    profile = AuthorityProfile(name="payments", max_consequence=0.4)
    plan = evaluate_gate(
        _tools(),
        ToolProposal("fs.read", {"path": "a.txt"}),
        GateSignals(
            authorized=0.95,
            sufficient_context=0.9,
            capability_present=0.92,
            confirmation_needed=0.05,
            consequence=0.9,
        ),
        authority_profile=profile,
    )
    assert plan.status == "abstain"
    assert plan.reason.startswith("escalate:")


def test_missing_grant_rejects() -> None:
    profile = AuthorityProfile(name="strict", min_authorized=0.95)
    plan = evaluate_gate(
        _tools(),
        ToolProposal("fs.read", {"path": "a.txt"}),
        GateSignals(
            authorized=0.2,
            sufficient_context=0.95,
            capability_present=0.95,
        ),
        authority_profile=profile,
    )
    assert plan.status == "blocked"
    assert plan.reason.startswith("reject:")


def test_missing_context_key_clarifies() -> None:
    profile = AuthorityProfile(
        name="mail",
        required_context_keys=("user_id", "thread_id"),
    )
    plan = evaluate_gate(
        _tools(),
        ToolProposal("fs.read", {"path": "a.txt"}),
        GateSignals(
            authorized=0.95,
            sufficient_context=0.95,
            capability_present=0.95,
            confirmation_needed=0.05,
            context={"user_id": "u1"},
        ),
        authority_profile=profile,
    )
    assert plan.status == "abstain"
    assert plan.reason.startswith("clarify:")
    assert "thread_id" in plan.reason


def test_profile_risk_reject() -> None:
    risk = score_result(
        ScoreQuestion(
            "risk",
            "Rate risk.",
            (ScoreLevel("low"), ScoreLevel("mid"), ScoreLevel("high")),
        ),
        {0: 0.0, 1: 0.0, 2: 1.0},
    )
    profile = AuthorityProfile(name="safe", max_risk_score=1.0)
    plan = evaluate_gate(
        _tools(),
        ToolProposal("fs.read", {"path": "a.txt"}),
        GateSignals(
            authorized=0.95,
            sufficient_context=0.95,
            capability_present=0.95,
            confirmation_needed=0.05,
            risk=risk,
        ),
        authority_profile=profile,
    )
    assert plan.status == "blocked"
    assert plan.reason.startswith("reject:")
