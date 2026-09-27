"""Score a tool proposal with the byte multitask scorer, then gate it.

Builds one shared-context Choice / Score / Noul row, runs
:class:`~akasha_model.multitask.MultiQuestionTinyScorer`, and converts the
heads into :class:`ChoiceResult` / :class:`GateSignals` for
:func:`evaluate_gate`. Still no executor.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.nn import functional as F

from .gate import (
    GateSignals,
    ToolProposal,
    choice_from_proposal,
    default_gate_planner,
    evaluate_gate,
)
from .host import HostOutcome, ToolHost, dispatch_plan
from .multitask import (
    MultiQuestionCollator,
    MultiQuestionExample,
    MultiQuestionTinyScorer,
    validate_multi,
)
from .multitask_eval import load_checkpoint
from .primitives import (
    ChoiceQuestion,
    ChoiceResult,
    OptionSpec,
    ScoreLevel,
    ScoreQuestion,
    choice_result,
    score_result,
)
from .tool_calling import ToolCallPlan, ToolCallPlanner, ToolSpec


RISK_LEVELS = ("low", "medium", "high")
NOUL_IDS = (
    "authorized",
    "sufficient_context",
    "capability_present",
    "confirmation_needed",
)


def gate_tool_options(tools: Mapping[str, ToolSpec]) -> list[dict[str, str]]:
    """Stable Choice options derived from the host catalog."""
    return [
        {"name": name, "description": tools[name].description}
        for name in sorted(tools)
    ]


def gate_questions_template(
    tools: Mapping[str, ToolSpec], *, route_label: int = 0,
) -> list[dict[str, Any]]:
    """Multitask questions used for authorize-tool-call rows and inference."""
    options = gate_tool_options(tools)
    if not 0 <= route_label < len(options):
        raise ValueError("route_label out of range for the tool catalog")
    return [
        {
            "id": "route",
            "type": "choice",
            "instructions": (
                "Choose which catalog tool should be authorized for execution "
                "given the proposed call and situation."
            ),
            "options": options,
            "label": route_label,
        },
        {
            "id": "risk",
            "type": "score",
            "instructions": (
                "Rate the operational risk of executing the proposed tool call."
            ),
            "levels": list(RISK_LEVELS),
            "label": 0,
        },
        {
            "id": "authorized",
            "type": "noul",
            "instructions": "Is the operation authorized in this situation?",
            "criteria": {
                "true": "Policy and user intent allow this tool call.",
                "false": "Authorization is missing or the call is out of policy.",
            },
            "label": 0,
        },
        {
            "id": "sufficient_context",
            "type": "noul",
            "instructions": "Is there enough context to execute safely?",
            "criteria": {
                "true": "Arguments and situation are clear enough to proceed.",
                "false": "Key details are missing or the request is ambiguous.",
            },
            "label": 0,
        },
        {
            "id": "capability_present",
            "type": "noul",
            "instructions": "Does the host hold the required capability?",
            "criteria": {
                "true": "The required capability is granted for this session.",
                "false": "The required capability is absent or revoked.",
            },
            "label": 0,
        },
        {
            "id": "confirmation_needed",
            "type": "noul",
            "instructions": "Is explicit human confirmation still required?",
            "criteria": {
                "true": "The call is irreversible or marked as needing confirmation.",
                "false": "No further human confirmation is required.",
            },
            "label": 0,
        },
    ]


def _dummy_example(
    context: Mapping[str, Any] | str,
    tools: Mapping[str, ToolSpec],
) -> MultiQuestionExample:
    payload_context: Any = context if isinstance(context, str) else dict(context)
    return validate_multi({
        "context": payload_context,
        "questions": gate_questions_template(tools, route_label=0),
    })


def score_proposal(
    model: MultiQuestionTinyScorer,
    collator: MultiQuestionCollator,
    tools: Mapping[str, ToolSpec],
    proposal: ToolProposal,
    *,
    context: Mapping[str, Any] | str,
    device: torch.device,
    confirmation_given: bool = False,
    use_model_choice: bool = False,
) -> tuple[ChoiceResult, GateSignals]:
    """Run the multitask scorer and pack planner inputs (no side effects).

    By default the Choice result is peaked on the System 2 ``proposal`` (the
    gate authorizes a proposed call). Pass ``use_model_choice=True`` to use the
    trained route head instead — useful once that head is strong enough.
    """
    example = _dummy_example(context, tools)
    batch = collator([example])
    moved = {
        kind: {name: value.to(device) for name, value in values.items()}
        for kind, values in batch.items()
    }
    model.eval()
    with torch.no_grad():
        outputs = model(moved)

    if use_model_choice:
        names = [option["name"] for option in gate_tool_options(tools)]
        choice_probs = F.softmax(outputs["choice"][0], dim=-1).tolist()
        choice_question = ChoiceQuestion(
            "route",
            "Choose which catalog tool should be authorized for execution.",
            tuple(OptionSpec(name, tools[name].description) for name in names),
        )
        choice = choice_result(
            choice_question,
            {name: float(prob) for name, prob in zip(names, choice_probs)},
        )
    else:
        choice = choice_from_proposal(tools, proposal)

    score_probs = F.softmax(outputs["score"][0, :len(RISK_LEVELS)], dim=-1)
    risk = score_result(
        ScoreQuestion(
            "risk",
            "Rate the operational risk of executing the proposed tool call.",
            tuple(ScoreLevel(level) for level in RISK_LEVELS),
        ),
        {index: float(score_probs[index]) for index in range(len(RISK_LEVELS))},
    )

    noul_values = torch.sigmoid(outputs["noul"]).tolist()
    noul_by_id = {
        question_id: float(value)
        for question_id, value in zip(NOUL_IDS, noul_values)
    }
    spec = tools.get(proposal.tool_name)
    # Only forward confirmation_needed when the catalog tool can require it.
    # Otherwise a noisy noul would false-positive-block safe tools like fs.read.
    confirmation_signal = None
    if spec is not None and (spec.requires_confirmation or spec.irreversible):
        confirmation_signal = noul_by_id["confirmation_needed"]
    signals = GateSignals(
        authorized=noul_by_id["authorized"],
        sufficient_context=noul_by_id["sufficient_context"],
        capability_present=noul_by_id["capability_present"],
        confirmation_needed=confirmation_signal,
        risk=risk,
        confirmation_given=confirmation_given,
    )
    return choice, signals


def plan_scored_proposal(
    model: MultiQuestionTinyScorer,
    collator: MultiQuestionCollator,
    tools: Mapping[str, ToolSpec],
    proposal: ToolProposal,
    *,
    context: Mapping[str, Any] | str,
    device: torch.device,
    confirmation_given: bool = False,
    planner: ToolCallPlanner | None = None,
    use_model_choice: bool = False,
) -> ToolCallPlan:
    """Score risk/nouls (and optionally route), then gate; never executes."""
    choice, signals = score_proposal(
        model, collator, tools, proposal, context=context, device=device,
        confirmation_given=confirmation_given,
        use_model_choice=use_model_choice,
    )
    return evaluate_gate(
        tools, proposal, signals,
        planner=planner or default_gate_planner(),
        choice=choice,
    )


def load_gate_multitask(path: str | Any, device: torch.device):
    """Load a multitask checkpoint saved by ``akasha-multitask-train``."""
    return load_checkpoint(path, device)


def run_scored_gated_call(
    model: MultiQuestionTinyScorer,
    collator: MultiQuestionCollator,
    tools: Mapping[str, ToolSpec],
    proposal: ToolProposal,
    host: ToolHost,
    *,
    context: Mapping[str, Any] | str,
    device: torch.device,
    confirmation_given: bool = False,
    planner: ToolCallPlanner | None = None,
    use_model_choice: bool = False,
) -> HostOutcome:
    """Score → gate → host dispatch. Executes only inside the supplied host."""
    plan = plan_scored_proposal(
        model, collator, tools, proposal,
        context=context, device=device,
        confirmation_given=confirmation_given,
        planner=planner,
        use_model_choice=use_model_choice,
    )
    return dispatch_plan(plan, host)


__all__ = [
    "NOUL_IDS",
    "RISK_LEVELS",
    "gate_questions_template",
    "gate_tool_options",
    "load_gate_multitask",
    "plan_scored_proposal",
    "run_scored_gated_call",
    "score_proposal",
]
