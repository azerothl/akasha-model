"""Small one-pass text and visual option scorers.

Path A (planner + host + outcomes + wire) imports without torch. Path B scorers,
training, and vision load lazily and require the ``torch`` extra::

    pip install "akasha-model[torch]"
"""

from __future__ import annotations

from typing import Any

from .primitives import (
    ChoiceQuestion,
    ChoiceResult,
    DecisionRequest,
    NoulQuestion,
    NoulResult,
    OptionSpec,
    ScoreLevel,
    ScoreQuestion,
    ScoreResult,
)
from .authority import AuthorityProfile
from .catalog import CatalogPolicy, check_catalog_policy
from .gate import (
    DEFAULT_MAX_RISK_SCORE,
    DEFAULT_MIN_CHOICE_CONFIDENCE,
    DEFAULT_MIN_CHOICE_PROBABILITY,
    DEFAULT_NOUL_THRESHOLD,
    GateSignals,
    ToolProposal,
    default_gate_planner,
    describe_plan,
    evaluate_gate,
)
from .host import (
    HostOutcome,
    ToolHost,
    describe_outcome,
    dispatch_plan,
    run_gated_call,
)
from .outcomes import (
    GateOutcomeRecord,
    append_outcome,
    load_outcomes,
    record_from_host_outcome,
    suggest_threshold_updates,
    summarize_outcomes,
)
from .tool_calling import ToolCallPlan, ToolCallPlanner, ToolSpec, validate_tool_arguments
from .audit import (
    GateAuditEnvelope,
    build_audit_envelope,
    content_hash,
    envelope_from_dict,
    envelope_to_dict,
    verify_envelope,
)
from .wire import (
    CONTRACT_VERSION,
    outcome_from_dict,
    outcome_to_dict,
    plan_from_dict,
    plan_to_dict,
    request_from_dict,
    request_to_dict,
    schema_path,
)
from .mix import (
    MIX_SCHEMA_VERSION,
    MixSessionState,
    PairMasking,
    TrackDescriptors,
    validate_mix_state,
)
from .mix_presets import MixPresetCatalog, MixSettingsProposal, propose_mix_settings
from .mix_proposal import (
    MixChange,
    MixProposalBatch,
    build_mix_audit_envelope,
    build_mix_proposal_batch,
)
from .offline import MODEL_ABSENT_STATUS, ModelAbsentError, resolve_local_checkpoint

__version__ = "0.2.0"

# Symbols that require the optional torch stack. Resolved via __getattr__.
_TORCH_EXPORTS = {
    "CHESS_OPTION_IDS",
    "DOOM_OPTION_IDS",
    "TOTAL_OPTIONS",
    "DecisionModel",
    "DoomScorerV2",
    "MultiQuestionCollator",
    "MultiQuestionDataset",
    "MultiQuestionTinyScorer",
    "convert_typed_row",
    "grpo_loss",
    "load_decision_checkpoint",
    "plan_scored_proposal",
    "run_scored_gated_call",
    "score_proposal",
}

__all__ = [
    "CHESS_OPTION_IDS", "DOOM_OPTION_IDS", "TOTAL_OPTIONS", "DoomScorerV2",
    "CONTRACT_VERSION", "GateAuditEnvelope",
    "ChoiceQuestion", "ChoiceResult", "DecisionRequest", "NoulQuestion",
    "NoulResult", "OptionSpec", "ScoreLevel", "ScoreQuestion", "ScoreResult",
    "DEFAULT_MAX_RISK_SCORE", "DEFAULT_MIN_CHOICE_CONFIDENCE",
    "DEFAULT_MIN_CHOICE_PROBABILITY", "DEFAULT_NOUL_THRESHOLD",
    "AuthorityProfile", "CatalogPolicy", "DecisionModel", "MultiQuestionCollator",
    "MultiQuestionDataset", "MultiQuestionTinyScorer", "GateOutcomeRecord",
    "GateSignals", "HostOutcome", "MIX_SCHEMA_VERSION", "MODEL_ABSENT_STATUS",
    "MixChange", "MixPresetCatalog", "MixProposalBatch", "ModelAbsentError",
    "MixSessionState", "MixSettingsProposal",
    "PairMasking", "ToolHost", "ToolProposal", "ToolCallPlan",
    "ToolCallPlanner", "ToolSpec", "TrackDescriptors", "append_outcome",
    "build_audit_envelope", "build_mix_audit_envelope", "build_mix_proposal_batch",
    "check_catalog_policy", "content_hash", "convert_typed_row",
    "default_gate_planner", "describe_outcome", "describe_plan",
    "dispatch_plan", "envelope_from_dict", "envelope_to_dict", "evaluate_gate",
    "grpo_loss",
    "load_decision_checkpoint", "load_outcomes", "outcome_from_dict",
    "outcome_to_dict", "plan_from_dict", "plan_scored_proposal", "plan_to_dict",
    "propose_mix_settings",
    "record_from_host_outcome", "request_from_dict", "request_to_dict",
    "resolve_local_checkpoint",
    "run_gated_call", "run_scored_gated_call", "schema_path", "score_proposal",
    "suggest_threshold_updates", "summarize_outcomes", "validate_mix_state",
    "validate_tool_arguments",
    "verify_envelope",
]


def __getattr__(name: str) -> Any:
    """Lazy-load torch-backed exports so Path A stays installable without torch."""
    if name not in _TORCH_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        if name in {"DecisionModel", "load_decision_checkpoint"}:
            from .decision import DecisionModel, load_decision_checkpoint

            values = {
                "DecisionModel": DecisionModel,
                "load_decision_checkpoint": load_decision_checkpoint,
            }
        elif name in {
            "MultiQuestionCollator",
            "MultiQuestionDataset",
            "MultiQuestionTinyScorer",
        }:
            from .multitask import (
                MultiQuestionCollator,
                MultiQuestionDataset,
                MultiQuestionTinyScorer,
            )

            values = {
                "MultiQuestionCollator": MultiQuestionCollator,
                "MultiQuestionDataset": MultiQuestionDataset,
                "MultiQuestionTinyScorer": MultiQuestionTinyScorer,
            }
        elif name == "grpo_loss":
            from .rlcd import grpo_loss

            values = {"grpo_loss": grpo_loss}
        elif name in {
            "plan_scored_proposal",
            "run_scored_gated_call",
            "score_proposal",
        }:
            from .gate_multitask import (
                plan_scored_proposal,
                run_scored_gated_call,
                score_proposal,
            )

            values = {
                "plan_scored_proposal": plan_scored_proposal,
                "run_scored_gated_call": run_scored_gated_call,
                "score_proposal": score_proposal,
            }
        elif name == "convert_typed_row":
            from .typed_decisions import convert_typed_row

            values = {"convert_typed_row": convert_typed_row}
        else:
            from .vision import CHESS_OPTION_IDS, DOOM_OPTION_IDS, TOTAL_OPTIONS, DoomScorerV2

            values = {
                "CHESS_OPTION_IDS": CHESS_OPTION_IDS,
                "DOOM_OPTION_IDS": DOOM_OPTION_IDS,
                "TOTAL_OPTIONS": TOTAL_OPTIONS,
                "DoomScorerV2": DoomScorerV2,
            }
    except ImportError as exc:  # pragma: no cover - exercised in gate-only CI
        raise ImportError(
            f"{name} requires the torch extra. "
            'Install with: pip install "akasha-model[torch]"'
        ) from exc
    globals().update(values)
    return values[name]
