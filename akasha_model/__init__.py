"""Small one-pass text and visual option scorers."""

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
from .decision import DecisionModel, load_decision_checkpoint
from .multitask import MultiQuestionCollator, MultiQuestionDataset, MultiQuestionTinyScorer
from .rlcd import grpo_loss
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
from .gate_multitask import plan_scored_proposal, run_scored_gated_call, score_proposal
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
from .typed_decisions import convert_typed_row
from .vision import CHESS_OPTION_IDS, DOOM_OPTION_IDS, TOTAL_OPTIONS, DoomScorerV2

__all__ = [
    "CHESS_OPTION_IDS", "DOOM_OPTION_IDS", "TOTAL_OPTIONS", "DoomScorerV2",
    "ChoiceQuestion", "ChoiceResult", "DecisionRequest", "NoulQuestion",
    "NoulResult", "OptionSpec", "ScoreLevel", "ScoreQuestion", "ScoreResult",
    "DEFAULT_MAX_RISK_SCORE", "DEFAULT_MIN_CHOICE_CONFIDENCE",
    "DEFAULT_MIN_CHOICE_PROBABILITY", "DEFAULT_NOUL_THRESHOLD",
    "DecisionModel", "MultiQuestionCollator", "MultiQuestionDataset",
    "MultiQuestionTinyScorer", "GateOutcomeRecord", "GateSignals",
    "HostOutcome", "ToolHost", "ToolProposal", "ToolCallPlan",
    "ToolCallPlanner", "ToolSpec", "append_outcome", "convert_typed_row",
    "default_gate_planner", "describe_outcome", "describe_plan",
    "dispatch_plan", "evaluate_gate", "grpo_loss", "load_decision_checkpoint",
    "load_outcomes", "plan_scored_proposal", "record_from_host_outcome",
    "run_gated_call", "run_scored_gated_call", "score_proposal",
    "suggest_threshold_updates", "summarize_outcomes",
    "validate_tool_arguments",
]

__version__ = "0.2.0"
