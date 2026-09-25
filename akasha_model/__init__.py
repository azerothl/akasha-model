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
from .gate import GateSignals, ToolProposal, describe_plan, evaluate_gate
from .tool_calling import ToolCallPlan, ToolCallPlanner, ToolSpec, validate_tool_arguments
from .typed_decisions import convert_typed_row
from .vision import CHESS_OPTION_IDS, DOOM_OPTION_IDS, TOTAL_OPTIONS, DoomScorerV2

__all__ = [
    "CHESS_OPTION_IDS", "DOOM_OPTION_IDS", "TOTAL_OPTIONS", "DoomScorerV2",
    "ChoiceQuestion", "ChoiceResult", "DecisionRequest", "NoulQuestion",
    "NoulResult", "OptionSpec", "ScoreLevel", "ScoreQuestion", "ScoreResult",
    "DecisionModel", "MultiQuestionCollator", "MultiQuestionDataset",
    "MultiQuestionTinyScorer", "GateSignals", "ToolProposal", "ToolCallPlan",
    "ToolCallPlanner", "ToolSpec", "convert_typed_row", "describe_plan",
    "evaluate_gate", "grpo_loss", "load_decision_checkpoint",
    "validate_tool_arguments",
]

__version__ = "0.2.0"
