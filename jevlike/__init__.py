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
from .multitask import MultiQuestionCollator, MultiQuestionDataset, MultiQuestionTinyScorer
from .tool_calling import ToolCallPlan, ToolCallPlanner, ToolSpec, validate_tool_arguments
from .vision import CHESS_OPTION_IDS, DOOM_OPTION_IDS, TOTAL_OPTIONS, DoomScorerV2

__all__ = [
    "CHESS_OPTION_IDS", "DOOM_OPTION_IDS", "TOTAL_OPTIONS", "DoomScorerV2",
    "ChoiceQuestion", "ChoiceResult", "DecisionRequest", "NoulQuestion",
    "NoulResult", "OptionSpec", "ScoreLevel", "ScoreQuestion", "ScoreResult",
    "MultiQuestionCollator", "MultiQuestionDataset", "MultiQuestionTinyScorer",
    "ToolCallPlan", "ToolCallPlanner", "ToolSpec", "validate_tool_arguments",
]

__version__ = "0.1.0"
