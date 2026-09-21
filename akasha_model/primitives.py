"""Typed decision primitives for bounded Choice, Score and Noul questions.

This module deliberately contains no model code.  It defines the stable
contract between an application (Akasha-OS) and a probabilistic scorer.  A
model may implement one or more primitives, while deterministic application
code remains responsible for thresholds, authorization and side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal


Primitive = Literal["choice", "score", "noul"]


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True)
class OptionSpec:
    """One bounded Choice option with semantic guidance for the model."""

    name: str
    description: str = ""
    not_for: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _non_empty(self.name, "option name")
        if any(not isinstance(item, str) or not item.strip()
               for item in (*self.not_for, *self.examples)):
            raise ValueError("option exclusions and examples must be non-empty strings")

    def model_text(self) -> str:
        """Stable textual representation used by the current scorer."""
        parts = [self.name]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.not_for:
            parts.append("Not for: " + "; ".join(self.not_for))
        if self.examples:
            parts.append("Examples: " + "; ".join(self.examples))
        return "\n".join(parts)


@dataclass(frozen=True)
class ChoiceQuestion:
    question_id: str
    instructions: str
    options: tuple[OptionSpec, ...]
    primitive: Primitive = "choice"

    def __post_init__(self) -> None:
        _non_empty(self.question_id, "question_id")
        _non_empty(self.instructions, "instructions")
        if self.primitive != "choice":
            raise ValueError("ChoiceQuestion primitive must be 'choice'")
        if len(self.options) < 2 or len(self.options) > 255:
            raise ValueError("Choice requires between 2 and 255 options")
        names = [option.name for option in self.options]
        if len(names) != len(set(names)):
            raise ValueError("Choice option names must be unique")


@dataclass(frozen=True)
class ScoreLevel:
    description: str

    def __post_init__(self) -> None:
        _non_empty(self.description, "score level description")


@dataclass(frozen=True)
class ScoreQuestion:
    question_id: str
    instructions: str
    levels: tuple[ScoreLevel, ...]
    primitive: Primitive = "score"

    def __post_init__(self) -> None:
        _non_empty(self.question_id, "question_id")
        _non_empty(self.instructions, "instructions")
        if self.primitive != "score":
            raise ValueError("ScoreQuestion primitive must be 'score'")
        if not 2 <= len(self.levels) <= 10:
            raise ValueError("Score requires between 2 and 10 ordered levels")


@dataclass(frozen=True)
class NoulQuestion:
    question_id: str
    instructions: str
    primitive: Primitive = "noul"
    criteria: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        _non_empty(self.question_id, "question_id")
        _non_empty(self.instructions, "instructions")
        if self.primitive != "noul":
            raise ValueError("NoulQuestion primitive must be 'noul'")
        if self.criteria is not None:
            if len(self.criteria) != 2 or any(not item.strip() for item in self.criteria):
                raise ValueError("Noul criteria must contain true and false descriptions")


@dataclass(frozen=True)
class DecisionRequest:
    state: str | dict[str, Any] | list[Any]
    questions: tuple[ChoiceQuestion | ScoreQuestion | NoulQuestion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state, (str, dict, list)):
            raise ValueError("state must be text, a JSON object or a JSON array")
        if not self.questions:
            raise ValueError("at least one question is required")
        ids = [question.question_id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question_id values must be unique")


@dataclass(frozen=True)
class ChoiceResult:
    selected: str | None
    probabilities: dict[str, float]
    confidence: float
    abstained: bool = False
    question_id: str = ""


@dataclass(frozen=True)
class ScoreResult:
    score: float
    legend: dict[int, str]
    probabilities: dict[int, float]
    confidence: float
    question_id: str = ""


@dataclass(frozen=True)
class NoulResult:
    probability: float
    question_id: str = ""


def _distribution_confidence(probabilities: dict[Any, float]) -> float:
    """Approximate concentration-based confidence.

    Normalised entropy is a transparent approximation that uses the whole
    distribution. Published System 1 models use private formulas; this one
    stays inspectable.
    """
    if len(probabilities) <= 1:
        return 1.0
    entropy = -sum(value * math.log(value) for value in probabilities.values() if value > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(probabilities))))


def choice_result(
    question: ChoiceQuestion,
    probabilities: dict[str, float],
    *,
    threshold: float = 0.0,
) -> ChoiceResult:
    """Build a validated Choice result and apply an explicit abstention rule."""
    names = {option.name for option in question.options}
    if set(probabilities) != names:
        raise ValueError("Choice probabilities must contain exactly all option names")
    if any(value < 0 or value > 1 for value in probabilities.values()):
        raise ValueError("Choice probabilities must be in [0, 1]")
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-4:
        raise ValueError("Choice probabilities must sum to 1")
    selected = max(probabilities, key=probabilities.get)
    confidence = _distribution_confidence(probabilities)
    abstained = confidence < threshold
    return ChoiceResult(
        selected=None if abstained else selected,
        probabilities=dict(probabilities),
        confidence=confidence,
        abstained=abstained,
        question_id=question.question_id,
    )


def score_result(
    question: ScoreQuestion,
    probabilities: dict[int, float],
) -> ScoreResult:
    """Convert an ordered Score distribution into its expected scalar value."""
    expected = set(range(len(question.levels)))
    if set(probabilities) != expected:
        raise ValueError("Score probabilities must contain exactly all level indices")
    total = sum(probabilities.values())
    if abs(total - 1.0) > 1e-4 or any(value < 0 for value in probabilities.values()):
        raise ValueError("Score probabilities must be non-negative and sum to 1")
    score = sum(level * probability for level, probability in probabilities.items())
    legend = {index: level.description for index, level in enumerate(question.levels)}
    confidence = _distribution_confidence(probabilities)
    return ScoreResult(score, legend, dict(probabilities), confidence, question.question_id)


def noul_result(question: NoulQuestion, probability: float) -> NoulResult:
    """Build a binary truth-probability result."""
    if not 0 <= probability <= 1:
        raise ValueError("Noul probability must be in [0, 1]")
    return NoulResult(probability=probability, question_id=question.question_id)
