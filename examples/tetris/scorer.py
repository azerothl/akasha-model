"""Path A placement scorer: heuristic features → Choice probabilities.

No torch / Hub. Optional Path B would swap ``score_placements`` for a MASK
scorer over the same placement IDs; keep this fallback for CI and first-run.
"""

from __future__ import annotations

import math
from typing import Sequence

from akasha_model import ChoiceQuestion, OptionSpec
from akasha_model.primitives import ChoiceResult, choice_result

from engine import Placement


def _softmax(scores: Sequence[float], temperature: float = 4.0) -> list[float]:
    if not scores:
        return []
    scale = max(temperature, 1e-6)
    shifted = [value / scale for value in scores]
    peak = max(shifted)
    exps = [math.exp(value - peak) for value in shifted]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [value / total for value in exps]


def build_choice_question(placements: Sequence[Placement]) -> ChoiceQuestion:
    if len(placements) < 2:
        raise ValueError("Choice requires at least two legal placements")
    options = tuple(
        OptionSpec(
            name=item.placement_id,
            description=(
                f"{item.piece} rot={item.rotation} col={item.column} "
                f"row={item.landing_row} lines={item.lines_cleared} "
                f"holes={item.holes} height={item.aggregate_height} "
                f"bump={item.bumpiness}"
            ),
            examples=(
                f"cells={list(item.occupied_cells)}",
            ),
        )
        for item in placements
    )
    return ChoiceQuestion(
        "placement",
        "Pick the best legal final lock for the active piece.",
        options,
    )


def score_placements(
    placements: Sequence[Placement],
    *,
    temperature: float = 4.0,
    threshold: float = 0.0,
) -> ChoiceResult:
    """Convert heuristic scores into a validated ChoiceResult."""
    question = build_choice_question(placements)
    probs = _softmax([item.heuristic_score for item in placements], temperature)
    mass = {
        item.placement_id: float(prob)
        for item, prob in zip(placements, probs, strict=True)
    }
    # Re-normalise after float rounding so choice_result accepts the mass.
    total = sum(mass.values())
    mass = {key: value / total for key, value in mass.items()}
    return choice_result(question, mass, threshold=threshold)


def top_alternatives(
    result: ChoiceResult,
    *,
    limit: int = 5,
) -> list[tuple[str, float]]:
    ordered = sorted(result.probabilities.items(), key=lambda item: item[1], reverse=True)
    return ordered[:limit]
