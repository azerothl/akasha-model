"""Path A placement scorer: heuristic features → Choice probabilities.

No torch / Hub. Optional Path B would swap ``score_placements`` for a MASK
scorer over the same placement IDs; keep this fallback for CI and first-run.

When ``next_piece`` is known (standard Tetris preview), each candidate's
score includes a one-ply look-ahead: after locking, take the best heuristic
score available for the upcoming piece. That feeds the preview into Choice.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from akasha_model import ChoiceQuestion, OptionSpec
from akasha_model.primitives import ChoiceResult, choice_result

from engine import Placement, apply_lock, empty_board, enumerate_placements


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


def lookahead_bonus(
    board_after: Sequence[Sequence[str | None]],
    next_piece: str | None,
    *,
    weight: float = 0.55,
) -> float:
    """Best Path A score for ``next_piece`` on ``board_after``, scaled."""
    if not next_piece:
        return 0.0
    upcoming = enumerate_placements(board_after, next_piece)
    if not upcoming:
        return -80.0 * weight
    return weight * max(item.heuristic_score for item in upcoming)


def effective_scores(
    placements: Sequence[Placement],
    board: Sequence[Sequence[str | None]] | None = None,
    *,
    next_piece: str | None = None,
) -> list[float]:
    """Heuristic score plus optional next-piece look-ahead."""
    base_board = board if board is not None else empty_board()
    scores: list[float] = []
    for item in placements:
        bonus = 0.0
        if next_piece:
            after, _cleared = apply_lock(base_board, item.piece, item.occupied_cells)
            bonus = lookahead_bonus(after, next_piece)
        scores.append(item.heuristic_score + bonus)
    return scores


def build_choice_question(
    placements: Sequence[Placement],
    *,
    next_piece: str | None = None,
) -> ChoiceQuestion:
    if len(placements) < 2:
        raise ValueError("Choice requires at least two legal placements")
    preview = (
        f" Next piece in preview: {next_piece}."
        if next_piece
        else ""
    )
    options = tuple(
        OptionSpec(
            name=item.placement_id,
            description=(
                f"Lock {item.piece} with rotation {item.rotation} at column "
                f"{item.column} (landing row {item.landing_row}); "
                f"clears {item.lines_cleared} line(s); holes={item.holes}, "
                f"height={item.aggregate_height}, bump={item.bumpiness}."
            ),
            examples=(
                f"cells={list(item.occupied_cells)}",
            ),
        )
        for item in placements
    )
    return ChoiceQuestion(
        "placement",
        (
            "Pick the best legal final lock for the active piece."
            f"{preview} Prefer setups that leave a good landing for the preview."
        ),
        options,
    )


def score_placements(
    placements: Sequence[Placement],
    *,
    temperature: float = 4.0,
    threshold: float = 0.0,
    board: Sequence[Sequence[str | None]] | None = None,
    next_piece: str | None = None,
) -> ChoiceResult:
    """Convert heuristic (+ optional next-piece) scores into a ChoiceResult."""
    question = build_choice_question(placements, next_piece=next_piece)
    scores = effective_scores(placements, board, next_piece=next_piece)
    probs = _softmax(scores, temperature)
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


def sample_placement(
    result: ChoiceResult,
    rng: random.Random,
) -> str:
    """Draw one placement id from the Choice probability mass."""
    if not result.probabilities:
        raise ValueError("no probabilities to sample")
    names = list(result.probabilities.keys())
    weights = [result.probabilities[name] for name in names]
    return rng.choices(names, weights=weights, k=1)[0]
