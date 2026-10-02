"""Smoke tests for the realtime Tetris demo (Path A, no torch)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from akasha_model import ChoiceQuestion

ROOT = Path(__file__).resolve().parent


@pytest.fixture()
def tetris_mods():
    sys.path.insert(0, str(ROOT))
    try:
        engine = importlib.import_module("engine")
        scorer = importlib.import_module("scorer")
        play = importlib.import_module("play")
        engine = importlib.reload(engine)
        scorer = importlib.reload(scorer)
        play = importlib.reload(play)
        yield engine, scorer, play
    finally:
        sys.path.pop(0)
        for name in ("engine", "scorer", "play"):
            sys.modules.pop(name, None)


def test_enumerate_and_verify_roundtrip(tetris_mods):
    engine, scorer, _play = tetris_mods
    board = engine.empty_board()
    placements = engine.enumerate_placements(board, "T")
    assert len(placements) >= 2
    ids = {item.placement_id for item in placements}
    assert len(ids) == len(placements)
    # Human-readable lock labels (not opaque p0/p1…).
    assert all("rot=" in item.placement_id and "col=" in item.placement_id for item in placements)
    assert all(item.placement_id.startswith("T ") for item in placements)
    result = scorer.score_placements(placements, board=board, next_piece="I")
    assert isinstance(result.selected, str)
    assert result.selected in ids
    verified = engine.verify_placement(board, "T", result.selected, placements)
    assert verified.ok
    assert verified.placement is not None
    next_board, cleared = engine.lock_verified(board, "T", verified.placement)
    assert cleared == 0
    assert any(cell == "T" for row in next_board for cell in row)


def test_host_rejects_unknown_id(tetris_mods):
    engine, _scorer, _play = tetris_mods
    board = engine.empty_board()
    placements = engine.enumerate_placements(board, "O")
    bad = engine.verify_placement(board, "O", "O rot=0 col=99", placements)
    assert not bad.ok
    assert "unknown" in bad.reason


def test_choice_question_matches_ids(tetris_mods):
    engine, scorer, _play = tetris_mods
    board = engine.empty_board()
    placements = engine.enumerate_placements(board, "I")
    question = scorer.build_choice_question(placements, next_piece="O")
    assert isinstance(question, ChoiceQuestion)
    assert {opt.name for opt in question.options} == {
        item.placement_id for item in placements
    }
    assert "Next piece in preview: O" in question.instructions


def test_seven_bag_peek_matches_next(tetris_mods):
    engine, _scorer, _play = tetris_mods
    bag = engine.SevenBag(seed=11)
    upcoming = bag.peek()
    assert upcoming in engine.BAG_ORDER
    assert bag.next_piece() == upcoming
    assert bag.peek() in engine.BAG_ORDER


def test_next_piece_changes_scores(tetris_mods):
    engine, scorer, _play = tetris_mods
    board = engine.empty_board()
    placements = engine.enumerate_placements(board, "T")
    plain = scorer.effective_scores(placements, board, next_piece=None)
    with_i = scorer.effective_scores(placements, board, next_piece="I")
    assert len(plain) == len(with_i)
    assert plain != with_i


def test_gravity_and_level_accelerate(tetris_mods):
    engine, _scorer, _play = tetris_mods
    assert engine.drop_interval_ms(0) == engine.BASE_DROP_MS
    assert engine.drop_interval_ms(5) < engine.drop_interval_ms(0)
    assert engine.drop_interval_ms(0, score=500) < engine.drop_interval_ms(0, score=0)
    assert engine.drop_interval_ms(100) == engine.MIN_DROP_MS
    assert engine.level_for_lines(0) == 0
    assert engine.level_for_lines(engine.LINES_PER_LEVEL) == 1
    assert engine.score_for_soft_drop(12) == 12
    board = engine.empty_board()
    active = engine.spawn_piece(board, "T")
    assert active is not None
    assert active.origin_row == 0
    fallen, locked = engine.soft_drop(board, active)
    assert not locked
    assert fallen is not None
    assert fallen.origin_row == 1


def test_plan_approach_shows_falling_rows(tetris_mods):
    engine, scorer, _play = tetris_mods
    board = engine.empty_board()
    placements = engine.enumerate_placements(board, "O")
    result = scorer.score_placements(placements, board=board, next_piece="I")
    target = next(item for item in placements if item.placement_id == result.selected)
    path = engine.plan_approach(board, "O", target)
    assert len(path) >= 2
    rows = [pose.origin_row for pose in path]
    assert max(rows) > min(rows)
    assert path[0].origin_row == 0


def test_play_trace_and_html(tetris_mods):
    _engine, _scorer, play = tetris_mods
    trace = play.run_game(pieces=6, seed=3)
    assert trace["path"] == "A"
    assert trace["scorer"] == "path_a_heuristic"
    assert trace["mode"] == "realtime_gravity"
    assert trace["pieces_requested"] == 6
    assert trace["sample_locks"] is True
    assert len(trace["steps"]) >= 1
    locked = [step for step in trace["steps"] if step["host"]["status"] == "locked"]
    assert locked
    for step in locked:
        assert step["choice"]["selected"]
        assert "rot=" in step["choice"]["selected"]
        assert step["choice"]["sampled"] is True
        assert step["choice"]["argmax"]
        assert step["host"]["verified"] is True
        assert "probabilities" in step["choice"]
        assert step.get("next_piece") in {"I", "O", "T", "S", "Z", "J", "L"}
        assert len(step["candidates"]) >= 2
        kinds = [frame["kind"] for frame in step["frames"]]
        assert "fall" in kinds
        assert "lock" in kinds
        assert step["drop_ms"] >= 1
        assert any(frame.get("next_piece") for frame in step["frames"])
    assert "final_level" in trace
    assert trace["final_score"] > 0
    assert trace["final_drop_ms"] < 800
    assert "label_legend" in trace
    html = play.render_html(trace)
    assert "path_a_heuristic" in html
    assert '"status":"locked"' in html or '"status": "locked"' in html
    assert "realtime_gravity" in html
    assert "Gravity" in html or "fall" in html.lower()
    assert "Placement probabilities" in html
    assert "next-grid" in html
    assert "rot=" in html
    assert "sampled ≠ argmax" in html
    assert "sample-note" in html
    assert "sampled locks" in html or "sample_locks" in html


def test_sampling_can_differ_from_argmax(tetris_mods):
    _engine, _scorer, play = tetris_mods
    # Seeded sample often locks a non-argmax placement within a short run.
    trace = play.run_game(pieces=20, seed=0, temperature=4.0, sample=True)
    locked = [step for step in trace["steps"] if step["host"]["status"] == "locked"]
    assert locked
    differs = [
        step for step in locked
        if step["choice"]["selected"] != step["choice"]["argmax"]
    ]
    assert differs, "expected at least one sampled lock ≠ argmax"
    step = differs[0]
    assert step["choice"]["sampled"] is True
    assert step["choice"]["selected"] in step["choice"]["all_probabilities"]
    assert step["choice"]["argmax"] in step["choice"]["all_probabilities"]
    assert (
        step["choice"]["all_probabilities"][step["choice"]["argmax"]]
        >= step["choice"]["all_probabilities"][step["choice"]["selected"]]
    )


def test_greedy_locks_argmax(tetris_mods):
    _engine, _scorer, play = tetris_mods
    trace = play.run_game(pieces=5, seed=4, sample=False)
    assert trace["sample_locks"] is False
    for step in trace["steps"]:
        if step["host"]["status"] != "locked":
            continue
        assert step["choice"]["sampled"] is False
        assert step["choice"]["selected"] == step["choice"]["argmax"]


def test_default_runs_until_game_over_or_safety(tetris_mods):
    _engine, _scorer, play = tetris_mods
    trace = play.run_game(seed=1, safety_max_pieces=12)
    assert trace["pieces_requested"] is None
    assert trace["stop_reason"] in {"game_over", "safety_cap"}
    assert len(trace["steps"]) >= 1
    assert len(trace["steps"]) <= 12


def test_sampling_can_reach_game_over(tetris_mods):
    _engine, _scorer, play = tetris_mods
    # High temperature + sampling should eventually stack out within the cap.
    trace = play.run_game(seed=2, temperature=14.0, safety_max_pieces=80, sample=True)
    assert trace["stop_reason"] in {"game_over", "safety_cap"}
    assert len(trace["steps"]) >= 1
    if trace["game_over"]:
        assert trace["steps"][-1]["host"]["status"] == "blocked"
