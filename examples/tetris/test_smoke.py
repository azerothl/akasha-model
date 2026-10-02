"""Smoke tests for the Tetris Choice demo (Path A, no torch)."""

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
    result = scorer.score_placements(placements)
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
    bad = engine.verify_placement(board, "O", "p999", placements)
    assert not bad.ok
    assert "unknown" in bad.reason


def test_choice_question_matches_ids(tetris_mods):
    engine, scorer, _play = tetris_mods
    board = engine.empty_board()
    placements = engine.enumerate_placements(board, "I")
    question = scorer.build_choice_question(placements)
    assert isinstance(question, ChoiceQuestion)
    assert {opt.name for opt in question.options} == {
        item.placement_id for item in placements
    }


def test_play_trace_and_html(tetris_mods):
    _engine, _scorer, play = tetris_mods
    trace = play.run_game(pieces=6, seed=3)
    assert trace["path"] == "A"
    assert trace["scorer"] == "path_a_heuristic"
    assert len(trace["steps"]) >= 1
    locked = [step for step in trace["steps"] if step["host"]["status"] == "locked"]
    assert locked
    for step in locked:
        assert step["choice"]["selected"]
        assert step["host"]["verified"] is True
        assert "probabilities" in step["choice"]
        assert len(step["candidates"]) >= 2
    html = play.render_html(trace)
    assert "path_a_heuristic" in html
    assert '"status":"locked"' in html or '"status": "locked"' in html
    assert "legal placements" in html.lower() or "placement" in html.lower()
