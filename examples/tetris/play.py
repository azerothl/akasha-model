"""Realtime Tetris demo: falling pieces + Choice over legal locks.

Pieces spawn at the top, Choice picks a verified placement, then the piece
rotates/shifts and falls with visible gravity. Drop speed accelerates as
lines / level rise.

Path A only (heuristic softmax). Optional Path B is not required; keep this
heuristic as the offline / CI fallback.

```sh
uv pip install -e '.[dev]'
python examples/tetris/play.py --open
```

Text-only: ``python examples/tetris/play.py --text --no-html``.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any

from engine import (
    ActivePiece,
    Placement,
    SevenBag,
    board_to_rows,
    drop_interval_ms,
    empty_board,
    enumerate_placements,
    ghost_cells,
    level_for_lines,
    lock_verified,
    plan_approach,
    score_for_clear,
    score_for_soft_drop,
    spawn_piece,
    verify_placement,
)
from scorer import score_placements, top_alternatives

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "visual.html"
DEFAULT_OUT = ROOT / "out" / "demo.html"


def _placement_payload(item: Placement) -> dict[str, Any]:
    return {
        "id": item.placement_id,
        "piece": item.piece,
        "rotation": item.rotation,
        "rotation_index": item.rotation_index,
        "column": item.column,
        "landing_row": item.landing_row,
        "origin_col": item.origin_col,
        "origin_row": item.origin_row,
        "occupied_cells": [list(cell) for cell in item.occupied_cells],
        "lines_cleared": item.lines_cleared,
        "holes": item.holes,
        "aggregate_height": item.aggregate_height,
        "bumpiness": item.bumpiness,
        "heuristic_score": round(item.heuristic_score, 3),
    }


def _active_payload(active: ActivePiece | None) -> dict[str, Any] | None:
    if active is None:
        return None
    return {
        "piece": active.piece,
        "rotation": active.rotation,
        "rotation_index": active.rotation_index,
        "origin_col": active.origin_col,
        "origin_row": active.origin_row,
        "cells": [list(cell) for cell in active.cells()],
    }


def _choice_payload(result) -> dict[str, Any]:
    return {
        "selected": result.selected,
        "confidence": round(result.confidence, 4),
        "probabilities": {
            key: round(value, 4)
            for key, value in top_alternatives(result, limit=8)
        },
        "all_probabilities": {
            key: round(value, 4)
            for key, value in result.probabilities.items()
        },
        "scorer": "path_a_heuristic",
    }


def _frame(
    *,
    kind: str,
    board_rows: list[str],
    score: int,
    lines_total: int,
    level: int,
    drop_ms: int,
    piece: str | None = None,
    active: ActivePiece | None = None,
    ghost: list[list[int]] | None = None,
    hold_ms: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "board": board_rows,
        "score": score,
        "lines_total": lines_total,
        "level": level,
        "drop_ms": drop_ms,
        "piece": piece,
        "active": _active_payload(active),
        "ghost": ghost or [],
    }
    if hold_ms is not None:
        payload["hold_ms"] = hold_ms
    return payload


def run_game(
    *,
    pieces: int = 18,
    seed: int = 7,
    temperature: float = 4.0,
) -> dict[str, Any]:
    """Play ``pieces`` locks (or until game over) with falling-frame traces."""
    board = empty_board()
    bag = SevenBag(seed=seed)
    steps: list[dict[str, Any]] = []
    score = 0
    lines_total = 0
    game_over = False

    for index in range(pieces):
        level = level_for_lines(lines_total)
        drop_ms = drop_interval_ms(level, score)
        piece = bag.next_piece()
        before = board_to_rows(board)
        spawned = spawn_piece(board, piece)
        candidates = enumerate_placements(board, piece)

        if spawned is None or len(candidates) < 2:
            game_over = True
            frames = [
                _frame(
                    kind="game_over",
                    board_rows=before,
                    score=score,
                    lines_total=lines_total,
                    level=level,
                    drop_ms=drop_ms,
                    piece=piece,
                    hold_ms=900,
                )
            ]
            steps.append(
                {
                    "index": index,
                    "piece": piece,
                    "board_before": before,
                    "candidates": [_placement_payload(item) for item in candidates],
                    "choice": None,
                    "host": {
                        "status": "blocked",
                        "reason": (
                            "spawn blocked"
                            if spawned is None
                            else "fewer than two legal placements — game over"
                        ),
                        "verified": False,
                    },
                    "board_after": before,
                    "score": score,
                    "lines_total": lines_total,
                    "level": level,
                    "drop_ms": drop_ms,
                    "frames": frames,
                }
            )
            break

        result = score_placements(candidates, temperature=temperature)
        selected = result.selected
        assert selected is not None
        verified = verify_placement(board, piece, selected, candidates)
        choice = _choice_payload(result)

        if not verified.ok or verified.placement is None:
            frames = [
                _frame(
                    kind="choice",
                    board_rows=before,
                    score=score,
                    lines_total=lines_total,
                    level=level,
                    drop_ms=drop_ms,
                    piece=piece,
                    active=spawned,
                    ghost=[list(cell) for cell in ghost_cells(board, spawned)],
                    hold_ms=700,
                )
            ]
            steps.append(
                {
                    "index": index,
                    "piece": piece,
                    "board_before": before,
                    "candidates": [_placement_payload(item) for item in candidates],
                    "choice": choice,
                    "host": {
                        "status": "rejected",
                        "reason": verified.reason,
                        "verified": False,
                    },
                    "board_after": before,
                    "score": score,
                    "lines_total": lines_total,
                    "level": level,
                    "drop_ms": drop_ms,
                    "frames": frames,
                }
            )
            game_over = True
            break

        target = verified.placement
        path = plan_approach(board, piece, target)
        frames: list[dict[str, Any]] = []

        # Soft-drop points for every gravity step (rows the piece falls).
        fall_rows = 0
        if path:
            fall_rows = max(0, path[-1].origin_row - path[0].origin_row)
        score += score_for_soft_drop(fall_rows)
        drop_ms = drop_interval_ms(level, score)

        # Brief pause so the Choice bars / target ghost are readable.
        frames.append(
            _frame(
                kind="choice",
                board_rows=before,
                score=score,
                lines_total=lines_total,
                level=level,
                drop_ms=drop_ms,
                piece=piece,
                active=spawned,
                ghost=[list(cell) for cell in target.occupied_cells],
                hold_ms=max(280, min(620, drop_ms)),
            )
        )

        prev: ActivePiece | None = None
        for pose in path:
            # Recompute tempo from live score so gravity visibly ramps mid-run.
            live_drop = drop_interval_ms(level, score)
            if prev is not None and pose.origin_row > prev.origin_row:
                kind = "fall"
                hold = live_drop
            elif prev is not None and pose.rotation_index != prev.rotation_index:
                kind = "rotate"
                hold = max(70, live_drop // 4)
            elif prev is not None and pose.origin_col != prev.origin_col:
                kind = "shift"
                hold = max(55, live_drop // 5)
            else:
                kind = "spawn"
                hold = max(90, live_drop // 3)
            frames.append(
                _frame(
                    kind=kind,
                    board_rows=before,
                    score=score,
                    lines_total=lines_total,
                    level=level,
                    drop_ms=live_drop,
                    piece=piece,
                    active=pose,
                    ghost=[list(cell) for cell in target.occupied_cells],
                    hold_ms=hold,
                )
            )
            prev = pose

        board, cleared = lock_verified(board, piece, target)
        lines_total += cleared
        level_after = level_for_lines(lines_total)
        score += score_for_clear(cleared, level)
        drop_after = drop_interval_ms(level_after, score)
        after = board_to_rows(board)

        frames.append(
            _frame(
                kind="lock",
                board_rows=after,
                score=score,
                lines_total=lines_total,
                level=level_after,
                drop_ms=drop_after,
                piece=piece,
                hold_ms=max(160, drop_after // 2),
            )
        )

        steps.append(
            {
                "index": index,
                "piece": piece,
                "board_before": before,
                "candidates": [_placement_payload(item) for item in candidates],
                "choice": choice,
                "host": {
                    "status": "locked",
                    "reason": verified.reason,
                    "verified": True,
                    "placement": _placement_payload(target),
                    "lines_cleared": cleared,
                },
                "board_after": after,
                "score": score,
                "lines_total": lines_total,
                "level": level_after,
                "drop_ms": drop_after,
                "frames": frames,
            }
        )

    final_level = level_for_lines(lines_total)
    return {
        "title": "Akasha Model — Tetris",
        "path": "A",
        "scorer": "path_a_heuristic",
        "mode": "realtime_gravity",
        "note": (
            "Random pieces fall from the top with visible gravity. "
            "Choice picks among engine-enumerated lock IDs (probs on the right); "
            "the host verifies cells, then the piece rotates, shifts, and drops. "
            "Tempo accelerates as score (and level) rise. Path A heuristic — no torch."
        ),
        "seed": seed,
        "pieces_requested": pieces,
        "game_over": game_over,
        "final_score": score,
        "final_lines": lines_total,
        "final_level": final_level,
        "final_drop_ms": drop_interval_ms(final_level, score),
        "steps": steps,
    }


def render_html(trace: dict[str, Any], template: Path = TEMPLATE) -> str:
    raw = template.read_text(encoding="utf-8")
    marker = "/* DATA */"
    if marker not in raw:
        raise SystemExit(f"template missing {marker!r}: {template}")
    payload = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    return raw.replace(marker, payload, 1)


def print_text(trace: dict[str, Any]) -> None:
    print(f"{trace['title']} · path {trace['path']} · {trace['scorer']} · {trace['mode']}")
    print(trace["note"])
    print()
    for step in trace["steps"]:
        choice = step.get("choice") or {}
        host = step["host"]
        print(
            f"--- piece {step['index'] + 1}: {step['piece']} · "
            f"level {step['level']} · drop {step['drop_ms']}ms ---"
        )
        print(f"legal placements: {len(step['candidates'])}")
        fall_frames = sum(1 for frame in step.get("frames", []) if frame.get("kind") == "fall")
        print(f"gravity frames: {fall_frames}")
        if choice:
            print(
                f"Choice selected={choice['selected']} "
                f"conf={choice['confidence']:.3f}"
            )
            top = ", ".join(
                f"{name}={prob:.3f}"
                for name, prob in choice["probabilities"].items()
            )
            print(f"top probs: {top}")
        print(f"host: {host['status']} — {host['reason']}")
        if host.get("verified") and host.get("placement"):
            place = host["placement"]
            print(
                f"lock {place['id']} rot={place['rotation']} "
                f"col={place['column']} row={place['landing_row']} "
                f"lines={host.get('lines_cleared', 0)}"
            )
        print(f"score={step['score']} lines={step['lines_total']}")
        print()
    print(
        f"done · steps={len(trace['steps'])} "
        f"score={trace['final_score']} lines={trace['final_lines']} "
        f"level={trace['final_level']} drop_ms={trace['final_drop_ms']} "
        f"game_over={trace['game_over']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pieces",
        type=int,
        default=18,
        help="Number of locks to attempt (default: 18)",
    )
    parser.add_argument("--seed", type=int, default=7, help="7-bag seed")
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Softmax temperature for the Path A heuristic (default: 4)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="HTML output path (default: examples/tetris/out/demo.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the written HTML in the default browser",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Print a paced text log instead of (or as well as) HTML",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Also write the structured game trace JSON",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip writing HTML (use with --text / --json)",
    )
    args = parser.parse_args(argv)

    if args.pieces < 1:
        raise SystemExit("--pieces must be >= 1")

    trace = run_game(
        pieces=args.pieces,
        seed=args.seed,
        temperature=args.temperature,
    )

    if args.text or args.no_html:
        print_text(trace)

    if not args.no_html:
        html = render_html(trace)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(html, encoding="utf-8")
        print(f"Wrote {args.out}")
        print(
            f"Steps: {len(trace['steps'])} · score {trace['final_score']} · "
            f"level {trace['final_level']} · drop {trace['final_drop_ms']}ms · "
            f"path {trace['path']}"
        )
        print("Open the HTML: falling blocks + accelerating tempo + Choice probs.")
        if args.open:
            webbrowser.open(args.out.resolve().as_uri())

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(trace, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
