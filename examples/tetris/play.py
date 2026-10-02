"""Realtime Tetris demo: falling pieces + Choice over legal locks.

Pieces spawn at the top, Choice picks a verified placement (using the next
piece preview in the heuristic), then the piece rotates/shifts and falls with
visible gravity. The game runs until game over (or an optional piece cap for
CI). Drop speed accelerates as lines / level rise.

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
import random
import secrets
import sys
import webbrowser
from pathlib import Path
from typing import Any

from engine import (
    ActivePiece,
    Placement,
    SPAWN_COL,
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
from scorer import sample_placement, score_placements, top_alternatives

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "visual.html"
DEFAULT_OUT = ROOT / "out" / "demo.html"

# Soft ceiling so a runaway heuristic game cannot write an unbounded HTML file.
# Default play has no artificial short cap: stop on game over (or this safety).
DEFAULT_SAFETY_MAX_PIECES = 120
SEED_MASK = 2**31


def resolve_seed(seed: int | None) -> tuple[int, str]:
    """Return ``(seed, source)``. ``None`` draws a fresh seed each call."""
    if seed is None:
        return secrets.randbelow(SEED_MASK), "random"
    if seed < 0:
        raise ValueError("seed must be >= 0")
    return int(seed), "fixed"


def _placement_payload(item: Placement) -> dict[str, Any]:
    return {
        "id": item.placement_id,
        "label": item.placement_id,
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


def _choice_payload(result, *, locked: str | None = None, sampled: bool = False) -> dict[str, Any]:
    selected = locked if locked is not None else result.selected
    return {
        "selected": selected,
        "argmax": result.selected,
        "sampled": sampled,
        "confidence": round(float(result.probabilities.get(selected, result.confidence)), 4),
        "probabilities": {
            key: round(value, 4)
            for key, value in top_alternatives(result, limit=10)
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
    next_piece: str | None = None,
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
        "next_piece": next_piece,
        "active": _active_payload(active),
        "ghost": ghost or [],
    }
    if hold_ms is not None:
        payload["hold_ms"] = hold_ms
    return payload


def run_game(
    *,
    pieces: int | None = None,
    seed: int | None = None,
    temperature: float = 4.0,
    safety_max_pieces: int = DEFAULT_SAFETY_MAX_PIECES,
    sample: bool = True,
) -> dict[str, Any]:
    """Play until game over (or ``pieces`` / safety cap).

    ``pieces=None`` means no intentional short demo cap — keep going until the
    spawn is blocked / fewer than two legal locks, the optional ``pieces``
    argument (CI), or ``safety_max_pieces``.

    ``seed=None`` (the default) draws a fresh 7-bag + sample seed so each run
    — and each ``play.py`` invocation / page regenerate — is a new sequence.
    Pass an integer to replay. Locks are **sampled** from the Choice mass
    unless ``sample=False`` (argmax / ``--greedy``).
    """
    seed, seed_source = resolve_seed(seed)
    board = empty_board()
    bag = SevenBag(seed=seed)
    rng = random.Random(seed ^ 0xA5A5)
    steps: list[dict[str, Any]] = []
    score = 0
    lines_total = 0
    game_over = False
    hit_cap = False

    if pieces is not None and pieces < 1:
        raise ValueError("pieces must be >= 1 when set")
    limit = pieces if pieces is not None else safety_max_pieces

    index = 0
    while index < limit:
        level = level_for_lines(lines_total)
        drop_ms = drop_interval_ms(level, score)
        piece = bag.next_piece()
        next_piece = bag.peek()
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
                    next_piece=next_piece,
                    hold_ms=900,
                )
            ]
            steps.append(
                {
                    "index": index,
                    "piece": piece,
                    "next_piece": next_piece,
                    "board_before": before,
                    "candidates": [_placement_payload(item) for item in candidates],
                    "choice": None,
                    "host": {
                        "status": "blocked",
                        "reason": (
                            "spawn blocked — game over"
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

        result = score_placements(
            candidates,
            temperature=temperature,
            board=board,
            next_piece=next_piece,
        )
        if sample:
            selected = sample_placement(result, rng)
        else:
            selected = result.selected
        assert selected is not None
        verified = verify_placement(board, piece, selected, candidates)
        choice = _choice_payload(result, locked=selected, sampled=sample)

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
                    next_piece=next_piece,
                    active=spawned,
                    ghost=[list(cell) for cell in ghost_cells(board, spawned)],
                    hold_ms=700,
                )
            ]
            steps.append(
                {
                    "index": index,
                    "piece": piece,
                    "next_piece": next_piece,
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
                next_piece=next_piece,
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
                    next_piece=next_piece,
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
                next_piece=next_piece,
                hold_ms=max(160, drop_after // 2),
            )
        )

        steps.append(
            {
                "index": index,
                "piece": piece,
                "next_piece": next_piece,
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
        index += 1
    else:
        # Loop exhausted without game over → intentional or safety cap.
        hit_cap = True
        if pieces is None:
            # Safety ceiling — treat as stopped rather than a true game over.
            hit_cap = True

    final_level = level_for_lines(lines_total)
    stop_reason = (
        "game_over"
        if game_over
        else ("piece_cap" if pieces is not None and hit_cap else "safety_cap" if hit_cap else "complete")
    )
    return {
        "title": "Akasha Model — Tetris",
        "path": "A",
        "scorer": "path_a_heuristic",
        "mode": "realtime_gravity",
        "note": (
            "Piece types come from a 7-bag (this run's seed). Spawn column is "
            "standard Tetris (centred, not random X). Choice samples a legal "
            "landing from the softmax unless greedy/argmax. Gravity is visible; "
            "the next-piece preview feeds a one-ply look-ahead. Host verifies "
            "cells, then the piece rotates, shifts, and drops. Tempo accelerates "
            "as score (and level) rise. Path A heuristic — no torch. "
            "This HTML is one recorded run; re-run play.py for a new bag."
        ),
        "seed": seed,
        "seed_source": seed_source,
        "spawn_col": SPAWN_COL,
        "spawn_x_random": False,
        "pieces_requested": pieces,
        "safety_max_pieces": safety_max_pieces,
        "sample_locks": sample,
        "stop_reason": stop_reason,
        "game_over": game_over,
        "final_score": score,
        "final_lines": lines_total,
        "final_level": final_level,
        "final_drop_ms": drop_interval_ms(final_level, score),
        "label_legend": (
            "Each bar label is a legal lock: "
            "'T rot=R col=3' means piece T, rotation R (0/R/2/L), leftmost column 3."
        ),
        "rng_hud": (
            f"Type: 7-bag ({seed_source} seed {seed}) · "
            f"spawn: col {SPAWN_COL} (standard, not random X) · "
            f"lock: {'softmax sample' if sample else 'greedy argmax'}"
        ),
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
    print(trace.get("rng_hud", ""))
    print(trace.get("label_legend", ""))
    print()
    for step in trace["steps"]:
        choice = step.get("choice") or {}
        host = step["host"]
        nxt = step.get("next_piece") or "—"
        print(
            f"--- piece {step['index'] + 1}: {step['piece']} "
            f"(next {nxt}) · level {step['level']} · drop {step['drop_ms']}ms ---"
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
        f"game_over={trace['game_over']} stop={trace.get('stop_reason')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pieces",
        type=int,
        default=None,
        help=(
            "Optional lock cap (CI / short runs). Default: play until game over "
            f"(safety ceiling {DEFAULT_SAFETY_MAX_PIECES})."
        ),
    )
    parser.add_argument(
        "--safety-max-pieces",
        type=int,
        default=DEFAULT_SAFETY_MAX_PIECES,
        help=(
            "Hard ceiling when --pieces is omitted "
            f"(default: {DEFAULT_SAFETY_MAX_PIECES})."
        ),
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Always lock the Choice argmax instead of sampling (may never game-over)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="7-bag + lock-sample seed. Default: a new random seed every run.",
    )
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

    if args.pieces is not None and args.pieces < 1:
        raise SystemExit("--pieces must be >= 1")
    if args.safety_max_pieces < 1:
        raise SystemExit("--safety-max-pieces must be >= 1")
    if args.seed is not None and args.seed < 0:
        raise SystemExit("--seed must be >= 0")

    trace = run_game(
        pieces=args.pieces,
        seed=args.seed,
        temperature=args.temperature,
        safety_max_pieces=args.safety_max_pieces,
        sample=not args.greedy,
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
            f"stop {trace.get('stop_reason')} · path {trace['path']}"
        )
        seed_src = trace.get("seed_source", "fixed")
        replay = (
            f"; replay with --seed {trace['seed']}"
            if seed_src == "random"
            else ""
        )
        print(f"Seed {trace['seed']} ({seed_src}{replay}).")
        print(
            "Open the HTML: falling blocks + next preview + Choice probs. "
            "Stop the replay anytime."
        )
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
