"""Paced Tetris demo: enumerate → Choice → host verify → lock.

Path A only (heuristic softmax). Writes a self-contained browser page so
you can follow legal placement IDs, probabilities, and host verification.

```sh
uv pip install -e '.[dev]'
python examples/tetris/play.py --open
```

Text-only paced log: ``python examples/tetris/play.py --text``.
Optional neural Path B is not required; document any future MASK scorer as an
extra and keep this heuristic as the offline / CI fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Any

from engine import (
    SevenBag,
    board_to_rows,
    empty_board,
    enumerate_placements,
    lock_verified,
    verify_placement,
)
from scorer import score_placements, top_alternatives

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "visual.html"
DEFAULT_OUT = ROOT / "out" / "demo.html"


def _placement_payload(item) -> dict[str, Any]:
    return {
        "id": item.placement_id,
        "piece": item.piece,
        "rotation": item.rotation,
        "column": item.column,
        "landing_row": item.landing_row,
        "occupied_cells": [list(cell) for cell in item.occupied_cells],
        "lines_cleared": item.lines_cleared,
        "holes": item.holes,
        "aggregate_height": item.aggregate_height,
        "bumpiness": item.bumpiness,
        "heuristic_score": round(item.heuristic_score, 3),
    }


def run_game(
    *,
    pieces: int = 12,
    seed: int = 7,
    temperature: float = 4.0,
) -> dict[str, Any]:
    """Play ``pieces`` locks (or until game over) and return a visual trace."""
    board = empty_board()
    bag = SevenBag(seed=seed)
    steps: list[dict[str, Any]] = []
    score = 0
    lines_total = 0
    game_over = False

    for index in range(pieces):
        piece = bag.next_piece()
        before = board_to_rows(board)
        candidates = enumerate_placements(board, piece)
        if len(candidates) < 2:
            game_over = True
            steps.append(
                {
                    "index": index,
                    "piece": piece,
                    "board_before": before,
                    "candidates": [_placement_payload(item) for item in candidates],
                    "choice": None,
                    "host": {
                        "status": "blocked",
                        "reason": "fewer than two legal placements — game over",
                        "verified": False,
                    },
                    "board_after": before,
                    "score": score,
                    "lines_total": lines_total,
                }
            )
            break

        result = score_placements(candidates, temperature=temperature)
        selected = result.selected
        assert selected is not None  # threshold defaults to 0.0
        verified = verify_placement(board, piece, selected, candidates)
        if not verified.ok or verified.placement is None:
            steps.append(
                {
                    "index": index,
                    "piece": piece,
                    "board_before": before,
                    "candidates": [_placement_payload(item) for item in candidates],
                    "choice": {
                        "selected": selected,
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
                    },
                    "host": {
                        "status": "rejected",
                        "reason": verified.reason,
                        "verified": False,
                    },
                    "board_after": before,
                    "score": score,
                    "lines_total": lines_total,
                }
            )
            game_over = True
            break

        board, cleared = lock_verified(board, piece, verified.placement)
        lines_total += cleared
        score += (0, 40, 100, 300, 1200)[cleared] if cleared <= 4 else 1200
        steps.append(
            {
                "index": index,
                "piece": piece,
                "board_before": before,
                "candidates": [_placement_payload(item) for item in candidates],
                "choice": {
                    "selected": selected,
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
                },
                "host": {
                    "status": "locked",
                    "reason": verified.reason,
                    "verified": True,
                    "placement": _placement_payload(verified.placement),
                    "lines_cleared": cleared,
                },
                "board_after": board_to_rows(board),
                "score": score,
                "lines_total": lines_total,
            }
        )

    return {
        "title": "Akasha Model — Tetris Choice demo",
        "path": "A",
        "scorer": "path_a_heuristic",
        "note": (
            "Engine enumerates legal placements → Choice over IDs → host "
            "verifies cells before lock. Path A heuristic softmax; no torch."
        ),
        "seed": seed,
        "pieces_requested": pieces,
        "game_over": game_over,
        "final_score": score,
        "final_lines": lines_total,
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
    print(f"{trace['title']} · path {trace['path']} · {trace['scorer']}")
    print(trace["note"])
    print()
    for step in trace["steps"]:
        choice = step.get("choice") or {}
        host = step["host"]
        print(f"--- piece {step['index'] + 1}: {step['piece']} ---")
        print(f"legal placements: {len(step['candidates'])}")
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
        f"game_over={trace['game_over']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pieces",
        type=int,
        default=12,
        help="Number of locks to attempt (default: 12)",
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
            f"path {trace['path']}"
        )
        print("Open the HTML to follow board → legal IDs → probs → host lock.")
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
