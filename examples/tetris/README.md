# Tetris realtime demo (Path A)

A playable-feeling Tetris example: **random pieces fall from the top** with
visible gravity, tempo accelerates as the level rises, and Akasha **Choice**
still picks among engine-enumerated legal locks (probability bars on the side).

1. **Engine** enumerates legal final placements for the active piece.
2. **Choice** picks among those placements (human-readable labels + probs),
   using the **next-piece preview** in a one-ply look-ahead.
3. **Animation** rotates / shifts the piece, then drops it row by row.
4. **Host** re-checks the ID and exact cells **before** locking.

The game runs **until game over** (or you press **Stop** in the HTML replay).
Locks are **sampled** from the Choice probabilities (seeded) so imperfect play
can stack out; `--greedy` forces argmax. `--pieces N` remains for short CI runs.

No torch / Hub. Scoring is a deterministic board heuristic turned into a
softmax — good enough for an offline game. A future MASK / Path B scorer
can replace `scorer.score_placements` over the same IDs; keep the heuristic as
the CI and first-run fallback.

## Run (visual, recommended)

```sh
uv pip install -e '.[dev]'
python examples/tetris/play.py --open
```

Writes `examples/tetris/out/demo.html`. **Play** auto-starts: watch pieces
spawn, next-box update, Choice bars update, then gravity pull the piece down.
Drop interval shrinks with level (lines cleared). Press **Stop** anytime.

Placement bar labels look like `T rot=R col=3` (piece, rotation `0/R/2/L`,
leftmost column) — not opaque `p0` / `p3` ids.

## Text-only

```sh
python examples/tetris/play.py --text --no-html
```

## Smoke

```sh
PYTHONPATH=examples/tetris pytest -q examples/tetris/test_smoke.py
```

## Files

| File | Role |
|------|------|
| `engine.py` | Board, 7-bag (+ peek), gravity, placement enum, host verify |
| `scorer.py` | Path A heuristic + next-piece look-ahead → Choice |
| `play.py` | Game until over + falling frames → HTML / text / JSON |
| `visual.html` | Browser UI: fall + next preview + probability bars |
| `test_smoke.py` | Physics + Choice + gravity + HTML smoke |

Coordinates: columns `0..9` left→right, rows `0..19` top→bottom. Gravity
starts at 800 ms/row and floors at 55 ms/row as **score** and level rise
(soft-drop points + line clears; `level = lines // 10`).
