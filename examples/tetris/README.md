# Tetris Choice demo (Path A)

A small paced Tetris example of the typed-choice loop:

1. **Engine** enumerates legal final placements for the active piece.
2. **Choice** picks among those placement IDs (with probabilities).
3. **Host** re-checks the ID and exact cells **before** locking.

No torch / Hub. Scoring is a deterministic board heuristic turned into a
softmax — good enough to play a short paced game offline. A future MASK /
Path B scorer can replace `scorer.score_placements` over the same IDs; keep
the heuristic as the CI and first-run fallback.

External comparison (not used in package or CLI names): the same enumerate →
Choice → verify pattern as the public TypeSafe Tetris playground.

## Run (visual, recommended)

```sh
uv pip install -e '.[dev]'
python examples/tetris/play.py --open
```

Writes `examples/tetris/out/demo.html`. Use **Play** / **Next lock** to follow
board → legal IDs → probability bars → host verified lock.

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
| `engine.py` | Board, 7-bag, placement enumeration, host verify + lock |
| `scorer.py` | Path A heuristic → `ChoiceQuestion` / `choice_result` |
| `play.py` | Paced game → HTML / text / JSON trace |
| `visual.html` | Browser UI for the decision loop |
| `test_smoke.py` | Physics + Choice + HTML smoke |

Coordinates: columns `0..9` left→right, rows `0..19` top→bottom. Placement
`column` is the leftmost occupied cell; `occupied_cells` are the exact four
cells after landing.
