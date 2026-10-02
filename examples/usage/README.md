# First-run usage demo

Path A walkthrough for a new user: **tool gate** (authorize / abstain /
block) plus a **ticket-router** Choice menu with an abstain threshold. CPU
only — no torch, no Hub download, no side effects.

## Visual (recommended)

Builds a self-contained browser page: state, typed Choice / Score / Noul
signals, probability bars, confidence, and ready / abstain / blocked
rationale.

```sh
uv pip install -e '.[dev]'
python examples/usage/visual_demo.py --open
```

Writes `examples/usage/out/demo.html` (gitignored). Open that file if you
skip `--open`.

## Text-only

```sh
python examples/usage/demo.py
```

| Part | What you see |
|------|----------------|
| Tool gate | `ready` safe read, `blocked` delete without confirm, `abstain` ambiguous mail |
| Ticket router | High-confidence refund route vs low-confidence escalate |

Deeper demos:

- [examples/gate](../gate/README.md) — full gate / host / outcomes / Path B tiny scorer
- [examples/film](../film/) — Doom/chess frame + matrix film from `play.py --trace`
- [docs/using-the-tool-gate.md](../../docs/using-the-tool-gate.md) — integrator guide
- [docs/use-cases.md](../../docs/use-cases.md) — train/predict ticket triage and other menus

These scripts use **scripted** Path A probabilities so first-run stays
torch-free. Training a real triage scorer is documented in use-cases §1
(`akasha-data` / `akasha-train` / `akasha-predict` with the `[torch]` extra).
