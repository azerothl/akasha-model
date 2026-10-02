# First-run usage demo

Smallest Path A walkthrough for a new user: **tool gate** (authorize /
abstain / block) plus a **ticket-router** Choice menu with an abstain
threshold. CPU only — no torch, no Hub download, no side effects.

```sh
uv pip install -e '.[dev]'
python examples/usage/demo.py
```

| Part | What you see |
|------|----------------|
| Tool gate | `ready` safe read, `blocked` delete without confirm, `abstain` ambiguous mail |
| Ticket router | High-confidence refund route vs low-confidence escalate |

Deeper demos:

- [examples/gate](../gate/README.md) — full gate / host / outcomes / Path B tiny scorer
- [docs/using-the-tool-gate.md](../../docs/using-the-tool-gate.md) — integrator guide
- [docs/use-cases.md](../../docs/use-cases.md) — train/predict ticket triage and other menus

This script uses **scripted** probabilities for the ticket menu so Path A stays
torch-free. Training a real triage scorer is documented in use-cases §1
(`akasha-data` / `akasha-train` / `akasha-predict` with the `[torch]` extra).
