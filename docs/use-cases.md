# Concrete use cases

Akasha Model scores a **bounded menu** in one pass: a context plus typed
questions (`choice`, `score`, `noul`). It does **not** write free-form answers
and it does **not** execute tools. Your application owns side effects.

The product wedge for Akasha OS is the **tool gate**. The scenarios below show
other menus the same scorer shape can drive, with copy-paste entry points that
already exist in this repository.

| Use case | Typical questions | Start here |
|----------|-------------------|------------|
| Tool authorization gate | choice + score + noul → `ready`/`abstain`/`blocked` | [using-the-tool-gate.md](using-the-tool-gate.md) |
| Support / ticket triage | one `choice` (+ abstain) | [§ Support triage](#1-support--ticket-triage) |
| Agent / OS action routing | one `choice` over OS actions | [§ OS action menu](#2-agent--os-action-menu) |
| Risk + auth checklist | `choice` + `score` + several `noul` | [§ Multi-signal checklist](#3-multi-signal-checklist) |
| High-stakes abstention | ensemble of choice scorers | [§ Ensemble abstention](#4-high-stakes-ensemble-abstention) |
| Next-click navigation | one `choice` over link titles | [§ Wikispeedia](#5-next-click-navigation-wikispeedia) |
| Screen controller (lab) | vision softmax over fixed buttons | [§ Vision lab](#6-screen-controller-lab) |

---

## 1. Support / ticket triage

**Situation.** A ticket or chat message arrives. You want a cheap first pass
that picks a queue (`refund`, `sales`, `technical support`) or abstains when
confidence is low so a human (or a larger model) takes over.

**Why this fits.** Options change per ticket; you need probabilities and an
explicit abstain threshold, not generated prose.

Train on single-choice JSONL, then score one new menu:

```sh
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'

akasha-data synthetic --output data/synthetic
akasha-train data/synthetic/train.jsonl \
  --validation data/synthetic/validation.jsonl \
  --output runs/synthetic.pt

akasha-predict runs/synthetic.pt \
  --context "The customer needs a refund. Order #4412 charged twice." \
  --option refund \
  --option sales \
  --option "technical support" \
  --option-description "Billing correction or money back." \
  --option-description "Pricing, upgrade, or purchase help." \
  --option-description "Product bug or how-to." \
  --min-confidence 0.75
```

JSONL shape (one label index per row):

```json
{"context":"The customer needs a refund.","options":["refund","sales","technical support"],"label":0}
```

**Host policy.** If `abstained` is true, escalate; otherwise route on
`selected`. Re-check business rules outside the model.

---

## 2. Agent / OS action menu

**Situation.** An agent runtime (Akasha OS or similar) has a structured state
(session, resource, trust, capabilities) and a catalogue of OS-level actions.
You want a one-pass ranker over that menu before any planner commits.

**Why this fits.** The option list is bounded and typed; labels come from a
routing policy, not from free text generation.

Generate the synthetic Akasha-OS-style benchmark and train a choice scorer:

```sh
akasha-data akasha-os --output data/akasha_os
akasha-train data/akasha_os/train.jsonl \
  --validation data/akasha_os/validation.jsonl \
  --output runs/akasha-os.pt
akasha-eval runs/akasha-os.pt data/akasha_os/test.jsonl
```

Splits are grouped by scenario family by default so near-duplicate contexts do
not leak into the test set. For a finer split:

```sh
akasha-data akasha-os --output data/akasha_os_context --split-by context
```

**Honest limit.** This dataset is a local synthetic benchmark inspired by
Akasha OS identifiers, not production telemetry. Treat test metrics as a
routing probe; put the [tool gate](using-the-tool-gate.md) in front of any
real side effect.

---

## 3. Multi-signal checklist

**Situation.** One request needs several answers at once: which route, how
risky, and yes/no checks (authorized? enough context? capability present?).

**Why this fits.** `choice`, `score`, and `noul` share one encoder forward on
the multitask / MASK paths. Your app combines the signals; the model does not
call tools.

Example row (abbreviated): one Choice, one Score, several Noul:

```json
{"context":{"request":"access a protected device","offline":true},"questions":[{"id":"route","type":"choice","instructions":"Choose the route.","options":[{"name":"deny","description":"Block the operation."},{"name":"allow","description":"Permit the operation."}],"label":0},{"id":"risk","type":"score","instructions":"Rate the risk.","levels":["low","high"],"label":1},{"id":"authorized","type":"noul","instructions":"Is it authorized?","criteria":{"true":"The user explicitly granted the capability.","false":"No explicit grant is present."},"label":0}]}
```

Train and evaluate the byte multitask scorer:

```sh
akasha-multitask-train data/akasha_os_multi/train.jsonl \
  --validation data/akasha_os_multi/validation.jsonl \
  --output runs/akasha-os-multitask.pt
akasha-multitask-eval runs/akasha-os-multitask.pt \
  data/akasha_os_multi/test.jsonl \
  --output reports/akasha_os_multi_metrics.json
```

Or train the MASK encoder with RLCD and score with `akasha-typed-eval` (do not
load a MASK checkpoint with `akasha-multitask-eval`).

**How this relates to the gate.** The authorize-tool-call path uses the same
typed heads, then `evaluate_gate` / `plan_scored_proposal` to turn them into
`ready` / `abstain` / `blocked`. For a checklist that is *not* a tool call,
consume the head outputs in your own policy code.

Python contracts for building questions in application code:

```python
from akasha_model import ChoiceQuestion, NoulQuestion, OptionSpec, ScoreLevel, ScoreQuestion

choice = ChoiceQuestion(
    "route", "Choose the safest route.",
    (OptionSpec("deny", "Block the operation."),
     OptionSpec("allow", "Permit the operation.")),
)
score = ScoreQuestion(
    "risk", "Rate the risk.",
    (ScoreLevel("low"), ScoreLevel("high")),
)
noul = NoulQuestion("authorized", "Is the operation authorized?")
```

---

## 4. High-stakes ensemble abstention

**Situation.** Wrong answers are costly (payments, irreversible deletes,
compliance). One checkpoint’s confidence is not enough; you want agreement
across seeds before acting.

**Why this fits.** `akasha-ensemble-predict` averages probabilities and can
abstain on disagreement or low mean confidence.

```sh
akasha-ensemble-predict \
  runs/akasha_os_ensemble/seed-11.pt \
  runs/akasha_os_ensemble/seed-23.pt \
  runs/akasha_os_ensemble/seed-37.pt \
  --context "User asks to fetch a public webpage." \
  --option tool.request \
  --option skill.invoke \
  --option capability.check
```

Evaluate agreement on a stress or held-out set with `akasha-ensemble-eval`.
`unanimous_accuracy` and `unanimous_coverage` describe the trade-off: fewer
decisions, higher reliability among the ones that remain.

**Host policy.** Abstain → ask the user or regenerate the proposal. Agree →
still run your permission checks (and preferably the [tool gate](using-the-tool-gate.md)
if the next step is an executable tool).

---

## 5. Next-click navigation (Wikispeedia)

**Situation.** Given a Wikipedia-style page and a list of outgoing link
titles, predict the human’s next click.

**Why this fits.** Classic variable-length choice menu; one pass over the
current candidates.

```sh
scripts/get_wikispeedia.sh
akasha-train data/wikispeedia/jsonl/train.jsonl \
  --validation data/wikispeedia/jsonl/validation.jsonl \
  --output runs/wikispeedia.pt
```

Data stay outside git. Cite West & Leskovec, *Human Wayfinding in Information
Networks*, WWW 2012, and review the [SNAP terms](https://snap.stanford.edu/data/wikispeedia.html).

Numbers in the root README under “What to expect” are historical local
experiments, not a guarantee for your quickstart run.

---

## 6. Screen controller (lab)

**Situation.** Score fixed controller buttons from a 160×120 frame (plus
motion). Doom uses option rows 0–6; chess uses 7–11 on the same 12-row table.

**Why this fits.** Same one-read-per-option head as text, over image patches.
This is a **research / film demo**, not a product claim of game competence.

```sh
uv pip install -e '.[games]'
python examples/doom/play.py examples/checkpoints/joint-imitation.pt \
  --episodes 1 --game-seconds 10 --device cpu
```

Details: [examples/doom](../examples/doom/README.md),
[examples/chess](../examples/chess/README.md). After useful checkpoints, run
`examples/doom/audit.py` — reward above random is not visual control if
shuffled-patch KL stays near zero.

---

## Choosing a path

| You need… | Prefer… |
|-----------|---------|
| Authorize a **proposed** tool call | [Tool gate guide](using-the-tool-gate.md) + `examples/gate` |
| One menu, text only, fast CPU starter | `akasha-train` / `akasha-predict` (byte choice) |
| Several typed questions on one state | `akasha-multitask-train` or `akasha-rlcd-train` |
| Public typed-decisions comparison | `akasha-typed-eval --prepare` then RLCD on Hub splits |
| Screen buttons | `akasha_model.vision` + Doom/chess examples |

Still start with the [root README quickstart](../README.md#quickstart) once, so
the install and CLI paths match the release test.
