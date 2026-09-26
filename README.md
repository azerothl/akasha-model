# Akasha Model

Train a small one-pass model that chooses among a changing list of typed options.

Akasha Model takes a piece of state (text or JSON) and a list of questions. Each question is `choice`, `score` or `noul`. It returns one probability distribution per question in a single forward pass, instead of writing an answer word by word.

The public comparison for this shape of model is TypeSafe [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). TypeSafe has not published its design. This repository is the decision scorer for Akasha OS: independent weights, independent training, and a deterministic tool-call planner that never executes side effects.

**Using the tool gate (authorize / abstain / block before your host runs a tool):** see [docs/using-the-tool-gate.md](docs/using-the-tool-gate.md) and [examples/gate](examples/gate/README.md). Host: `akasha_model.host` (`run_gated_call`); outcomes: `akasha_model.outcomes` (`python examples/gate/outcomes_demo.py`).

**Concrete use cases beyond the gate** (support triage, OS action menus, multi-signal checklists, ensemble abstention, Wikispeedia, vision lab): [docs/use-cases.md](docs/use-cases.md).

## Demo

The same option-attention head can score controller buttons from image patches. [This ten-second film](docs/jevre-demo-10s-bgm.mp4) joins two selected five-second windows: live `deadly_corridor` combat on the seven Doom buttons, then a chess controller walking to and playing moves with five keys. The diagram shows the tensors used for each decision. The Doom window came from the supplied joint checkpoint, which averaged 0.60 kills and -97.50 reward across its ten recorded episodes. The chess window came from the stronger chess-only checkpoint, which scored 4 wins, 46 draws and 0 losses in 50 sampled games against a random mover, but 0 wins, 2 draws and 48 losses against Stockfish level 0. The windows were selected for activity and are not typical-play or competence claims.

<video src="docs/jevre-demo-10s-bgm.mp4" controls width="960"></video>

Install the game extras and record a fresh 640 by 480 Doom trace from the released joint checkpoint:

```sh
uv pip install -e '.[games]'
python examples/doom/play.py examples/checkpoints/joint-imitation.pt --episodes 10 --game-seconds 35.3 --device cpu --capture-resolution 640x480 --output runs/doom.mp4 --trace runs/doom-trace.json
```

Render the trace in the same visual layout. This writes a silent film because the author-owned soundtrack source is not part of the repository.

```sh
(cd examples/film && npm install && npx playwright install chromium)
examples/film/make-film.sh runs/doom-trace.json runs/doom-film.mp4 10
```

The release includes the [Doom example](examples/doom/README.md), the [chess example](examples/chess/README.md), the single-game checkpoints and the shared 12-option checkpoint. Both games import the visual scorer from `akasha_model.vision`; there is no second model copy in either example.

## Architecture

The production text path encodes each question as one MASK-marker sequence:

```
[CLS] choice question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

A bidirectional encoder (tiny transformer, or a **trained** BERT such as `bert-base-uncased` / ModernBERT) reads the whole sequence. A shared scorer reads the hidden state at each `[MASK]` and returns one logit per option. `choice`, `score` and `noul` share that head; noul is the two-way pair `false` / `true`.

The encoder is trained, not frozen. Options are written before the state, so a full option block can truncate the context: keep `--max-len` larger than `--head-max-len` (768 for the tiny encoder; **512 for `bert-base-uncased`**, which cannot index longer sequences). High-cardinality menus also keep `--option-max-len`; raise `--head-max-len` when a question has dozens of options.

A cheaper byte-encoder scorer remains available for CPU smoke tests and the original JSONL choice format.

## Data format

Use one JSON object per line:

```json
{"context":"The customer needs a refund.","options":["refund","sales","technical support"],"label":0}
```

`label` is the zero-based index of the correct option. Each row may have a different number of options, with a minimum of two.

An option can also be an object:

```json
{"context":"The user asks for a webpage.","options":[{"name":"tool.request","description":"Use the network tool for an external fetch."},{"name":"skill.invoke","description":"Use an installed local workflow."}],"label":0}
```

The package exposes three typed decision contracts in `akasha_model.primitives`:

```python
from akasha_model import ChoiceQuestion, NoulQuestion, OptionSpec, ScoreLevel, ScoreQuestion

choice = ChoiceQuestion(
    "route", "Choose the safest route.",
    (OptionSpec("tool.request", "Use the network tool."),
     OptionSpec("skill.invoke", "Use a local skill.")),
)
score = ScoreQuestion(
    "risk", "Rate the risk.",
    (ScoreLevel("low"), ScoreLevel("high")),
)
noul = NoulQuestion("authorized", "Is the operation authorized?")
```

These contracts validate bounded options, ordered score levels, probability distributions and explicit abstention thresholds. Authorization, side effects and workflow decisions remain application code rather than model output.

### Tool calling: planning, not execution

The multitask model does not emit a JSON function call and has no tool
executor. `Choice` selects a tool from a bounded catalogue, `Score` estimates
a risk level, and `Noul` supplies binary signals such as authorization,
capability presence, or sufficient context. The application must still
validate arguments, permissions and human confirmation before any side
effect.

`akasha_model.tool_calling.ToolCallPlanner` is that deterministic boundary:
it returns a `ready`, `abstain` or `blocked` plan, validates a subset of the
parameter JSON schema, and never runs the tool. Akasha OS must supply its
own executor after a `ready` plan and repeat its final checks.

The host-facing helper is `evaluate_gate` (and `plan_scored_proposal` when a
multitask scorer fills Choice / Score / Noul). Default thresholds live in
`akasha_model.gate` (`DEFAULT_MIN_CHOICE_PROBABILITY=0.55`,
`DEFAULT_MIN_CHOICE_CONFIDENCE=0.50`, `DEFAULT_NOUL_THRESHOLD=0.70`,
`DEFAULT_MAX_RISK_SCORE=1.5`). See [examples/gate](examples/gate/README.md) for
the scripted demo, authorize-tool-call JSONL, tiny train loop, and go/no-go
checks on false positives.

```python
from akasha_model import ToolCallPlanner, ToolSpec, evaluate_gate, ToolProposal, GateSignals

spec = ToolSpec(
    "fs.read", "Read a file",
    parameters={"type": "object", "required": ["path"],
                "properties": {"path": {"type": "string"}},
                "additionalProperties": False},
    required_capability="workspace_access",
)
plan = evaluate_gate(
    {spec.name: spec, "fs.delete": ToolSpec("fs.delete")},
    ToolProposal("fs.read", {"path": "notes.txt"}),
    GateSignals(authorized=0.95, sufficient_context=0.9, capability_present=0.92),
)
if plan.status == "ready":
    akasha_executor.execute(plan.tool_name, plan.arguments)
```

### Multi-question training

The shared-context model accepts several atomic questions in one JSONL row.
The encoder is shared, while `Choice`, `Score` and `Noul` have independent
heads on the byte path. The MASK+BERT path scores every type at `[MASK]`
markers in one sequence.

```json
{"context":{"request":"access a protected device","offline":true},"questions":[{"id":"route","type":"choice","instructions":"Choose the route.","options":[{"name":"deny","description":"Block the operation."},{"name":"allow","description":"Permit the operation."}],"label":0},{"id":"risk","type":"score","instructions":"Rate the risk.","levels":["low","high"],"label":1},{"id":"authorized","type":"noul","instructions":"Is it authorized?","criteria":{"true":"The user explicitly granted the capability.","false":"No explicit grant is present."},"label":0}]}
```

Train the byte multitask scorer:

```sh
akasha-multitask-train data/akasha_os_multi/train.jsonl \
  --validation data/akasha_os_multi/validation.jsonl \
  --output runs/akasha-os-multitask.pt
```

Train the MASK encoder with RLCD (proper-scoring reward + GRPO group baseline) on the same leakage-audited splits. The tiny encoder is CPU-friendly; pass `--encoder hf --hf-model bert-base-uncased` to train BERT.

```sh
akasha-rlcd-train data/akasha_os_multi/train.jsonl \
  --validation data/akasha_os_multi/validation.jsonl \
  --encoder tiny --group-size 4 \
  --output runs/akasha-rlcd.pt
```

Akasha OS rows typically hold **one `choice`, one `score` and nine `noul`**. RLCD scores every question in the row, so noul can dominate the gradient. GRPO validation loss is not accuracy. Score a MASK checkpoint with `akasha-typed-eval` (not `akasha-multitask-eval`, which only loads the byte scorer):

```sh
akasha-typed-eval runs/akasha-rlcd.pt \
  --data data/akasha_os_multi/test.jsonl \
  --output reports/akasha_os_multi_rlcd.json
```

`label` is an option index for `Choice` and a level index for `Score`. Score
levels are ordered descriptions, numbered internally from 0; they are not
arbitrary numeric measurements. A `Noul` returns the probability that the
condition in its instructions is true, with values near 0.5 representing
uncertainty. Optional `target` arrays store soft gold distributions for RLCD.

Evaluate the **byte** multitask scorer on the held-out test and stress sets:

```powershell
akasha-multitask-eval `
  runs\akasha-os-multitask.pt `
  data\akasha_os_multi\test.jsonl `
  --stress data\akasha_os_multi\stress.jsonl `
  --output reports\akasha_os_multi_metrics.json
```

The report includes accuracy and calibration for `Choice`, ordinal error for
`Score`, AUROC/Brier/log loss for `Noul`, coverage-risk tables, and stress
results broken down by `stress_type`.

The multi-question defaults are sized for structured Akasha states:
`context_tokens=768`, `question_tokens=512` and `option_tokens=384`.
Lower values may silently truncate criteria and source-derived descriptions.
`Noul` uses `0` or `1`. Rows may contain any subset of the three question
types; absent heads are skipped for that batch.

## Quickstart

Run these commands from the repository root. They create local synthetic data, train on it, evaluate the saved model and score one new menu.

```sh
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'

akasha-data synthetic --output data/synthetic
akasha-train data/synthetic/train.jsonl \
  --validation data/synthetic/validation.jsonl \
  --output runs/synthetic.pt
akasha-eval runs/synthetic.pt data/synthetic/test.jsonl
akasha-predict runs/synthetic.pt \
  --context "Choose the exact badge amber badger. Badge: amber badger." \
  --option "azure crane" \
  --option "amber badger" \
  --option "gold heron"
```

At inference time, option descriptions can be supplied and the application can
request abstention below a confidence threshold:

```sh
akasha-predict runs/synthetic.pt \
  --context "The user asks to fetch a public webpage." \
  --option tool.request \
  --option skill.invoke \
  --option-description "Use the network tool for an external fetch." \
  --option-description "Use an installed local workflow." \
  --min-confidence 0.75
```

The evaluation prints top-1 accuracy, which is the fraction of correct first choices. Top-3 accuracy is the fraction with the right answer among the three highest scores. It also reports negative log-likelihood (NLL), Brier score and expected calibration error (ECE). ECE compares confidence with observed accuracy. The command also prints a shuffled-context control, which pairs each menu with the wrong context. A useful model should beat that control.

## Calibrate a trained model

Use a separate calibration split that was not used to train or select the model. Temperature scaling changes the sharpness of the probabilities without changing the ranking of the options:

```sh
akasha-calibrate runs/synthetic.pt data/synthetic/validation.jsonl \
  --output runs/synthetic-calibrated.pt
akasha-eval runs/synthetic-calibrated.pt data/synthetic/test.jsonl
```

The fitted temperature is stored in the checkpoint and applied automatically by `akasha-predict` and `akasha-eval`. Compare NLL, Brier score and ECE on the held-out test set before deciding whether calibration helped. Calibration improves the reliability of probabilities; it does not necessarily improve top-1 accuracy.

Training can also include a calibration-aware objective. The default weight is zero, which keeps the original training behavior. A small positive weight adds the multiclass Brier score to the cross-entropy objective:

```sh
akasha-train data/synthetic/train.jsonl \
  --validation data/synthetic/validation.jsonl \
  --output runs/synthetic-calibration-aware.pt \
  --calibration-weight 0.1
```

Tune this weight on a validation split. Do not assume that a lower calibration loss improves accuracy, and always compare the final held-out NLL, Brier score and ECE against the unmodified baseline.

## Leakage-resistant splits

When related rows share the same context, split by context rather than by row. The helper below combines JSONL files and keeps every identical context in exactly one split:

```sh
python scripts/split_by_context.py data/akasha_os data/akasha_os_grouped
```

Train and evaluate against `data/akasha_os_grouped`. Check that the reported context sets do not overlap before interpreting the test metrics.

To probe robustness without contaminating training, derive an evaluation-only stress set from the grouped test split:

```sh
python scripts/make_stress_eval.py \
  data/akasha_os_grouped/test.jsonl \
  data/akasha_os_grouped/stress.jsonl
akasha-eval runs/akasha_os_grouped/baseline.pt \
  data/akasha_os_grouped/stress.jsonl
```

The stress set contains ambiguous-context and irrelevant-noise variants. Its labels inherit the original test labels, so treat it as a robustness probe rather than a replacement for a human-labelled benchmark.

## Ensemble abstention

For uncertainty-sensitive decisions, train several checkpoints with different seeds and evaluate their agreement:

```sh
akasha-ensemble-eval \
  runs/akasha_os_ensemble/seed-11.pt \
  runs/akasha_os_ensemble/seed-23.pt \
  runs/akasha_os_ensemble/seed-37.pt \
  --data data/akasha_os_grouped/stress.jsonl
```

The ensemble averages probabilities and reports the fraction of examples on which every model agrees. A conservative application can abstain when the models disagree; `unanimous_accuracy` measures the accuracy of the remaining decisions and `unanimous_coverage` measures how often a decision is still returned.

Use the same policy at inference time:

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

The command abstains when the ensemble disagrees or when its mean confidence is below the configured threshold.

## Use your own data

1. Export train, validation and test JSONL files in the format above.
2. Keep all options that the model will see at prediction time in each row.
3. Split related records together. For example, keep all records for one customer or one target page in one split. This prevents near-duplicates from leaking into the test set.
4. Run `akasha-train` (single-choice JSONL), `akasha-multitask-train` (byte Choice/Score/Noul) or `akasha-rlcd-train` (MASK+RLCD) with your train and validation files.
5. Evaluate once on a held-out test file that was never used for training or model selection. Use `akasha-eval` for single-choice checkpoints, `akasha-multitask-eval` for the byte multitask scorer, and `akasha-typed-eval --data` for MASK/RLCD checkpoints.

The default byte encoder truncates context to 192 bytes and each option to 32 bytes. Raise `--context-tokens` or `--option-tokens` when your text needs more room. Training supports CPU, Apple MPS for a Mac GPU, and CUDA for an NVIDIA GPU through `--device`.

### Akasha-OS-style routing data

The repository can generate a deterministic, synthetic policy dataset inspired
by [azerothl/akasha-os](https://github.com/azerothl/akasha-os): the context
describes a task, surface, resource and trust state; the options are possible
OS-level actions; and `label` identifies the action selected by a small routing
policy. It covers sessions, memory, scheduled tasks, agents, skills, tools,
model packs and capability checks. It is a local synthetic benchmark, not a
dump of Akasha OS telemetry.

```sh
akasha-data akasha-os --output data/akasha_os
akasha-train data/akasha_os/train.jsonl \
  --validation data/akasha_os/validation.jsonl \
  --output runs/akasha-os.pt
akasha-eval runs/akasha-os.pt data/akasha_os/test.jsonl
```

Splits are grouped by scenario family by default: one family (for example
Canvas, memory or device) stays entirely in a single file. For a finer
split, group by family and runtime signal instead (GPU, permissions,
network, audit, and so on):

```sh
akasha-data akasha-os --output data/akasha_os_context --split-by context
```

That keeps the same context situation, or a near-duplicate variant, from
appearing in both training and evaluation.

For the v2 dataset derived from a local source checkout, with abstention and
audit:

```powershell
.venv\Scripts\python.exe scripts\generate_akasha_dataset.py `
  --akasha-root <path-to-akasha-os> `
  --output data\akasha_os_v2 `
  --seed 20260918 `
  --total 30000
.venv\Scripts\python.exe scripts\audit_akasha_dataset.py `
  --input data\akasha_os_v2
```

`stress.jsonl` is evaluation-only. The full report is in
`reports/akasha_dataset_report.md`.

### Akasha OS — source-derived multitask dataset

To train the `Choice`, `Score` and `Noul` heads separately on scenarios
derived from identifiers that actually exist in a local Akasha OS checkout:

```powershell
.venv\Scripts\python.exe scripts\generate_akasha_multitask_dataset.py `
  --akasha-root <path-to-akasha-os> `
  --output data\akasha_os_multi `
  --seed 20260918 `
  --total 30000
.venv\Scripts\python.exe -m akasha_model.rlcd `
  data\akasha_os_multi\train.jsonl `
  --validation data\akasha_os_multi\validation.jsonl `
  --encoder tiny `
  --output runs\akasha_os_multi_rlcd.pt
.venv\Scripts\python.exe -m akasha_model.typed_decisions `
  runs\akasha_os_multi_rlcd.pt `
  --data data\akasha_os_multi\test.jsonl `
  --output reports\akasha_os_multi_rlcd.json
```

This version writes 24,000 training rows, 3,000 validation rows, 3,000 test
rows and 3,000 separate `stress` rows. Each row has one Choice, one Score
and nine Noul questions. Provenance, families and the audit are in
`reports/akasha_multitask_dataset_report.md`.

## Compare on typed-decisions

[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
is the public 2,000-decision benchmark used by Laya (0.766) and the published
Jev 1.13.0 number (0.727). Official Hub splits are 1,200 train rows and 400
test rows (2,000 test questions). They are kept as-is; this command does not
regroup rows across those splits.

A MASK checkpoint trained only on Akasha OS is a transfer run, not the
published comparison. Train on the Hub train split, then score the Hub test
split. The Hub has no validation file: use a subset of train for
`--validation` if you need early stopping, and report test only with
`akasha-typed-eval`.

```sh
uv pip install -e '.[eval,transformers]'
akasha-typed-eval --prepare data/typed_decisions
akasha-rlcd-train data/typed_decisions/train.jsonl \
  --validation data/typed_decisions/train.jsonl \
  --encoder hf --hf-model bert-base-uncased \
  --max-len 512 --batch-size 1 \
  --output runs/akasha-typed.pt
akasha-typed-eval runs/akasha-typed.pt \
  --data data/typed_decisions/test.jsonl \
  --output reports/typed_decisions.json
```

The report prints accuracy, soft accuracy, Brier, ECE and score MAE next to
those published comparison figures. A number here is agreement with a teacher
model, not a claim of correctness, and it is not a substitute for the Akasha OS
anti-leakage audit.

## Use a trained BERT encoder

Install the optional dependency and name any MASK-capable encoder from Hugging Face. Unlike the frozen-encoder choice scorer, RLCD updates the encoder weights.

```sh
uv pip install -e '.[transformers]'
akasha-rlcd-train data/akasha_os_multi/train.jsonl \
  --validation data/akasha_os_multi/validation.jsonl \
  --output runs/akasha-bert.pt \
  --encoder hf \
  --hf-model bert-base-uncased \
  --max-len 512 \
  --batch-size 1
```

`bert-base-uncased` position embeddings stop at 512 tokens; a larger `--max-len`
warns and can index past the table. `--batch-size 4` at 768 tokens does not fit
a 16 GB GPU with group-size 4. `--device auto` picks CUDA, MPS or CPU.
`--hf-model answerdotai/ModernBERT-base` is the
closer match to Laya's published checkpoints. The checkpoint stores the trained
encoder and scorer. Loading it needs access to the same tokenizer name.

The older frozen-encoder choice path remains: `akasha-train --encoder hf`.

## Wikispeedia example

[`scripts/get_wikispeedia.sh`](scripts/get_wikispeedia.sh) downloads the public SNAP archives and builds next-click JSONL files. The data stay outside this repository.

```sh
scripts/get_wikispeedia.sh
akasha-train data/wikispeedia/jsonl/train.jsonl \
  --validation data/wikispeedia/jsonl/validation.jsonl \
  --output runs/wikispeedia.pt
```

Cite Robert West and Jure Leskovec, *Human Wayfinding in Information Networks*, WWW 2012. Review the source data terms on the [SNAP dataset page](https://snap.stanford.edu/data/wikispeedia.html).

## What to expect

In the experiments that led to the original starter, the one-pass scorer reached about 98% accuracy on synthetic menus. On target-disjoint Wikispeedia next-click data, a frozen Qwen2.5-0.5B encoder plus the scorer reached 26%, against about 8% for shuffled and random-encoder controls. A small model trained from scratch on 40,000 clicks reached 29%. At eight options, one pass was about 100 times faster than a small decoder forced to write 400 tokens.

These numbers describe local experiments, not this quickstart run. They are not a claim of parity with Jev.

## Limitations

- This is a research starter for Akasha OS, not a copy of any commercial System 1 API.
- Accuracy depends on data quality, split quality and the encoder.
- The byte encoder is cheap but weak on language meaning.
- The pretrained path may download a large model and needs more memory.
- One-pass scoring requires the complete option list before prediction.
- The speed comparison used a small local decoder rather than a large commercial model.
- High-cardinality questions still need enough `--head-max-len` tokens per option, and enough `--max-len` left for the state after the option block.
- MASK/RLCD and byte-multitask checkpoints are different files. `akasha-multitask-eval` cannot load a MASK run; use `akasha-typed-eval`.
- Noul-heavy multitask rows can leave `choice` near chance even when overall GRPO loss falls. Always report per-head accuracy.

## License

Code is released under the [MIT License](LICENSE). Downloaded datasets and pretrained models keep their own terms.
