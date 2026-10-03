# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Mix MASK T8 training recipe (`scripts/train_mix_mask.py`, `--mix-compact-state`,
  [`docs/mix-mask-train.md`](docs/mix-mask-train.md)): compact dB/LUFS state so
  tiny MASK can see numbers; T4 judge vs majority prior. Weights stay in `runs/`.
  `gate-tiny.pt` is explicitly not a mix checkpoint (#82).
- Mix Option B import pipeline (human labels schema, anti-leak audit, no audio,
  no invented annotator rows) (#83).
- ONNX export scaffold `scripts/export_mix_mask_onnx.py` that waits on a
  number-sensitive mix MASK (#84).

## [0.3.0] - 2026-10-02

Second tagged release: mix Path A surface, first-run demos, presentation site,
and offline Path B helpers. Install from PyPI or pin a git tag:

```text
pip install akasha-model==0.3.0
# or
akasha-model @ git+https://github.com/azerothl/akasha-model.git@v0.3.0
```

### Added

- First-run Path A usage demo (`examples/usage/demo.py`): tool gate
  ready/abstain/blocked plus a ticket-router Choice abstain walkthrough;
  CPU-only, no torch / Hub. README “Try it” pointer and CI smoke.
- Visual first-run walkthrough (`examples/usage/visual_demo.py` +
  `visual.html`): browser page with state, typed Choice/Score/Noul bars,
  confidence, and ready/abstain/blocked rationale; shared `decision_trace.py`.
- Tetris realtime demo (`examples/tetris/`): falling pieces with visible
  gravity, level-based accelerating drop tempo, Path A Choice over legal locks
  with human-readable labels and probability bars, next-piece preview feeding
  one-ply look-ahead, play until game over (or Stop), host verify before lock;
  browser UI (`play.py --open`) plus text fallback; CPU-only.
- Mix descriptor contract (`akasha_model.mix`, schema + docs): host-computed
  dB/LUFS/centroid/band/crest/correlation/masking JSON state for Choice/Score/Noul
  mix questions; no audio in the package (#56).
- Mix preset catalogue (`akasha_model.mix_presets`): ordered Choice/Score picks
  map to explicit gain/pan/EQ/compressor values via versioned JSON; confidence
  threshold is a required host parameter (#57).
- Mix proposal batches (`akasha_model.mix_proposal`): confirmable before/after
  change sets with one-action undo and audit envelopes via `build_audit_envelope`
  (#61).
- Synthetic mix JSONL generator + anti-leakage audit (Option A written rules,
  per-row licence/source fields, no audio) (#58).
- Mix number-sensitivity protocol (`scripts/evaluate_mix_number_sensitivity.py`):
  baseline / shuffled / perturbed / removed-number controls with ECE and
  coverage-risk; MASK tiny + BERT encoders; CPU and CUDA 16GB latency/RSS/VRAM
  reports plus French instruction probe (#59).
- Weight/data licence dossier (`docs/licences-poids.md`) for shipped checkpoints
  and base encoders; Gabriel review marked pending before commercial delivery
  (#60).
- Offline Path B helpers (`akasha_model.offline`): local `.pt` load with
  network denial, clear model-absent status; ONNX/GGUF export deferred until a
  mix MASK checkpoint exists (#62).
- Presentation site with GitHub Pages deploy (`site/`,
  https://azerothl.github.io/akasha-model/) (#79, #80).

### Changed

- Tetris demo (`examples/tetris/play.py`): default is a **new random 7-bag +
  lock-sample seed each run** (HUD shows `seed N (this run)`); `--seed N`
  remains for reproducible replays. Spawn column stays standard Tetris, not
  random X.

### Docs

- README: clarify PyPI vs checkout contents, gate-first framing, schema-first
  MASK layouts, Path A catalog/authority/wire/audit pointers, and
  `akasha-multitask-calibrate`.

## [0.2.0] - 2026-09-27

First tagged release for Akasha OS and other Path A / Path C hosts.

Install from PyPI or pin a git tag (preferred over feature branches):

```text
pip install akasha-model==0.2.0
# or
akasha-model @ git+https://github.com/azerothl/akasha-model.git@v0.2.0
```

### Path A — scripted gate (no checkpoint)

- `evaluate_gate`, `GateSignals`, `ToolProposal`, `default_gate_planner`
- `ToolCallPlanner` / `ToolSpec` / `ToolCallPlan` (`ready` / `abstain` / `blocked`)
- Default thresholds: `DEFAULT_MIN_CHOICE_PROBABILITY`, `DEFAULT_MIN_CHOICE_CONFIDENCE`,
  `DEFAULT_NOUL_THRESHOLD`, `DEFAULT_MAX_RISK_SCORE`

### Path C — host dispatch (no built-in executor)

- `ToolHost` protocol, `dispatch_plan`, `run_gated_call`
- `HostOutcome` / `HostAction` (`executed`, `skipped_abstain`, `skipped_blocked`,
  `rejected_by_host`)

### Path D — outcomes (threshold suggestions only)

- `append_outcome`, `summarize_outcomes`, `suggest_threshold_updates`
- Suggestions never auto-write package `DEFAULT_*` constants

### Path B — scored gate (torch)

- `plan_scored_proposal`, `score_proposal`, `run_scored_gated_call`
- Demo checkpoint: `examples/gate/checkpoints/gate-tiny.pt` (demo-only)

### Docs

- [docs/using-the-tool-gate.md](docs/using-the-tool-gate.md)
- [examples/gate/README.md](examples/gate/README.md)

[0.3.0]: https://github.com/azerothl/akasha-model/releases/tag/v0.3.0
[0.2.0]: https://github.com/azerothl/akasha-model/releases/tag/v0.2.0
