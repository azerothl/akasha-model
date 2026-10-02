# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Mix descriptor contract (`akasha_model.mix`, schema + docs): host-computed
  dB/LUFS/centroid/band/crest/correlation/masking JSON state for Choice/Score/Noul
  mix questions; no audio in the package (#56).
- Mix preset catalogue (`akasha_model.mix_presets`): ordered Choice/Score picks
  map to explicit gain/pan/EQ/compressor values via versioned JSON; confidence
  threshold is a required host parameter (#57).

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

[0.2.0]: https://github.com/azerothl/akasha-model/releases/tag/v0.2.0
