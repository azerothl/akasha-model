# Mix training/eval dataset (synthetic Option A)

Generator: [`scripts/generate_mix_dataset.py`](../scripts/generate_mix_dataset.py)  
Audit: [`scripts/audit_mix_dataset.py`](../scripts/audit_mix_dataset.py)

## What is implemented

**Option A — written rules** (`label_mode=option_a_written_rules`, `rules_version=mix-rules-v1`).

Descriptors are **synthetic numbers** (no stems, no MedleyDB/MUSDB/etc.). Labels
are produced by the documented heuristics in the generator (`rule_preset`,
`rule_vocal_gain_level`, `rule_masked`).

**What tests measure:** agreement with these written rules — **not** mix quality
or human preference.

Option B (human annotations): schema + import pipeline in
[`docs/contracts/mix-option-b.md`](contracts/mix-option-b.md). **No human labels
are shipped.** Format fixtures are `annotation_status=format_only` and must not
be used as training data.

MASK mix training recipe (T8): [`docs/mix-mask-train.md`](mix-mask-train.md).
`gate-tiny.pt` is not a mix checkpoint.

## Provenance fields (per row)

| Field | Meaning |
|-------|---------|
| `source` | `akasha-synthetic-mix-rules` |
| `licence` / `label_licence` / `descriptor_licence` | `CC0-1.0` |
| `rules_version` | rule table id |
| `context_key` | split key (session id) |

`licence_report.json` summarises proportions by licence/source. **No audio** is
written; keep outputs under `data/` (gitignored).

## Usage

```sh
python scripts/generate_mix_dataset.py --output data/mix_synth --rows 120
python scripts/audit_mix_dataset.py --input data/mix_synth
```

Splits are hashed by `context_key` (same idea as `scripts/split_by_context.py`).
