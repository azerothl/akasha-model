# Mix Option B labels (human annotations)

Option A (`scripts/generate_mix_dataset.py`) labels rows with **written rules**.
Tests on Option A measure **rule agreement**, not mix quality.

Option B is **human** Choice / Score / Noul labels on the same T1 descriptors
and T2 question shapes. This repository ships the **schema and import
pipeline**, not a labelled corpus. Do not invent annotator labels.

| Artifact | Path |
|----------|------|
| JSON Schema | [`akasha_model/schemas/mix-option-b-labels.schema.json`](../../akasha_model/schemas/mix-option-b-labels.schema.json) |
| Import | [`scripts/import_mix_option_b.py`](../../scripts/import_mix_option_b.py) |
| Audit | [`scripts/audit_mix_dataset.py`](../../scripts/audit_mix_dataset.py) (same `context_key` anti-leak as Option A) |
| Format fixture | [`tests/contracts/fixtures/mix-option-b-format.jsonl`](../../tests/contracts/fixtures/mix-option-b-format.jsonl) |

## Row fields

Required: `context` (mix-descriptors v1), `questions` (choice preset, score
levels, noul masking), `context_key`, `source`, `licence`, `label_licence`,
`label_mode=option_b_human`.

Recommended: `descriptor_licence`, `annotator_hash` (non-identifying), 
`annotation_status=human`.

`annotation_status=format_only` is **only** for schema tests. Import refuses
those rows unless `--allow-format-fixture`. Never train MASK weights on
format fixtures.

**No audio** in JSONL or git (`audio` / `wav` / `stems` / … keys fail import
and audit).

## What tests measure

| Set | Metric |
|-----|--------|
| Option A JSONL | Agreement with `mix-rules-v1` |
| Option B JSONL | Agreement with **annotators** on the typed questions |
| Neither | Mix quality, loudness taste, or “sounds better” |

## Import

```sh
python scripts/import_mix_option_b.py \
  --input path/to/human.jsonl \
  --output data/mix_option_b
python scripts/audit_mix_dataset.py --input data/mix_option_b
```

Record label licences in [`licences-poids.md`](licences-poids.md) **before**
training any checkpoint on Option B. Gabriel review (#60) still pending for
commercial delivery.
