# Zeroshot full-scale probe report (optional Path B)

**Date:** 2026-09-30  
**Issue:** [#46](https://github.com/azerothl/akasha-model/issues/46)  
**Status:** offline run verified locally — **not** a release blocker; **not** a
gate / System 1 cutover.

## Command

```sh
python scripts/generate_zeroshot_decision_dataset.py \
  --output data/zeroshot_decision \
  --seed 20260922 \
  --train 50000 --val 5000 --test-domain 5000 --test-labels 5000 --stress 5000

python scripts/audit_zeroshot_decision_dataset.py \
  --input data/zeroshot_decision
```

Generated JSONL stays under `data/` (gitignored). Do not mix Hub
typed-decisions into this train set.

## Audit summary (seed `20260922`)

| Check | Result |
|-------|--------|
| `ok` | `true` |
| Critical leakage | none |
| `split_group` collisions with train | 0 (val / test_domain / test_labels) |
| Domain holdout | `pure_threshold` held out; train domains: access_control, industrial_safety, logistics_priority, support_triage |
| Arbitrary-label share (train) | ≈ 0.601 |
| Row counts | train 50k / val 5k / each test 5k / stress 5k |

`gates.json` template fields (unseen-domain accuracy, ablations, ECE) remain
null until a separate Path B train/eval pass fills them — out of scope for this
ticket.

## Scope notes

- Current generator domain pool is **five** synthetic domains (not the older
  prose target of ≥40). Expanding domains is optional polish, not a gate
  identity requirement.
- Smoke CI continues to use tiny row overrides (see §9 of the dataset spec).
- Gate product identity remains Path A authorization (`evaluate_gate`); this
  dataset is an optional description-driven Path B quality probe only.
