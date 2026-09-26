# Dataset spec: zero-shot conditional decisions

**Status:** draft · **Date:** 2026-09-22  
**Repo:** `azerothl/akasha-model`  
**Related:** ADR 0010 (akasha-os decision layer), MASK+RLCD path, `ToolCallPlanner`

## 1. Goal

Train and evaluate a **general conditional judgement** skill:

$$
P(\text{option}_i \mid \text{state},\ \text{question},\ \text{description}(\text{option}_i))
$$

not a fixed-label classifier over Akasha action ids.

A checkpoint passes this spec only if it can decide well when:

1. option **names** were never seen in training (arbitrary / invented labels),
2. the **correct rule** is stated only in the state or option descriptions,
3. calibration tracks confidence (high $P$ ≈ high empirical accuracy).

This is the open analogue of the Jev mental model: options are part of the
**input**, not part of a frozen class table. Adding a new Akasha tool or agent
must not require a retrain if its description is supplied at inference.

## 2. Non-goals

- Memorizing Akasha inventory ids (`fs.read`, `web.search`, …) as class indices.
- Replacing hard gates (caps, PolicyEngine, confirmations, placement).
- Claiming parity with commercial Jev / published typed-decisions leaderboards
  without running those evals separately (`akasha-typed-eval`).
- Teaching facts that are **not** present in the state (see §8 failure modes).

## 3. Relationship to existing datasets

| Dataset | Role after this spec |
|---|---|
| `data/akasha_os_multi*` | **In-domain** routing / policy simulation; keep for product fit |
| `data/synthetic`, Wikispeedia | Smoke / transfer probes |
| Hub `LocalLLaMA/typed-decisions` | External comparison only |
| **`data/zeroshot_decision/` (new)** | Gate for “general decision” competence |

Akasha-OS multitask data remains useful, but it is **insufficient alone**:
stable Choice templates and recurring action names can be memorized. Zero-shot
splits below are mandatory before ADR 0010 Phase C cutover.

## 4. Row schema

Compatible with multitask JSONL (`validate_multi` / MASK RLCD rows):

```json
{
  "id": "zs_train_000042",
  "domain": "industrial_safety",
  "split_group": "family_or_scenario_id",
  "context": {
    "narrative": "…",
    "facts": {"temperature_c": 91, "vibration": "high"},
    "policy": "Reduce speed when vibration is high and temperature > 85; shut down if temperature > 100."
  },
  "questions": [
    {
      "id": "action",
      "type": "choice",
      "instructions": "Choose the action that matches the policy.",
      "options": [
        {"name": "continue", "description": "Machine operates normally."},
        {"name": "banana", "description": "Reduce speed when operating conditions are becoming unstable."},
        {"name": "shutdown", "description": "Immediately stop when conditions may damage the machine."}
      ],
      "label": 1
    },
    {
      "id": "risk",
      "type": "score",
      "instructions": "Rate operational risk given the facts and policy.",
      "levels": ["low", "elevated", "critical"],
      "label": 1
    },
    {
      "id": "policy_sufficient",
      "type": "noul",
      "instructions": "Is the policy text sufficient to decide without outside knowledge?",
      "criteria": {
        "true": "The policy fully determines the correct action from the facts.",
        "false": "Missing definitions or contradictory rules."
      },
      "label": 1
    }
  ],
  "meta": {
    "label_surface": "arbitrary",
    "rule_location": "context.policy",
    "gold_depends_on": ["facts.temperature_c", "facts.vibration", "context.policy"]
  }
}
```

Rules:

- `choice` option **names** may be opaque (`banana`, `opt_7`, UUID). Meaning lives
  in `description` and/or `context.policy`.
- Gold `label` must be computable by a **deterministic teacher** from
  `context` + option texts alone (no LLM teacher for the primary train set).
- Prefer 1 Choice + 1 Score + a small Noul set; use `--type-balance` when Noul
  count would dominate RLCD (§7).

## 5. Generation recipe

### 5.1 Domain bank

Maintain ≥ 40 synthetic domains, e.g.:

- Akasha-like OS routing (tools, agents, caps) **with renamed labels**
- customer support triage, code review assignment, game style / agent pick
- industrial / medical / logistics policies (fictional units OK if defined)
- pure logic: thresholds, boolean combinations, priority lists

Each domain ships:

- a fact schema,
- a policy template language (deterministic),
- distractor option generators,
- optional stress transforms.

### 5.2 Instance construction

For each row:

1. Sample domain + fact vector.
2. Sample or compose a **policy string** that uniquely determines the gold Choice
   (and Score / Noul when present).
3. Build candidate options:
   - **semantic names** (easy),
   - **arbitrary names** + correct descriptions (hard / target),
   - **misleading names** (name suggests wrong action; description correct),
   - **swapped descriptions** stress (eval only).
4. Shuffle option order; record gold index after shuffle.
5. Assign `split_group` so near-duplicates stay in one split.

Target mix in **train**:

| Slice | Share | Intent |
|---|---:|---|
| Arbitrary labels + full descriptions | ≥ 40% | Force description reading |
| Misleading names | ≥ 15% | Break lexical shortcuts |
| Semantic names | ≤ 30% | Easy curriculum |
| Underspecified / abstain-friendly | 10–15% | Calibration & Noul |

### 5.3 Split protocol (anti-leakage)

Hard requirements (fail audit if violated):

1. **Category disjointness (Choice):** no gold option *description embedding
   cluster* or canonical rule id used in train may appear as gold in test.
   Practically: hold out entire **domains** or **policy families** for test.
2. **Label-string disjointness:** test Choice option *names* must be fresh
   draws (generator seed space disjoint from train).
3. **State collision:** zero exact/normalized context collisions across splits
   (reuse `audit_akasha_*` style checks).
4. **Stress is eval-only:** never train on `stress.jsonl`.

Recommended layout:

```text
data/zeroshot_decision/
  train.jsonl
  validation.jsonl
  test_unseen_domain.jsonl      # domains never in train
  test_unseen_labels.jsonl      # same domain family, new names/rules holdout
  stress.jsonl
  audit.json
  GENERATOR.md                  # seed, version, command line
```

Suggested scale (v1): train 50k / val 5k / each test 5k / stress 5k rows.

## 6. Evaluation protocol

Run with `akasha-typed-eval` (MASK) or multitask eval (byte), **per head**.

### 6.1 Metrics (gates)

| Metric | Split | Gate (v1 proposal) |
|---|---|---|
| Choice accuracy | `test_unseen_domain` | ≥ random + margin; track vs in-domain Akasha test |
| Choice accuracy | `test_unseen_labels` | Must stay close to unseen_domain (no collapse) |
| Description ablation | same rows, descriptions stripped | Accuracy → ~chance (proves dependence on text) |
| Name-only ablation | descriptions removed, names kept | Must **not** retain high accuracy on arbitrary-name slice |
| ECE / Brier (Choice & Noul) | held-out | Report; aim ECE ↓ after calibration / RLCD |
| Reliability at $P\in[0.75,0.85]$ | held-out | Empirical accuracy within ±0.10 of bin mean |
| Score MAE / ordinal error | held-out | Report per domain |
| Abstain quality | stress / underspecified | Higher abstain or lower confidence when Noul says insufficient |

### 6.2 Required ablations

1. **Shuffle option order** — accuracy stable.
2. **Paraphrase descriptions** (eval templates) — limited drop.
3. **Negate policy** — prediction must flip with gold.
4. **Remove policy from context** — accuracy → chance; confidence should fall.

### 6.3 Product-facing probe (Akasha)

Separate small suite: invent agents/tools (`Aether`, `Forge`, `Sentinel`, …)
**absent from train**, described only in-menu; route user tasks. This is a
demo/probe, not the statistical gate.

## 7. Training guidance

- Prefer MASK+RLCD (`akasha-rlcd-train`) with `--type-balance` when rows are
  Noul-heavy.
- Curriculum: start with semantic names, increase arbitrary / misleading share.
- Calibrate on val that was not used for early stopping selection when reporting
  final ECE.
- Do not mix Hub typed-decisions into the zero-shot train set if you want a
  clean external comparison later.

## 8. Failure modes to document in `audit.json`

| Failure | Detection |
|---|---|
| Shortcut on option order | Position-biased gold without shuffle audit |
| Shortcut on option name lexicon | High name-only ablation accuracy on arbitrary slice |
| Leakage of policy families | Overlap of `split_group` / domain ids across splits |
| Unanswerable gold | Teacher cannot derive label from context alone |
| Undefined jargon | State uses terms without definitions (invalid train row) |

Undefined jargon **without** definitions is allowed only in stress rows whose
expected behavior is abstain / low confidence / `policy_sufficient=false`.

## 9. Generator CLI (to implement)

```text
python scripts/generate_zeroshot_decision_dataset.py \
  --output data/zeroshot_decision \
  --seed 20260922 \
  --train 50000 --val 5000 --test-domain 5000 --test-labels 5000 --stress 5000

python scripts/audit_zeroshot_decision_dataset.py \
  --input data/zeroshot_decision
```

Audit must emit: collision counts, domain holdout proof, arbitrary-label share,
ablation hooks sample ids, and a machine-readable `gates.json` template.

## 10. Acceptance for ADR 0010 Phase C

Shadow (Phase A) may use Akasha-OS multitask alone. **Gated cutover (Phase C)**
additionally requires:

1. This dataset generated + audited with zero critical leakage findings.
2. Unseen-domain and unseen-label Choice accuracy reported side-by-side with
   in-domain Akasha accuracy.
3. Description-ablation and name-only ablation results published in `reports/`.
4. ECE (or reliability diagram) after calibration attached to the cutover note.

Until then, treat strong in-domain Akasha numbers as **product fit**, not proof
of general System One competence.

## 11. References

- README — MASK architecture, RLCD, typed-decisions comparison
- `reports/akasha_multitask_dataset_report.md` — anti-leakage audit style
- Branch `cursor/rlcd-type-balance` — `--type-balance` for multitask RLCD
- akasha-os ADR 0010 — decision layer integration / Phase A–D
