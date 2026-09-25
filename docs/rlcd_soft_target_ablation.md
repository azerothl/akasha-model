# Proper-reward / RLCD ablation on typed-decisions

This experiment measures two independent choices: the policy term (`1` or `0`)
and the training target (the supplied distribution or a one-hot label). It uses
the same `bert-base-uncased` initial checkpoint for all four arms of each seed,
type-balanced loss, four option-order views per training question, and seeds
7, 17 and 27. The reward remains Akasha's log + spherical + ordinal ranked
score in every arm. These soft targets come from `typed-decisions`; this is not
a reproduction of the teacher-permutation labeling process in system-one-270m.

Prepare the official data once, if it is not already available:

```powershell
uv pip install -e '.[eval,transformers]'
akasha-typed-eval --prepare data/typed_decisions
```

Inspect the deterministic train/validation partition without training:

```powershell
python scripts/run_rlcd_ablation.py --prepare-only
```

Run the twelve sequential training and evaluation jobs:

```powershell
python scripts/run_rlcd_ablation.py --device cuda
```

The default batch size is one source state, which expands to its questions and
four option-order views; this fits the 16 GB GPU used for development. Each
checkpoint is selected using only the internal validation subset of the
official train split. The official test data is checked for split integrity
during preparation; its labels are used for model evaluation after each
checkpoint is written. A failed or interrupted run can be resumed with the
same command. The `runs/rlcd_ablation/` directory contains the source hashes,
split manifest, shared initial checkpoint for each seed, training logs,
checkpoints, question-level test records and `comparison.json`.

Reports retain the historical `ece`, which uses normalized entropy as
confidence. `ece_maxprob` is a separate 15-bin ECE using the predicted option's
probability and correctness against the supplied label. `accuracy` retains its
historical target-argmax definition; `label_accuracy` uses the supplied label.
NLL and Brier compare with the full supplied target distribution. Permutation
stability uses eight seeded orderings per question, aligns probabilities to
original option identities, and reports top-choice flip rate and mean
Jensen-Shannon divergence. Binary questions necessarily repeat orderings.

`comparison.json` contains every seed's metrics, seed means, paired 95%
bootstrap intervals for each arm's difference from `rlcd_hard`, and direct
contrasts for policy, soft-target and interaction effects. The bootstrap
unit is the original state/row; all questions from a state are resampled
together. The intervals describe uncertainty over held-out states conditional
on the three chosen seeds. Lower Brier, NLL, ECE and instability are better;
higher accuracy is better.

To rebuild only the aggregate report from existing run records:

```powershell
python scripts/run_rlcd_ablation.py --report-only
```

The data, checkpoints and reports in `runs/` remain outside Git. No paid API
is used.
