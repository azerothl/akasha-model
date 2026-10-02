# Mix number-sensitivity report (issue #59 / T4)

Protocol: `mix-number-sensitivity-v1`  
Data: T3 synthetic Option A (`data/mix_synth`, 400 rows generated, **63 test** rows)  
Hardware: NVIDIA GeForce RTX 4080 SUPER (**16375.5 MiB** VRAM)  
Checkpoints (local, not committed): `runs/mix-mask-tiny.pt` (12 epochs),  
`runs/mix-mask-bert.pt` (BERT `bert-base-uncased`, best of 3 epochs, Hub offline).

Raw JSON: [`mix_number_sensitivity_cpu.json`](mix_number_sensitivity_cpu.json),  
[`mix_number_sensitivity_cuda.json`](mix_number_sensitivity_cuda.json).

**No pass/fail thresholds** — values as measured.

## Finding

`rule_prior` drops when numbers are shuffled / perturbed / removed, so the
controls are discriminative. Short MASK tiny and BERT fine-tunes on this Option A
set **match the majority prior** on every control (≈0.52 / 0.64 / 0.52) — they do
**not** yet show number sensitivity. That is a measured negative result for these
checkpoints, not a protocol failure.

## Accuracy by control (choice / score / noul)

Identical on CPU and CUDA for accuracy (same weights).

| Encoder | baseline | shuffled | perturbed | numbers removed |
|---------|----------|----------|-----------|-----------------|
| rule_prior | 1.000 / 1.000 / 1.000 | 0.698 / 0.683 / 1.000 | 0.905 / 0.825 / 0.952 | 0.524 / 0.635 / 0.476 |
| majority_prior | 0.524 / 0.635 / 0.476 | same | same | same |
| MASK tiny | 0.524 / 0.635 / 0.524 | same | same | same |
| MASK BERT | 0.524 / 0.635 / 0.524 | same | same | same |

Choice ECE (baseline): rule 0.000 · majority 0.184 · tiny 0.398 · BERT 0.472.

## Latency and memory (baseline)

| Encoder | CPU ms/row | CPU RSS MiB | CUDA ms/row | CUDA RSS MiB | CUDA peak alloc MiB |
|---------|------------|-------------|-------------|--------------|---------------------|
| rule_prior | 0.003 | 504 | 0.003 | 504 | — |
| majority_prior | 0.001 | 504 | 0.001 | 504 | — |
| MASK tiny | 22.1 | 825 | 6.9 | 1132 | 358 |
| MASK BERT | 395 | 1729 | 11.6 | 2175 | 841 |

## French (`--french-probe`, MASK tiny)

Instructions/options rewritten to French; labels unchanged.

| Head | Δ accuracy (FR − EN) |
|------|----------------------|
| choice | −0.238 |
| score | −0.556 |
| noul | 0.000 |

French question text changes tiny predictions on choice/score; noul unchanged on
this run. Descriptors remain language-agnostic JSON numbers.

## Reproduce

See [`docs/mix-number-sensitivity.md`](../docs/mix-number-sensitivity.md).
