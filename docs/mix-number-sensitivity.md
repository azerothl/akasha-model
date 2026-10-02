# Mix number-sensitivity protocol (issue #59)

Script: [`scripts/evaluate_mix_number_sensitivity.py`](../scripts/evaluate_mix_number_sensitivity.py)

## Controls

| Control | What changes |
|---------|----------------|
| `baseline` | Original synthetic descriptors |
| `shuffled_context` | Track blocks permuted across rows |
| `numbers_perturbed` | Multiplicative/additive noise on numeric fields |
| `numbers_removed` | Numeric fields set to `null` |

Questions and labels stay fixed so a drop vs baseline indicates **number/context sensitivity**.

## Metrics (per head: choice / score / noul)

- accuracy
- ECE (10 bins)
- coverage / risk at confidence thresholds 0.50–0.95
- latency (s, ms/row) and RSS (MiB)

## Encoders shipped in the CPU recipe

| Encoder | Role |
|---------|------|
| `rule_prior` | Teacher rules from the T3 generator |
| `majority_prior` | Constant prior (balanced / unchanged / not-masked) |

**Not in default CI:** BERT / ModernBERT and a full **GPU 16GB** pass — need CUDA
hardware and local/offline weights. The JSON report records
`torch_cuda_available` and explicit `device_notes`.

## French

Descriptor units are language-agnostic. Question **instructions** may be French;
this protocol documents FR behaviour but does **not** claim a measured FR vs EN
accuracy gap until a trained checkpoint is evaluated (see report `french` block
and `tests/test_mix_number_sensitivity_smoke.py`).

## Reproduce

```sh
python scripts/generate_mix_dataset.py --output data/mix_synth --rows 120
python scripts/evaluate_mix_number_sensitivity.py \
  --data data/mix_synth \
  --output reports/mix_number_sensitivity.json
```

Values are published as measured — **no pass/fail threshold** in the protocol.
