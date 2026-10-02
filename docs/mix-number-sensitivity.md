# Mix number-sensitivity protocol (issue #59)

Script: [`scripts/evaluate_mix_number_sensitivity.py`](../scripts/evaluate_mix_number_sensitivity.py)

Published measurements (no pass/fail thresholds):  
[`reports/mix_number_sensitivity_cpu.json`](../reports/mix_number_sensitivity_cpu.json),  
[`reports/mix_number_sensitivity_cuda.json`](../reports/mix_number_sensitivity_cuda.json),  
summary below and in [`reports/mix_number_sensitivity.md`](../reports/mix_number_sensitivity.md).

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
- latency (s, ms/row), process RSS (MiB), and on CUDA peak allocated VRAM (MiB)

## Encoders

| Encoder | Role |
|---------|------|
| `rule_prior` | Teacher rules from the T3 generator (oracle for Option A) |
| `majority_prior` | Constant prior (balanced / unchanged / not-masked) |
| `mask:<path>` | MASK `DecisionModel` checkpoint (`tiny` or HF `bert-base-uncased`) |

Train local checkpoints (not committed; `runs/` is gitignored):

```sh
python scripts/generate_mix_dataset.py --output data/mix_synth --rows 400
python -m akasha_model.rlcd data/mix_synth/train.jsonl \
  --validation data/mix_synth/validation.jsonl \
  --encoder tiny --epochs 12 --device cuda \
  --output runs/mix-mask-tiny.pt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m akasha_model.rlcd \
  data/mix_synth/train.jsonl \
  --validation data/mix_synth/validation.jsonl \
  --encoder hf --hf-model bert-base-uncased --epochs 3 --batch-size 4 \
  --device cuda --output runs/mix-mask-bert.pt
```

## French

Descriptor units are language-agnostic. Pass `--french-probe` to rewrite Choice /
Score / Noul instructions (and option text) to French while keeping labels fixed,
then report per-head accuracy deltas vs English on the first MASK encoder.

## Reproduce

```sh
python scripts/generate_mix_dataset.py --output data/mix_synth --rows 400
python scripts/evaluate_mix_number_sensitivity.py \
  --data data/mix_synth --device cpu --offline-hub --french-probe \
  --encoders "rule_prior,majority_prior,mask:runs/mix-mask-tiny.pt,mask:runs/mix-mask-bert.pt" \
  --output reports/mix_number_sensitivity_cpu.json
python scripts/evaluate_mix_number_sensitivity.py \
  --data data/mix_synth --device cuda --offline-hub --french-probe \
  --encoders "rule_prior,majority_prior,mask:runs/mix-mask-tiny.pt,mask:runs/mix-mask-bert.pt" \
  --output reports/mix_number_sensitivity_cuda.json
```

Values are published as measured — **no pass/fail threshold** in the protocol.
