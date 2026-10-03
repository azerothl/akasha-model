# Mix MASK training recipe (issue #82 / T8)

T4 ([#59](https://github.com/azerothl/akasha-model/issues/59)) measured tiny and
BERT MASK checkpoints trained on raw mix JSON: they matched the **majority
prior** on every control. That is **not** a publishable mix MASK.

`examples/gate/checkpoints/gate-tiny.pt` is a **tool-gate** demo (byte
multitask). It is **not** a mix checkpoint. Do not evaluate it as T8.

## What this recipe changes

Schema-first MASK puts options before state. Compact JSON descriptors are still
hundreds of bytes; the tiny **byte** tokenizer then clips dB/LUFS. Training
with `--mix-compact-state` renders `mix-compact-v1` text (rms/lufs/masking
first). Host contracts stay on schema v1 JSON; compact text is a MASK encoding
only, stored on the checkpoint as `config.mix_compact_state`.

Weights stay in `runs/` (gitignored). Do not commit private mix `.pt` files.

## CPU (tiny encoder)

```sh
python scripts/generate_mix_dataset.py --output data/mix_synth --rows 240
python scripts/audit_mix_dataset.py --input data/mix_synth
python scripts/train_mix_mask.py --data data/mix_synth --device cpu \
  --output runs/mix-mask-tiny.pt --rows 240 --epochs 20
```

`train_mix_mask.py` calls RLCD with `--mix-compact-state --mask-layout state_first`
`--type-balance --target-mode one_hot --group-size 1`, then T4
(`scripts/evaluate_mix_number_sensitivity.py`). Exit 0 only if the T4 judge
(`majority_sensitivity_verdict`) sees ≥0.08 accuracy over majority **and** a
≥0.08 drop on `numbers_removed` or `numbers_perturbed` on that head.

A CPU probe (120 Option A rows, 8 tiny epochs) moved **score** and **noul** when
numbers were stripped (drops 0.57 / 0.21 on 14 test rows) but stayed within one
example of the majority prior on baseline — **not** a publishable mix MASK.
BERT/GPU and/or Option B labels are the next capacity/data steps, not a
different protocol.

## GPU / BERT (optional, not required for the recipe)

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m akasha_model.rlcd \
  data/mix_synth/train.jsonl \
  --validation data/mix_synth/validation.jsonl \
  --encoder hf --hf-model bert-base-uncased --epochs 3 --batch-size 4 \
  --device cuda --mix-compact-state --type-balance --target-mode one_hot \
  --output runs/mix-mask-bert.pt
```

## ONNX / GGUF

T10 ([#84](https://github.com/azerothl/akasha-model/issues/84)) waits on a T8
checkpoint that passes the judge. Scaffolding:
`scripts/export_mix_mask_onnx.py` (refuses `gate-tiny.pt` and non-sensitive
T4 reports). The tiny `DecisionModel` graph **can** export when the optional
`onnx` package is installed; that is not acceptance. GGUF is out of scope
(not a llama.cpp LLM). Offline serving until T8 remains local `.pt`
([path-b-offline.md](path-b-offline.md)).
