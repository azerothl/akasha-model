# Mix assistant examples

Path A helpers for the song-maker mix-assistant wedge. **No audio** is loaded or
stored here — only JSON descriptors and typed questions.

| File | Role |
|------|------|
| [`sample_questions.jsonl`](sample_questions.jsonl) | One multitask row: Choice preset, Score gain/pan, Noul masking |
| [`presets.v1.json`](presets.v1.json) | Versioned MixPresetCatalog (levels → dB / pan / EQ / compressor) |
| [`proposal_demo.py`](proposal_demo.py) | Confirmable batch + undo + audit envelope (no DAW apply) |
| Number sensitivity | [`docs/mix-number-sensitivity.md`](../../docs/mix-number-sensitivity.md) |
| Dataset | [`docs/mix-dataset.md`](../../docs/mix-dataset.md) — `scripts/generate_mix_dataset.py` |
| Contract | [`docs/contracts/mix-descriptors.md`](../../docs/contracts/mix-descriptors.md) |
| Presets | [`docs/contracts/mix-presets.md`](../../docs/contracts/mix-presets.md) |

```python
from pathlib import Path
from akasha_model.mix import load_mix_jsonl, validate_mix_state

rows = load_mix_jsonl(Path(__file__).parent / "sample_questions.jsonl")
assert rows[0].questions[0].primitive == "choice"
validate_mix_state(rows[0].state)  # type: ignore[arg-type]
```

Labels in the sample are **format fixtures**, not mix-quality claims.
