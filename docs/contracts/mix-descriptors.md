# Mix descriptors → decision state

Contract for hosts that turn **deterministic audio analysis** into JSON state for
Akasha Choice / Score / Noul questions (song-maker « Assistant de mix unique »).

| Artifact | Path |
|----------|------|
| JSON Schema | [`akasha_model/schemas/mix-descriptors.schema.json`](../../akasha_model/schemas/mix-descriptors.schema.json) |
| Python validators | `akasha_model.mix` (`TrackDescriptors`, `MixSessionState`, `validate_mix_state`) |
| Example JSONL | [`examples/mix/sample_questions.jsonl`](../../examples/mix/sample_questions.jsonl) |
| Schema version | **1** (`schema_version: 1`) |

## Non-negotiables

- **No audio in the package.** Waveforms, stems, and proprietary datasets stay in
  the host. Only rounded numeric descriptors enter `DecisionRequest.state`.
- Descriptors are computed by **host-owned deterministic code**. The model never
  invents dB / LUFS values; it only scores bounded options over the serialized state.
- Path A remains importable without torch. `akasha_model.mix` has no torch import.

## Descriptor fields

Per track (`tracks[]`):

| Field | Unit | Bounds | Rounding |
|-------|------|--------|----------|
| `track_id` | token | non-empty, `[A-Za-z0-9_.:-]+` | — |
| `rms_db` | dBFS | [-120, 0] | 0.1 |
| `lufs` | LUFS | [-70, 0] | 0.1 |
| `spectral_centroid_hz` | Hz | [20, 20000] | 1 |
| `band_energies_db` | dBFS ×4 | each [-120, 0] | 0.1 |
| `crest_factor_db` | dB | [0, 40] | 0.1 |
| `stereo_correlation` | — | [-1, 1] | 0.01 |

Fixed band order for `band_energies_db`: **low / low-mid / high-mid / high**
(host defines exact Hz edges; the model only sees the four numbers).

Optional `pair_masking[]` entries: `masker_id`, `maskee_id`, `masking_index` in
[0, 1] (0.01). Track ids in pairs must exist in `tracks`.

Optional session `sample_rate_hz` in [8000, 192000].

## Question types (mix contract)

Hosts attach the validated descriptor object as `context` / `state` and ask:

1. **Choice — preset.** Pick one named mix preset from a bounded catalogue
   (e.g. `vocal_forward`, `balanced`, `bass_heavy`). Option names are catalogue
   keys; descriptions may cite intended use. See issue T2 for mapping names →
   numeric DSP values.
2. **Score — ordered level per parameter.** One Score question per continuous
   control (gain, pan, EQ band, compressor ratio/threshold). Levels are ordered
   ordinals (e.g. `much quieter` … `much louder`); T2 maps level index → dB / ratio.
3. **Noul — masked track?** Binary “is track A masked by track B?” using
   `pair_masking` and criteria strings in the question.

## Example multitask JSONL row

See [`examples/mix/sample_questions.jsonl`](../../examples/mix/sample_questions.jsonl).
Rows use the shared multitask shape (`context` + `questions[]` with
`choice` / `score` / `noul`). Labels in the sample are synthetic for format
checks only — they are not mix-quality ground truth.

## Validation

```python
from akasha_model.mix import validate_mix_state, mix_decision_request_from_row
from akasha_model.primitives import DecisionRequest

state = validate_mix_state({...})          # raises ValueError on OOB
request: DecisionRequest = mix_decision_request_from_row(row)
```

Out-of-bounds values, wrong band arity, duplicate `track_id`, and unknown
masking pair ids fail closed.
