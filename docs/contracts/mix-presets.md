# Mix preset catalogue (ordered levels → numeric values)

Bridge between bounded model outputs and explicit DSP numbers. The model never
invents a continuous dB / pan / ratio; it only picks a **named preset** or an
**ordered Score index**. This catalogue maps those picks to values.

| Artifact | Path |
|----------|------|
| Catalogue JSON | [`examples/mix/presets.v1.json`](../../examples/mix/presets.v1.json) |
| Python API | `akasha_model.mix_presets` (`MixPresetCatalog`, `propose_mix_settings`) |
| Descriptors | [`mix-descriptors.md`](mix-descriptors.md) |

`catalog_version`: **1**.

## Discretization limits

From `discretization` in the JSON:

| Control | Step (catalogue) | Score levels |
|---------|------------------|--------------|
| Gain (dB) | 1.5 dB typical; ladder uses ±3 / ±6 | max 10 (`ScoreQuestion`) |
| Pan | 0.25–0.5 | usually 5 (hard L…hard R) |
| EQ bands | 1.0 dB | four fixed bands, listed explicitly in presets |
| Compressor | ratio + threshold listed per preset | optional Score ladders later |

Finer control requires a denser ladder (still ≤10 levels) or more presets — not
free-form floats from the model.

## Bounds

Catalogue `bounds` reject values outside:

- `gain_db` ∈ [-24, 24]
- `pan` ∈ [-1, 1]
- `eq_db` ∈ [-12, 12] (each band)
- `compressor_ratio` ∈ [1, 20]
- `compressor_threshold_db` ∈ [-60, 0]

## Confidence threshold

`propose_mix_settings(..., min_confidence=…)` **requires** an explicit
`min_confidence`. The package does **not** ship a default — hosts set it after
measurement (issue T4). Abstention returns `status="abstain"` with empty
settings; nothing is applied inside the package.

## Usage

```python
from akasha_model.mix_presets import MixPresetCatalog, propose_mix_settings
from akasha_model.primitives import ChoiceQuestion, OptionSpec, choice_result

catalog = MixPresetCatalog.from_json_path("examples/mix/presets.v1.json")
question = ChoiceQuestion(
    "preset", "Choose a mix preset.",
    tuple(OptionSpec(name) for name in catalog.preset_names()),
)
choice = choice_result(question, {
    "balanced": 0.1, "bass_heavy": 0.1, "vocal_forward": 0.8,
})
proposal = propose_mix_settings(catalog, choice=choice, min_confidence=0.55)
assert proposal.status == "ready"
assert proposal.settings["vocal.gain_db"] == 3.0
```

Side effects (applying faders) stay in the host after confirmation (issue T7).
