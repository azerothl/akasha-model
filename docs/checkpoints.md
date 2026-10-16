# Checkpoints: formats, loading and calibration

Akasha ships three checkpoint families. They do **not** share one scorer
contract, so the loader reports what a checkpoint can really do instead of
promising a common evaluation or calibration path.

| Architecture | Built by | Input contract | Capabilities |
|---|---|---|---|
| `mask` | `akasha-rlcd-train`, `scripts/train_mix_mask.py` | MASK-marker sequences (`MaskCollator`) | choice, score, noul, per-type temperature, RLCD |
| `single_choice` | `akasha-train` | one context + one option list (`ByteCollator`) | choice, single temperature |
| `byte_multitask` | `akasha-multitask-train` | shared state + typed questions (`MultiQuestionCollator`) | choice, score, noul, multitask calibration |

## Loading

Use the factory when the family is not known in advance:

```python
from akasha_model import load_any_checkpoint

loaded = load_any_checkpoint("runs/model.pt", device)
loaded.architecture   # "mask" | "single_choice" | "byte_multitask"
loaded.info.capabilities
loaded.model          # family-specific scorer; contracts differ per row above
```

Family loaders keep working unchanged (`load_decision_checkpoint`,
`akasha_model.model.load_checkpoint`, `akasha_model.multitask_eval.load_checkpoint`).

New checkpoints are stamped with `format="akasha-checkpoint"`,
`format_version=1`, `architecture` and `capabilities`, without removing the
historical fields (`kind`, `config`, `state_dict`). Legacy checkpoints are
detected from those historical fields; a payload whose architecture cannot be
detected is refused with `CheckpointFormatError`.

## Safety

Payloads load with `torch.load(..., weights_only=True)` by default. A file
that needs full pickle is refused; pass `allow_unsafe_pickle=True` only for a
checkpoint you fully trust — pickle deserialisation executes code. Loading is
local-only: nothing is downloaded. See `SECURITY.md`.

## MASK temperatures (#87)

`DecisionModel` holds one temperature per question type
(`[choice, score, noul]`). Temperatures divide logits **at inference only**
(`eval()` mode), never during training or resume, so training losses and
RLCD rewards always see raw logits. A positive temperature preserves the
ranking inside each question; `T=1` (the default for legacy checkpoints)
leaves predictions identical.

Calibrate on a held-out split that is neither train, selection nor test:

```sh
akasha-mask-calibrate runs/mask.pt data/calibration.jsonl --output runs/mask-calibrated.pt
```

This writes a checkpoint whose `config["temperature"]` and `temperature`
buffer agree, and records the fitting split under `calibration.fit_split`.
It does not change gate thresholds or deterministic rules.

## Ordinal score loss (#86)

RLCD groups permute option order for robustness. The ranked probability
(CDF) term of `proper_reward` is computed in the canonical level order:
`MaskCollator` emits `option_order` for every row and `grpo_loss` forwards
it, so permuting the displayed levels no longer changes the loss. An error
towards a far level costs more than an error towards an adjacent level.
