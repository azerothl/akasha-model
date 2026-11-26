"""Tests for the unified, safe checkpoint factory (#92) and MASK calibration (#87)."""

from __future__ import annotations

import pytest
import torch

from akasha_model.checkpoints import (
    ARCH_BYTE_MULTITASK,
    ARCH_MASK,
    ARCH_SINGLE_CHOICE,
    CheckpointFormatError,
    inspect_payload,
    load_any_checkpoint,
    load_checkpoint_payload,
    stamp_payload,
)
from akasha_model.decision import DecisionModel, TinyEncoder, save_decision_checkpoint
from akasha_model.mask_calibrate import fit_temperature


class _PickleOnly:
    """Module-level object that weights_only loading must refuse."""

    def __init__(self) -> None:
        self.value = 1


def _mask_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = DecisionModel(
        TinyEncoder(hidden_size=64, layers=1, heads=4, dropout=0.0),
        head_layers=0, dropout=0.0,
    )
    config = {
        "encoder": "tiny", "width": 64, "encoder_layers": 1,
        "encoder_heads": 4, "head_layers": 0, "dropout": 0.0,
    }
    path = tmp_path / "mask.pt"
    save_decision_checkpoint(path, model, config)
    return path


def test_inspect_detects_each_architecture_and_rejects_unknown():
    mask = inspect_payload({"kind": "mask", "config": {}, "state_dict": {}})
    assert mask.architecture == ARCH_MASK and mask.legacy
    single = inspect_payload({
        "config": {"context_tokens": 8, "option_tokens": 4}, "state_dict": {},
    })
    assert single.architecture == ARCH_SINGLE_CHOICE
    multi = inspect_payload({
        "config": {"context_tokens": 8, "question_tokens": 4}, "state_dict": {},
    })
    assert multi.architecture == ARCH_BYTE_MULTITASK
    with pytest.raises(CheckpointFormatError):
        inspect_payload({"config": {}, "state_dict": {}})
    with pytest.raises(CheckpointFormatError):
        inspect_payload({"architecture": "nope", "config": {}, "state_dict": {}})


def test_stamp_preserves_existing_fields_and_adds_metadata():
    payload = {"kind": "mask", "config": {"encoder": "tiny"}, "state_dict": {}, "custom": 1}
    stamped = stamp_payload(payload, ARCH_MASK)
    assert stamped["custom"] == 1 and stamped["kind"] == "mask"
    assert stamped["format_version"] == 1 and stamped["architecture"] == ARCH_MASK
    assert "choice" in stamped["capabilities"]


def test_load_any_checkpoint_roundtrip_mask(tmp_path):
    path = _mask_checkpoint(tmp_path)
    loaded = load_any_checkpoint(path, torch.device("cpu"))
    assert loaded.architecture == ARCH_MASK
    assert loaded.tokenizer is not None and loaded.collator is None
    assert loaded.info.format_version == 1 and not loaded.info.legacy
    assert loaded.config["temperature"] == [1.0, 1.0, 1.0]


def test_legacy_mask_checkpoint_without_stamp_still_loads(tmp_path):
    path = _mask_checkpoint(tmp_path)
    payload = load_checkpoint_payload(path)
    legacy = {key: value for key, value in payload.items()
              if key not in {"format", "format_version", "architecture", "capabilities"}}
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    loaded = load_any_checkpoint(legacy_path, torch.device("cpu"))
    assert loaded.architecture == ARCH_MASK and loaded.info.legacy


def test_missing_and_unknown_format_are_refused(tmp_path):
    with pytest.raises(CheckpointFormatError):
        load_checkpoint_payload(tmp_path / "absent.pt")
    bad = tmp_path / "bad.pt"
    torch.save({"config": {}, "state_dict": {}}, bad)
    with pytest.raises(CheckpointFormatError):
        load_any_checkpoint(bad, torch.device("cpu"))


def test_unsafe_pickle_is_refused_by_default(tmp_path):
    path = tmp_path / "evil.pt"
    torch.save({"config": {"kind": "mask"}, "state_dict": {}, "x": _PickleOnly()}, path)
    with pytest.raises(CheckpointFormatError):
        load_checkpoint_payload(path)
    payload = load_checkpoint_payload(path, allow_unsafe_pickle=True)
    assert payload["config"]["kind"] == "mask"


def test_fit_temperature_is_positive_and_reduces_nll():
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    labels = torch.tensor([0, 0, 1, 1])
    temperature = fit_temperature(logits * 4, labels)
    assert temperature > 0
    before = torch.nn.functional.cross_entropy(logits * 4, labels)
    after = torch.nn.functional.cross_entropy(logits * 4 / temperature, labels)
    assert float(after) <= float(before) + 1e-5
    assert fit_temperature(torch.empty(0, 2), torch.empty(0, dtype=torch.long)) == 1.0



@pytest.mark.parametrize("legacy", [False, True])
def test_single_choice_factory_preserves_predictions_and_temperature(tmp_path, legacy):
    from akasha_model.data import ChoiceExample
    from akasha_model.model import make_system

    device = torch.device("cpu")
    config = {
        "encoder": "tiny", "width": 8, "rank": 4, "context_tokens": 16,
        "option_tokens": 8, "temperature": 2.0,
    }
    source, collator = make_system(config, device)
    payload = {"config": config, "state_dict": source.state_dict()}
    path = tmp_path / "single.pt"
    torch.save(payload if legacy else stamp_payload(payload, ARCH_SINGLE_CHOICE), path)
    loaded = load_any_checkpoint(path, device)
    examples = [ChoiceExample("choose a", ("a", "b"), 0)]
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(loaded.collator(examples)), source(collator(examples)))
    assert loaded.architecture == ARCH_SINGLE_CHOICE
    assert loaded.info.legacy == legacy
    assert loaded.model.temperature == 2.0


@pytest.mark.parametrize("legacy", [False, True])
def test_byte_multitask_factory_preserves_typed_predictions(tmp_path, legacy):
    from akasha_model.multitask import (
        MultiQuestionCollator,
        MultiQuestionTinyScorer,
        apply_multitask_calibration,
        validate_multi,
    )

    config = {"width": 8, "rank": 4, "context_tokens": 16, "question_tokens": 16}
    config["calibration"] = {"choice_temperature": 2.0, "score_temperature": 3.0}
    source = MultiQuestionTinyScorer(8, 4, 16, 16)
    payload = {"config": config, "state_dict": source.state_dict()}
    path = tmp_path / "multitask.pt"
    torch.save(payload if legacy else stamp_payload(payload, ARCH_BYTE_MULTITASK), path)
    loaded = load_any_checkpoint(path, torch.device("cpu"))
    batch = MultiQuestionCollator(context_tokens=16, question_tokens=16)([
        validate_multi({
            "context": "choose a",
            "questions": [
                {"id": "a", "type": "choice", "instructions": "Choose", "options": ["a", "b"], "label": 0},
                {"id": "b", "type": "score", "instructions": "Rate", "levels": ["low", "high"], "label": 1},
                {"id": "c", "type": "noul", "instructions": "Is it valid?", "label": 1},
            ],
        }),
    ])
    with torch.inference_mode():
        original = apply_multitask_calibration(source(batch), config["calibration"])
        restored = apply_multitask_calibration(loaded.model(batch), loaded.config["calibration"])
        assert set(restored) == {"choice", "score", "noul"}
        for kind in restored:
            torch.testing.assert_close(restored[kind], original[kind])
    assert loaded.architecture == ARCH_BYTE_MULTITASK
    assert loaded.info.legacy == legacy


@pytest.mark.parametrize("metadata", [
    {"format": "other", "format_version": 1},
    {"format": "akasha-checkpoint", "format_version": 2},
    {"format": "akasha-checkpoint", "format_version": True},
])
def test_factory_rejects_unknown_format_metadata(metadata):
    with pytest.raises(CheckpointFormatError, match="format"):
        inspect_payload({"kind": "mask", "config": {}, "state_dict": {}, **metadata})


def test_factory_resolves_hf_encoder_without_network(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    import akasha_model.model as single_choice

    config = {"encoder": "hf", "hf_model": "test/encoder", "context_tokens": 8, "option_tokens": 4}
    payload = stamp_payload({"config": config, "state_dict": {}}, ARCH_SINGLE_CHOICE)
    checkpoint = tmp_path / "hf.pt"
    torch.save(payload, checkpoint)
    encoder_dir = tmp_path / "cached-encoder"
    encoder_dir.mkdir()

    def download(*, repo_id, local_files_only):
        assert repo_id == "test/encoder"
        assert local_files_only is True
        return encoder_dir

    def build(data, device):
        assert data["config"]["hf_model"] == str(encoder_dir.resolve())
        return None, None

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(single_choice, "build_single_choice", build)
    loaded = load_any_checkpoint(checkpoint, torch.device("cpu"))
    assert loaded.config == config
