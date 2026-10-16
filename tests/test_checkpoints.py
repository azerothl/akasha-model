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
