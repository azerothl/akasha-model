"""Offline Path B load guards (#62)."""

from __future__ import annotations

from pathlib import Path

import pytest

from akasha_model.offline import (
    MODEL_ABSENT_STATUS,
    ModelAbsentError,
    deny_network,
    resolve_local_checkpoint,
)

GATE_TINY = Path(__file__).resolve().parents[1] / "examples" / "gate" / "checkpoints" / "gate-tiny.pt"


def test_missing_checkpoint_user_status() -> None:
    with pytest.raises(ModelAbsentError) as excinfo:
        resolve_local_checkpoint(Path("/no/such/model.pt"))
    assert str(excinfo.value) == MODEL_ABSENT_STATUS
    assert "torch.load" not in str(excinfo.value).lower()


def test_resolve_existing_demo_checkpoint() -> None:
    assert GATE_TINY.is_file()
    assert resolve_local_checkpoint(GATE_TINY) == GATE_TINY


def test_deny_network_blocks_socket() -> None:
    import socket

    with deny_network(True):
        with pytest.raises(OSError, match="network disabled"):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_offline_load_roundtrip_when_torch_available() -> None:
    torch = pytest.importorskip("torch")
    from akasha_model.offline import load_offline_torch_checkpoint, measure_rss_mb

    rss0 = measure_rss_mb()
    payload, meta = load_offline_torch_checkpoint(GATE_TINY, deny_net=True)
    assert meta.bytes > 0
    assert isinstance(payload, dict)
    assert "state_dict" in payload or any(
        isinstance(v, torch.Tensor) for v in payload.values()
    )
    assert measure_rss_mb() >= rss0


def test_offline_module_importable_without_torch() -> None:
    import akasha_model.offline as offline

    source = Path(offline.__file__).read_text(encoding="utf-8")
    # Top-level import must not require torch; torch is imported inside load_*
    assert "import torch" not in source.split("def load_offline_torch_checkpoint")[0]
