"""Offline Path B loading helpers — no Hub/network at inference time.

Hosts (e.g. Tauri) must ship the tokenizer/weights beside the app. This module
refuses to fetch remote assets and raises a clear, non-technical status when a
local checkpoint is missing.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# User-facing status (no raw filesystem exception as the primary message).
MODEL_ABSENT_STATUS = (
    "Le modèle de décision n'est pas installé. "
    "Ajoutez le fichier de poids fourni avec l'application, puis réessayez."
)
MODEL_ABSENT_STATUS_EN = (
    "The decision model is not installed. "
    "Add the weight file shipped with the application, then try again."
)


class ModelAbsentError(FileNotFoundError):
    """Raised when a required local checkpoint path is missing."""

    def __init__(self, path: str | Path, *, status: str = MODEL_ABSENT_STATUS) -> None:
        self.path = str(path)
        self.status = status
        super().__init__(status)

    def __str__(self) -> str:
        return self.status


@dataclass(frozen=True)
class OfflineLoadResult:
    path: str
    bytes: int
    format: str


def resolve_local_checkpoint(path: str | Path | None) -> Path:
    """Return an existing local checkpoint path or raise ModelAbsentError."""
    if path is None or str(path).strip() == "":
        raise ModelAbsentError("<unset>", status=MODEL_ABSENT_STATUS)
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise ModelAbsentError(candidate, status=MODEL_ABSENT_STATUS)
    return candidate


@contextmanager
def deny_network(enabled: bool = True) -> Iterator[None]:
    """Best-effort guard: block common outbound sockets while loading weights.

    This is a safety net for hosts that accidentally leave HF Hub online. It is
    not a sandbox. Prefer shipping weights on disk and never calling
    ``from_pretrained`` at inference time.
    """
    if not enabled:
        yield
        return
    import socket

    original = socket.socket

    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise OSError("network disabled for offline Path B load")

    socket.socket = _blocked  # type: ignore[assignment]
    # Also clear common Hub online env if set
    old_hf = os.environ.get("HF_HUB_OFFLINE")
    old_trans = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        socket.socket = original  # type: ignore[assignment]
        if old_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hf
        if old_trans is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_trans


def load_offline_torch_checkpoint(
    path: str | Path,
    *,
    map_location: str = "cpu",
    deny_net: bool = True,
) -> tuple[Any, OfflineLoadResult]:
    """Load a local ``.pt`` payload with optional network denial.

    Requires the ``torch`` extra. Does not download tokenizer or Hub weights.
    """
    resolved = resolve_local_checkpoint(path)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise RuntimeError(
            "Path B offline load requires torch; install akasha-model[torch]"
        ) from exc
    with deny_network(deny_net):
        payload = torch.load(resolved, map_location=map_location, weights_only=False)
    meta = OfflineLoadResult(
        path=str(resolved),
        bytes=resolved.stat().st_size,
        format="torch_pt",
    )
    return payload, meta


def measure_rss_mb() -> float:
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)


__all__ = [
    "MODEL_ABSENT_STATUS",
    "MODEL_ABSENT_STATUS_EN",
    "ModelAbsentError",
    "OfflineLoadResult",
    "deny_network",
    "load_offline_torch_checkpoint",
    "measure_rss_mb",
    "resolve_local_checkpoint",
]
