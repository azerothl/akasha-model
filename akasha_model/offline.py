"""Offline Path B loading helpers — no Hub/network at inference time.

Hosts (e.g. Tauri) must ship the tokenizer/weights beside the app. This module
refuses to fetch remote assets and raises a clear, non-technical status when a
local checkpoint is missing.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:
    import resource as _resource
except ImportError:  # Windows CPython has no resource module
    _resource = None

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
    """Peak resident set size in MiB (best-effort; 0.0 if unavailable)."""
    if _resource is not None:
        usage = float(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        if sys.platform == "darwin":
            return round(usage / (1024.0 * 1024.0), 2)
        return round(usage / 1024.0, 2)
    if sys.platform.startswith("win"):
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        psapi = ctypes.WinDLL("psapi")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_mem = psapi.GetProcessMemoryInfo
        get_mem.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        get_mem.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if get_mem(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1024.0 * 1024.0), 2)
    return 0.0


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
