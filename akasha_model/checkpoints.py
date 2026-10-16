"""Versioned checkpoint metadata and a single loading factory.

Three checkpoint families exist and keep distinct output contracts, so this
module does not pretend they share one scorer API:

* ``mask`` — :class:`akasha_model.decision.DecisionModel` (MASK markers);
* ``single_choice`` — the one-pass scorer from :mod:`akasha_model.model`;
* ``byte_multitask`` — :class:`akasha_model.multitask.MultiQuestionTinyScorer`.

New checkpoints are stamped with explicit ``format`` / ``format_version`` /
``architecture`` / ``capabilities`` metadata. Legacy checkpoints (no stamp)
are detected from their historical ``kind``/``config``/``state_dict`` shape;
an unrecognised payload is refused with :class:`CheckpointFormatError`.

Loading is safe by default: payloads are read with
``torch.load(..., weights_only=True)``. The unsafe pickle path is only taken
when the caller explicitly passes ``allow_unsafe_pickle=True`` for a trusted
legacy file. Nothing here downloads anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINT_FORMAT = "akasha-checkpoint"
FORMAT_VERSION = 1

ARCH_MASK = "mask"
ARCH_SINGLE_CHOICE = "single_choice"
ARCH_BYTE_MULTITASK = "byte_multitask"
ARCHITECTURES = (ARCH_MASK, ARCH_SINGLE_CHOICE, ARCH_BYTE_MULTITASK)

CAPABILITIES: dict[str, tuple[str, ...]] = {
    ARCH_MASK: ("choice", "score", "noul", "per_type_temperature", "rlcd"),
    ARCH_SINGLE_CHOICE: ("choice", "single_temperature"),
    ARCH_BYTE_MULTITASK: ("choice", "score", "noul", "multitask_calibration"),
}


class CheckpointFormatError(ValueError):
    """Raised when a checkpoint is missing, unknown, or unsafe to load."""


@dataclass(frozen=True)
class CheckpointInfo:
    architecture: str
    format_version: int | None
    capabilities: tuple[str, ...]
    legacy: bool


@dataclass
class LoadedCheckpoint:
    """Result of :func:`load_any_checkpoint`; only one of tokenizer/collator is set."""

    architecture: str
    model: Any
    config: dict[str, Any]
    info: CheckpointInfo
    tokenizer: Any = None
    collator: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


def inspect_payload(payload: Any) -> CheckpointInfo:
    """Detect the architecture of a checkpoint payload without building a model."""
    if not isinstance(payload, dict):
        raise CheckpointFormatError("checkpoint payload must be a mapping")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise CheckpointFormatError("checkpoint has no 'config' mapping")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise CheckpointFormatError("checkpoint has no 'state_dict' mapping")

    declared = payload.get("architecture") or payload.get("kind") or config.get("kind")
    if declared is not None:
        if declared not in ARCHITECTURES:
            raise CheckpointFormatError(f"unknown checkpoint architecture: {declared!r}")
        architecture = str(declared)
    elif "question_tokens" in config or "noul_head.weight" in state_dict:
        architecture = ARCH_BYTE_MULTITASK
    elif "context_tokens" in config and "option_tokens" in config:
        architecture = ARCH_SINGLE_CHOICE
    else:
        raise CheckpointFormatError(
            "cannot detect checkpoint architecture; expected a 'mask', "
            "'single_choice' or 'byte_multitask' checkpoint"
        )
    version = payload.get("format_version")
    return CheckpointInfo(
        architecture=architecture,
        format_version=int(version) if version is not None else None,
        capabilities=tuple(payload.get("capabilities") or CAPABILITIES[architecture]),
        legacy=version is None,
    )


def stamp_payload(payload: dict[str, Any], architecture: str) -> dict[str, Any]:
    """Return ``payload`` stamped with explicit, versioned checkpoint metadata."""
    if architecture not in ARCHITECTURES:
        raise CheckpointFormatError(f"unknown checkpoint architecture: {architecture!r}")
    stamped = dict(payload)
    stamped.update(
        format=CHECKPOINT_FORMAT,
        format_version=FORMAT_VERSION,
        architecture=architecture,
        capabilities=list(CAPABILITIES[architecture]),
    )
    return stamped


def load_checkpoint_payload(
    path: str | Path, *, allow_unsafe_pickle: bool = False,
) -> dict[str, Any]:
    """Load a local checkpoint payload, refusing unsafe pickle by default."""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise CheckpointFormatError(f"checkpoint not found: {candidate}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise RuntimeError(
            "checkpoint loading requires the torch extra; "
            'install with: pip install "akasha-model[torch]"'
        ) from exc
    try:
        payload = torch.load(candidate, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_unsafe_pickle:
            raise CheckpointFormatError(
                "checkpoint could not be loaded with weights_only=True; "
                "refusing unsafe pickle deserialisation. If you fully trust "
                "this file, pass allow_unsafe_pickle=True explicitly."
            ) from exc
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointFormatError("checkpoint payload must be a mapping")
    return payload


def load_any_checkpoint(
    path: str | Path,
    device: Any,
    *,
    allow_unsafe_pickle: bool = False,
) -> LoadedCheckpoint:
    """Load any supported checkpoint family through one factory.

    The returned :class:`LoadedCheckpoint` exposes the architecture and its
    real capabilities; callers must still use the family-specific contract
    (MASK markers, single-choice batch, or multitask batch).
    """
    payload = load_checkpoint_payload(path, allow_unsafe_pickle=allow_unsafe_pickle)
    info = inspect_payload(payload)
    config = payload["config"]
    if info.architecture == ARCH_MASK:
        from .decision import build_mask_model

        model, tokenizer = build_mask_model(payload, device)
        return LoadedCheckpoint(
            info.architecture, model, config, info, tokenizer=tokenizer,
        )
    if info.architecture == ARCH_SINGLE_CHOICE:
        from .model import build_single_choice

        model, collator = build_single_choice(payload, device)
        return LoadedCheckpoint(
            info.architecture, model, config, info, collator=collator,
        )
    from .multitask_eval import build_byte_multitask

    model = build_byte_multitask(payload, device)
    return LoadedCheckpoint(info.architecture, model, config, info)


__all__ = [
    "ARCHITECTURES",
    "ARCH_BYTE_MULTITASK",
    "ARCH_MASK",
    "ARCH_SINGLE_CHOICE",
    "CAPABILITIES",
    "CHECKPOINT_FORMAT",
    "FORMAT_VERSION",
    "CheckpointFormatError",
    "CheckpointInfo",
    "LoadedCheckpoint",
    "inspect_payload",
    "load_any_checkpoint",
    "load_checkpoint_payload",
    "stamp_payload",
]
