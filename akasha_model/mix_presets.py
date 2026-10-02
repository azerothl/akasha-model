"""Discrete mix preset catalogue: ordered levels → explicit numeric values.

The model only selects bounded Choice names or Score indices. This module is the
deterministic bridge to gain/pan/EQ/compressor numbers. No side effects: it
never applies settings to a DAW.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .primitives import ChoiceResult, ScoreResult

CATALOG_VERSION = 1


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class ScoreLadderLevel:
    description: str
    value: float

    def __post_init__(self) -> None:
        _non_empty(self.description, "score level description")
        object.__setattr__(self, "value", _finite(self.value, "score level value"))


@dataclass(frozen=True)
class ScoreLadder:
    question_id: str
    target: str
    unit: str
    levels: tuple[ScoreLadderLevel, ...]

    def __post_init__(self) -> None:
        _non_empty(self.question_id, "score ladder question_id")
        _non_empty(self.target, "score ladder target")
        _non_empty(self.unit, "score ladder unit")
        if not 2 <= len(self.levels) <= 10:
            raise ValueError("score ladder requires between 2 and 10 levels")

    def value_at(self, level_index: int) -> float:
        if not isinstance(level_index, int) or isinstance(level_index, bool):
            raise ValueError("level_index must be an int")
        if not 0 <= level_index < len(self.levels):
            raise ValueError(
                f"level_index {level_index} outside [0, {len(self.levels) - 1}]"
            )
        return self.levels[level_index].value


@dataclass(frozen=True)
class MixPreset:
    name: str
    description: str
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        _non_empty(self.name, "preset name")
        object.__setattr__(self, "values", dict(self.values))


@dataclass(frozen=True)
class MixSettingsProposal:
    """Non-executing settings map derived from model results + catalogue."""

    status: str  # "ready" | "abstain"
    reason: str
    settings: Mapping[str, Any]
    sources: Mapping[str, str]

    @property
    def executable(self) -> bool:
        return self.status == "ready"


@dataclass(frozen=True)
class MixPresetCatalog:
    """Associates Choice options and Score levels with explicit DSP values."""

    catalog_version: int
    presets: Mapping[str, MixPreset]
    score_ladders: Mapping[str, ScoreLadder]
    bounds: Mapping[str, tuple[float, float]]
    discretization: Mapping[str, Any]
    description: str = ""

    def __post_init__(self) -> None:
        if self.catalog_version != CATALOG_VERSION:
            raise ValueError(
                f"unsupported catalog_version {self.catalog_version}; "
                f"this package speaks {CATALOG_VERSION}"
            )
        if not self.presets:
            raise ValueError("catalogue needs at least one preset")
        object.__setattr__(self, "presets", dict(self.presets))
        object.__setattr__(self, "score_ladders", dict(self.score_ladders))
        object.__setattr__(self, "bounds", dict(self.bounds))
        object.__setattr__(self, "discretization", dict(self.discretization))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MixPresetCatalog:
        presets_raw = data.get("presets")
        if not isinstance(presets_raw, Mapping) or not presets_raw:
            raise ValueError("presets must be a non-empty object")
        presets = {
            name: MixPreset(
                name=name,
                description=str(item.get("description", "")),
                values=dict(item.get("values", {})),
            )
            for name, item in presets_raw.items()
        }
        ladders_raw = data.get("score_ladders", {})
        if not isinstance(ladders_raw, Mapping):
            raise ValueError("score_ladders must be an object")
        ladders: dict[str, ScoreLadder] = {}
        for question_id, item in ladders_raw.items():
            levels_raw = item.get("levels")
            if not isinstance(levels_raw, list):
                raise ValueError(f"score ladder {question_id!r} needs levels")
            levels = tuple(
                ScoreLadderLevel(str(level["description"]), float(level["value"]))
                for level in levels_raw
            )
            ladders[question_id] = ScoreLadder(
                question_id=question_id,
                target=str(item["target"]),
                unit=str(item.get("unit", "")),
                levels=levels,
            )
        bounds_raw = data.get("bounds", {})
        if not isinstance(bounds_raw, Mapping):
            raise ValueError("bounds must be an object")
        bounds: dict[str, tuple[float, float]] = {}
        for key, pair in bounds_raw.items():
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise ValueError(f"bounds[{key!r}] must be [low, high]")
            bounds[key] = (_finite(pair[0], f"bounds[{key}].low"),
                           _finite(pair[1], f"bounds[{key}].high"))
        return cls(
            catalog_version=int(data.get("catalog_version", CATALOG_VERSION)),
            presets=presets,
            score_ladders=ladders,
            bounds=bounds,
            discretization=dict(data.get("discretization", {})),
            description=str(data.get("description", "")),
        )

    @classmethod
    def from_json_path(cls, path: str | Path) -> MixPresetCatalog:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("catalogue JSON must be an object")
        return cls.from_dict(payload)

    def preset_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.presets))

    def values_for_preset(self, name: str) -> Mapping[str, Any]:
        preset = self.presets.get(name)
        if preset is None:
            raise KeyError(f"unknown preset {name!r}")
        self._check_values(preset.values)
        return dict(preset.values)

    def values_for_score_level(self, question_id: str, level_index: int) -> tuple[str, float]:
        ladder = self.score_ladders.get(question_id)
        if ladder is None:
            raise KeyError(f"unknown score ladder {question_id!r}")
        value = ladder.value_at(level_index)
        self._check_scalar(ladder.target, value)
        return ladder.target, value

    def _bound_key_for_target(self, target: str) -> str | None:
        if target.endswith(".gain_db") or target == "gain_db":
            return "gain_db"
        if target.endswith(".pan") or target == "pan":
            return "pan"
        if target.endswith(".eq_db") or target == "eq_db":
            return "eq_db"
        if target.endswith(".compressor_ratio") or target == "compressor_ratio":
            return "compressor_ratio"
        if target.endswith(".compressor_threshold_db") or target == "compressor_threshold_db":
            return "compressor_threshold_db"
        return None

    def _check_scalar(self, target: str, value: float) -> None:
        key = self._bound_key_for_target(target)
        if key is None or key not in self.bounds:
            return
        low, high = self.bounds[key]
        if value < low or value > high:
            raise ValueError(f"{target}={value} outside bounds [{low}, {high}]")

    def _check_values(self, values: Mapping[str, Any]) -> None:
        for target, raw in values.items():
            if isinstance(raw, list):
                key = self._bound_key_for_target(target)
                if key == "eq_db" and "eq_db" in self.bounds:
                    low, high = self.bounds["eq_db"]
                    for index, item in enumerate(raw):
                        number = _finite(item, f"{target}[{index}]")
                        if number < low or number > high:
                            raise ValueError(
                                f"{target}[{index}]={number} outside [{low}, {high}]"
                            )
                continue
            self._check_scalar(target, _finite(raw, target))


def _nearest_level_index(probabilities: Mapping[int, float]) -> int:
    return max(probabilities, key=probabilities.get)


def propose_mix_settings(
    catalog: MixPresetCatalog,
    *,
    choice: ChoiceResult | None = None,
    scores: Sequence[ScoreResult] = (),
    min_confidence: float,
    current_settings: Mapping[str, Any] | None = None,
) -> MixSettingsProposal:
    """Map model results to explicit settings. Pure: no I/O, no DAW apply.

    ``min_confidence`` is required (no package default). Hosts should set it
    after the first measurement campaign (see issue T4).
    """
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    settings: dict[str, Any] = dict(current_settings or {})
    sources: dict[str, str] = {}

    if choice is not None:
        if choice.abstained or choice.selected is None:
            return MixSettingsProposal(
                status="abstain",
                reason="choice abstained",
                settings={},
                sources={},
            )
        if choice.confidence < min_confidence:
            return MixSettingsProposal(
                status="abstain",
                reason=(
                    f"choice confidence {choice.confidence:.3f} "
                    f"below min_confidence {min_confidence:.3f}"
                ),
                settings={},
                sources={},
            )
        try:
            preset_values = catalog.values_for_preset(choice.selected)
        except KeyError as exc:
            return MixSettingsProposal(
                status="abstain",
                reason=str(exc),
                settings={},
                sources={},
            )
        settings.update(preset_values)
        for key in preset_values:
            sources[key] = f"preset:{choice.selected}"

    for score in scores:
        if score.confidence < min_confidence:
            return MixSettingsProposal(
                status="abstain",
                reason=(
                    f"score {score.question_id!r} confidence {score.confidence:.3f} "
                    f"below min_confidence {min_confidence:.3f}"
                ),
                settings={},
                sources={},
            )
        level = _nearest_level_index(score.probabilities)
        try:
            target, value = catalog.values_for_score_level(score.question_id, level)
        except KeyError as exc:
            return MixSettingsProposal(
                status="abstain",
                reason=str(exc),
                settings={},
                sources={},
            )
        settings[target] = value
        sources[target] = f"score:{score.question_id}:{level}"

    if not settings:
        return MixSettingsProposal(
            status="abstain",
            reason="no choice or score results provided",
            settings={},
            sources={},
        )
    return MixSettingsProposal(
        status="ready",
        reason="mapped from catalogue",
        settings=settings,
        sources=sources,
    )


def default_catalog_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "mix" / "presets.v1.json"


__all__ = [
    "CATALOG_VERSION",
    "MixPreset",
    "MixPresetCatalog",
    "MixSettingsProposal",
    "ScoreLadder",
    "ScoreLadderLevel",
    "default_catalog_path",
    "propose_mix_settings",
]
