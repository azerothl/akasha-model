"""Mix descriptors and (later) preset / proposal helpers — Path A, no torch.

Hosts compute numeric descriptors outside this package. Audio never enters
akasha-model; only rounded JSON state is validated here for Choice/Score/Noul.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .primitives import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    OptionSpec,
    ScoreLevel,
    ScoreQuestion,
)

MIX_SCHEMA_VERSION = 1
BAND_COUNT = 4
BAND_NAMES = ("low", "low_mid", "high_mid", "high")

# Bounds match akasha_model/schemas/mix-descriptors.schema.json
RMS_DB_RANGE = (-120.0, 0.0)
LUFS_RANGE = (-70.0, 0.0)
CENTROID_HZ_RANGE = (20.0, 20000.0)
BAND_DB_RANGE = (-120.0, 0.0)
CREST_DB_RANGE = (0.0, 40.0)
CORR_RANGE = (-1.0, 1.0)
MASKING_RANGE = (0.0, 1.0)
SAMPLE_RATE_RANGE = (8000, 192000)


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _in_range(value: float, low: float, high: float, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < low or number > high:
        raise ValueError(f"{name}={number} outside [{low}, {high}]")
    return number


def round_db(value: float) -> float:
    return round(float(value), 1)


def round_hz(value: float) -> float:
    return float(round(float(value)))


def round_corr(value: float) -> float:
    return round(float(value), 2)


def round_masking(value: float) -> float:
    return round(float(value), 2)


@dataclass(frozen=True)
class TrackDescriptors:
    """Per-track numeric descriptors (host-computed, model-consumable)."""

    track_id: str
    rms_db: float
    lufs: float
    spectral_centroid_hz: float
    band_energies_db: tuple[float, float, float, float]
    crest_factor_db: float
    stereo_correlation: float

    def __post_init__(self) -> None:
        _non_empty(self.track_id, "track_id")
        if not all(ch.isalnum() or ch in "._:-" for ch in self.track_id):
            raise ValueError("track_id has invalid characters")
        object.__setattr__(self, "rms_db", round_db(
            _in_range(self.rms_db, *RMS_DB_RANGE, "rms_db")))
        object.__setattr__(self, "lufs", round_db(
            _in_range(self.lufs, *LUFS_RANGE, "lufs")))
        object.__setattr__(
            self,
            "spectral_centroid_hz",
            round_hz(_in_range(
                self.spectral_centroid_hz, *CENTROID_HZ_RANGE, "spectral_centroid_hz",
            )),
        )
        if len(self.band_energies_db) != BAND_COUNT:
            raise ValueError(f"band_energies_db must have {BAND_COUNT} values")
        bands = tuple(
            round_db(_in_range(v, *BAND_DB_RANGE, f"band_energies_db[{i}]"))
            for i, v in enumerate(self.band_energies_db)
        )
        object.__setattr__(self, "band_energies_db", bands)
        object.__setattr__(self, "crest_factor_db", round_db(
            _in_range(self.crest_factor_db, *CREST_DB_RANGE, "crest_factor_db")))
        object.__setattr__(self, "stereo_correlation", round_corr(
            _in_range(self.stereo_correlation, *CORR_RANGE, "stereo_correlation")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "rms_db": self.rms_db,
            "lufs": self.lufs,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "band_energies_db": list(self.band_energies_db),
            "crest_factor_db": self.crest_factor_db,
            "stereo_correlation": self.stereo_correlation,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrackDescriptors:
        bands = data.get("band_energies_db")
        if not isinstance(bands, Sequence) or isinstance(bands, (str, bytes)):
            raise ValueError("band_energies_db must be a sequence of 4 numbers")
        return cls(
            track_id=str(data["track_id"]),
            rms_db=float(data["rms_db"]),
            lufs=float(data["lufs"]),
            spectral_centroid_hz=float(data["spectral_centroid_hz"]),
            band_energies_db=tuple(float(v) for v in bands),  # type: ignore[arg-type]
            crest_factor_db=float(data["crest_factor_db"]),
            stereo_correlation=float(data["stereo_correlation"]),
        )


@dataclass(frozen=True)
class PairMasking:
    masker_id: str
    maskee_id: str
    masking_index: float

    def __post_init__(self) -> None:
        _non_empty(self.masker_id, "masker_id")
        _non_empty(self.maskee_id, "maskee_id")
        if self.masker_id == self.maskee_id:
            raise ValueError("masker_id and maskee_id must differ")
        object.__setattr__(self, "masking_index", round_masking(
            _in_range(self.masking_index, *MASKING_RANGE, "masking_index")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "masker_id": self.masker_id,
            "maskee_id": self.maskee_id,
            "masking_index": self.masking_index,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PairMasking:
        return cls(
            masker_id=str(data["masker_id"]),
            maskee_id=str(data["maskee_id"]),
            masking_index=float(data["masking_index"]),
        )


@dataclass(frozen=True)
class MixSessionState:
    """Validated mix descriptor payload used as DecisionRequest.state."""

    session_id: str
    tracks: tuple[TrackDescriptors, ...]
    pair_masking: tuple[PairMasking, ...] = ()
    sample_rate_hz: int | None = None
    schema_version: int = MIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty(self.session_id, "session_id")
        if self.schema_version != MIX_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported mix schema_version {self.schema_version}; "
                f"this package speaks {MIX_SCHEMA_VERSION}"
            )
        if not self.tracks:
            raise ValueError("tracks must be non-empty")
        if len(self.tracks) > 64:
            raise ValueError("at most 64 tracks")
        ids = [t.track_id for t in self.tracks]
        if len(ids) != len(set(ids)):
            raise ValueError("track_id values must be unique")
        known = set(ids)
        for pair in self.pair_masking:
            if pair.masker_id not in known or pair.maskee_id not in known:
                raise ValueError(
                    f"pair_masking references unknown track "
                    f"({pair.masker_id!r}, {pair.maskee_id!r})"
                )
        if self.sample_rate_hz is not None:
            rate = self.sample_rate_hz
            if not isinstance(rate, int) or isinstance(rate, bool):
                raise ValueError("sample_rate_hz must be an integer")
            if rate < SAMPLE_RATE_RANGE[0] or rate > SAMPLE_RATE_RANGE[1]:
                raise ValueError(
                    f"sample_rate_hz={rate} outside {SAMPLE_RATE_RANGE}"
                )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "tracks": [t.to_dict() for t in self.tracks],
            "pair_masking": [p.to_dict() for p in self.pair_masking],
        }
        if self.sample_rate_hz is not None:
            payload["sample_rate_hz"] = self.sample_rate_hz
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MixSessionState:
        tracks_raw = data.get("tracks")
        if not isinstance(tracks_raw, list):
            raise ValueError("tracks must be a list")
        pairs_raw = data.get("pair_masking", [])
        if pairs_raw is None:
            pairs_raw = []
        if not isinstance(pairs_raw, list):
            raise ValueError("pair_masking must be a list")
        return cls(
            session_id=str(data["session_id"]),
            tracks=tuple(TrackDescriptors.from_dict(item) for item in tracks_raw),
            pair_masking=tuple(PairMasking.from_dict(item) for item in pairs_raw),
            sample_rate_hz=data.get("sample_rate_hz"),
            schema_version=int(data.get("schema_version", MIX_SCHEMA_VERSION)),
        )


def validate_mix_state(data: Mapping[str, Any] | MixSessionState) -> MixSessionState:
    """Validate and normalise a mix descriptor payload. Raises ValueError on OOB."""
    if isinstance(data, MixSessionState):
        return data
    return MixSessionState.from_dict(data)


def mix_schema_path() -> Path:
    return Path(__file__).resolve().parent / "schemas" / "mix-descriptors.schema.json"


def example_mix_state() -> MixSessionState:
    """Fixed fixture used by docs and tests (no audio)."""
    return MixSessionState(
        session_id="demo-mix-1",
        sample_rate_hz=48000,
        tracks=(
            TrackDescriptors(
                track_id="vocal",
                rms_db=-18.0,
                lufs=-14.0,
                spectral_centroid_hz=1800.0,
                band_energies_db=(-22.0, -16.0, -14.0, -20.0),
                crest_factor_db=12.0,
                stereo_correlation=0.15,
            ),
            TrackDescriptors(
                track_id="bass",
                rms_db=-12.0,
                lufs=-11.0,
                spectral_centroid_hz=120.0,
                band_energies_db=(-8.0, -14.0, -28.0, -40.0),
                crest_factor_db=6.0,
                stereo_correlation=0.92,
            ),
        ),
        pair_masking=(
            PairMasking(masker_id="bass", maskee_id="vocal", masking_index=0.35),
        ),
    )


def example_mix_questions() -> tuple[
    ChoiceQuestion, ScoreQuestion, ScoreQuestion, NoulQuestion,
]:
    """Canonical mix question shapes (preset choice, levels, masking noul)."""
    choice = ChoiceQuestion(
        "preset",
        "Choose a mix preset for this session.",
        (
            OptionSpec(
                "vocal_forward",
                "Lift the lead vocal relative to the bed.",
                not_for=("instrumental beds without a lead",),
            ),
            OptionSpec(
                "balanced",
                "Keep relative balances close to the current descriptors.",
            ),
            OptionSpec(
                "bass_heavy",
                "Emphasise low-end energy.",
                not_for=("sparse acoustic mixes",),
            ),
        ),
    )
    gain = ScoreQuestion(
        "vocal.gain",
        "Ordered vocal gain adjustment relative to current level.",
        (
            ScoreLevel("much quieter"),
            ScoreLevel("quieter"),
            ScoreLevel("unchanged"),
            ScoreLevel("louder"),
            ScoreLevel("much louder"),
        ),
    )
    pan = ScoreQuestion(
        "vocal.pan",
        "Ordered vocal pan position.",
        (
            ScoreLevel("hard left"),
            ScoreLevel("left"),
            ScoreLevel("center"),
            ScoreLevel("right"),
            ScoreLevel("hard right"),
        ),
    )
    masked = NoulQuestion(
        "vocal_masked_by_bass",
        "Is the vocal masked by the bass in a way that needs a corrective move?",
        criteria=(
            "Masking index and spectral overlap indicate the vocal is masked.",
            "The vocal remains intelligible relative to the bass.",
        ),
    )
    return choice, gain, pan, masked


def example_decision_request() -> DecisionRequest:
    choice, gain, pan, masked = example_mix_questions()
    return DecisionRequest(example_mix_state().to_dict(), (choice, gain, pan, masked))


def _option_from_payload(option: Any) -> OptionSpec:
    if isinstance(option, str):
        return OptionSpec(option)
    if isinstance(option, Mapping):
        return OptionSpec(
            str(option["name"]),
            str(option.get("description", "")),
            tuple(option.get("not_for", ())),
            tuple(option.get("examples", ())),
        )
    raise ValueError("Choice options must be strings or objects")


def question_from_jsonl_item(item: Mapping[str, Any]) -> (
    ChoiceQuestion | ScoreQuestion | NoulQuestion
):
    """Build a primitives question from a multitask JSONL question object."""
    kind = item.get("type")
    question_id = _non_empty(str(item.get("id", "")), "question id")
    instructions = _non_empty(str(item.get("instructions", "")), "instructions")
    if kind == "choice":
        options = item.get("options")
        if not isinstance(options, list):
            raise ValueError("Choice options must be a list")
        return ChoiceQuestion(
            question_id, instructions, tuple(_option_from_payload(o) for o in options),
        )
    if kind == "score":
        levels = item.get("levels")
        if not isinstance(levels, list):
            raise ValueError("Score levels must be a list")
        specs: list[ScoreLevel] = []
        for level in levels:
            if isinstance(level, str):
                specs.append(ScoreLevel(level))
            elif isinstance(level, Mapping):
                specs.append(ScoreLevel(str(level.get("description", level.get("name", "")))))
            else:
                raise ValueError("Score levels must be strings or objects")
        return ScoreQuestion(question_id, instructions, tuple(specs))
    if kind == "noul":
        criteria = item.get("criteria")
        pair: tuple[str, str] | None = None
        if criteria is not None:
            if not isinstance(criteria, Mapping):
                raise ValueError("Noul criteria must be an object")
            pair = (str(criteria["true"]), str(criteria["false"]))
        return NoulQuestion(question_id, instructions, criteria=pair)
    raise ValueError("question type must be choice, score or noul")


def mix_decision_request_from_row(row: Mapping[str, Any]) -> DecisionRequest:
    """Validate mix context + questions from a multitask JSONL row."""
    context = row.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("mix JSONL context must be a mix descriptor object")
    state = validate_mix_state(context)
    questions_raw = row.get("questions")
    if not isinstance(questions_raw, list) or not questions_raw:
        raise ValueError("mix JSONL needs a non-empty questions list")
    questions = tuple(question_from_jsonl_item(item) for item in questions_raw)
    return DecisionRequest(state.to_dict(), questions)


def load_mix_jsonl(path: str | Path) -> list[DecisionRequest]:
    rows: list[DecisionRequest] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(mix_decision_request_from_row(json.loads(line)))
    if not rows:
        raise ValueError(f"no mix JSONL rows in {path}")
    return rows


__all__ = [
    "BAND_COUNT",
    "BAND_NAMES",
    "MIX_SCHEMA_VERSION",
    "MixSessionState",
    "PairMasking",
    "TrackDescriptors",
    "example_decision_request",
    "example_mix_questions",
    "example_mix_state",
    "load_mix_jsonl",
    "mix_decision_request_from_row",
    "mix_schema_path",
    "question_from_jsonl_item",
    "round_corr",
    "round_db",
    "round_hz",
    "round_masking",
    "validate_mix_state",
]
