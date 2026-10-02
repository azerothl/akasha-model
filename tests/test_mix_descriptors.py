"""Mix descriptor schema and JSONL contract (#56)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from akasha_model.mix import (
    MIX_SCHEMA_VERSION,
    MixSessionState,
    PairMasking,
    TrackDescriptors,
    example_decision_request,
    example_mix_state,
    load_mix_jsonl,
    mix_decision_request_from_row,
    mix_schema_path,
    validate_mix_state,
)
from akasha_model.primitives import ChoiceQuestion, NoulQuestion, ScoreQuestion

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "mix" / "sample_questions.jsonl"


def test_schema_file_is_published() -> None:
    path = mix_schema_path()
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["properties"]["schema_version"]["const"] == MIX_SCHEMA_VERSION
    assert payload["$defs"]["track"]["properties"]["rms_db"]["minimum"] == -120


def test_example_state_roundtrip() -> None:
    state = example_mix_state()
    again = validate_mix_state(state.to_dict())
    assert again == state
    assert again.tracks[0].rms_db == -18.0


def test_example_decision_request_uses_primitives() -> None:
    request = example_decision_request()
    assert isinstance(request.state, dict)
    assert request.state["schema_version"] == MIX_SCHEMA_VERSION
    kinds = [q.primitive for q in request.questions]
    assert kinds.count("choice") >= 1
    assert kinds.count("score") >= 1
    assert kinds.count("noul") >= 1
    assert all(
        isinstance(q, (ChoiceQuestion, ScoreQuestion, NoulQuestion))
        for q in request.questions
    )


def test_sample_jsonl_validates_via_primitives() -> None:
    rows = load_mix_jsonl(SAMPLE)
    assert len(rows) == 1
    request = rows[0]
    assert isinstance(request.questions[0], ChoiceQuestion)
    assert isinstance(request.questions[1], ScoreQuestion)
    assert isinstance(request.questions[-1], NoulQuestion)
    validate_mix_state(request.state)  # type: ignore[arg-type]


def test_rejects_rms_out_of_bounds() -> None:
    data = example_mix_state().to_dict()
    data["tracks"][0]["rms_db"] = 3.0
    with pytest.raises(ValueError, match="rms_db"):
        validate_mix_state(data)


def test_rejects_wrong_band_arity() -> None:
    data = example_mix_state().to_dict()
    data["tracks"][0]["band_energies_db"] = [-10.0, -10.0]
    with pytest.raises(ValueError, match="band_energies_db"):
        validate_mix_state(data)


def test_rejects_unknown_masking_pair() -> None:
    data = example_mix_state().to_dict()
    data["pair_masking"] = [
        {"masker_id": "vocal", "maskee_id": "ghost", "masking_index": 0.2},
    ]
    with pytest.raises(ValueError, match="unknown track"):
        validate_mix_state(data)


def test_rejects_correlation_oob() -> None:
    with pytest.raises(ValueError, match="stereo_correlation"):
        TrackDescriptors(
            track_id="x",
            rms_db=-20,
            lufs=-18,
            spectral_centroid_hz=1000,
            band_energies_db=(-20, -20, -20, -20),
            crest_factor_db=8,
            stereo_correlation=1.5,
        )


def test_rejects_masking_oob() -> None:
    with pytest.raises(ValueError, match="masking_index"):
        PairMasking("a", "b", 1.2)


def test_mix_module_has_no_torch_import() -> None:
    import akasha_model.mix as mix

    source = Path(mix.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source


def test_row_requires_object_context() -> None:
    with pytest.raises(ValueError, match="mix descriptor object"):
        mix_decision_request_from_row({
            "context": "not-json-descriptors",
            "questions": [],
        })


def test_dataclass_constructs() -> None:
    state = MixSessionState(
        session_id="s",
        tracks=(
            TrackDescriptors(
                "kick", -10, -12, 80, (-6, -20, -35, -45), 8, 0.99,
            ),
        ),
    )
    assert state.to_dict()["tracks"][0]["track_id"] == "kick"
