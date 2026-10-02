"""MixPresetCatalog: discrete levels → explicit DSP values (#57)."""

from __future__ import annotations

from pathlib import Path

import pytest

from akasha_model.mix_presets import (
    MixPresetCatalog,
    default_catalog_path,
    propose_mix_settings,
)
from akasha_model.primitives import (
    ChoiceQuestion,
    OptionSpec,
    ScoreLevel,
    ScoreQuestion,
    choice_result,
    score_result,
)

CATALOG = default_catalog_path()


@pytest.fixture(scope="module")
def catalog() -> MixPresetCatalog:
    return MixPresetCatalog.from_json_path(CATALOG)


def test_catalog_loads_and_lists_presets(catalog: MixPresetCatalog) -> None:
    names = catalog.preset_names()
    assert "vocal_forward" in names
    assert "balanced" in names
    values = catalog.values_for_preset("balanced")
    assert values["vocal.gain_db"] == 0.0
    assert values["vocal.pan"] == 0.0


def test_score_ladder_maps_index(catalog: MixPresetCatalog) -> None:
    target, value = catalog.values_for_score_level("vocal.gain", 3)
    assert target == "vocal.gain_db"
    assert value == 3.0


def test_score_ladder_rejects_oob_index(catalog: MixPresetCatalog) -> None:
    with pytest.raises(ValueError, match="level_index"):
        catalog.values_for_score_level("vocal.gain", 99)


def test_bounds_reject_extreme_preset_mutation(catalog: MixPresetCatalog) -> None:
    data = {
        "catalog_version": 1,
        "presets": {
            "broken": {
                "description": "oob",
                "values": {"vocal.gain_db": 100.0},
            }
        },
        "score_ladders": {},
        "bounds": {"gain_db": [-24.0, 24.0]},
        "discretization": {},
    }
    broken = MixPresetCatalog.from_dict(data)
    with pytest.raises(ValueError, match="outside bounds"):
        broken.values_for_preset("broken")


def test_propose_ready_from_choice(catalog: MixPresetCatalog) -> None:
    question = ChoiceQuestion(
        "preset",
        "Choose.",
        tuple(OptionSpec(n) for n in catalog.preset_names()),
    )
    probs = {n: 0.05 for n in catalog.preset_names()}
    probs["vocal_forward"] = 1.0 - 0.05 * (len(probs) - 1)
    choice = choice_result(question, probs)
    proposal = propose_mix_settings(catalog, choice=choice, min_confidence=0.5)
    assert proposal.status == "ready"
    assert proposal.settings["vocal.gain_db"] == 3.0
    assert proposal.sources["vocal.gain_db"].startswith("preset:")


def test_propose_abstains_on_low_confidence(catalog: MixPresetCatalog) -> None:
    question = ChoiceQuestion(
        "preset",
        "Choose.",
        tuple(OptionSpec(n) for n in catalog.preset_names()),
    )
    # Near-uniform → low confidence
    n = len(catalog.preset_names())
    probs = {name: 1.0 / n for name in catalog.preset_names()}
    choice = choice_result(question, probs)
    proposal = propose_mix_settings(catalog, choice=choice, min_confidence=0.9)
    assert proposal.status == "abstain"
    assert proposal.settings == {}


def test_propose_requires_min_confidence(catalog: MixPresetCatalog) -> None:
    with pytest.raises(TypeError):
        propose_mix_settings(catalog, choice=None)  # type: ignore[call-arg]


def test_propose_from_score_ladder(catalog: MixPresetCatalog) -> None:
    question = ScoreQuestion(
        "vocal.gain",
        "Gain.",
        tuple(ScoreLevel(level.description) for level in catalog.score_ladders["vocal.gain"].levels),
    )
    probs = {i: 0.05 for i in range(5)}
    probs[4] = 0.8
    score = score_result(question, probs)
    proposal = propose_mix_settings(catalog, scores=[score], min_confidence=0.4)
    assert proposal.status == "ready"
    assert proposal.settings["vocal.gain_db"] == 6.0


def test_catalog_path_exists() -> None:
    assert CATALOG.is_file()


def test_mix_presets_no_torch() -> None:
    import akasha_model.mix_presets as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source
