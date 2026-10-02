"""Mix proposal batch: confirmation + one-action undo (#61)."""

from __future__ import annotations

from pathlib import Path

from akasha_model.audit import verify_envelope
from akasha_model.mix_presets import MixPresetCatalog, default_catalog_path, propose_mix_settings
from akasha_model.mix_proposal import (
    MIX_APPLY_TOOL,
    apply_changes_locally,
    build_mix_audit_envelope,
    build_mix_proposal_batch,
    mix_batch_as_tool_plan,
)
from akasha_model.primitives import ChoiceQuestion, OptionSpec, choice_result


def _ready_batch():
    catalog = MixPresetCatalog.from_json_path(default_catalog_path())
    question = ChoiceQuestion(
        "preset",
        "Choose.",
        tuple(OptionSpec(n) for n in catalog.preset_names()),
    )
    probs = {n: 0.05 for n in catalog.preset_names()}
    probs["vocal_forward"] = 1.0 - 0.05 * (len(probs) - 1)
    choice = choice_result(question, probs)
    proposal = propose_mix_settings(catalog, choice=choice, min_confidence=0.5)
    current = {
        "vocal.gain_db": 0.0,
        "vocal.pan": 0.0,
        "bass.gain_db": 0.0,
    }
    return build_mix_proposal_batch(proposal, current_settings=current)


def test_batch_ready_with_undo_inverse() -> None:
    batch = _ready_batch()
    assert batch.status == "ready"
    assert batch.changes
    # undo then re-apply restores
    settings = {c.target: c.before for c in batch.changes}
    applied = apply_changes_locally(settings, batch.changes)
    restored = apply_changes_locally(applied, batch.undo)
    assert restored == settings


def test_package_never_mutates_input_mapping() -> None:
    batch = _ready_batch()
    original = {c.target: c.before for c in batch.changes}
    snapshot = dict(original)
    apply_changes_locally(original, batch.changes)
    assert original == snapshot


def test_abstain_when_settings_unchanged() -> None:
    catalog = MixPresetCatalog.from_json_path(default_catalog_path())
    question = ChoiceQuestion(
        "preset",
        "Choose.",
        tuple(OptionSpec(n) for n in catalog.preset_names()),
    )
    probs = {n: 0.05 for n in catalog.preset_names()}
    probs["balanced"] = 1.0 - 0.05 * (len(probs) - 1)
    choice = choice_result(question, probs)
    proposal = propose_mix_settings(catalog, choice=choice, min_confidence=0.5)
    # Current already matches balanced preset values for overlapping keys
    current = dict(proposal.settings)
    batch = build_mix_proposal_batch(proposal, current_settings=current)
    assert batch.status == "abstain"
    assert batch.changes == ()


def test_tool_plan_and_audit_envelope() -> None:
    batch = _ready_batch()
    plan = mix_batch_as_tool_plan(batch)
    assert plan.status == "ready"
    assert plan.tool_name == MIX_APPLY_TOOL.name
    assert plan.arguments is not None
    assert "undo" in plan.arguments
    envelope = build_mix_audit_envelope(batch, request_id="mix-demo-1")
    assert verify_envelope(envelope)
    assert envelope.request_id == "mix-demo-1"
    assert envelope.plan["status"] == "ready"


def test_tool_requires_confirmation() -> None:
    assert MIX_APPLY_TOOL.requires_confirmation is True


def test_mix_proposal_no_torch() -> None:
    import akasha_model.mix_proposal as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "from torch" not in source
