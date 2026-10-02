"""Demo: mix proposal batch with confirmation + undo (no DAW side effects)."""

from __future__ import annotations

import json
from pathlib import Path

from akasha_model.audit import verify_envelope
from akasha_model.mix_presets import MixPresetCatalog, propose_mix_settings
from akasha_model.mix_proposal import (
    apply_changes_locally,
    build_mix_audit_envelope,
    build_mix_proposal_batch,
)
from akasha_model.primitives import ChoiceQuestion, OptionSpec, choice_result

ROOT = Path(__file__).resolve().parent


def main() -> None:
    catalog = MixPresetCatalog.from_json_path(ROOT / "presets.v1.json")
    question = ChoiceQuestion(
        "preset",
        "Choose a mix preset.",
        tuple(OptionSpec(n) for n in catalog.preset_names()),
    )
    probs = {n: 0.05 for n in catalog.preset_names()}
    probs["vocal_forward"] = 1.0 - 0.05 * (len(probs) - 1)
    choice = choice_result(question, probs)
    proposal = propose_mix_settings(catalog, choice=choice, min_confidence=0.55)
    current = {"vocal.gain_db": 0.0, "bass.gain_db": 0.0, "vocal.pan": 0.0}
    batch = build_mix_proposal_batch(proposal, current_settings=current)
    print(json.dumps(batch.to_dict(), indent=2, sort_keys=True))
    envelope = build_mix_audit_envelope(batch, request_id="demo-mix-batch")
    assert verify_envelope(envelope)
    print("audit_ok", envelope.content_sha256[:16], "…")
    # Host-side preview only — package does not apply.
    if batch.executable:
        preview = apply_changes_locally(current, batch.changes)
        undone = apply_changes_locally(preview, batch.undo)
        expected = {**current, **{c.target: c.before for c in batch.changes}}
        assert undone == expected
        print("preview_keys", sorted(preview))
        print("undo_restores", True)


if __name__ == "__main__":
    main()
