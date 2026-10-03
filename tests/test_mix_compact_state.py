"""Compact mix MASK state and T4 verdict helper (#82)."""

from __future__ import annotations

from akasha_model.mix import (
    COMPACT_MIX_STATE_VERSION,
    example_mix_state,
    format_compact_mix_state,
    looks_like_mix_descriptors,
    majority_sensitivity_verdict,
)
from akasha_model.sequence import serialize_state


def test_compact_state_keeps_db_and_is_shorter_than_json() -> None:
    state = example_mix_state().to_dict()
    compact = format_compact_mix_state(state)
    raw = serialize_state(state)
    assert compact.startswith(COMPACT_MIX_STATE_VERSION)
    assert "rms_db=-18" in compact
    assert "lufs=-14" in compact
    assert "mask bass>vocal" in compact
    assert len(compact) < len(raw)
    assert looks_like_mix_descriptors(state)


def test_compact_state_renders_t4_nulls() -> None:
    state = example_mix_state().to_dict()
    state["tracks"][0]["rms_db"] = None
    state["pair_masking"][0]["masking_index"] = None
    compact = format_compact_mix_state(state)
    assert "rms_db=null" in compact
    assert "mask bass>vocal null" in compact


def test_verdict_rejects_majority_clone() -> None:
    head = {"accuracy": 0.52}
    majority_heads = {"choice": head, "score": {"accuracy": 0.64}, "noul": {"accuracy": 0.5}}
    report = {
        "encoders": {
            "majority_prior": {
                "baseline": {"heads": majority_heads},
            },
            "mask:demo.pt": {
                "baseline": {"heads": majority_heads},
                "numbers_removed": {"heads": majority_heads},
                "numbers_perturbed": {"heads": majority_heads},
            },
        }
    }
    verdict = majority_sensitivity_verdict(report)
    assert verdict["ok"] is False
    assert verdict["beats_majority"] is False


def test_verdict_accepts_number_drop() -> None:
    majority = {
        "choice": {"accuracy": 0.52},
        "score": {"accuracy": 0.64},
        "noul": {"accuracy": 0.50},
    }
    strong = {
        "choice": {"accuracy": 0.80},
        "score": {"accuracy": 0.64},
        "noul": {"accuracy": 0.50},
    }
    dropped = {
        "choice": {"accuracy": 0.52},
        "score": {"accuracy": 0.64},
        "noul": {"accuracy": 0.50},
    }
    report = {
        "encoders": {
            "majority_prior": {"baseline": {"heads": majority}},
            "mask:mix.pt": {
                "baseline": {"heads": strong},
                "numbers_removed": {"heads": dropped},
                "numbers_perturbed": {"heads": strong},
            },
        }
    }
    verdict = majority_sensitivity_verdict(report)
    assert verdict["ok"] is True
    assert verdict["heads"]["choice"]["gap_vs_majority"] >= 0.08
