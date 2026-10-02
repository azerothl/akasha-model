#!/usr/bin/env python3
"""Generate a synthetic mix-assistant JSONL dataset (Option A: written rules).

Labels follow versioned heuristic rules over host-style numeric descriptors.
This measures **agreement with the documented rules**, not mix quality.

No audio is generated or written. Output defaults under ``data/`` (gitignored).
Each row carries ``source`` and ``licence`` for the label + descriptor origin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

RULES_VERSION = "mix-rules-v1"
# Synthetic descriptors + rule labels — not third-party stems.
LABEL_LICENCE = "CC0-1.0"
DESCRIPTOR_LICENCE = "CC0-1.0"
SOURCE = "akasha-synthetic-mix-rules"

PRESETS = ("vocal_forward", "balanced", "bass_heavy")
GAIN_LEVELS = (
    "much quieter",
    "quieter",
    "unchanged",
    "louder",
    "much louder",
)


def _bucket(context_key: str, seed: str) -> float:
    digest = hashlib.sha256(f"{seed}:{context_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _round1(value: float) -> float:
    return round(value, 1)


def rule_preset(vocal_lufs: float, bass_lufs: float, masking: float) -> str:
    """Written rule (versioned): map descriptor geometry → preset name."""
    if masking >= 0.45 and vocal_lufs < bass_lufs - 2:
        return "vocal_forward"
    if bass_lufs <= vocal_lufs - 3:
        return "bass_heavy"
    return "balanced"


def rule_vocal_gain_level(vocal_rms: float, masking: float) -> int:
    """Ordered gain level index 0..4 from RMS + masking."""
    if masking >= 0.5 and vocal_rms < -20:
        return 4  # much louder
    if masking >= 0.35 and vocal_rms < -16:
        return 3  # louder
    if vocal_rms > -10:
        return 1  # quieter
    if vocal_rms > -8:
        return 0  # much quieter
    return 2  # unchanged


def rule_masked(masking: float) -> int:
    return 1 if masking >= 0.40 else 0


def make_row(index: int, rng: random.Random) -> dict:
    vocal_rms = _round1(rng.uniform(-28, -8))
    bass_rms = _round1(rng.uniform(-22, -6))
    vocal_lufs = _round1(vocal_rms + rng.uniform(2, 6))
    bass_lufs = _round1(bass_rms + rng.uniform(1, 5))
    masking = round(rng.uniform(0.05, 0.85), 2)
    session_id = f"synth-mix-{index:04d}"
    context = {
        "schema_version": 1,
        "session_id": session_id,
        "sample_rate_hz": 48000,
        "tracks": [
            {
                "track_id": "vocal",
                "rms_db": vocal_rms,
                "lufs": min(0.0, vocal_lufs),
                "spectral_centroid_hz": float(rng.randint(800, 3500)),
                "band_energies_db": [
                    _round1(vocal_rms - 4),
                    _round1(vocal_rms),
                    _round1(vocal_rms + 2),
                    _round1(vocal_rms - 2),
                ],
                "crest_factor_db": _round1(rng.uniform(6, 16)),
                "stereo_correlation": round(rng.uniform(-0.2, 0.4), 2),
            },
            {
                "track_id": "bass",
                "rms_db": bass_rms,
                "lufs": min(0.0, bass_lufs),
                "spectral_centroid_hz": float(rng.randint(60, 200)),
                "band_energies_db": [
                    _round1(bass_rms + 2),
                    _round1(bass_rms - 2),
                    _round1(bass_rms - 12),
                    _round1(bass_rms - 20),
                ],
                "crest_factor_db": _round1(rng.uniform(4, 12)),
                "stereo_correlation": round(rng.uniform(0.7, 1.0), 2),
            },
        ],
        "pair_masking": [
            {
                "masker_id": "bass",
                "maskee_id": "vocal",
                "masking_index": masking,
            }
        ],
    }
    preset = rule_preset(vocal_lufs, bass_lufs, masking)
    gain_label = rule_vocal_gain_level(vocal_rms, masking)
    masked_label = rule_masked(masking)
    return {
        "id": session_id,
        "context": context,
        "context_key": session_id,
        "source": SOURCE,
        "licence": LABEL_LICENCE,
        "descriptor_licence": DESCRIPTOR_LICENCE,
        "label_licence": LABEL_LICENCE,
        "rules_version": RULES_VERSION,
        "label_mode": "option_a_written_rules",
        "questions": [
            {
                "id": "preset",
                "type": "choice",
                "instructions": "Choose a mix preset for this session.",
                "options": [
                    {"name": name, "description": f"Preset {name}."}
                    for name in PRESETS
                ],
                "label": PRESETS.index(preset),
            },
            {
                "id": "vocal.gain",
                "type": "score",
                "instructions": "Ordered vocal gain adjustment.",
                "levels": list(GAIN_LEVELS),
                "label": gain_label,
            },
            {
                "id": "vocal_masked_by_bass",
                "type": "noul",
                "instructions": "Is the vocal masked by the bass?",
                "criteria": {
                    "true": "Masking index indicates the vocal is masked.",
                    "false": "The vocal remains clear relative to the bass.",
                },
                "label": masked_label,
            },
        ],
    }


def write_licence_report(rows: list[dict], path: Path) -> dict:
    by_licence = Counter(row["licence"] for row in rows)
    by_source = Counter(row["source"] for row in rows)
    report = {
        "rows": len(rows),
        "label_mode": "option_a_written_rules",
        "rules_version": RULES_VERSION,
        "measures": "agreement with written rules (not mix quality)",
        "proportions_by_licence": dict(by_licence),
        "proportions_by_source": dict(by_source),
        "audio_committed": False,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def split_rows(
    rows: list[dict],
    *,
    seed: str,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for row in rows:
        value = _bucket(row["context_key"], seed)
        if value < test_ratio:
            split = "test"
        elif value < test_ratio + validation_ratio:
            split = "validation"
        else:
            split = "train"
        splits[split].append(row)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/mix_synth"))
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-seed", default="mix-context-split-v1")
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    if args.rows < 3:
        raise SystemExit("--rows must be >= 3")
    rng = random.Random(args.seed)
    rows = [make_row(i, rng) for i in range(args.rows)]
    args.output.mkdir(parents=True, exist_ok=True)
    all_path = args.output / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    splits = split_rows(
        rows,
        seed=args.split_seed,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )
    for name, items in splits.items():
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in items:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = write_licence_report(rows, args.output / "licence_report.json")
    print(json.dumps({
        "output": str(args.output),
        "rows": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "licence_report": report,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
