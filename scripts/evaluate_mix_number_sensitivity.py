#!/usr/bin/env python3
"""Number-sensitivity evaluation protocol for mix descriptors (issue #59).

Controls (same questions, altered context):
  - baseline: original descriptors
  - shuffled_context: permute track blocks across rows
  - numbers_perturbed: multiplicative/additive noise on numeric fields
  - numbers_removed: numeric fields set to null (scorer must tolerate / degrade)

Reports per head (choice / score / noul): accuracy, ECE, coverage/risk, plus
wall-clock latency and process RSS.

Encoders in this script (CPU-friendly, no Hub download):
  - rule_prior: written rules from ``generate_mix_dataset.py`` (teacher)
  - majority_prior: constant prior (balanced / unchanged / not-masked)

BERT / ModernBERT and GPU 16GB runs are documented as optional follow-ups when
hardware and offline weights are available (see report ``device_notes`` /
``docs/mix-number-sensitivity.md``).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
ROOT = Path(__file__).resolve().parents[1]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ece(confidence: list[float], correct: list[float], bins: int = 10) -> float:
    if not confidence:
        return 0.0
    total = len(confidence)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            i for i, value in enumerate(confidence)
            if lower <= value < upper or (index == bins - 1 and value == 1.0)
        ]
        if selected:
            accuracy = _mean([correct[i] for i in selected])
            average_confidence = _mean([confidence[i] for i in selected])
            error += len(selected) / total * abs(accuracy - average_confidence)
    return error


def _coverage_risk(confidence: list[float], correct: list[float]) -> dict[str, dict[str, float]]:
    result = {}
    for threshold in THRESHOLDS:
        selected = [i for i, value in enumerate(confidence) if value >= threshold]
        coverage = len(selected) / max(len(confidence), 1)
        accuracy = _mean([correct[i] for i in selected]) if selected else 0.0
        result[str(threshold)] = {
            "coverage": coverage,
            "accuracy": accuracy,
            "risk": 1.0 - accuracy if selected else 0.0,
        }
    return result


def _rss_mb() -> float:
    # Linux: KiB
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_rules_module():
    spec = importlib.util.spec_from_file_location(
        "generate_mix_dataset", ROOT / "scripts" / "generate_mix_dataset.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def perturb_numbers(context: dict, rng: random.Random, scale: float = 0.15) -> dict:
    payload = copy.deepcopy(context)
    for track in payload.get("tracks", []):
        for key in ("rms_db", "lufs", "crest_factor_db"):
            if isinstance(track.get(key), (int, float)):
                track[key] = round(float(track[key]) * (1 + rng.uniform(-scale, scale)), 1)
        if isinstance(track.get("spectral_centroid_hz"), (int, float)):
            track["spectral_centroid_hz"] = float(
                max(20, min(20000, round(
                    float(track["spectral_centroid_hz"]) * (1 + rng.uniform(-scale, scale)),
                )))
            )
        if isinstance(track.get("band_energies_db"), list):
            track["band_energies_db"] = [
                round(float(v) * (1 + rng.uniform(-scale, scale)), 1)
                if isinstance(v, (int, float)) else v
                for v in track["band_energies_db"]
            ]
        if isinstance(track.get("stereo_correlation"), (int, float)):
            track["stereo_correlation"] = round(
                max(-1.0, min(1.0, float(track["stereo_correlation"]) + rng.uniform(-scale, scale))),
                2,
            )
    for pair in payload.get("pair_masking", []):
        if isinstance(pair.get("masking_index"), (int, float)):
            pair["masking_index"] = round(
                max(0.0, min(1.0, float(pair["masking_index"]) + rng.uniform(-scale, scale))),
                2,
            )
    return payload


def remove_numbers(context: dict) -> dict:
    payload = copy.deepcopy(context)
    for track in payload.get("tracks", []):
        for key in ("rms_db", "lufs", "spectral_centroid_hz", "crest_factor_db", "stereo_correlation"):
            if key in track:
                track[key] = None
        if "band_energies_db" in track:
            track["band_energies_db"] = [None for _ in track["band_energies_db"]]
    for pair in payload.get("pair_masking", []):
        if "masking_index" in pair:
            pair["masking_index"] = None
    return payload


def shuffle_track_blocks(rows: list[dict], rng: random.Random) -> list[dict]:
    out = [copy.deepcopy(row) for row in rows]
    tracks = [row["context"]["tracks"] for row in out]
    order = list(range(len(tracks)))
    rng.shuffle(order)
    for index, row in enumerate(out):
        row["context"]["tracks"] = copy.deepcopy(tracks[order[index]])
        ids = {t["track_id"] for t in row["context"]["tracks"]}
        row["context"]["pair_masking"] = [
            p for p in row["context"].get("pair_masking", [])
            if p.get("masker_id") in ids and p.get("maskee_id") in ids
        ]
    return out


def build_controls(rows: list[dict], seed: int = 0) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    baseline = [copy.deepcopy(r) for r in rows]
    perturbed = []
    removed = []
    for row in baseline:
        item = copy.deepcopy(row)
        item["context"] = perturb_numbers(item["context"], rng)
        perturbed.append(item)
        item2 = copy.deepcopy(row)
        item2["context"] = remove_numbers(item2["context"])
        removed.append(item2)
    return {
        "baseline": baseline,
        "shuffled_context": shuffle_track_blocks(baseline, random.Random(seed + 1)),
        "numbers_perturbed": perturbed,
        "numbers_removed": removed,
    }


def _labels(row: dict) -> dict[str, int]:
    return {q["type"]: int(q["label"]) for q in row["questions"]}


def _summarize(store: dict[str, dict[str, list]], *, elapsed_s: float, n_rows: int) -> dict[str, Any]:
    heads = {}
    for head, values in store.items():
        heads[head] = {
            "examples": len(values.get("correct", [])),
            "accuracy": _mean(values.get("correct", [])),
            "ece": _ece(values.get("confidence", []), values.get("correct", [])),
            "mean_confidence": _mean(values.get("confidence", [])),
            "coverage_risk": _coverage_risk(
                values.get("confidence", []), values.get("correct", []),
            ),
        }
    return {
        "heads": heads,
        "latency_s": round(elapsed_s, 4),
        "latency_ms_per_row": round(1000.0 * elapsed_s / max(n_rows, 1), 3),
        "rss_mb": _rss_mb(),
    }


def _extract_numbers(row: dict) -> tuple[float, float, float, float]:
    context = row["context"]
    tracks = {t["track_id"]: t for t in context.get("tracks", [])}
    vocal = tracks.get("vocal", {})
    bass = tracks.get("bass", {})
    masking = 0.0
    for pair in context.get("pair_masking", []):
        if pair.get("maskee_id") == "vocal" and isinstance(pair.get("masking_index"), (int, float)):
            masking = float(pair["masking_index"])
    vocal_lufs = float(vocal["lufs"]) if isinstance(vocal.get("lufs"), (int, float)) else -70.0
    bass_lufs = float(bass["lufs"]) if isinstance(bass.get("lufs"), (int, float)) else -70.0
    vocal_rms = float(vocal["rms_db"]) if isinstance(vocal.get("rms_db"), (int, float)) else -120.0
    return vocal_lufs, bass_lufs, vocal_rms, masking


def evaluate_rule_prior(rows: list[dict], rules) -> dict[str, Any]:
    store: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    t0 = time.perf_counter()
    for row in rows:
        vocal_lufs, bass_lufs, vocal_rms, masking = _extract_numbers(row)
        preset = rules.rule_preset(vocal_lufs, bass_lufs, masking)
        gain = rules.rule_vocal_gain_level(vocal_rms, masking)
        masked = rules.rule_masked(masking)
        labels = _labels(row)
        store["choice"]["correct"].append(float(rules.PRESETS.index(preset) == labels["choice"]))
        store["choice"]["confidence"].append(1.0)
        store["score"]["correct"].append(float(gain == labels["score"]))
        store["score"]["confidence"].append(1.0)
        store["noul"]["correct"].append(float(masked == labels["noul"]))
        store["noul"]["confidence"].append(1.0)
    return _summarize(store, elapsed_s=time.perf_counter() - t0, n_rows=len(rows))


def evaluate_majority_prior(rows: list[dict], rules) -> dict[str, Any]:
    """Constant prior: balanced / unchanged / not-masked with fixed confidence."""
    store: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    t0 = time.perf_counter()
    choice_pred = rules.PRESETS.index("balanced")
    for row in rows:
        labels = _labels(row)
        store["choice"]["correct"].append(float(choice_pred == labels["choice"]))
        store["choice"]["confidence"].append(0.34)
        store["score"]["correct"].append(float(2 == labels["score"]))
        store["score"]["confidence"].append(0.34)
        store["noul"]["correct"].append(float(0 == labels["noul"]))
        store["noul"]["confidence"].append(0.5)
    return _summarize(store, elapsed_s=time.perf_counter() - t0, n_rows=len(rows))


def french_probe_note() -> dict[str, Any]:
    return {
        "status": "documented_not_benchmarked",
        "numeric_descriptors": (
            "Units are language-agnostic; FR and EN hosts share mix-descriptors schema v1."
        ),
        "question_instructions": (
            "French instructions were not accuracy-benchmarked in this CPU report. "
            "A smoke check (see tests) only asserts that FR instruction strings still "
            "build valid Score/Choice/Noul primitives."
        ),
        "follow_up": "Re-run controls with FR instructions once a trained checkpoint exists.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/mix_number_sensitivity.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--encoders",
        default="rule_prior,majority_prior",
        help="comma list from: rule_prior,majority_prior",
    )
    args = parser.parse_args()

    data_path = args.data
    rows = _load_jsonl(data_path / "test.jsonl" if data_path.is_dir() else data_path)
    controls = build_controls(rows, seed=args.seed)
    rules = _load_rules_module()

    device_notes = [
        f"requested_device={args.device}",
        "This run publishes CPU metrics for rule_prior and majority_prior.",
        "GPU 16GB + BERT/ModernBERT measurements are NOT included here "
        "(hardware / Hub weights blocked in default CI).",
    ]
    cuda = False
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    device_notes.append(f"torch_cuda_available={cuda}")

    report: dict[str, Any] = {
        "protocol": "mix-number-sensitivity-v1",
        "data": str(data_path),
        "rows": len(rows),
        "device": "cpu",
        "device_notes": device_notes,
        "french": french_probe_note(),
        "encoders": {},
    }
    dispatch = {
        "rule_prior": evaluate_rule_prior,
        "majority_prior": evaluate_majority_prior,
    }
    for name in [e.strip() for e in args.encoders.split(",") if e.strip()]:
        if name not in dispatch:
            report["encoders"][name] = {"error": f"unknown encoder {name}"}
            continue
        report["encoders"][name] = {
            control: dispatch[name](control_rows, rules)
            for control, control_rows in controls.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
