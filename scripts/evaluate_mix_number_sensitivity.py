#!/usr/bin/env python3
"""Number-sensitivity evaluation protocol for mix descriptors (issue #59).

Controls (same questions, altered context):
  - baseline: original descriptors
  - shuffled_context: permute track blocks across rows
  - numbers_perturbed: multiplicative/additive noise on numeric fields
  - numbers_removed: numeric fields set to null (scorer must tolerate / degrade)

Reports per head (choice / score / noul): accuracy, ECE, coverage/risk, plus
wall-clock latency, process RSS, and (on CUDA) peak allocated VRAM.

Encoders:
  - rule_prior / majority_prior: CPU-friendly baselines (no Hub download)
  - mask:<path>: MASK DecisionModel checkpoint (tiny or HF/BERT)

French: optional ``--french-probe`` rewrites question text to FR and records
baseline accuracy vs the English instructions on the same rows.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from akasha_model.offline import measure_rss_mb


THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
ROOT = Path(__file__).resolve().parents[1]

FRENCH_REWRITE = {
    "preset": {
        "instructions": "Choisissez un préréglage de mix pour cette session.",
        "options": [
            {"name": "vocal_forward", "description": "Mettre la voix en avant."},
            {"name": "balanced", "description": "Garder un équilibre neutre."},
            {"name": "bass_heavy", "description": "Renforcer les graves."},
        ],
    },
    "vocal.gain": {
        "instructions": "Ajustement ordonné du gain de la voix.",
        "levels": [
            "beaucoup plus bas",
            "plus bas",
            "inchangé",
            "plus haut",
            "beaucoup plus haut",
        ],
    },
    "vocal_masked_by_bass": {
        "instructions": "La voix est-elle masquée par la basse ?",
        "criteria": {
            "true": "Oui, le masquage est problématique.",
            "false": "Non, la voix reste intelligible.",
        },
    },
}


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


def rewrite_french(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = copy.deepcopy(row)
        questions = []
        for question in item["questions"]:
            patch = FRENCH_REWRITE.get(question["id"])
            if patch is None:
                questions.append(question)
                continue
            updated = dict(question)
            updated.update(patch)
            questions.append(updated)
        item["questions"] = questions
        out.append(item)
    return out


def _labels(row: dict) -> dict[str, int]:
    return {q["type"]: int(q["label"]) for q in row["questions"]}


def _summarize(
    store: dict[str, dict[str, list]],
    *,
    elapsed_s: float,
    n_rows: int,
    rss_mb: float,
    cuda_peak_mb: float | None = None,
) -> dict[str, Any]:
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
    payload: dict[str, Any] = {
        "heads": heads,
        "latency_s": round(elapsed_s, 4),
        "latency_ms_per_row": round(1000.0 * elapsed_s / max(n_rows, 1), 3),
        "rss_mb": rss_mb,
    }
    if cuda_peak_mb is not None:
        payload["cuda_peak_allocated_mb"] = round(cuda_peak_mb, 2)
    return payload


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
    return _summarize(
        store,
        elapsed_s=time.perf_counter() - t0,
        n_rows=len(rows),
        rss_mb=measure_rss_mb(),
    )


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
    return _summarize(
        store,
        elapsed_s=time.perf_counter() - t0,
        n_rows=len(rows),
        rss_mb=measure_rss_mb(),
    )


def _cuda_peak_mb(device) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not hasattr(device, "type") or device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)


def evaluate_mask_checkpoint(
    rows: list[dict],
    checkpoint: Path,
    device_name: str,
    batch_size: int = 8,
) -> dict[str, Any]:
    import torch
    from akasha_model.decision import load_decision_checkpoint
    from akasha_model.model import select_device
    from akasha_model.typed_decisions import evaluate_typed_records

    device = select_device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    model, tokenizer, config = load_decision_checkpoint(checkpoint, device)
    t0 = time.perf_counter()
    summary, records = evaluate_typed_records(
        model, tokenizer, rows, device,
        config.get("max_len", 768),
        config.get("head_max_len", 384),
        config.get("option_max_len", 48),
        batch_size=batch_size,
        permutations=1,
    )
    elapsed = time.perf_counter() - t0
    store: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        kind = record["kind"]
        store[kind]["correct"].append(float(record["label_correct"]))
        store[kind]["confidence"].append(float(record["confidence"]))
    result = _summarize(
        store,
        elapsed_s=elapsed,
        n_rows=len(rows),
        rss_mb=measure_rss_mb(),
        cuda_peak_mb=_cuda_peak_mb(device),
    )
    result["typed_summary"] = {
        name: {
            "accuracy": summary[name]["accuracy"],
            "ece": summary[name]["ece"],
            "examples": summary[name]["examples"],
        }
        for name in ("all", "choice", "score", "noul")
        if name in summary
    }
    result["checkpoint"] = str(checkpoint)
    result["encoder_config"] = {
        "encoder": config.get("encoder"),
        "hf_model": config.get("hf_model"),
        "width": config.get("width"),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def french_probe_note(
    measured: dict[str, Any] | None = None,
) -> dict[str, Any]:
    note = {
        "numeric_descriptors": (
            "Units are language-agnostic; FR and EN hosts share mix-descriptors schema v1."
        ),
        "question_instructions": (
            "French instructions rewrite the Choice/Score/Noul text while keeping "
            "labels fixed; accuracy is compared to the English baseline on the same rows."
        ),
    }
    if measured is None:
        note["status"] = "documented_not_benchmarked"
        note["follow_up"] = (
            "Pass --french-probe with a MASK checkpoint to measure FR vs EN accuracy."
        )
        return note
    note["status"] = "measured"
    note["measured"] = measured
    return note


def _parse_encoders(raw: str) -> list[tuple[str, Path | None]]:
    items: list[tuple[str, Path | None]] = []
    for part in raw.split(","):
        name = part.strip()
        if not name:
            continue
        if name.startswith("mask:"):
            path = Path(name.split(":", 1)[1])
            items.append((f"mask:{path.name}", path))
        else:
            items.append((name, None))
    return items


def _device_meta(device_name: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "requested_device": device_name,
        "torch_cuda_available": False,
        "gpu_name": None,
        "gpu_memory_total_mb": None,
    }
    try:
        import torch
        meta["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            meta["gpu_name"] = props.name
            meta["gpu_memory_total_mb"] = round(props.total_memory / (1024.0 * 1024.0), 1)
    except Exception:
        pass
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/mix_number_sensitivity.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument(
        "--encoders",
        default="rule_prior,majority_prior",
        help="comma list: rule_prior,majority_prior,mask:/path/to.pt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--french-probe",
        action="store_true",
        help="measure FR vs EN instruction accuracy on the first MASK encoder",
    )
    parser.add_argument(
        "--offline-hub",
        action="store_true",
        help="set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE before loading HF encoders",
    )
    args = parser.parse_args()

    if args.offline_hub:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    data_path = args.data
    rows = _load_jsonl(data_path / "test.jsonl" if data_path.is_dir() else data_path)
    controls = build_controls(rows, seed=args.seed)
    rules = _load_rules_module()
    device_meta = _device_meta(args.device)

    device_notes = [
        f"requested_device={args.device}",
        f"torch_cuda_available={device_meta['torch_cuda_available']}",
    ]
    if device_meta["gpu_name"]:
        device_notes.append(
            f"gpu={device_meta['gpu_name']} "
            f"({device_meta['gpu_memory_total_mb']} MiB total)"
        )
    else:
        # Always record GPU status so CPU CI smoke tests can assert on it.
        device_notes.append(
            "GPU not detected for this run; CUDA 16GB measurements live in "
            "reports/mix_number_sensitivity_cuda.json when published."
        )
    device_notes.append(
        "Values published as measured — no pass/fail threshold in this protocol."
    )

    report: dict[str, Any] = {
        "protocol": "mix-number-sensitivity-v1",
        "data": str(data_path),
        "rows": len(rows),
        "device": args.device,
        "device_meta": device_meta,
        "device_notes": device_notes,
        "french": french_probe_note(),
        "encoders": {},
    }

    prior_dispatch: dict[str, Callable[[list[dict], Any], dict[str, Any]]] = {
        "rule_prior": evaluate_rule_prior,
        "majority_prior": evaluate_majority_prior,
    }
    mask_checkpoints: list[tuple[str, Path]] = []

    for name, path in _parse_encoders(args.encoders):
        if name in prior_dispatch:
            report["encoders"][name] = {
                control: prior_dispatch[name](control_rows, rules)
                for control, control_rows in controls.items()
            }
            continue
        if path is None:
            report["encoders"][name] = {"error": f"unknown encoder {name}"}
            continue
        if not path.is_file():
            report["encoders"][name] = {"error": f"missing checkpoint {path}"}
            continue
        mask_checkpoints.append((name, path))
        report["encoders"][name] = {
            control: evaluate_mask_checkpoint(
                control_rows, path, args.device, batch_size=args.batch_size,
            )
            for control, control_rows in controls.items()
        }

    if args.french_probe:
        if not mask_checkpoints:
            report["french"] = french_probe_note({
                "error": "--french-probe requires at least one mask: checkpoint",
            })
            report["french"]["status"] = "error"
        else:
            name, path = mask_checkpoints[0]
            en = evaluate_mask_checkpoint(
                controls["baseline"], path, args.device, batch_size=args.batch_size,
            )
            fr = evaluate_mask_checkpoint(
                rewrite_french(controls["baseline"]), path, args.device,
                batch_size=args.batch_size,
            )
            measured = {
                "encoder": name,
                "checkpoint": str(path),
                "english_heads": en["heads"],
                "french_heads": fr["heads"],
                "delta_accuracy": {
                    head: round(
                        fr["heads"][head]["accuracy"] - en["heads"][head]["accuracy"],
                        4,
                    )
                    for head in ("choice", "score", "noul")
                },
            }
            report["french"] = french_probe_note(measured)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "rows": len(rows),
        "device": args.device,
        "encoders": list(report["encoders"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
