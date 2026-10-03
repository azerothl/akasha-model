#!/usr/bin/env python3
"""CPU-friendly MASK mix training recipe (issue #82 / T8).

``examples/gate/checkpoints/gate-tiny.pt`` is a **gate** demo scorer, not a mix
MASK checkpoint. Mix weights stay under ``runs/`` (gitignored).

T4 (#59) showed short RLCD runs on raw JSON state matching the majority prior.
This recipe uses mix-compact-v1 state so dB/LUFS/masking tokens are not clipped
out of the byte sequence, then evaluates ``scripts/evaluate_mix_number_sensitivity.py``.

A checkpoint is mix-publishable only if T4 shows a clear gap vs majority prior
on baseline vs numbers_removed / numbers_perturbed. This script prints that
verdict; it does not lower the bar.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from akasha_model.mix import majority_sensitivity_verdict

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "generate_mix_dataset.py"
AUDIT = ROOT / "scripts" / "audit_mix_dataset.py"
EVAL = ROOT / "scripts" / "evaluate_mix_number_sensitivity.py"
FORBIDDEN_OUTPUT_MARKERS = ("gate-tiny", "examples/gate/checkpoints")


def _run(argv: list[str]) -> None:
    print("+ " + " ".join(argv), flush=True)
    subprocess.run(argv, check=True, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mix_synth"))
    parser.add_argument("--output", type=Path, default=Path("runs/mix-mask-tiny.pt"))
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("reports/mix_number_sensitivity_t8.json"),
    )
    args = parser.parse_args()
    output = args.output
    marker = str(output).replace("\\", "/")
    if any(item in marker for item in FORBIDDEN_OUTPUT_MARKERS):
        print(
            "refusing to overwrite gate-tiny / gate demo checkpoints; use runs/",
            file=sys.stderr,
        )
        return 2

    python = sys.executable
    if not args.skip_train:
        _run([python, str(GEN), "--output", str(args.data), "--rows", str(args.rows)])
        _run([python, str(AUDIT), "--input", str(args.data)])
        _run([
            python, "-m", "akasha_model.rlcd",
            str(args.data / "train.jsonl"),
            "--validation", str(args.data / "validation.jsonl"),
            "--encoder", "tiny",
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--group-size", "1",
            "--type-balance",
            "--target-mode", "one_hot",
            "--mix-compact-state",
            "--mask-layout", "state_first",
            "--learning-rate", "3e-4",
            "--encoder-learning-rate", "3e-4",
            "--output", str(output),
        ])

    if args.skip_eval:
        print(json.dumps({"checkpoint": str(output), "eval": "skipped"}))
        return 0
    if not output.is_file():
        print(json.dumps({"ok": False, "error": f"missing checkpoint {output}"}))
        return 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _run([
        python, str(EVAL),
        "--data", str(args.data),
        "--device", args.device,
        "--encoders", f"rule_prior,majority_prior,mask:{output}",
        "--output", str(args.report),
    ])
    report = json.loads(args.report.read_text(encoding="utf-8"))
    verdict = majority_sensitivity_verdict(report)
    print(json.dumps({
        "checkpoint": str(output),
        "gate_tiny_is_not_mix": True,
        "weights_in_git": False,
        "verdict": verdict,
        "report": str(args.report),
    }, sort_keys=True, indent=2))
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
