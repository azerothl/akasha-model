"""Smoke-scale zeroshot generator + audit (optional Path B probe)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_generate_and_audit_smoke(tmp_path: Path) -> None:
    out = tmp_path / "zeroshot_decision"
    gen = subprocess.run(
        [
            sys.executable,
            "scripts/generate_zeroshot_decision_dataset.py",
            "--output",
            str(out),
            "--seed",
            "20260922",
            "--train",
            "40",
            "--val",
            "10",
            "--test-domain",
            "10",
            "--test-labels",
            "10",
            "--stress",
            "10",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert gen.returncode == 0, gen.stderr
    audit = subprocess.run(
        [
            sys.executable,
            "scripts/audit_zeroshot_decision_dataset.py",
            "--input",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert audit.returncode == 0, audit.stdout + audit.stderr
    report = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["domain_holdout_ok"] is True
    assert report["arbitrary_label_share_train"] >= 0.3
