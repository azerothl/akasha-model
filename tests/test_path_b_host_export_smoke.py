"""Smoke: host authorize-export → multitask splits (no private traffic)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "examples" / "gate"
FIXTURE = GATE / "fixtures" / "host_authorize_export.jsonl"


def test_import_authorize_export_smoke(tmp_path: Path) -> None:
    out = tmp_path / "from_host"
    env = {**os.environ, "PYTHONPATH": str(GATE)}
    proc = subprocess.run(
        [
            sys.executable,
            str(GATE / "import_authorize_export.py"),
            "--input",
            str(FIXTURE),
            "--output",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert sum(payload["counts"].values()) == 6
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert "gate-tiny.pt is demo-only" in manifest["note"]
    assert any(path.exists() and path.stat().st_size > 0 for path in out.glob("*.jsonl"))
