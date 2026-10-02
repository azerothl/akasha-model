"""Smoke tests for synthetic mix dataset generator + anti-leakage audit (#58)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from akasha_model.mix import mix_decision_request_from_row

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "generate_mix_dataset.py"
AUDIT = ROOT / "scripts" / "audit_mix_dataset.py"


@pytest.fixture()
def mix_dir(tmp_path: Path) -> Path:
    out = tmp_path / "mix_synth"
    result = subprocess.run(
        [sys.executable, str(GEN), "--output", str(out), "--rows", "40", "--seed", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out.joinpath("train.jsonl").is_file()
    assert "option_a_written_rules" in result.stdout or (
        json.loads((out / "licence_report.json").read_text())["label_mode"]
        == "option_a_written_rules"
    )
    return out


def test_generator_writes_licence_report(mix_dir: Path) -> None:
    report = json.loads((mix_dir / "licence_report.json").read_text(encoding="utf-8"))
    assert report["audio_committed"] is False
    assert report["proportions_by_licence"]["CC0-1.0"] == report["rows"]
    assert report["measures"].startswith("agreement with written rules")


def test_rows_validate_as_mix_decision_requests(mix_dir: Path) -> None:
    line = (mix_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert row["source"]
    assert row["licence"]
    request = mix_decision_request_from_row(row)
    assert request.questions


def test_audit_passes(mix_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(AUDIT), "--input", str(mix_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["ok"] is True


def test_audit_detects_context_leak(mix_dir: Path, tmp_path: Path) -> None:
    leak = tmp_path / "leaky"
    leak.mkdir()
    train = (mix_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(train[0])
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        (leak / name).write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(AUDIT), "--input", str(leak)],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["ok"] is False
    assert any(item.startswith("context_key_leak:") for item in payload["critical"])
