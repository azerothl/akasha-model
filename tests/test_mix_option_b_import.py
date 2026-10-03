"""Option B human-label import pipeline (#83). No invented annotator labels."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from akasha_model.mix import (
    OPTION_B_LABEL_MODE,
    example_mix_state,
    mix_option_b_schema_path,
    row_has_audio_payload,
)

ROOT = Path(__file__).resolve().parents[1]
IMPORT = ROOT / "scripts" / "import_mix_option_b.py"
AUDIT = ROOT / "scripts" / "audit_mix_dataset.py"
FIXTURE = ROOT / "tests" / "contracts" / "fixtures" / "mix-option-b-format.jsonl"


def _format_row(session_id: str) -> dict:
    state = example_mix_state().to_dict()
    state["session_id"] = session_id
    return {
        "id": session_id,
        "context": state,
        "context_key": session_id,
        "source": "akasha-schema-fixture",
        "licence": "CC0-1.0",
        "label_licence": "CC0-1.0",
        "descriptor_licence": "CC0-1.0",
        "label_mode": OPTION_B_LABEL_MODE,
        "annotation_status": "format_only",
        "questions": [
            {
                "id": "preset",
                "type": "choice",
                "instructions": "Choose a mix preset for this session.",
                "options": [
                    {"name": "vocal_forward", "description": "a"},
                    {"name": "balanced", "description": "b"},
                    {"name": "bass_heavy", "description": "c"},
                ],
                "label": 1,
            },
            {
                "id": "vocal.gain",
                "type": "score",
                "instructions": "Ordered vocal gain adjustment.",
                "levels": ["much quieter", "quieter", "unchanged", "louder", "much louder"],
                "label": 2,
            },
            {
                "id": "vocal_masked_by_bass",
                "type": "noul",
                "instructions": "Is the vocal masked by the bass?",
                "criteria": {"true": "masked", "false": "clear"},
                "label": 0,
            },
        ],
    }


def test_option_b_schema_file_exists() -> None:
    path = mix_option_b_schema_path()
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["properties"]["label_mode"]["const"] == OPTION_B_LABEL_MODE


def test_format_fixture_is_not_human() -> None:
    rows = [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    for row in rows:
        assert row["annotation_status"] == "format_only"
        assert row["source"] == "akasha-schema-fixture"
        assert row["label_mode"] == OPTION_B_LABEL_MODE


def test_import_refuses_format_fixture_as_human(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable, str(IMPORT),
            "--input", str(FIXTURE),
            "--output", str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False


def test_import_accepts_format_fixture_flag(tmp_path: Path) -> None:
    out = tmp_path / "imported"
    result = subprocess.run(
        [
            sys.executable, str(IMPORT),
            "--input", str(FIXTURE),
            "--output", str(out),
            "--allow-format-fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    report = json.loads((out / "licence_report.json").read_text(encoding="utf-8"))
    assert report["audio_committed"] is False
    assert "annotator agreement" in report["measures"]
    assert report["annotation_status"]["format_only"] == payload["rows"]


def test_import_and_audit_many_format_rows(tmp_path: Path) -> None:
    raw = tmp_path / "rows.jsonl"
    rows = [_format_row(f"mix-b-{index:03d}") for index in range(40)]
    raw.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8",
    )
    out = tmp_path / "split"
    subprocess.run(
        [
            sys.executable, str(IMPORT),
            "--input", str(raw),
            "--output", str(out),
            "--allow-format-fixture",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    audit = subprocess.run(
        [sys.executable, str(AUDIT), "--input", str(out)],
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(audit.stdout)
    assert audit.returncode == 0
    assert payload["ok"] is True


def test_import_rejects_audio_key(tmp_path: Path) -> None:
    row = _format_row("mix-b-audio")
    row["audio"] = "vocal.wav"
    assert row_has_audio_payload(row)
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable, str(IMPORT),
            "--input", str(path),
            "--output", str(tmp_path / "out"),
            "--allow-format-fixture",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "audio" in result.stdout
