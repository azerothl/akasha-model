"""Smoke tests for mix number-sensitivity protocol (#59)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from akasha_model.primitives import (
    ChoiceQuestion,
    NoulQuestion,
    OptionSpec,
    ScoreLevel,
    ScoreQuestion,
)

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "generate_mix_dataset.py"
EVAL = ROOT / "scripts" / "evaluate_mix_number_sensitivity.py"


def test_protocol_runs_on_synth_data(tmp_path: Path) -> None:
    data = tmp_path / "mix"
    subprocess.run(
        [sys.executable, str(GEN), "--output", str(data), "--rows", "36", "--seed", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    out = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable, str(EVAL),
            "--data", str(data),
            "--output", str(out),
            "--encoders", "rule_prior,majority_prior",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["protocol"] == "mix-number-sensitivity-v1"
    assert "rule_prior" in report["encoders"]
    assert "majority_prior" in report["encoders"]
    baseline = report["encoders"]["rule_prior"]["baseline"]["heads"]
    assert baseline["choice"]["accuracy"] == 1.0
    assert "ece" in baseline["choice"]
    assert "coverage_risk" in baseline["choice"]
    assert "french" in report
    assert any("GPU" in note or "gpu" in note.lower() for note in report["device_notes"])


def test_french_instructions_build_primitives() -> None:
    choice = ChoiceQuestion(
        "preset",
        "Choisissez un préréglage de mix pour cette session.",
        (
            OptionSpec("vocal_forward", "Mettre la voix en avant."),
            OptionSpec("balanced", "Garder un équilibre neutre."),
        ),
    )
    score = ScoreQuestion(
        "vocal.gain",
        "Ajustement ordonné du gain de la voix.",
        (ScoreLevel("plus bas"), ScoreLevel("inchangé"), ScoreLevel("plus haut")),
    )
    noul = NoulQuestion(
        "masque",
        "La voix est-elle masquée par la basse ?",
        criteria=(
            "Oui, le masquage est problématique.",
            "Non, la voix reste intelligible.",
        ),
    )
    assert choice.instructions.startswith("Choisissez")
    assert score.levels[0].description == "plus bas"
    assert noul.criteria is not None
