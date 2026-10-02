"""Smoke: licences dossier lists required checkpoint rows (#60)."""

from __future__ import annotations

from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "docs" / "licences-poids.md"


def test_licences_doc_exists_and_covers_encoders() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "bert-base-uncased" in text
    assert "ModernBERT" in text
    assert "gate-tiny.pt" in text
    assert "Gabriel" in text
    assert "pending" in text.lower()
    assert "CC0-1.0" in text or "synthétique" in text.lower()
