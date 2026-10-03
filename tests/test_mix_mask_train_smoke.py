"""Tiny MASK can use compact mix numbers (training smoke for #82)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from akasha_model.decision import (
    ByteTokenizer,
    DecisionModel,
    MaskCollator,
    TinyEncoder,
    collate_items,
)
from akasha_model.multitask import validate_multi
from akasha_model.rlcd import grpo_loss

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "scripts" / "export_mix_mask_onnx.py"
TRAIN = ROOT / "scripts" / "train_mix_mask.py"


def _noul_row(session: str, masking: float, label: int) -> dict:
    return {
        "context": {
            "schema_version": 1,
            "session_id": session,
            "sample_rate_hz": 48000,
            "tracks": [
                {
                    "track_id": "vocal",
                    "rms_db": -18.0,
                    "lufs": -14.0,
                    "spectral_centroid_hz": 1800.0,
                    "band_energies_db": [-22.0, -16.0, -14.0, -20.0],
                    "crest_factor_db": 12.0,
                    "stereo_correlation": 0.15,
                },
                {
                    "track_id": "bass",
                    "rms_db": -12.0,
                    "lufs": -11.0,
                    "spectral_centroid_hz": 120.0,
                    "band_energies_db": [-8.0, -14.0, -28.0, -40.0],
                    "crest_factor_db": 6.0,
                    "stereo_correlation": 0.92,
                },
            ],
            "pair_masking": [
                {"masker_id": "bass", "maskee_id": "vocal", "masking_index": masking},
            ],
        },
        "questions": [
            {
                "id": "vocal_masked_by_bass",
                "type": "noul",
                "instructions": "Is the vocal masked by the bass?",
                "criteria": {"true": "masked", "false": "clear"},
                "label": label,
            }
        ],
    }


def test_tiny_mask_learns_masking_from_compact_state() -> None:
    high = validate_multi(_noul_row("hi", 0.8, 1))
    low = validate_multi(_noul_row("lo", 0.1, 0))
    model = DecisionModel(
        TinyEncoder(hidden_size=32, layers=1, heads=4, max_len=192, dropout=0.0),
        head_layers=1, dropout=0.0,
    )
    collator = MaskCollator(
        ByteTokenizer(), 192, 64, 16, group_size=1,
        layout="state_first", mix_compact_state=True, target_mode="one_hot",
    )
    batch = collator([high, low, high, low])
    optimiser = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(40):
        loss = grpo_loss(
            model(
                batch["input_ids"], batch["attention_mask"],
                batch["marker_pos"], batch["marker_mask"], batch["qtype"],
            ),
            batch, 0.5, 1.0, 1.0,
        )["total"]
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for example in (high, low):
            item = collator.encode_question(example.context, example.questions[0])
            packed = collate_items([[item]], ByteTokenizer.pad_token_id)
            logits = model(
                packed["input_ids"], packed["attention_mask"],
                packed["marker_pos"], packed["marker_mask"], packed["qtype"],
            )
            pred = int(logits[0, :len(item["markers"])].argmax().item())
            if pred == item["label"]:
                correct += 1
    assert correct == 2


def test_export_refuses_gate_tiny() -> None:
    result = subprocess.run(
        [
            sys.executable, str(EXPORT),
            "--checkpoint", str(ROOT / "examples/gate/checkpoints/gate-tiny.pt"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["issue"] == 82


def test_export_refuses_missing_mix_checkpoint(tmp_path: Path) -> None:
    missing = tmp_path / "mix-mask-tiny.pt"
    result = subprocess.run(
        [sys.executable, str(EXPORT), "--checkpoint", str(missing)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert "missing checkpoint" in payload["blocker"]


def test_train_script_refuses_gate_tiny_output() -> None:
    result = subprocess.run(
        [
            sys.executable, str(TRAIN),
            "--output", "examples/gate/checkpoints/gate-tiny.pt",
            "--skip-train", "--skip-eval",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
