#!/usr/bin/env python3
"""ONNX export for a mix MASK checkpoint (issue #84 / T10).

Blocked until T8 (#82) produces a number-sensitive mix MASK. This script:

- refuses ``gate-tiny.pt`` (gate byte scorer, not mix MASK)
- refuses missing / non-MASK checkpoints
- refuses export when ``--require-mix-compact`` (default) and the checkpoint
  was not trained with mix-compact-v1
- optional ``--sensitivity-report`` must show T4 number sensitivity

GGUF is not used: the tiny MASK graph is a custom encoder, not a llama.cpp LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from akasha_model.decision import DecisionModel, load_decision_checkpoint
from akasha_model.mix import majority_sensitivity_verdict
from akasha_model.model import select_device


def _is_gate_tiny(path: Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    return "gate-tiny" in text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/mix-mask-tiny.onnx"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sensitivity-report", type=Path)
    parser.add_argument(
        "--require-mix-compact",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if _is_gate_tiny(args.checkpoint):
        print(json.dumps({
            "ok": False,
            "blocker": "gate-tiny.pt is the Path B gate demo, not a mix MASK checkpoint",
            "issue": 82,
        }))
        return 2
    if not args.checkpoint.is_file():
        print(json.dumps({
            "ok": False,
            "blocker": f"missing checkpoint {args.checkpoint}",
            "issue": 82,
            "hint": "Train with scripts/train_mix_mask.py; keep weights in runs/",
        }))
        return 2

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kind = payload.get("kind") or (payload.get("config") or {}).get("kind")
    if kind != "mask":
        print(json.dumps({
            "ok": False,
            "blocker": f"checkpoint kind={kind!r} is not MASK",
            "issue": 82,
        }))
        return 2
    device = select_device(args.device)
    model, _tokenizer, config = load_decision_checkpoint(args.checkpoint, device)
    if not isinstance(model, DecisionModel):
        print(json.dumps({"ok": False, "blocker": "not a DecisionModel", "issue": 82}))
        return 2
    if args.require_mix_compact and not config.get("mix_compact_state"):
        print(json.dumps({
            "ok": False,
            "blocker": "checkpoint was not trained with --mix-compact-state (T8 recipe)",
            "issue": 82,
        }))
        return 2
    if args.sensitivity_report is not None:
        if not args.sensitivity_report.is_file():
            print(json.dumps({
                "ok": False,
                "blocker": f"missing T4 report {args.sensitivity_report}",
                "issue": 82,
            }))
            return 2
        report = json.loads(args.sensitivity_report.read_text(encoding="utf-8"))
        verdict = majority_sensitivity_verdict(report)
        if not verdict.get("ok"):
            print(json.dumps({
                "ok": False,
                "blocker": "T4 report is not number-sensitive; T10 waits on #82",
                "verdict": verdict,
                "issue": 82,
            }))
            return 2

    model.eval()
    max_len = int(config.get("max_len", 128))
    width_opts = 8
    dummy_ids = torch.zeros(1, max_len, dtype=torch.long, device=device)
    dummy_mask = torch.ones(1, max_len, dtype=torch.long, device=device)
    dummy_pos = torch.zeros(1, width_opts, dtype=torch.long, device=device)
    dummy_mmask = torch.zeros(1, width_opts, dtype=torch.bool, device=device)
    dummy_mmask[0, 0] = True
    dummy_qtype = torch.zeros(1, dtype=torch.long, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            (dummy_ids, dummy_mask, dummy_pos, dummy_mmask, dummy_qtype),
            str(args.output),
            input_names=[
                "input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype",
            ],
            output_names=["logits"],
            dynamo=False,
            opset_version=17,
        )
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "blocker": f"ONNX export failed: {exc}",
            "issue": 84,
            "fallback": "keep local .pt via akasha_model.offline until the graph exports cleanly",
        }))
        return 1
    print(json.dumps({
        "ok": True,
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "note": "ONNX is derived from a mix MASK .pt; do not commit private weights",
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
