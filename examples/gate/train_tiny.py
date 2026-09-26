"""Train a tiny multitask scorer on authorize-tool-call JSONL."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/gate"))
    parser.add_argument("--output", type=Path, default=Path("runs/gate-tiny.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train = args.data / "train.jsonl"
    validation = args.data / "validation.jsonl"
    if not train.is_file() or not validation.is_file():
        raise SystemExit(
            f"missing {train} or {validation}; run "
            "python examples/gate/generate_data.py first",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "akasha_model.multitask_train",
        str(train),
        "--validation", str(validation),
        "--output", str(args.output),
        "--width", str(args.width),
        "--rank", str(args.width),
        "--context-tokens", "384",
        "--question-tokens", "256",
        "--option-tokens", "128",
        "--epochs", str(args.epochs),
        "--batch-size", "16",
        "--learning-rate", "3e-3",
        "--device", args.device,
        "--seed", "7",
    ]
    print(json.dumps({"command": command}))
    subprocess.check_call(command)


if __name__ == "__main__":
    main()
