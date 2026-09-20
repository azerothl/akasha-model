"""Build stress-test rows from a leakage-resistant held-out JSONL split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rng = random.Random(args.seed)
    stress = []
    for row in rows:
        target = row["options"][row["label"]]
        distractors = [option for option in row["options"] if option != target]
        distractor = rng.choice(distractors)

        ambiguous = dict(row)
        ambiguous["context"] = (
            f"{row['context']} Evidence is incomplete: {target} and {distractor} "
            "are both plausible next actions. Prefer the primary intent, but defer "
            "if confidence is low."
        )
        ambiguous["stress_type"] = "ambiguous"
        stress.append(ambiguous)

        noisy = dict(row)
        noisy["context"] = (
            f"{row['context']} Irrelevant telemetry: queue=busy, cache=stale, "
            "retry=enabled, operator=away, clock=uncertain."
        )
        noisy["stress_type"] = "irrelevant_noise"
        stress.append(noisy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in stress:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"source_rows": len(rows), "stress_rows": len(stress)}))


if __name__ == "__main__":
    main()
