#!/usr/bin/env python3
"""Generate optional Path B zero-shot conditional decision JSONL (smoke → full).

This is a **quality probe** for description-driven decisions — not a gate
identity / JevBench cutover requirement. Data stays under ``data/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DOMAINS = (
    "industrial_safety",
    "support_triage",
    "logistics_priority",
    "access_control",
    "pure_threshold",
)


def _row(
    *,
    row_id: str,
    domain: str,
    split_group: str,
    temp: int,
    vibration_high: bool,
    label_surface: str,
    rng: random.Random,
) -> dict:
    # Deterministic teacher: reduce if vib high and temp>85; shutdown if temp>100.
    if temp > 100:
        action = "shutdown"
        risk_label = 2
    elif vibration_high and temp > 85:
        action = "reduce"
        risk_label = 1
    else:
        action = "continue"
        risk_label = 0

    descriptions = {
        "continue": "Machine operates normally under the stated policy.",
        "reduce": "Reduce speed when operating conditions are becoming unstable.",
        "shutdown": "Immediately stop when conditions may damage the machine.",
    }
    if label_surface == "semantic":
        names = ["continue", "reduce", "shutdown"]
    elif label_surface == "misleading":
        names = ["shutdown", "continue", "reduce"]  # names swapped vs meaning slots
        # Keep descriptions tied to true actions; shuffle assignment below.
        name_for = {
            "continue": "shutdown",
            "reduce": "continue",
            "shutdown": "reduce",
        }
        names = [name_for[a] for a in ("continue", "reduce", "shutdown")]
    else:  # arbitrary
        names = [f"opt_{rng.randint(1000, 9999)}" for _ in range(3)]

    actions = ["continue", "reduce", "shutdown"]
    options = [
        {"name": names[i], "description": descriptions[actions[i]]}
        for i in range(3)
    ]
    order = list(range(3))
    rng.shuffle(order)
    options = [options[i] for i in order]
    gold = order.index(actions.index(action))

    policy = (
        "Reduce speed when vibration is high and temperature > 85; "
        "shut down if temperature > 100; otherwise continue."
    )
    return {
        "id": row_id,
        "domain": domain,
        "split_group": split_group,
        "context": {
            "narrative": f"Sensor report in domain {domain}.",
            "facts": {
                "temperature_c": temp,
                "vibration": "high" if vibration_high else "normal",
            },
            "policy": policy,
        },
        "questions": [
            {
                "id": "action",
                "type": "choice",
                "instructions": "Choose the action that matches the policy.",
                "options": options,
                "label": gold,
            },
            {
                "id": "risk",
                "type": "score",
                "instructions": "Rate operational risk given the facts and policy.",
                "levels": ["low", "elevated", "critical"],
                "label": risk_label,
            },
            {
                "id": "policy_sufficient",
                "type": "noul",
                "instructions": "Is the policy text sufficient to decide without outside knowledge?",
                "criteria": {
                    "true": "The policy fully determines the correct action from the facts.",
                    "false": "Missing definitions or contradictory rules.",
                },
                "label": 1,
            },
        ],
        "meta": {
            "label_surface": label_surface,
            "rule_location": "context.policy",
            "gold_depends_on": [
                "facts.temperature_c",
                "facts.vibration",
                "context.policy",
            ],
        },
    }


def _surface(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.45:
        return "arbitrary"
    if roll < 0.60:
        return "misleading"
    if roll < 0.85:
        return "semantic"
    return "arbitrary"


def generate_split(
    *,
    name: str,
    count: int,
    seed: int,
    domain_pool: tuple[str, ...],
    id_prefix: str,
) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        domain = domain_pool[index % len(domain_pool)]
        # Prefix with split name so train/val/test groups never collide.
        split_group = f"{name}:{domain}:family_{index % max(1, count // 50)}"
        rows.append(
            _row(
                row_id=f"{id_prefix}_{index:06d}",
                domain=domain,
                split_group=split_group,
                temp=rng.randint(60, 120),
                vibration_high=rng.random() < 0.45,
                label_surface=_surface(rng),
                rng=rng,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--train", type=int, default=50000)
    parser.add_argument("--val", type=int, default=5000)
    parser.add_argument("--test-domain", type=int, default=5000)
    parser.add_argument("--test-labels", type=int, default=5000)
    parser.add_argument("--stress", type=int, default=5000)
    args = parser.parse_args()

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    # Hold out last domain for test-domain split.
    train_domains = DOMAINS[:-1]
    holdout_domain = (DOMAINS[-1],)

    splits = {
        "train.jsonl": generate_split(
            name="train", count=args.train, seed=args.seed,
            domain_pool=train_domains, id_prefix="zs_train",
        ),
        "val.jsonl": generate_split(
            name="val", count=args.val, seed=args.seed + 1,
            domain_pool=train_domains, id_prefix="zs_val",
        ),
        "test_domain.jsonl": generate_split(
            name="test_domain", count=args.test_domain, seed=args.seed + 2,
            domain_pool=holdout_domain, id_prefix="zs_tdom",
        ),
        "test_labels.jsonl": generate_split(
            name="test_labels", count=args.test_labels, seed=args.seed + 3,
            domain_pool=train_domains, id_prefix="zs_tlab",
        ),
        "stress.jsonl": generate_split(
            name="stress", count=args.stress, seed=args.seed + 4,
            domain_pool=DOMAINS, id_prefix="zs_stress",
        ),
    }
    for filename, rows in splits.items():
        path = out / filename
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    meta = {
        "seed": args.seed,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "train_domains": list(train_domains),
        "holdout_domain": list(holdout_domain),
        "note": "Optional Path B probe; not a gate identity / JevBench cutover.",
    }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
