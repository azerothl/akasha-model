"""Create a leakage-safe train augmentation for difficult Akasha scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

try:
    from scripts.generate_akasha_dataset import _select_actions, extract_inventory, stable
    from scripts.generate_akasha_multitask_dataset_v3 import STRESS_TYPES, family, make_row, needs_network, words
except ModuleNotFoundError:  # Direct execution: ``python scripts/foo.py``.
    from generate_akasha_dataset import _select_actions, extract_inventory, stable
    from generate_akasha_multitask_dataset_v3 import STRESS_TYPES, family, make_row, needs_network, words


def build(input_path: Path, output_path: Path, akasha_root: Path, seed: int,
          extra: int, action_limit: int = 160,
          stress_types: tuple[str, ...] = STRESS_TYPES) -> dict:
    if not stress_types:
        raise ValueError("stress_types must contain at least one scenario")
    inventory_rows = extract_inventory(akasha_root)
    inventory = {item["id"]: item for item in inventory_rows}
    actions = _select_actions(inventory_rows, min(action_limit, len(inventory_rows)))
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata_path = input_path.with_name(input_path.stem + ".metadata.jsonl")
    metadata = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()] if metadata_path.exists() else [{} for _ in rows]
    added_metadata = []
    network_actions = [action for action in actions if needs_network(action)] or actions
    install_actions = [action for action in actions if "install" in words(action) or family(action) == "module"] or actions
    for index in range(extra):
        stress_type = stress_types[index % len(stress_types)]
        action_pool = network_actions if stress_type == "offline_network" else install_actions if stress_type == "unsigned_install" else actions
        action = action_pool[index % len(action_pool)]
        row_seed = stable(f"{seed}:train_stress:{action}:{stress_type}:{index}")
        row, item = make_row(row_seed, action, stress_type, actions, Random(row_seed), stress_type)
        item["family"] = f"train_stress:{stress_type}:{index % 64}"
        item["source_refs"] = inventory.get(item["oracle_action"], {}).get("source_files", [])
        rows.append(row)
        added_metadata.append(item)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    output_metadata = output_path.with_name(output_path.stem + ".metadata.jsonl")
    with output_metadata.open("w", encoding="utf-8") as handle:
        for item in metadata + added_metadata:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"base_rows": len(rows) - extra, "added_rows": extra, "total_rows": len(rows), "output": str(output_path), "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/akasha_os_multi_v3/train_augmented.jsonl"))
    parser.add_argument("--akasha-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--extra", type=int, default=6000)
    parser.add_argument("--action-limit", type=int, default=160)
    parser.add_argument(
        "--stress-types", nargs="+", default=list(STRESS_TYPES),
        help="scenario cycle for added hard rows; defaults to all standard stress types",
    )
    args = parser.parse_args()
    print(json.dumps(build(
        args.input, args.output, args.akasha_root, args.seed, args.extra,
        args.action_limit, tuple(args.stress_types),
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
