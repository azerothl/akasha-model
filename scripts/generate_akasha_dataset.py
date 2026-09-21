"""Generate a grouped, abstention-aware Akasha OS decision dataset.

The source of truth is a local Akasha OS checkout. Action identifiers are
extracted from Rust string constants and shipped module manifests; no action is
invented by the generator. The JSONL rows keep the three-field schema
``context``, ``options``, ``label``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from random import Random

ABSTAIN = "__abstain__"
PREFIXES = (
    "agent.", "chat.", "canvas.", "cap.", "create.", "device.", "feedback.",
    "harness.", "health.", "mcp.", "media.", "mem.", "memory.", "model.",
    "module.", "notes.", "plan.", "provider.", "room.", "schedule.",
    "secrets.", "session.", "skill.", "tasks.", "tool.", "update.", "user.",
)
QUOTED_ID_RE = re.compile(r"[\"']([a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_.-]*)+)[\"']")
RUST_CONST_RE = re.compile(r'pub const\s+[A-Z0-9_]+\s*:\s*&str\s*=\s*"([^"]+)"')
YAML_TOOL_RE = re.compile(r"^\s*-?\s*name:\s*([a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)+)\s*$")


def stable(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def _candidate(value: str) -> bool:
    return value.startswith(PREFIXES) and len(value) <= 80 and not value.endswith(".")


def extract_inventory(root: Path) -> list[dict]:
    found: dict[str, dict] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".rs", ".yaml", ".yml", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Quoted literals are used by bus handlers, tool dispatch, tests and
        # UI routing. We intentionally do not scan arbitrary dotted tokens:
        # Rust field accesses such as `agent.tools.len` are not actions.
        values = set(QUOTED_ID_RE.findall(text))
        values.update(RUST_CONST_RE.findall(text))
        if path.suffix.lower() in {".yaml", ".yml"}:
            values.update(YAML_TOOL_RE.findall(text))
        for value in values:
            if not _candidate(value):
                continue
            rel = str(path.relative_to(root)).replace("\\", "/")
            entry = found.setdefault(value, {
                "id": value,
                "description": "Identifier extracted from Akasha OS source or manifest.",
                "source_files": [],
                "preconditions": ["the owning service or module is available"],
                "permissions": ["the capability required by the owning handler"],
                "side_effects": ["depends on the handler; inspect the cited source before execution"],
                "similar_actions": [],
                "abstention_reasons": ["unknown state, missing capability, or contradictory request"],
                "positive_examples": [],
                "negative_examples": [],
            })
            if rel not in entry["source_files"]:
                entry["source_files"].append(rel)
    actions = sorted(found.values(), key=lambda item: item["id"])
    for entry in actions:
        prefix = entry["id"].split(".", 1)[0]
        entry["similar_actions"] = [x["id"] for x in actions if x["id"].split(".", 1)[0] == prefix and x["id"] != entry["id"]][:8]
        entry["positive_examples"] = [f"request related to {entry['id']} with capability granted"]
        entry["negative_examples"] = [f"request related to {entry['id']} while capability is absent"]
    if not actions:
        raise RuntimeError(f"no Akasha action identifiers found under {root}")
    return actions


def _verb(action: str) -> str:
    return action.replace(".", " ").replace("_", " ")


def _families(actions: list[str]) -> list[dict]:
    rows = []
    for action_index, action in enumerate(actions):
        for variant, kind in enumerate(("normal", "contrast", "noisy", "abstain")):
            rows.append({"id": f"family-{action_index:03d}-{variant}", "action": action, "kind": kind})
    for index, kind in enumerate(("ambiguous", "contradictory", "out_of_distribution", "dangerous")):
        rows.append({"id": f"abstain-{index:03d}", "action": ABSTAIN, "kind": kind})
    return rows


def _select_actions(inventory: list[dict], limit: int) -> list[str]:
    """Select a broad action vocabulary while keeping the full inventory."""
    groups: dict[str, list[str]] = {}
    for item in inventory:
        action = item["id"]
        groups.setdefault(action.split(".", 1)[0], []).append(action)
    for values in groups.values():
        values.sort()
    selected: list[str] = []
    while len(selected) < min(limit, len(inventory)):
        progressed = False
        for prefix in sorted(groups):
            if len(selected) >= limit:
                break
            if groups[prefix]:
                selected.append(groups[prefix].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _context(rng: Random, action: str, kind: str, index: int) -> str:
    surface = rng.choice(("chat", "agents", "background", "cli", "module", "canvas", "integration", "memory", "models"))
    resource = rng.choice(("local", "network", "gpu", "filesystem", "secrets", "external-service"))
    trust = rng.choice(("trusted", "restricted", "untrusted"))
    capability = rng.choice(("granted", "absent", "refused", "expired", "unknown"))
    state = rng.choice(("ready", "busy", "offline", "missing", "stale", "degraded", "failed"))
    signal = rng.choice(("queue busy", "disk low", "GPU pressure", "retryable failure", "permission pending", "audit clean", "no network"))
    phrasing = rng.choice((
        "User asks to {verb}.",
        "Route this request: {verb}.",
        "The operator wants to {verb}.",
        "Incoming intent concerns: {verb}.",
        "A background job proposes: {verb}.",
    ))
    if kind == "missing":
        capability = rng.choice(("unknown", "absent", "expired"))
        state = rng.choice(("missing", "offline", "stale"))
    elif kind == "contrast":
        signal = "similar action also plausible"
    elif kind == "abstain":
        trust = "restricted"
        signal = "human confirmation pending"
    elif kind in {"ambiguous", "contradictory", "out_of_distribution", "dangerous"}:
        action = ABSTAIN
        capability = rng.choice(("unknown", "refused", "absent"))
        signal = {
            "ambiguous": "two targets named",
            "contradictory": "offline but network required",
            "out_of_distribution": "unsupported request",
            "dangerous": "destructive effect not confirmed",
        }[kind]
    return (
        f"{phrasing.format(verb=_verb(action))} surface={surface}; resource={resource}; "
        f"trust={trust}; capability={capability}; state={state}; signal={signal}; "
        f"trace={index}."
    )


def generate(root: Path, output: Path, seed: int, total: int, max_actions: int) -> dict:
    inventory = extract_inventory(root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "action_inventory.json").write_text(json.dumps({
        "source_root": str(root), "action_count": len(inventory), "actions": inventory,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    actions = _select_actions(inventory, max_actions)
    families = _families(actions)
    rng = Random(seed)
    rng.shuffle(families)
    stress_families = [f for f in families if f["action"] == ABSTAIN or f["kind"] != "normal"]
    # Keep every action's four scenario variants available for the grouped
    # train/validation/test split. The same difficult families are also used
    # as held-out stress cases, which are never used for training.
    ordinary_families = [f for f in families if f["action"] != ABSTAIN]
    split_families = {"train": [], "validation": [], "test": []}
    for i, family in enumerate(ordinary_families):
        split = ("train", "validation", "test")[i % 3] if i % 3 else "train"
        split_families[split].append(family)
    # Keep all action classes in every normal split when the inventory allows it.
    for action in actions:
        owned = [f for f in ordinary_families if f["action"] == action]
        for split, family in zip(("train", "validation", "test"), owned[:3]):
            for current in split_families.values():
                if family in current:
                    current.remove(family)
            split_families[split].append(family)
    target_counts = {"train": round(total * .70), "validation": round(total * .15), "test": total - round(total * .70) - round(total * .15)}
    rows = {"train": [], "validation": [], "test": [], "stress": []}
    metadata = {key: [] for key in rows}
    all_families = {key: [f["id"] for f in value] for key, value in split_families.items()}
    for split, family_rows in split_families.items():
        per_family = max(1, (target_counts[split] + len(family_rows) - 1) // max(1, len(family_rows)))
        for family in family_rows:
            for n in range(per_family):
                if len(rows[split]) >= target_counts[split]:
                    break
                local = Random(seed + stable(f"{family['id']}:{n}"))
                item = {"context": _context(local, family["action"], family["kind"], n), "options": [], "label": 0}
                target = family["action"] if family["kind"] in {"normal", "contrast", "noisy"} else ABSTAIN
                candidates = [target] + local.sample([a for a in actions if a != target], min(7, len(actions) - 1))
                local.shuffle(candidates)
                item["options"] = candidates
                item["label"] = candidates.index(target)
                rows[split].append(item)
                metadata[split].append({"family": family["id"], "kind": family["kind"], "target": target})
    for n in range(max(1, round(total * .10))):
        family = stress_families[n % len(stress_families)]
        local = Random(seed + 991 * n)
        target = family["action"] if family["kind"] in {"contrast", "noisy"} else ABSTAIN
        candidates = [target] + local.sample(actions, min(7, len(actions)))
        local.shuffle(candidates)
        rows["stress"].append({"context": _context(local, target, family["kind"], n), "options": candidates, "label": candidates.index(target)})
        metadata["stress"].append({"family": family["id"], "kind": family["kind"], "target": target})
    for split, payloads in rows.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        (output / f"{split}.metadata.jsonl").write_text("\n".join(json.dumps(x) for x in metadata[split]) + "\n", encoding="utf-8")
    manifest = {"seed": seed, "total": total, "action_count_used": len(actions), "split_by": "scenario_family", "families": all_families}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"rows": {key: len(value) for key, value in rows.items()}, "families": {key: len(value) for key, value in all_families.items()}, "actions": len(actions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--akasha-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/akasha_os_v2"))
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--total", type=int, default=30000)
    parser.add_argument("--max-actions", type=int, default=160)
    args = parser.parse_args()
    print(json.dumps(generate(args.akasha_root, args.output, args.seed, args.total, args.max_actions), indent=2))


if __name__ == "__main__":
    main()
