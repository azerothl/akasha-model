"""Strict anti-leakage and schema audit for the Akasha OS v2 dataset."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.generate_akasha_dataset import extract_inventory
except ModuleNotFoundError:  # Direct execution: ``python scripts/foo.py``.
    from generate_akasha_dataset import extract_inventory
from jevlike.multitask import validate_multi


FORBIDDEN_KEYS = {"target_action", "required_for_candidate", "oracle_action", "stress_type", "variant", "source", "case_id", "label", "prompt_injection_signal"}
SOURCE_PATH_RE = re.compile(r"(?:crates|src|examples|modules|\.rs|\.yaml|\.yml|akasha[-_]?os)", re.I)


def flattened(value: object) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.append((str(key), str(item)))
            found.extend(flattened(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(flattened(item))
    return found


def semantic_context_key(context: object) -> str:
    """Keep observable policy evidence while ignoring only synthetic noise."""
    if not isinstance(context, dict):
        return json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    projected = {key: value for key, value in context.items() if key != "contextual_factors"}
    return json.dumps(projected, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit(root: Path, akasha_root: Path | None = None) -> dict:
    inventory = extract_inventory(akasha_root) if akasha_root else json.loads((root / "action_inventory.json").read_text(encoding="utf-8"))["actions"]
    action_names = sorted({item["id"] for item in inventory}, key=len, reverse=True)
    report: dict = {"counts": {}, "question_types": Counter(), "labels": {"choice": Counter(), "score": Counter(), "noul": Counter()}, "labels_by_question_id": defaultdict(Counter), "violations": [], "excluded": [], "collisions": {}, "families": {}, "lengths": {"context": [], "question": [], "option": []}, "truncation": {"context_bytes_768": 0, "question_bytes_512": 0, "option_bytes_384": 0}, "ambiguous_cases": 0, "abstention_cases": 0, "stress_types": Counter(), "ood": {}, "semantic_ambiguity": {}}
    states: dict[str, set[str]] = {}
    question_instances: dict[str, set[str]] = {}
    families: dict[str, set[str]] = {}
    question_ids = ("route", "operation_risk", "authorized", "capability_present", "confirmation_needed", "protected_resource", "requires_network", "irreversible", "prompt_injection", "sufficient_context", "offline_compatible")
    request_labels: dict[str, dict[str, defaultdict(set)]] = {}
    duplicate_option_rows: dict[str, int] = {}
    metadata_mismatches: dict[str, int] = {}
    for split in ("train", "validation", "test", "stress", "ood"):
        data_path = root / f"{split}.jsonl"
        meta_path = root / f"{split}.metadata.jsonl"
        states[split], question_instances[split], families[split] = set(), set(), set()
        data_lines = [line for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        metadata = [json.loads(line) for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        request_labels[split] = {question_id: defaultdict(set) for question_id in question_ids}
        duplicate_option_rows[split] = 0
        metadata_mismatches[split] = abs(len(data_lines) - len(metadata))
        valid = 0
        for line_no, raw in enumerate(data_lines, 1):
            try:
                payload = json.loads(raw)
                example = validate_multi(payload)
            except Exception as exc:
                report["excluded"].append({"split": split, "line": line_no, "reason": str(exc)})
                continue
            valid += 1
            context = payload["context"]
            state_serialized = json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            normalized = re.sub(r"\s+", " ", state_serialized).casefold()
            states[split].add(normalized)
            question_instances[split].add(normalized + "|" + json.dumps(payload["questions"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            if line_no <= len(metadata):
                families[split].add(metadata[line_no - 1].get("family", ""))
                if metadata[line_no - 1].get("variant") in {"ambiguous", "contradictory_state", "unknown_action", "prompt_injection"}:
                    report["ambiguous_cases"] += 1
                if split == "stress":
                    report["stress_types"][metadata[line_no - 1].get("stress_type", "unknown")] += 1
            serialized_context = json.dumps(context, ensure_ascii=False, sort_keys=True)
            for key, value in flattened(context):
                if key in FORBIDDEN_KEYS:
                    report["violations"].append({"split": split, "line": line_no, "kind": "forbidden_context_key", "value": key})
                if SOURCE_PATH_RE.search(value) and (".rs" in value or "/" in value or "\\" in value):
                    report["violations"].append({"split": split, "line": line_no, "kind": "source_reference_in_context", "value": value[:200]})
            for action_name in action_names:
                if action_name.casefold() in serialized_context.casefold():
                    report["violations"].append({"split": split, "line": line_no, "kind": "action_name_in_context", "value": action_name})
            context_length = len(serialized_context.encode("utf-8"))
            report["lengths"]["context"].append(context_length)
            if context_length > 768:
                report["truncation"]["context_bytes_768"] += 1
            for question in example.questions:
                report["question_types"][question["type"]] += 1
                report["labels"][question["type"]][str(question["label"])] += 1
                report["labels_by_question_id"][question["id"]][str(question["label"])] += 1
                if question["id"] in request_labels[split]:
                    request = semantic_context_key(context)
                    if question["type"] == "choice":
                        raw_question = next(item for item in payload["questions"] if item.get("id") == question["id"])
                        raw_options = raw_question["options"]
                        target = raw_options[question["label"]]
                        target_name = target.get("name", "") if isinstance(target, dict) else target
                        request_labels[split][question["id"]][request].add(target_name)
                        descriptions = [item.get("description", "") if isinstance(item, dict) else "" for item in raw_options]
                        if descriptions.count(descriptions[question["label"]]) > 1:
                            duplicate_option_rows[split] += 1
                    else:
                        request_labels[split][question["id"]][request].add(str(question["label"]))
                question_text = question.get("question_text", question["instructions"])
                question_length = len(question_text.encode("utf-8"))
                report["lengths"]["question"].append(question_length)
                if question_length > 512:
                    report["truncation"]["question_bytes_512"] += 1
                if question["type"] == "choice" and question["options"][question["label"]].split("\n", 1)[0] in {"other", "insufficient_context"}:
                    report["abstention_cases"] += 1
                for option in question.get("options", []) if question["type"] == "choice" else question.get("levels", []):
                    text = option if isinstance(option, str) else json.dumps(option, ensure_ascii=False)
                    option_length = len(text.encode("utf-8"))
                    report["lengths"]["option"].append(option_length)
                    if option_length > 384:
                        report["truncation"]["option_bytes_384"] += 1
        report["counts"][split] = valid
        report["families"][split] = len(families[split])
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"), ("train", "ood"), ("validation", "ood"), ("test", "ood")):
        report["collisions"][f"{left}_vs_{right}"] = {
            "state_exact_or_normalized": len(states[left] & states[right]),
            "question_instances": len(question_instances[left] & question_instances[right]),
            "families_shared": len((families[left] - {""}) & (families[right] - {""})),
        }
    for key, values in report["lengths"].items():
        if values:
            report["lengths"][key] = {"min": min(values), "mean": round(sum(values) / len(values), 2), "max": max(values)}
        else:
            report["lengths"][key] = {"min": 0, "mean": 0, "max": 0}
    report["question_types"] = dict(report["question_types"])
    report["labels"] = {key: dict(value) for key, value in report["labels"].items()}
    report["labels_by_question_id"] = {key: dict(value) for key, value in report["labels_by_question_id"].items()}
    report["stress_types"] = dict(report["stress_types"])
    report["ood"] = {"rows": report["counts"].get("ood", 0), "families": report["families"].get("ood", 0)}
    report["semantic_ambiguity"] = {
        split: {
            "semantic_context_groups_with_multiple_targets": {
                question_id: sum(len(labels) > 1 for labels in groups.values())
                for question_id, groups in request_labels[split].items()
            },
            "duplicate_target_description_rows": duplicate_option_rows[split],
            "metadata_count_mismatch": metadata_mismatches[split],
        }
        for split in request_labels
    }
    report["quality_gate"] = {
        "formal_schema_and_leakage": not report["excluded"] and not report["violations"],
        "no_cross_split_context_collisions": all(item["state_exact_or_normalized"] == 0 for item in report["collisions"].values()),
        "no_request_label_conflicts": all(
            value == 0
            for split in report["semantic_ambiguity"].values()
            for question_id, value in split["semantic_context_groups_with_multiple_targets"].items()
            if question_id in {"route", "operation_risk", "requires_network", "irreversible"}
        ),
        "no_duplicate_choice_target_descriptions": all(value["duplicate_target_description_rows"] == 0 for value in report["semantic_ambiguity"].values()),
        "metadata_aligned": all(value["metadata_count_mismatch"] == 0 for value in report["semantic_ambiguity"].values()),
    }
    report["ok"] = all(report["quality_gate"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--akasha-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.input, args.akasha_root)
    destination = args.output or args.input / "audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
