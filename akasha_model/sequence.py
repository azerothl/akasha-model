"""MASK-marker sequences for one-pass typed decisions.

Format: ``[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1
... [SEP] <state> [SEP]``. Each option is scored at its ``[MASK]`` position.
"""

from __future__ import annotations

import json
from typing import Any

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def render_options(question: dict[str, Any]) -> list[str]:
    """Render option texts in label-index order. Noul is always [false, true]."""
    kind, criteria = question["t"], question.get("crit")
    if kind == "choice":
        return [
            key if value is None or value == "" else f"{key}: {render_criterion(value)}"
            for key, value in criteria.items()
        ]
    if kind == "score":
        return [
            f"level {index}: {render_criterion(criterion)}"
            for index, criterion in enumerate(criteria)
        ]
    criteria = criteria or {}
    false_text, true_text = criteria.get("false"), criteria.get("true")
    return [
        "false: " + (render_criterion(false_text) if false_text not in (None, "")
                     else "no, the statement does not hold"),
        "true: " + (render_criterion(true_text) if true_text not in (None, "")
                    else "yes, the statement holds"),
    ]


def mask_question_from_row(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a multitask JSONL question into the MASK-sequence spec."""
    kind = payload["type"]
    instructions = payload["instructions"]
    if kind == "choice":
        specs = payload.get("option_specs")
        if specs:
            criteria = {
                item["name"]: item.get("description") or None for item in specs
            }
        else:
            criteria = {}
            for option in payload["options"]:
                if isinstance(option, dict):
                    criteria[option["name"]] = option.get("description") or None
                else:
                    criteria[str(option)] = None
        question = {"t": "choice", "ins": instructions, "crit": criteria}
    elif kind == "score":
        question = {"t": "score", "ins": instructions, "crit": list(payload["levels"])}
    else:
        question = {"t": "noul", "ins": instructions, "crit": payload.get("criteria") or {}}
    question["label"] = int(payload["label"])
    if payload.get("target") is not None:
        question["target"] = [float(value) for value in payload["target"]]
    return question


def option_target(question: dict[str, Any], order: list[int] | None = None) -> list[float]:
    """Gold distribution aligned to ``order`` (default: natural option order)."""
    options = render_options(question)
    width = len(options)
    if question.get("target") is not None:
        target = list(question["target"])
        if len(target) != width:
            raise ValueError("target length must match the option count")
    else:
        target = [0.0] * width
        label = int(question["label"])
        if question["t"] == "noul":
            target[label] = 1.0
        else:
            target[label] = 1.0
    total = sum(target)
    if total <= 0:
        raise ValueError("target must contain positive mass")
    target = [value / total for value in target]
    if order is None:
        return target
    return [target[index] for index in order]


def build_sequence(
    tokenizer,
    state: str | dict[str, Any] | list[Any],
    question: dict[str, Any],
    max_len: int = 768,
    head_max_len: int = 384,
    option_order: list[int] | None = None,
    option_max_len: int = 48,
    truncate_left: bool = False,
) -> tuple[list[int], list[int]]:
    """Tokenise one typed question. Returns ``(input_ids, mask_marker_positions)``."""
    mask_token = tokenizer.mask_token
    options = render_options(question)
    order = option_order if option_order is not None else list(range(len(options)))
    instructions = str(question["ins"]).replace(mask_token, " ")
    head_ids = tokenizer(
        f"{question['t']} question: {instructions}", add_special_tokens=False,
    )["input_ids"]
    option_ids = []
    for index in order:
        text = " " + options[index].replace(mask_token, " ")
        option_ids.append(
            [tokenizer.mask_token_id]
            + tokenizer(text, add_special_tokens=False)["input_ids"][:option_max_len]
        )
    option_budget = head_max_len - sum(len(item) for item in option_ids)
    if option_budget < 16:
        per_option = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        option_ids = [item[:per_option] for item in option_ids]
        option_budget = head_max_len - sum(len(item) for item in option_ids)
    head_ids = head_ids[: max(8, option_budget)]
    ids = [tokenizer.cls_token_id] + head_ids + [tokenizer.sep_token_id]
    markers = []
    for option in option_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    state_ids = tokenizer(
        serialize_state(state).replace(mask_token, " "), add_special_tokens=False,
    )["input_ids"]
    state_ids = state_ids[-room:] if truncate_left else state_ids[:room]
    ids = ids + state_ids + [tokenizer.sep_token_id]
    return ids[:max_len], [marker for marker in markers if marker < max_len]
