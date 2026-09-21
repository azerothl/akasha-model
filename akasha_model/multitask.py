"""Shared-context multi-question scorer.

The model shares a text encoder across all questions in a decision request,
then dispatches to independent Choice, Score and Noul heads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .data import _bytes
from .primitives import (
    ChoiceQuestion, NoulQuestion, OptionSpec, ScoreLevel, ScoreQuestion,
)


State = str | dict[str, Any] | list[Any]


def serialize_state(state: State) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class MultiQuestionExample:
    context: State
    questions: tuple[dict[str, Any], ...]


def _question(payload: dict[str, Any]) -> dict[str, Any]:
    kind = payload.get("type")
    question_id = payload.get("id")
    instructions = payload.get("instructions", "")
    if kind not in {"choice", "score", "noul"}:
        raise ValueError("question type must be choice, score or noul")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("each question needs a non-empty id")
    if not isinstance(instructions, str) or not instructions:
        raise ValueError("each question needs non-empty instructions")
    result = dict(payload)
    result["type"] = kind
    if kind == "choice":
        options = payload.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= 255:
            raise ValueError("Choice requires between 2 and 255 options")
        specs = []
        for option in options:
            if isinstance(option, str):
                specs.append(OptionSpec(option))
            elif isinstance(option, dict):
                specs.append(OptionSpec(
                    option.get("name"), option.get("description", ""),
                    tuple(option.get("not_for", ())), tuple(option.get("examples", ())),
                ))
            else:
                raise ValueError("Choice options must be strings or objects")
        label = payload.get("label")
        if not isinstance(label, int) or not 0 <= label < len(specs):
            raise ValueError("Choice label must be an option index")
        # Store normalized model text so collators do not need to understand
        # every option metadata field again.
        result["options"] = [spec.model_text() for spec in specs]
        result["option_specs"] = [
            {"name": spec.name, "description": spec.description,
             "not_for": list(spec.not_for), "examples": list(spec.examples)}
            for spec in specs
        ]
        ChoiceQuestion(question_id, instructions, tuple(specs))
    elif kind == "score":
        levels = payload.get("levels")
        if not isinstance(levels, list) or not 2 <= len(levels) <= 10:
            raise ValueError("Score requires between 2 and 10 levels")
        parsed = tuple(
            ScoreLevel(item if isinstance(item, str) else item["description"])
            for item in levels if isinstance(item, (str, dict))
        )
        if len(parsed) != len(levels):
            raise ValueError("Score levels must be strings or objects")
        ScoreQuestion(question_id, instructions, parsed)
        label = payload.get("label")
        if not isinstance(label, int) or not 0 <= label < len(parsed):
            raise ValueError("Score label must be a level index")
        result["levels"] = [level.description for level in parsed]
        result["question_text"] = instructions + " Levels: " + "; ".join(
            level.description for level in parsed
        )
    else:
        label = payload.get("label")
        if not isinstance(label, (bool, int)) or int(label) not in (0, 1):
            raise ValueError("Noul label must be boolean or 0/1")
        criteria = payload.get("criteria")
        if criteria is not None:
            if (not isinstance(criteria, dict) or
                    not isinstance(criteria.get("true"), str) or
                    not isinstance(criteria.get("false"), str)):
                raise ValueError("Noul criteria must contain true and false strings")
            result["question_text"] = (
                instructions + " True: " + criteria["true"] +
                " False: " + criteria["false"]
            )
    result["id"] = question_id
    result["instructions"] = instructions
    result["label"] = int(result["label"])
    target = payload.get("target")
    if target is not None:
        if not isinstance(target, list) or not target:
            raise ValueError("target must be a non-empty list of probabilities")
        values = [float(item) for item in target]
        if any(value < 0 for value in values):
            raise ValueError("target probabilities must be non-negative")
        result["target"] = values
    return result


def validate_multi(payload: dict[str, Any]) -> MultiQuestionExample:
    context = payload.get("context")
    questions = payload.get("questions")
    if isinstance(context, str):
        if not context.strip():
            raise ValueError("multi-question rows need a non-empty context")
    elif not isinstance(context, (dict, list)):
        raise ValueError("context must be text, a JSON object or a JSON array")
    if not isinstance(questions, list) or not questions:
        raise ValueError("multi-question rows need at least one question")
    normalized = tuple(_question(item) for item in questions if isinstance(item, dict))
    if len(normalized) != len(questions):
        raise ValueError("questions must contain objects")
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique per context")
    return MultiQuestionExample(context, normalized)


class MultiQuestionDataset(Dataset[MultiQuestionExample]):
    def __init__(self, path: str | Path) -> None:
        with Path(path).open(encoding="utf-8") as handle:
            self.examples = [validate_multi(json.loads(line)) for line in handle
                             if line.strip()]
        if not self.examples:
            raise ValueError(f"no examples in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> MultiQuestionExample:
        return self.examples[index]


def _pad(rows: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, rows))
    values = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for row, tokens in enumerate(rows):
        values[row, :len(tokens)] = torch.tensor(tokens)
    return values, values.ne(pad_id)


def _group_batch(items: list[dict[str, Any]], question_tokens: int,
                 option_tokens: int) -> dict[str, torch.Tensor]:
    questions = [_bytes(item.get("question_text", item["instructions"]), question_tokens)
                 for item in items]
    question_ids, question_mask = _pad(questions)
    result: dict[str, torch.Tensor] = {
        "question_ids": question_ids, "question_mask": question_mask,
        "state_indices": torch.tensor([item["state_index"] for item in items], dtype=torch.long),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
    }
    if items[0]["type"] == "choice":
        option_rows = [[_bytes(option, option_tokens) for option in item["options"]]
                       for item in items]
        max_options = max(map(len, option_rows))
        max_tokens = max(len(tokens) for row in option_rows for tokens in row)
        option_ids = torch.zeros((len(items), max_options, max_tokens), dtype=torch.long)
        option_mask = torch.zeros((len(items), max_options), dtype=torch.bool)
        option_token_mask = torch.zeros_like(option_ids, dtype=torch.bool)
        for row, options in enumerate(option_rows):
            option_mask[row, :len(options)] = True
            for column, tokens in enumerate(options):
                option_ids[row, column, :len(tokens)] = torch.tensor(tokens)
                option_token_mask[row, column, :len(tokens)] = True
        result.update(option_ids=option_ids, option_mask=option_mask,
                      option_token_mask=option_token_mask)
    elif items[0]["type"] == "score":
        level_rows = [[_bytes(level, option_tokens) for level in item["levels"]]
                      for item in items]
        max_levels = max(map(len, level_rows))
        max_tokens = max(len(tokens) for row in level_rows for tokens in row)
        level_ids = torch.zeros((len(items), max_levels, max_tokens), dtype=torch.long)
        level_token_mask = torch.zeros_like(level_ids, dtype=torch.bool)
        level_mask = torch.zeros((len(items), max_levels), dtype=torch.bool)
        for row, levels in enumerate(level_rows):
            level_mask[row, :len(levels)] = True
            for column, tokens in enumerate(levels):
                level_ids[row, column, :len(tokens)] = torch.tensor(tokens)
                level_token_mask[row, column, :len(tokens)] = True
        result.update(level_ids=level_ids, level_token_mask=level_token_mask,
                      level_mask=level_mask)
    return result


class MultiQuestionCollator:
    def __init__(self, context_tokens: int = 768, question_tokens: int = 512,
                 option_tokens: int = 384, max_score_levels: int = 10) -> None:
        self.context_tokens = context_tokens
        self.question_tokens = question_tokens
        self.option_tokens = option_tokens
        self.max_score_levels = max_score_levels

    def __call__(self, examples: list[MultiQuestionExample]) -> dict[str, dict[str, torch.Tensor]]:
        groups = {kind: [] for kind in ("choice", "score", "noul")}
        states = [_bytes(serialize_state(example.context), self.context_tokens)
                  for example in examples]
        state_ids, state_mask = _pad(states)
        for state_index, example in enumerate(examples):
            for question in example.questions:
                item = dict(question)
                item["state_index"] = state_index
                groups[question["type"]].append(item)
        result = {
            kind: _group_batch(items, self.question_tokens, self.option_tokens)
            for kind, items in groups.items() if items
        }
        result["states"] = {"context_ids": state_ids, "context_mask": state_mask}
        if "score" in result:
            mask = result["score"]["level_mask"]
            if mask.shape[1] > self.max_score_levels:
                raise ValueError("a Score question exceeds max_score_levels")
        return result


class MultiQuestionTinyScorer(nn.Module):
    """A shared byte encoder with independent typed decision heads."""

    def __init__(self, width: int = 64, rank: int = 64,
                 context_tokens: int = 768, question_tokens: int = 512,
                 max_score_levels: int = 10, encoder: str = "mean",
                 lexical_multiplier: float = 64.0) -> None:
        super().__init__()
        if encoder not in {"mean", "local", "lexical"}:
            raise ValueError("encoder must be mean, local or lexical")
        self.embedding = nn.Embedding(257, width, padding_idx=0)
        self.context_position = nn.Embedding(context_tokens, width)
        self.question_position = nn.Embedding(question_tokens, width)
        self.encoder = encoder
        self.lexical_multiplier = float(lexical_multiplier)
        if encoder == "local":
            # Keep the byte vocabulary but recover short-range ordering. The
            # mean encoder remains the default for old checkpoints/API users.
            self.local_encoder = nn.Sequential(
                nn.Conv1d(width, width, kernel_size=5, padding=2,
                          groups=width, bias=False),
                nn.GELU(),
                nn.Conv1d(width, width, kernel_size=1, bias=False),
            )
            self.local_scale = nn.Parameter(torch.tensor(0.25))
        if encoder == "lexical":
            # A learned weight for an input-only n-gram overlap feature. This
            # helps match unseen action descriptions compositionally without
            # exposing labels, action ids or OOD metadata to the model.
            self.lexical_scale = nn.Parameter(torch.tensor(0.25))
        self.fuse = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh())
        self.option_projection = nn.Linear(width, rank, bias=False)
        self.context_projection = nn.Linear(width, rank, bias=False)
        self.score_level_projection = nn.Linear(width, rank, bias=False)
        self.noul_head = nn.Linear(width, 1)
        self.max_score_levels = max_score_levels

    def _pool(self, ids: torch.Tensor, mask: torch.Tensor,
              positions: nn.Embedding | None = None) -> torch.Tensor:
        values = self.embedding(ids)
        if positions is not None:
            position_ids = torch.arange(ids.shape[1], device=ids.device)
            values = values + positions(position_ids)
        weights = mask.unsqueeze(-1).to(values.dtype)
        values = values * weights
        if self.encoder == "local":
            local = self.local_encoder(values.transpose(1, 2)).transpose(1, 2)
            values = (values + self.local_scale.tanh() * local) * weights
        return (values * weights).sum(1) / weights.sum(1).clamp_min(1)

    @staticmethod
    def _ngram_hist(ids: torch.Tensor, mask: torch.Tensor,
                    n: int = 5, buckets: int = 16384) -> torch.Tensor:
        """Build a compact hashed byte n-gram histogram per sequence."""
        if ids.shape[-1] < n:
            return torch.zeros(*ids.shape[:-1], buckets,
                               device=ids.device, dtype=torch.float32)
        windows = ids.unfold(-1, n, 1)
        valid = mask.unfold(-1, n, 1).all(-1)
        hashes = windows[..., 0]
        for index in range(1, n):
            hashes = (hashes * 257 + windows[..., index]) % buckets
        counts = torch.zeros(*ids.shape[:-1], buckets,
                             device=ids.device, dtype=torch.float32)
        counts.scatter_add_(-1, hashes,
                            valid.to(counts.dtype))
        return F.normalize(counts, p=2, dim=-1)

    def _lexical_overlap(self, context_ids: torch.Tensor,
                         context_mask: torch.Tensor,
                         option_ids: torch.Tensor,
                         option_mask: torch.Tensor) -> torch.Tensor:
        context_hist = self._ngram_hist(context_ids, context_mask)
        shape = option_ids.shape
        option_hist = self._ngram_hist(
            option_ids.reshape(-1, shape[-1]),
            option_mask.reshape(-1, shape[-1]),
        ).reshape(shape[0], shape[1], -1)
        return (option_hist * context_hist.unsqueeze(1)).sum(-1)

    def _base(self, context: torch.Tensor,
              question_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        question = self._pool(question_batch["question_ids"], question_batch["question_mask"],
                              self.question_position)
        state = context[question_batch["state_indices"]]
        return self.fuse(torch.cat((state, question), dim=-1))

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        # Encode every state once, then fan it out to the independent heads.
        context = self._pool(
            batch["states"]["context_ids"], batch["states"]["context_mask"],
            self.context_position,
        )
        if "choice" in batch:
            data = batch["choice"]
            base = self._base(context, data)
            option_shape = data["option_ids"].shape
            options = self._pool(
                data["option_ids"].reshape(-1, option_shape[-1]),
                data["option_token_mask"].reshape(-1, option_shape[-1]),
            ).reshape(option_shape[0], option_shape[1], -1)
            queries = self.option_projection(options)
            keys = self.context_projection(base).unsqueeze(1)
            logits = (queries * keys).sum(-1) / rank_scale(queries.shape[-1])
            if self.encoder == "lexical":
                overlap = self._lexical_overlap(
                    batch["states"]["context_ids"],
                    batch["states"]["context_mask"],
                    data["option_ids"],
                    data["option_token_mask"],
                )
                logits = logits + self.lexical_scale * (
                    self.lexical_multiplier * overlap
                )
            outputs["choice"] = logits.masked_fill(~data["option_mask"], -1e4)
        if "score" in batch:
            data = batch["score"]
            base = self._base(context, data)
            level_tokens = self.embedding(data["level_ids"])
            level_weights = data["level_token_mask"].unsqueeze(-1).to(level_tokens.dtype)
            levels = (level_tokens * level_weights).sum(2) / level_weights.sum(2).clamp_min(1)
            queries = self.context_projection(base).unsqueeze(1)
            keys = self.score_level_projection(levels)
            outputs["score"] = (queries * keys).sum(-1) / rank_scale(keys.shape[-1])
            outputs["score"] = outputs["score"].masked_fill(
                ~data["level_mask"], -1e4,
            )
        if "noul" in batch:
            outputs["noul"] = self.noul_head(
                self._base(context, batch["noul"])
            ).squeeze(-1)
        return outputs


def rank_scale(width: int) -> float:
    return max(float(width) ** 0.5, 1.0)


def multitask_loss(outputs: dict[str, torch.Tensor],
                   batch: dict[str, dict[str, torch.Tensor]],
                   score_levels: int | None = None) -> dict[str, torch.Tensor]:
    """Return per-head losses and a summed ``total`` loss."""
    losses: dict[str, torch.Tensor] = {}
    if "choice" in outputs:
        losses["choice"] = F.cross_entropy(outputs["choice"], batch["choice"]["labels"])
    if "score" in outputs:
        logits = outputs["score"]
        labels = batch["score"]["labels"]
        if "level_mask" in batch["score"]:
            level_mask = batch["score"]["level_mask"][:, :logits.shape[1]]
            logits = logits.masked_fill(~level_mask, -1e4)
        elif score_levels is not None:
            logits = logits[:, :score_levels]
        losses["score"] = F.cross_entropy(logits, labels)
    if "noul" in outputs:
        losses["noul"] = F.binary_cross_entropy_with_logits(
            outputs["noul"], batch["noul"]["labels"].float(),
        )
    if not losses:
        raise ValueError("batch contains no supported question types")
    losses["total"] = sum(losses.values())
    return losses


def apply_multitask_calibration(
    outputs: dict[str, torch.Tensor], calibration: dict[str, float] | None,
) -> dict[str, torch.Tensor]:
    """Apply post-hoc calibration without changing model parameters."""
    if not calibration:
        return outputs
    calibrated = dict(outputs)
    choice_temperature = max(float(calibration.get("choice_temperature", 1.0)), 1e-4)
    score_temperature = max(float(calibration.get("score_temperature", 1.0)), 1e-4)
    noul_scale = float(calibration.get("noul_scale", 1.0))
    noul_bias = float(calibration.get("noul_bias", 0.0))
    if "choice" in calibrated:
        calibrated["choice"] = calibrated["choice"] / choice_temperature
    if "score" in calibrated:
        calibrated["score"] = calibrated["score"] / score_temperature
    if "noul" in calibrated:
        calibrated["noul"] = calibrated["noul"] * noul_scale + noul_bias
    return calibrated
