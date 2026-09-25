"""Trainable encoder plus MASK-marker decision head."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from .sequence import QTYPES, build_sequence, mask_question_from_row, option_target


class ByteTokenizer:
    """Dependency-free tokenizer so MASK sequences can train and test on CPU."""

    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    unk_token_id = 4
    mask_token = "[MASK]"
    vocab_size = 261

    def encode(self, text: str) -> list[int]:
        return [byte + 5 for byte in text.encode("utf-8", errors="replace")]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        ids = self.encode(text)
        if add_special_tokens:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return {"input_ids": ids}


class TinyEncoder(nn.Module):
    """Small bidirectional transformer used when a pretrained BERT is not loaded."""

    def __init__(
        self, vocab_size: int = 261, hidden_size: int = 64, layers: int = 2,
        heads: int = 4, max_len: int = 768, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position = nn.Embedding(max_len, hidden_size)
        layer = nn.TransformerEncoderLayer(
            hidden_size, heads, 4 * hidden_size, dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token(input_ids) + self.position(positions.clamp_max(self.max_len - 1))
        padding = ~attention_mask.bool()
        hidden = self.transformer(hidden, src_key_padding_mask=padding)
        return SimpleNamespace(last_hidden_state=hidden)


class DecisionModel(nn.Module):
    """Bidirectional encoder backbone + typed MASK scorer."""

    def __init__(
        self, encoder: nn.Module, head_layers: int = 2, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        width = encoder.config.hidden_size
        heads = max(1, width // 64)
        layer = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout, batch_first=True, norm_first=True,
        )
        self.head = (
            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers > 0 else None
        )
        self.type_embedding = nn.Embedding(3, width)
        self.scorer = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1),
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
        marker_pos: torch.Tensor, marker_mask: torch.Tensor, qtype: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask,
        ).last_hidden_state
        hidden = hidden + self.type_embedding(qtype)[:, None, :]
        if self.head is not None:
            padding = ~attention_mask.bool()
            for layer in self.head.layers:
                hidden = layer(hidden, src_key_padding_mask=padding)
        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
        markers = torch.gather(hidden, 1, index)
        logits = self.scorer(markers).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


def collate_items(batch: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor] | None:
    items = [item for group in batch for item in (group if isinstance(group, list) else [group])]
    if not items:
        return None
    length = max(len(item["ids"]) for item in items)
    width = max(len(item["markers"]) for item in items)
    ids = torch.full((len(items), length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(items), length), dtype=torch.long)
    marker_pos = torch.zeros((len(items), width), dtype=torch.long)
    marker_mask = torch.zeros((len(items), width), dtype=torch.bool)
    target = torch.zeros((len(items), width), dtype=torch.float32)
    labels = []
    qtypes = []
    for row, item in enumerate(items):
        ids[row, :len(item["ids"])] = torch.tensor(item["ids"])
        attention[row, :len(item["ids"])] = 1
        count = len(item["markers"])
        marker_pos[row, :count] = torch.tensor(item["markers"])
        marker_mask[row, :count] = True
        aligned = item["target"]
        target[row, :len(aligned)] = torch.tensor(aligned, dtype=torch.float32)
        labels.append(item.get("label", -1))
        qtypes.append(item["qtype"])
    return {
        "input_ids": ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "target": target,
        "qtype": torch.tensor(qtypes, dtype=torch.long),
        "label": torch.tensor(labels, dtype=torch.long),
    }


class MaskCollator:
    def __init__(
        self, tokenizer, max_len: int = 768, head_max_len: int = 384,
        option_max_len: int = 48, group_size: int = 1, seed: int = 0,
        target_mode: str = "provided",
    ) -> None:
        if target_mode not in {"provided", "one_hot"}:
            raise ValueError("target_mode must be provided or one_hot")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.option_max_len = option_max_len
        self.group_size = max(1, group_size)
        self.seed = seed
        self.target_mode = target_mode

    def encode_question(
        self, state: Any, payload: dict[str, Any], option_order: list[int] | None = None,
    ) -> dict[str, Any]:
        question = mask_question_from_row(payload)
        ids, markers = build_sequence(
            self.tokenizer, state, question, self.max_len, self.head_max_len,
            option_order, self.option_max_len,
        )
        if self.target_mode == "one_hot":
            question = {**question, "target": None}
        options = option_target(question, option_order)
        if len(markers) != len(options):
            raise ValueError(
                f"options exceed head_max_len={self.head_max_len}; "
                "raise head_max_len or option_max_len"
            )
        label = int(question["label"])
        if option_order is not None:
            label = option_order.index(label)
        return {
            "ids": ids,
            "markers": markers,
            "target": options,
            "qtype": QTYPES[question["t"]],
            "label": label,
        }

    def _orders(self, count: int, example_index: int) -> list[list[int]]:
        natural = list(range(count))
        orders = [natural]
        generator = torch.Generator().manual_seed(self.seed + 104729 * example_index)
        for _ in range(self.group_size - 1):
            if count > 1:
                orders.append(torch.randperm(count, generator=generator).tolist())
            else:
                orders.append(natural)
        return orders

    def __call__(self, examples: list[Any]) -> dict[str, torch.Tensor]:
        grouped: list[list[dict[str, Any]]] = []
        for example_index, example in enumerate(examples):
            group: list[dict[str, Any]] = []
            for question in example.questions:
                spec = mask_question_from_row(question)
                count = len(option_target(spec))
                for order in self._orders(count, example_index):
                    group.append(self.encode_question(example.context, question, order))
            grouped.append(group)
        batch = collate_items(grouped, self.tokenizer.pad_token_id)
        if batch is None:
            raise ValueError("empty MASK batch")
        batch["group_size"] = torch.tensor(self.group_size)
        return batch


def make_tokenizer(config: dict[str, Any]):
    if config.get("encoder", "tiny") == "tiny":
        return ByteTokenizer()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["hf_model"])
    if tokenizer.mask_token_id is None:
        raise ValueError("MASK encoder requires a tokenizer with a [MASK] token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.sep_token or tokenizer.eos_token
    if tokenizer.cls_token_id is None:
        tokenizer.cls_token = tokenizer.bos_token or tokenizer.pad_token
    if tokenizer.sep_token_id is None:
        tokenizer.sep_token = tokenizer.eos_token or tokenizer.pad_token
    return tokenizer


def make_encoder(config: dict[str, Any]) -> nn.Module:
    if config.get("encoder", "tiny") == "tiny":
        return TinyEncoder(
            vocab_size=ByteTokenizer.vocab_size,
            hidden_size=config.get("width", 64),
            layers=config.get("encoder_layers", 2),
            heads=config.get("encoder_heads", 4),
            max_len=config.get("max_len", 768),
        )
    from transformers import AutoModel

    return AutoModel.from_pretrained(config["hf_model"])


def make_decision_model(config: dict[str, Any], device: torch.device) -> tuple[DecisionModel, Any]:
    tokenizer = make_tokenizer(config)
    model = DecisionModel(
        make_encoder(config),
        head_layers=config.get("head_layers", 2),
        dropout=config.get("dropout", 0.1),
    )
    temperature = config.get("temperature", [1.0, 1.0, 1.0])
    model.temperature.copy_(torch.tensor(temperature, dtype=model.temperature.dtype))
    return model.to(device), tokenizer


def load_decision_checkpoint(path: str | Path, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model, tokenizer = make_decision_model(config, device)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise ValueError(f"checkpoint mismatch: unexpected={unexpected}")
    if any(not name.startswith("encoder.") for name in missing):
        raise ValueError(f"checkpoint mismatch: missing={missing}")
    return model, tokenizer, config


def save_decision_checkpoint(
    path: str | Path, model: DecisionModel, config: dict[str, Any], extra: dict | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "mask",
        "config": config,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, output)
