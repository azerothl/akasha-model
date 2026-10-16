"""Regression tests for #86 (ordinal score loss) and #87 (MASK temperatures)."""

from __future__ import annotations

import torch

from akasha_model.decision import (
    ByteTokenizer,
    DecisionModel,
    MaskCollator,
    TinyEncoder,
)
from akasha_model.rewards import proper_reward
from akasha_model.rlcd import grpo_loss
from akasha_model.sequence import QTYPES


def _score_reward(reported, target, option_order=None):
    return proper_reward(
        torch.tensor([reported], dtype=torch.float32),
        torch.tensor([target], dtype=torch.float32),
        torch.tensor([QTYPES["score"]]),
        torch.ones(1, len(reported), dtype=torch.bool),
        option_order=(
            None if option_order is None else torch.tensor([option_order])
        ),
    )


def test_proper_reward_score_is_invariant_to_option_permutation():
    canonical = _score_reward([0.6, 0.3, 0.1], [1.0, 0.0, 0.0])
    order = [2, 0, 1]
    permuted_reported = [[0.6, 0.3, 0.1][index] for index in order]
    permuted_target = [[1.0, 0.0, 0.0][index] for index in order]
    permuted = _score_reward(permuted_reported, permuted_target, order)
    assert torch.allclose(canonical, permuted, atol=1e-6)


def test_proper_reward_penalises_far_score_error_more_than_adjacent():
    adjacent = _score_reward([0.5, 0.5, 0.0], [1.0, 0.0, 0.0])
    far = _score_reward([0.5, 0.0, 0.5], [1.0, 0.0, 0.0])
    assert float(adjacent) > float(far)


def test_grpo_loss_uses_collator_option_order():
    # One score question displayed in permuted order [2, 0, 1]; group_size=1.
    logits = torch.tensor([[2.0, 0.0, 1.0]])
    batch = {
        "marker_mask": torch.ones(1, 3, dtype=torch.bool),
        "qtype": torch.tensor([QTYPES["score"]]),
        "target": torch.tensor([[0.0, 1.0, 0.0]]),
        "option_order": torch.tensor([[2, 0, 1]]),
        "group_size": torch.tensor(1),
    }
    losses = grpo_loss(logits, batch, 0.5, 1.0, 1.0)
    assert torch.isfinite(losses["total"])


def test_mask_collator_records_option_order_and_realigns_target():
    collator = MaskCollator(ByteTokenizer())
    item = collator.encode_question(
        "state",
        {
            "type": "score",
            "instructions": "rate it",
            "levels": ["low", "mid", "high"],
            "label": 0,
            "target": [1.0, 0.0, 0.0],
        },
        option_order=[2, 0, 1],
    )
    assert item["order"] == [2, 0, 1]
    assert item["target"] == [0.0, 1.0, 0.0]
    assert item["label"] == 1


def _tiny_model() -> DecisionModel:
    torch.manual_seed(0)
    encoder = TinyEncoder(hidden_size=64, layers=1, heads=4, dropout=0.0)
    return DecisionModel(encoder, head_layers=0, dropout=0.0)


def _forward(model: DecisionModel) -> torch.Tensor:
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "marker_pos": torch.zeros(1, 1, dtype=torch.long),
        "marker_mask": torch.ones(1, 1, dtype=torch.bool),
        "qtype": torch.tensor([QTYPES["score"]]),
    }
    return model(**batch)


def test_temperature_applies_in_eval_only_and_preserves_identity_at_one():
    model = _tiny_model()
    model.eval()
    raw = _forward(model)
    model.set_temperature([1.0, 2.0, 1.0])
    calibrated = _forward(model)
    assert torch.allclose(calibrated, raw / 2.0, atol=1e-5)
    model.train()
    assert torch.allclose(_forward(model), raw, atol=1e-5)


def test_temperature_must_be_strictly_positive():
    model = _tiny_model()
    for bad in ([1.0, 0.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0]):
        try:
            model.set_temperature(bad)
        except ValueError:
            continue
        raise AssertionError(f"temperature {bad} should be rejected")
