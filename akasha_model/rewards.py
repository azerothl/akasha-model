"""Strictly proper scoring rewards used by RLCD."""

from __future__ import annotations

import math

import numpy as np
import torch

from .sequence import QTYPE_NAMES, QTYPES


def _to_canonical_order(values: torch.Tensor, option_order: torch.Tensor) -> torch.Tensor:
    """Scatter a permuted ``[..., K]`` tensor back to canonical option order.

    ``option_order[..., p]`` is the canonical index of the option shown at
    position ``p``, so ``canonical[..., option_order[p]] == values[..., p]``.
    """
    if values.shape[-1] != option_order.shape[-1]:
        raise ValueError("option_order width must match the option count")
    index = option_order.to(device=values.device, dtype=torch.long)
    expanded = torch.broadcast_to(index, values.shape)
    canonical = torch.zeros_like(values)
    return canonical.scatter(-1, expanded, values)


def proper_reward(
    reported: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    spherical_weight: float = 0.5,
    ranked_weight: float = 1.0,
    log_floor: float = -9.21,
    option_order: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log score + spherical score, minus ranked probability score on ordinal heads.

    ``reported`` is a distribution over options, shape ``[..., N, K]`` or ``[N, K]``.
    ``target`` is a one-hot or soft gold distribution, shape ``[N, K]``.

    The ranked (CDF) term is only meaningful in the canonical ordinal order
    of ``score`` levels. When options were permuted for augmentation/RLCD,
    pass ``option_order`` (canonical index per displayed position, as emitted
    by ``MaskCollator``) so both distributions are restored to that order
    before the CDF is taken; permuting the display order then leaves the
    reward unchanged. Without ``option_order`` the inputs are assumed to
    already be in canonical order.
    """
    reported = reported * mask
    log_reported = torch.log(reported.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * log_reported).sum(-1)
    spherical = (target * reported).sum(-1) / reported.norm(dim=-1).clamp_min(1e-9)
    reward = log_score + spherical_weight * spherical
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        ranked_reported, ranked_target, ranked_mask = reported, target, mask
        if option_order is not None:
            ranked_reported = _to_canonical_order(reported, option_order)
            ranked_target = _to_canonical_order(target, option_order)
            ranked_mask = _to_canonical_order(mask.to(reported.dtype), option_order) > 0
        count = ranked_mask.sum(-1).clamp(min=2).float()
        cdf_reported = torch.cumsum(ranked_reported, -1)
        cdf_target = torch.cumsum(ranked_target, -1)
        ranked = (((cdf_reported - cdf_target) ** 2) * ranked_mask).sum(-1) / (count - 1)
        reward = reward - ranked_weight * ranked * is_score
    return reward


def confidence_from_probs(probabilities: np.ndarray, count: int) -> float:
    """Normalised Shannon entropy: ``1 - H(p) / log(k)``."""
    if count < 2:
        return 1.0
    values = probabilities[:count]
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - entropy / math.log(count), 0.0, 1.0))


def temperature_bucket(qtype: int, count: int) -> str:
    size = "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
    return f"{QTYPE_NAMES[int(qtype)]}:{size}"
