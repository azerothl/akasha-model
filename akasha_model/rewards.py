"""Strictly proper scoring rewards used by RLCD."""

from __future__ import annotations

import math

import numpy as np
import torch

from .sequence import QTYPE_NAMES, QTYPES


def proper_reward(
    reported: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    spherical_weight: float = 0.5,
    ranked_weight: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Log score + spherical score, minus ranked probability score on ordinal heads.

    ``reported`` is a distribution over options, shape ``[..., N, K]`` or ``[N, K]``.
    ``target`` is a one-hot or soft gold distribution, shape ``[N, K]``.
    """
    reported = reported * mask
    log_reported = torch.log(reported.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * log_reported).sum(-1)
    spherical = (target * reported).sum(-1) / reported.norm(dim=-1).clamp_min(1e-9)
    reward = log_score + spherical_weight * spherical
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        count = mask.sum(-1).clamp(min=2).float()
        cdf_reported = torch.cumsum(reported, -1)
        cdf_target = torch.cumsum(target, -1)
        ranked = (((cdf_reported - cdf_target) ** 2) * mask).sum(-1) / (count - 1)
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
