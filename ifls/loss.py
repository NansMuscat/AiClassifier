"""Loss functions for hierarchical IFLS classification."""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss: FL(p) = -α(1-p)^γ · log(p)

    gamma=0 reduces to standard weighted cross-entropy.
    gamma=2 is the value from the original paper and works well here.

    Args:
        gamma: focusing parameter.
        weight: optional per-class weight tensor (shape: [num_classes]).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def compute_class_weights(
    indices: List[int],
    num_classes: int,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Inverse-frequency weights, clamped to avoid pathological values
    for near-empty segments.

    weight[c] = total / (num_classes * count[c])
    Clipped at max_weight.
    """
    counts  = Counter(indices)
    total   = len(indices)
    weights = torch.ones(num_classes)
    for cls, cnt in counts.items():
        weights[cls] = total / (num_classes * cnt)
    return weights.clamp(max=max_weight)
