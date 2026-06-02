"""
Multi-head IFLS classifier.

Architecture:
  DistilBERT backbone (LoRA fine-tuned)
    └─ [CLS] embedding (768-dim)
         ├─ Head: group      (linear → GELU → dropout → linear)
         ├─ Head: family
         ├─ Head: subfamily
         └─ Head: segment

The combined loss is a weighted sum of four FocalLoss terms, one per level.
segment logits are returned as the primary logits for HuggingFace Trainer metrics.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from ifls.loss import FocalLoss


LEVELS = ("group", "family", "subfamily", "segment")


def _classification_head(hidden_size: int, num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, num_classes),
    )


class IFLSMultiHeadClassifier(PreTrainedModel):
    """
    Args:
        config: HuggingFace config (used by PreTrainedModel; backbone loaded separately).
        num_classes: dict mapping level name → number of classes.
        level_weights: dict mapping level name → loss weight (should sum to 1).
        focal_gamma: gamma for focal loss.
        focal_weights: optional dict mapping level name → per-class weight tensor.
        head_dropout: dropout inside each classification head.
    """

    _tied_weights_keys = []

    @property
    def all_tied_weights_keys(self) -> dict:
        return {}

    def __init__(
        self,
        config,
        num_classes: Dict[str, int] = None,
        level_weights: Dict[str, float] = None,
        focal_gamma: float = None,
        focal_weights: Optional[Dict[str, torch.Tensor]] = None,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__(config)
        num_classes   = num_classes   if num_classes   is not None else config.num_classes
        level_weights = level_weights if level_weights is not None else config.level_weights
        focal_gamma   = focal_gamma   if focal_gamma   is not None else getattr(config, "focal_gamma", 2.0)
        self.level_weights = level_weights

        self.backbone = AutoModel.from_config(config)
        hidden = self.backbone.config.hidden_size

        self.heads = nn.ModuleDict({
            lvl: _classification_head(hidden, num_classes[lvl], head_dropout)
            for lvl in LEVELS
        })

        fw = focal_weights or {}
        self.focal_losses = nn.ModuleDict({
            lvl: FocalLoss(gamma=focal_gamma, weight=fw.get(lvl))
            for lvl in LEVELS
        })

    # ── forward (training + eval via Trainer) ─────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_group: Optional[torch.Tensor] = None,
        label_family: Optional[torch.Tensor] = None,
        label_subfamily: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:

        cls_emb = self._encode(input_ids, attention_mask)
        logits  = {lvl: head(cls_emb) for lvl, head in self.heads.items()}

        loss = None
        if labels is not None:
            all_labels = {
                "group":     label_group,
                "family":    label_family,
                "subfamily": label_subfamily,
                "segment":   labels,
            }
            loss = sum(
                self.level_weights[lvl] * self.focal_losses[lvl](logits[lvl], lbl)
                for lvl, lbl in all_labels.items()
            )

        # Trainer uses .logits for metric computation → return segment logits
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits["segment"],
        )

    # ── inference helper ──────────────────────────────────────────────────

    def forward_all(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return logits for all four levels. Used by cascaded inference."""
        cls_emb = self._encode(input_ids, attention_mask)
        return {lvl: head(cls_emb) for lvl, head in self.heads.items()}

    # ── private ───────────────────────────────────────────────────────────

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]   # [B, H]
