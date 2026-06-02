"""
Cascaded inference with hierarchy-constrained prediction.

At each level the valid candidates are restricted to children of the
predicted parent, masking all others to -inf before softmax.
Temperature scaling is applied at the segment level only (the level
that was calibrated).

All outputs use original DB long IDs, not model indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from ifls.calibrate import TemperatureScaler
from ifls.hierarchy import HierarchyEncoder
from ifls.model import IFLSMultiHeadClassifier
from ifls.normalize import normalize


@dataclass
class Prediction:
    text: str
    # DB long IDs (what callers care about)
    segment_id:   int
    subfamily_id: int
    family_id:    int
    group_id:     int
    # Per-level confidence after softmax (and temperature scaling for segment)
    confidence_segment:   float
    confidence_subfamily: float
    confidence_family:    float
    confidence_group:     float
    # Flag when segment confidence is below the configured threshold
    uncertain: bool


@torch.no_grad()
def predict_batch(
    texts: List[str],
    model: IFLSMultiHeadClassifier,
    tokenizer: PreTrainedTokenizerBase,
    hierarchy_encoder: HierarchyEncoder,
    scaler: TemperatureScaler,
    max_length: int = 32,
    confidence_threshold: float = 0.5,
    device: str = "cpu",
) -> List[Prediction]:
    """
    Run cascaded inference on a list of raw product name strings.

    Args:
        texts: raw product names (normalisation is applied here).
        model: trained IFLSMultiHeadClassifier in eval mode.
        tokenizer: matching tokenizer.
        hierarchy_encoder: fitted HierarchyEncoder.
        scaler: fitted TemperatureScaler.
        max_length: tokenizer max length (should match training).
        confidence_threshold: predictions below this are flagged as uncertain.
        device: "cpu" or "cuda".

    Returns:
        List of Prediction dataclasses, one per input text.
    """
    model.eval()
    model.to(device)
    scaler.to(device)

    normalized = [normalize(t) for t in texts]

    enc = tokenizer(
        normalized,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    all_logits = model.forward_all(input_ids, attention_mask)
    # all_logits: {level: [B, num_classes]}

    results: List[Prediction] = []
    batch_size = input_ids.shape[0]
    he = hierarchy_encoder

    for i in range(batch_size):

        # ── GROUP (unconstrained) ──────────────────────────────────────────
        g_logits = all_logits["group"][i]
        g_idx    = int(g_logits.argmax())
        g_prob   = float(F.softmax(g_logits, dim=-1).max())

        # ── FAMILY (children of predicted group) ──────────────────────────
        f_logits = _mask_to_valid(
            all_logits["family"][i],
            he.parent_to_children["family"].get(g_idx, set()),
        )
        f_idx  = int(f_logits.argmax())
        f_prob = float(F.softmax(f_logits, dim=-1)[f_idx])

        # ── SUBFAMILY (children of predicted family) ───────────────────────
        s_logits = _mask_to_valid(
            all_logits["subfamily"][i],
            he.parent_to_children["subfamily"].get(f_idx, set()),
        )
        s_idx  = int(s_logits.argmax())
        s_prob = float(F.softmax(s_logits, dim=-1)[s_idx])

        # ── SEGMENT (children of predicted subfamily + temperature scaling) ─
        seg_raw     = all_logits["segment"][i].unsqueeze(0)
        seg_cal     = scaler(seg_raw).squeeze(0)
        seg_logits  = _mask_to_valid(
            seg_cal,
            he.parent_to_children["segment"].get(s_idx, set()),
        )
        seg_idx  = int(seg_logits.argmax())
        seg_prob = float(F.softmax(seg_logits, dim=-1)[seg_idx])

        results.append(Prediction(
            text=texts[i],
            segment_id=he.encoders["segment"].to_id(seg_idx),
            subfamily_id=he.encoders["subfamily"].to_id(s_idx),
            family_id=he.encoders["family"].to_id(f_idx),
            group_id=he.encoders["group"].to_id(g_idx),
            confidence_segment=seg_prob,
            confidence_subfamily=s_prob,
            confidence_family=f_prob,
            confidence_group=g_prob,
            uncertain=seg_prob < confidence_threshold,
        ))

    return results


def _mask_to_valid(logits: torch.Tensor, valid_indices: set) -> torch.Tensor:
    """
    Set all logits except valid_indices to -inf.
    If valid_indices is empty (unknown parent), return logits unchanged
    so the model falls back to unconstrained argmax.
    """
    if not valid_indices:
        return logits
    mask = torch.full_like(logits, float("-inf"))
    idx  = torch.tensor(list(valid_indices), dtype=torch.long, device=logits.device)
    mask[idx] = logits[idx]
    return mask
