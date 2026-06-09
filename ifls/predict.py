"""
Inference with temperature-scaled confidence.

All outputs use original DB long IDs, not model indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ifls.calibrate import TemperatureScaler
from ifls.hierarchy import HierarchyEncoder
from ifls.normalize import normalize


@dataclass
class Prediction:
    text: str
    segment_id:   int
    subfamily_id: int
    family_id:    int
    group_id:     int
    confidence:   float
    uncertain:    bool


@torch.no_grad()
def predict_batch(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    hierarchy_encoder: HierarchyEncoder,
    scaler: TemperatureScaler,
    max_length: int = 32,
    confidence_threshold: float = 0.5,
    device: str = "cpu",
) -> List[Prediction]:
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

    logits      = model(input_ids=input_ids, attention_mask=attention_mask).logits
    cal_logits  = scaler(logits)
    probs       = F.softmax(cal_logits, dim=-1)
    seg_idxs    = probs.argmax(dim=-1).tolist()
    confidences = probs.max(dim=-1).values.tolist()

    results: List[Prediction] = []
    for text, seg_idx, conf in zip(texts, seg_idxs, confidences):
        path = hierarchy_encoder.segment_idx_to_path(seg_idx)
        results.append(Prediction(
            text=text,
            segment_id=path["segment_id"],
            subfamily_id=path["subfamily_id"],
            family_id=path["family_id"],
            group_id=path["group_id"],
            confidence=conf,
            uncertain=conf < confidence_threshold,
        ))

    return results
