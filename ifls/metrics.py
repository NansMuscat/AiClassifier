"""Evaluation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report


def compute_metrics_fn(eval_pred):
    """Passed directly to HuggingFace Trainer. Reports segment accuracy."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


def full_report(
    true_segment_indices: list[int],
    pred_segment_indices: list[int],
    segment_id_labels: list[str] | None = None,
) -> str:
    """
    Human-readable classification report for the segment level.

    Args:
        true_segment_indices: ground-truth model indices.
        pred_segment_indices: predicted model indices.
        segment_id_labels: optional list mapping index → DB ID string for display.
    """
    return classification_report(
        true_segment_indices,
        pred_segment_indices,
        target_names=segment_id_labels,
        zero_division=0,
    )
