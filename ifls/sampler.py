"""Hierarchical weighted sampler.

Assigns each training example a weight of 1/segment_count so the DataLoader
draws segments roughly uniformly regardless of how many products each contains.
"""

from __future__ import annotations

from collections import Counter
from typing import List

import pandas as pd
from torch.utils.data import WeightedRandomSampler

from ifls.hierarchy import HierarchyEncoder


def make_hierarchical_sampler(
    df: pd.DataFrame,
    hierarchy_encoder: HierarchyEncoder,
) -> WeightedRandomSampler:
    """
    Returns a WeightedRandomSampler sized to len(df).

    Args:
        df: training dataframe with segment_id column.
        hierarchy_encoder: fitted encoder (used to convert IDs to indices).
    """
    segment_indices: List[int] = [
        hierarchy_encoder.encoders["segment"].to_idx(int(sid))
        for sid in df["segment_id"].tolist()
    ]
    counts = Counter(segment_indices)
    weights = [1.0 / counts[s] for s in segment_indices]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
