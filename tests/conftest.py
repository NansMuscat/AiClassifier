"""Shared pytest fixtures."""

import pandas as pd
import pytest

from ifls.hierarchy import HierarchyEncoder


@pytest.fixture(scope="module")
def tiny_df():
    """
    Minimal synthetic dataframe with 3 groups, 6 families, 12 subfamilies, 24 segments.
    IDs are arbitrary longs to mimic real DB PKs.
    """
    rows = []
    for g in range(3):
        for f in range(2):
            for s in range(2):
                for seg in range(2):
                    rows.append({
                        "product_name": f"product g{g} f{f} s{s} seg{seg}",
                        "text":         f"product g{g} f{f} s{s} seg{seg}",
                        "group_id":     1000 + g,
                        "family_id":    2000 + g * 2 + f,
                        "subfamily_id": 3000 + g * 4 + f * 2 + s,
                        "segment_id":   4000 + g * 8 + f * 4 + s * 2 + seg,
                    })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted_encoder(tiny_df):
    return HierarchyEncoder().fit(tiny_df)
