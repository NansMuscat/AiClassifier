import json
import tempfile
from pathlib import Path

import pytest

from ifls.hierarchy import HierarchyEncoder


def test_num_classes(fitted_encoder):
    assert fitted_encoder.num_classes("group") == 3
    assert fitted_encoder.num_classes("family") == 6
    assert fitted_encoder.num_classes("subfamily") == 12
    assert fitted_encoder.num_classes("segment") == 24


def test_roundtrip_db_ids(fitted_encoder):
    """to_idx then to_id should return the original DB ID."""
    for db_id in [1000, 1001, 1002]:
        idx = fitted_encoder.encoders["group"].to_idx(db_id)
        assert fitted_encoder.encoders["group"].to_id(idx) == db_id


def test_indices_are_contiguous(fitted_encoder):
    """Model indices must be 0-based and contiguous."""
    for lvl in ("group", "family", "subfamily", "segment"):
        n = fitted_encoder.num_classes(lvl)
        indices = {
            fitted_encoder.encoders[lvl].to_idx(db_id)
            for db_id in [
                fitted_encoder.encoders[lvl].to_id(i) for i in range(n)
            ]
        }
        assert indices == set(range(n))


def test_parent_child_consistency(fitted_encoder):
    """Every child_to_parent entry must appear in the corresponding parent_to_children."""
    for child_lvl, c2p in fitted_encoder.child_to_parent.items():
        p2c = fitted_encoder.parent_to_children[child_lvl]
        for child_idx, parent_idx in c2p.items():
            assert child_idx in p2c[parent_idx], (
                f"{child_lvl}: child {child_idx} maps to parent {parent_idx} "
                f"but is not in parent_to_children"
            )


def test_no_cross_branch_children(fitted_encoder):
    """Children of one parent should not appear under another parent."""
    p2c = fitted_encoder.parent_to_children["family"]
    all_children = [c for children in p2c.values() for c in children]
    assert len(all_children) == len(set(all_children)), "Duplicate children across parents"


def test_save_and_load(fitted_encoder, tmp_path):
    path = tmp_path / "hierarchy.json"
    fitted_encoder.save(path)

    loaded = HierarchyEncoder.load(path)
    assert loaded.num_classes("segment") == fitted_encoder.num_classes("segment")

    # Spot-check one roundtrip
    db_id = 4000
    original_idx = fitted_encoder.encoders["segment"].to_idx(db_id)
    loaded_idx   = loaded.encoders["segment"].to_idx(db_id)
    assert original_idx == loaded_idx


def test_encode_row(fitted_encoder, tiny_df):
    row = tiny_df.iloc[0]
    encoded = fitted_encoder.encode_row(row)
    assert set(encoded.keys()) == {"group", "family", "subfamily", "segment"}
    for lvl, idx in encoded.items():
        assert 0 <= idx < fitted_encoder.num_classes(lvl)
