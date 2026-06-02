"""
Hierarchy encoding and tree structure.

Responsibility:
  - Map arbitrary DB long IDs → contiguous 0-based model indices (and back).
  - Store parent/child relationships for cascaded inference.
  - Serialize/deserialize from a single hierarchy.json file.

The CSV is expected to have columns:
  group_id, family_id, subfamily_id, segment_id  (all arbitrary longs)

Internally the model works with indices 0..N-1.
All public-facing outputs (predictions, exports) use the original DB IDs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

LEVELS = ("group", "family", "subfamily", "segment")


class LevelEncoder:
    """Bidirectional map between DB long IDs and contiguous model indices."""

    def __init__(self) -> None:
        self._id_to_idx: Dict[int, int] = {}
        self._idx_to_id: Dict[int, int] = {}

    def fit(self, ids: List[int]) -> "LevelEncoder":
        for idx, db_id in enumerate(sorted(set(ids))):
            self._id_to_idx[db_id] = idx
            self._idx_to_id[idx] = db_id
        return self

    def to_idx(self, db_id: int) -> int:
        return self._id_to_idx[db_id]

    def to_id(self, idx: int) -> int:
        return self._idx_to_id[idx]

    def num_classes(self) -> int:
        return len(self._id_to_idx)

    # ── serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        # Store as list of [db_id, idx] pairs sorted by idx for readability
        return {"mapping": [[self._idx_to_id[i], i] for i in range(self.num_classes())]}

    @classmethod
    def from_dict(cls, data: Dict) -> "LevelEncoder":
        enc = cls()
        for db_id, idx in data["mapping"]:
            enc._id_to_idx[db_id] = idx
            enc._idx_to_id[idx] = db_id
        return enc


class HierarchyEncoder:
    """
    Full hierarchy encoder for all four IFLS levels.

    After fit():
      - self.encoders[level]  — LevelEncoder for that level
      - self.child_to_parent  — {child_level: {child_idx: parent_idx}}
      - self.parent_to_children — {child_level: {parent_idx: set(child_idx)}}

    child_level is the *child* in the pair, e.g.:
      child_to_parent["family"][family_idx] = group_idx
      parent_to_children["family"][group_idx] = {family_idx, ...}
    """

    def __init__(self) -> None:
        self.encoders: Dict[str, LevelEncoder] = {lvl: LevelEncoder() for lvl in LEVELS}
        # Keyed by child level name
        self.child_to_parent: Dict[str, Dict[int, int]] = {}
        self.parent_to_children: Dict[str, Dict[int, Set[int]]] = {}

    # ── fitting ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "HierarchyEncoder":
        """
        df must contain columns: group_id, family_id, subfamily_id, segment_id
        (all arbitrary long integers).
        """
        for lvl in LEVELS:
            self.encoders[lvl].fit(df[f"{lvl}_id"].tolist())

        pairs = [
            ("group",     "family"),
            ("family",    "subfamily"),
            ("subfamily", "segment"),
        ]
        for parent_lvl, child_lvl in pairs:
            c2p: Dict[int, int] = {}
            p2c: Dict[int, Set[int]] = defaultdict(set)
            for _, row in df[[f"{parent_lvl}_id", f"{child_lvl}_id"]].drop_duplicates().iterrows():
                p_idx = self.encoders[parent_lvl].to_idx(int(row[f"{parent_lvl}_id"]))
                c_idx = self.encoders[child_lvl].to_idx(int(row[f"{child_lvl}_id"]))
                c2p[c_idx] = p_idx
                p2c[p_idx].add(c_idx)
            self.child_to_parent[child_lvl] = c2p
            self.parent_to_children[child_lvl] = {k: v for k, v in p2c.items()}

        return self

    # ── encode / decode convenience ───────────────────────────────────────

    def encode_row(self, row: pd.Series) -> Dict[str, int]:
        """Return model indices for all four levels from a dataframe row."""
        return {lvl: self.encoders[lvl].to_idx(int(row[f"{lvl}_id"])) for lvl in LEVELS}

    def decode_segment(self, segment_idx: int) -> int:
        """Return the DB segment ID for a model index."""
        return self.encoders["segment"].to_id(segment_idx)

    def num_classes(self, level: str) -> int:
        return self.encoders[level].num_classes()

    # ── serialisation ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data: Dict = {
            "encoders": {lvl: self.encoders[lvl].to_dict() for lvl in LEVELS},
            "child_to_parent": {
                child_lvl: {str(c): p for c, p in mapping.items()}
                for child_lvl, mapping in self.child_to_parent.items()
            },
            "parent_to_children": {
                child_lvl: {str(p): list(children) for p, children in mapping.items()}
                for child_lvl, mapping in self.parent_to_children.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HierarchyEncoder":
        data = json.loads(Path(path).read_text())
        enc = cls()
        enc.encoders = {lvl: LevelEncoder.from_dict(data["encoders"][lvl]) for lvl in LEVELS}
        enc.child_to_parent = {
            child_lvl: {int(c): p for c, p in mapping.items()}
            for child_lvl, mapping in data["child_to_parent"].items()
        }
        enc.parent_to_children = {
            child_lvl: {int(p): set(children) for p, children in mapping.items()}
            for child_lvl, mapping in data["parent_to_children"].items()
        }
        return enc
