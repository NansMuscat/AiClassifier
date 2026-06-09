"""
Hierarchy encoding for IFLS levels.

Responsibility:
  - Map arbitrary DB long IDs → contiguous 0-based model indices (and back).
  - Provide a full-path lookup: segment_idx → {group_id, family_id, subfamily_id, segment_id}.
  - Serialize/deserialize from a single hierarchy.json file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

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

    def to_dict(self) -> Dict:
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
    Encoder for all four IFLS levels.

    After fit():
      - self.encoders[level]  — LevelEncoder for that level
      - self._path_lookup     — segment_idx → {group_id, family_id, subfamily_id, segment_id}
    """

    def __init__(self) -> None:
        self.encoders: Dict[str, LevelEncoder] = {lvl: LevelEncoder() for lvl in LEVELS}
        self._path_lookup: Dict[int, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> "HierarchyEncoder":
        for lvl in LEVELS:
            self.encoders[lvl].fit(df[f"{lvl}_id"].tolist())

        path_cols = ["group_id", "family_id", "subfamily_id", "segment_id"]
        for _, row in df[path_cols].drop_duplicates().iterrows():
            seg_idx = self.encoders["segment"].to_idx(int(row["segment_id"]))
            self._path_lookup[seg_idx] = {
                "group_id":     int(row["group_id"]),
                "family_id":    int(row["family_id"]),
                "subfamily_id": int(row["subfamily_id"]),
                "segment_id":   int(row["segment_id"]),
            }

        return self

    def num_classes(self, level: str) -> int:
        return self.encoders[level].num_classes()

    def segment_idx_to_path(self, segment_idx: int) -> Dict[str, int]:
        """Return the full IFLS path (all DB IDs) for a segment model index."""
        return self._path_lookup[segment_idx]

    def save(self, path: str | Path) -> None:
        data = {
            "encoders":    {lvl: self.encoders[lvl].to_dict() for lvl in LEVELS},
            "path_lookup": {str(k): v for k, v in self._path_lookup.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "HierarchyEncoder":
        data = json.loads(Path(path).read_text())
        enc = cls()
        enc.encoders = {lvl: LevelEncoder.from_dict(data["encoders"][lvl]) for lvl in LEVELS}
        enc._path_lookup = {int(k): v for k, v in data["path_lookup"].items()}
        return enc
