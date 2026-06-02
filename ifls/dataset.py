"""PyTorch Dataset for IFLS hierarchical classification."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ifls.hierarchy import HierarchyEncoder


class IFLSDataset(Dataset):
    """
    Each item returns:
      input_ids, attention_mask  — tokenized product name
      label_group, label_family, label_subfamily, label_segment  — model indices
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        hierarchy_encoder: HierarchyEncoder,
        max_length: int = 32,
    ) -> None:
        self.texts = df["text"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-encode all labels to avoid doing it inside __getitem__
        encoded = [hierarchy_encoder.encode_row(row) for _, row in df.iterrows()]
        self.labels: Dict[str, torch.Tensor] = {
            lvl: torch.tensor([e[lvl] for e in encoded], dtype=torch.long)
            for lvl in ("group", "family", "subfamily", "segment")
        }

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":        enc["input_ids"].squeeze(0),
            "attention_mask":   enc["attention_mask"].squeeze(0),
            "label_group":      self.labels["group"][idx],
            "label_family":     self.labels["family"][idx],
            "label_subfamily":  self.labels["subfamily"][idx],
            "labels":           self.labels["segment"][idx],
        }
