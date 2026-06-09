"""PyTorch Dataset for IFLS segment classification."""

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
      labels                     — segment model index
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
        seg_enc = hierarchy_encoder.encoders["segment"]
        self.labels = torch.tensor(
            [seg_enc.to_idx(int(sid)) for sid in df["segment_id"]],
            dtype=torch.long,
        )

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
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         self.labels[idx],
        }
