"""HuggingFace Trainer subclass with hierarchical sampling."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Trainer


class HierarchicalTrainer(Trainer):
    """Injects a WeightedRandomSampler so segments are drawn uniformly."""

    def __init__(self, *args, sampler: WeightedRandomSampler, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sampler = sampler

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._sampler,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
