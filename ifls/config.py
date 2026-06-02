"""Load and expose the YAML config as a typed dataclass."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    csv_path: str
    text_column: str
    hierarchy_json: str
    encoders_dir: str
    train_ratio: float
    calib_ratio: float
    random_seed: int
    max_rows: int = None  # if set, truncate CSV after loading (dev/smoke use)


@dataclass
class ModelConfig:
    backbone: str
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]


@dataclass
class LossConfig:
    focal_gamma: float
    level_weights: Dict[str, float]
    max_class_weight: float


@dataclass
class TrainingConfig:
    output_dir: str
    batch_size: int
    learning_rate: float
    epochs: int
    fp16: bool
    dataloader_num_workers: int
    save_total_limit: int
    max_steps: int = -1  # if > 0, overrides epochs (dev/smoke use)


@dataclass
class CalibrationConfig:
    temperature_lr: float
    temperature_max_iter: int
    temperature_output: str


@dataclass
class InferenceConfig:
    confidence_threshold: float


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    inference: InferenceConfig

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        return Config(
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            loss=LossConfig(**raw["loss"]),
            training=TrainingConfig(**raw["training"]),
            calibration=CalibrationConfig(**raw["calibration"]),
            inference=InferenceConfig(**raw["inference"]),
        )
