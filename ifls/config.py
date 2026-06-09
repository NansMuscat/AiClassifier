"""Load and expose the YAML config as a typed dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class DataConfig:
    csv_path: str
    text_column: str
    hierarchy_json: str
    encoders_dir: str
    train_ratio: float
    calib_ratio: float
    random_seed: int
    max_rows: int = None


@dataclass
class ModelConfig:
    backbone: str
    max_length: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]


@dataclass
class TrainingConfig:
    output_dir: str
    batch_size: int
    learning_rate: float
    epochs: int
    fp16: bool
    dataloader_num_workers: int
    save_total_limit: int
    label_smoothing: float = 0.1
    max_steps: int = -1
    warmup_ratio: float = 0.05


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
    training: TrainingConfig
    calibration: CalibrationConfig
    inference: InferenceConfig

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        return Config(
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            calibration=CalibrationConfig(**raw["calibration"]),
            inference=InferenceConfig(**raw["inference"]),
        )
