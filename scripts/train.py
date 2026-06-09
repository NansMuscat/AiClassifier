"""
Training entry point.

Usage:
    python scripts/train.py --config config/base.yaml

Steps:
  1. Load config
  2. Load and normalize the CSV
  3. Fit HierarchyEncoder, save hierarchy.json
  4. Split into train / calibration / test
  5. Build IFLSDataset
  6. Build AutoModelForSequenceClassification with LoRA
  7. Train with HuggingFace Trainer (standard CE + label smoothing)
  8. Run temperature calibration on the calib split
  9. Save model, scaler, and a JSON results summary
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from ifls.calibrate import TemperatureScaler
from ifls.config import Config
from ifls.dataset import IFLSDataset
from ifls.hierarchy import HierarchyEncoder
from ifls.metrics import compute_metrics_fn
from ifls.normalize import normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/base.yaml")
    p.add_argument("--resume", default=None, help="Checkpoint dir to resume from")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── 1. Load & normalise ────────────────────────────────────────────────
    log.info(f"Loading CSV: {cfg.data.csv_path}")
    df = pd.read_csv(cfg.data.csv_path, sep=";")

    required_cols = {"product_name", "group_id", "family_id", "subfamily_id", "segment_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    if cfg.data.max_rows:
        df = df.head(cfg.data.max_rows)
        log.info(f"Truncated to {cfg.data.max_rows:,} rows (dev mode)")

    is_cased_french = (
        "camembert" in cfg.model.backbone.lower()
        or "roberta" in cfg.model.backbone.lower()
    )
    df["text"] = (
        df[cfg.data.text_column]
        .astype(str)
        .map(lambda t: normalize(t, strip_accents=not is_cased_french))
    )
    log.info(f"Loaded {len(df):,} products")

    # ── 2. Hierarchy encoder ───────────────────────────────────────────────
    log.info("Fitting hierarchy encoder...")
    he = HierarchyEncoder().fit(df)
    he.save(cfg.data.hierarchy_json)
    num_segments = he.num_classes("segment")
    log.info(f"Segments: {num_segments}")

    # ── 3. Split ───────────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=cfg.data.random_seed).reset_index(drop=True)
    n       = len(df)
    n_train = int(n * cfg.data.train_ratio)
    n_calib = int(n * cfg.data.calib_ratio)

    train_df = df.iloc[:n_train]
    calib_df = df.iloc[n_train: n_train + n_calib]
    test_df  = df.iloc[n_train + n_calib:]
    log.info(f"Split — train:{len(train_df):,}  calib:{len(calib_df):,}  test:{len(test_df):,}")

    # ── 4. Tokenizer & datasets ────────────────────────────────────────────
    log.info(f"Loading tokenizer: {cfg.model.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone, use_fast=False)

    train_ds = IFLSDataset(train_df, tokenizer, he, cfg.model.max_length)
    calib_ds = IFLSDataset(calib_df, tokenizer, he, cfg.model.max_length)
    test_ds  = IFLSDataset(test_df,  tokenizer, he, cfg.model.max_length)

    # ── 5. Model ───────────────────────────────────────────────────────────
    log.info(f"Building model: {cfg.model.backbone} ({num_segments} classes)")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.backbone,
        num_labels=num_segments,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        target_modules=cfg.model.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 6. Trainer ─────────────────────────────────────────────────────────
    ckpt_strategy = "steps" if cfg.training.max_steps > 0 else "epoch"
    ckpt_steps    = max(1, cfg.training.max_steps // 2) if cfg.training.max_steps > 0 else None

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir,
        learning_rate=cfg.training.learning_rate,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.batch_size,
        num_train_epochs=cfg.training.epochs,
        max_steps=cfg.training.max_steps,
        eval_strategy=ckpt_strategy,
        eval_steps=ckpt_steps,
        save_strategy=ckpt_strategy,
        save_steps=ckpt_steps,
        save_total_limit=cfg.training.save_total_limit,
        fp16=cfg.training.fp16 and device == "cuda",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        warmup_ratio=cfg.training.warmup_ratio,
        label_smoothing_factor=cfg.training.label_smoothing,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics_fn,
    )

    log.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── 7. Temperature calibration ─────────────────────────────────────────
    log.info("Calibrating temperature on held-out calibration set...")
    model.eval()
    model.to(device)

    all_logits, all_labels = [], []
    calib_loader = DataLoader(calib_ds, batch_size=cfg.training.batch_size)

    with torch.no_grad():
        for batch in calib_loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_logits.append(out.logits.cpu())
            all_labels.append(batch["labels"])

    scaler = TemperatureScaler()
    scaler.calibrate(
        torch.cat(all_logits),
        torch.cat(all_labels),
        lr=cfg.calibration.temperature_lr,
        max_iter=cfg.calibration.temperature_max_iter,
    )
    scaler.save(cfg.calibration.temperature_output)
    log.info(f"Temperature scaler saved to {cfg.calibration.temperature_output}")

    # ── 8. Save model ──────────────────────────────────────────────────────
    save_dir = Path(cfg.training.output_dir) / "best"
    log.info(f"Merging LoRA weights and saving to {save_dir}...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    log.info("Model and tokenizer saved.")

    # ── 9. Save results summary ────────────────────────────────────────────
    eval_results = trainer.evaluate(eval_dataset=test_ds)
    summary = {
        "config":        args.config,
        "num_products":  len(df),
        "num_segments":  num_segments,
        "eval_accuracy": eval_results.get("eval_accuracy"),
        "eval_loss":     eval_results.get("eval_loss"),
        "temperature":   scaler.temperature.item(),
    }
    results_path = save_dir / "results.json"
    results_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Results: {summary}")


if __name__ == "__main__":
    main()
