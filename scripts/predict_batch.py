"""
Batch inference entry point.

Usage:
    python scripts/predict_batch.py \
        --config config/base.yaml \
        --input data/new_products.csv \
        --output data/predictions.csv \
        --model models/best \
        --scaler models/best/temperature.pt

Input CSV must have a column matching config.data.text_column (default: product_name).
Output CSV adds: segment_id, subfamily_id, family_id, group_id, confidence, uncertain.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from ifls.calibrate import TemperatureScaler
from ifls.config import Config
from ifls.hierarchy import HierarchyEncoder
from ifls.predict import predict_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="config/base.yaml")
    p.add_argument("--input",      required=True, help="CSV with product names")
    p.add_argument("--output",     required=True, help="Output CSV path")
    p.add_argument("--model",      default="models/best")
    p.add_argument("--scaler",     default="models/best/temperature.pt")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Loading hierarchy encoder...")
    he = HierarchyEncoder.load(cfg.data.hierarchy_json)

    log.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    scaler = TemperatureScaler.load(args.scaler)

    log.info(f"Reading input: {args.input}")
    df    = pd.read_csv(args.input)
    texts = df[cfg.data.text_column].astype(str).tolist()

    log.info(f"Running inference on {len(texts):,} products...")
    all_preds = []
    for start in range(0, len(texts), args.batch_size):
        chunk = texts[start: start + args.batch_size]
        preds = predict_batch(
            texts=chunk,
            model=model,
            tokenizer=tokenizer,
            hierarchy_encoder=he,
            scaler=scaler,
            max_length=cfg.model.max_length,
            confidence_threshold=cfg.inference.confidence_threshold,
            device=device,
        )
        all_preds.extend(preds)
        if (start // args.batch_size) % 10 == 0:
            log.info(f"  {start + len(chunk):,} / {len(texts):,}")

    pred_df = pd.DataFrame([
        {
            "segment_id":   p.segment_id,
            "subfamily_id": p.subfamily_id,
            "family_id":    p.family_id,
            "group_id":     p.group_id,
            "confidence":   round(p.confidence, 4),
            "uncertain":    p.uncertain,
        }
        for p in all_preds
    ])

    out_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    log.info(f"Saved {len(out_df):,} rows to {args.output}")

    uncertain_count = pred_df["uncertain"].sum()
    log.info(f"Uncertain predictions: {uncertain_count:,} ({uncertain_count/len(out_df):.1%})")


if __name__ == "__main__":
    main()
