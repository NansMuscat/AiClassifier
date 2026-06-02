"""
Tests for cascaded inference.
Uses the tiny synthetic hierarchy + a randomly-initialized model (no training needed).
Checks structure and contract, not prediction quality.
"""

import torch
import pytest
from transformers import AutoConfig, AutoTokenizer

from ifls.calibrate import TemperatureScaler
from ifls.model import IFLSMultiHeadClassifier
from ifls.predict import predict_batch, _mask_to_valid


MODEL_NAME = "distilbert-base-uncased"


@pytest.fixture(scope="module")
def model_and_tools(fitted_encoder):
    num_classes  = {lvl: fitted_encoder.num_classes(lvl) for lvl in ("group", "family", "subfamily", "segment")}
    level_weights = {"group": 0.10, "family": 0.20, "subfamily": 0.30, "segment": 0.40}
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.name_or_path = MODEL_NAME
    m = IFLSMultiHeadClassifier(cfg, num_classes, level_weights)
    m.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    scaler    = TemperatureScaler()   # default T=1 (identity)

    return m, tokenizer, scaler


def test_predict_batch_returns_correct_count(model_and_tools, fitted_encoder):
    model, tokenizer, scaler = model_and_tools
    texts = ["yaourt nature", "poulet roti", "beurre demi-sel"]
    preds = predict_batch(texts, model, tokenizer, fitted_encoder, scaler)
    assert len(preds) == 3


def test_predict_segment_id_is_valid_db_id(model_and_tools, fitted_encoder):
    model, tokenizer, scaler = model_and_tools
    preds = predict_batch(["jambon blanc"], model, tokenizer, fitted_encoder, scaler)
    p = preds[0]
    valid_segment_ids = {
        fitted_encoder.encoders["segment"].to_id(i)
        for i in range(fitted_encoder.num_classes("segment"))
    }
    assert p.segment_id in valid_segment_ids


def test_predict_confidence_is_probability(model_and_tools, fitted_encoder):
    model, tokenizer, scaler = model_and_tools
    preds = predict_batch(["fromage blanc"], model, tokenizer, fitted_encoder, scaler)
    p = preds[0]
    for conf in (p.confidence_segment, p.confidence_family, p.confidence_subfamily, p.confidence_group):
        assert 0.0 <= conf <= 1.0


def test_predict_uncertain_flag(model_and_tools, fitted_encoder):
    model, tokenizer, scaler = model_and_tools
    preds = predict_batch(
        ["x"],
        model,
        tokenizer,
        fitted_encoder,
        scaler,
        confidence_threshold=0.99,   # force uncertain
    )
    assert preds[0].uncertain is True


def test_mask_to_valid_restricts_logits():
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    valid  = {0, 2}
    masked = _mask_to_valid(logits, valid)
    assert masked[1] == float("-inf")
    assert masked[3] == float("-inf")
    assert masked[0] == 1.0
    assert masked[2] == 3.0


def test_mask_to_valid_empty_set_returns_unchanged():
    logits = torch.tensor([1.0, 2.0, 3.0])
    result = _mask_to_valid(logits, set())
    assert torch.equal(result, logits)
