"""
Fast CPU-only model smoke tests.
These catch shape regressions and wiring errors without needing a GPU.
"""

import torch
import pytest
from transformers import AutoConfig, AutoTokenizer

from ifls.model import IFLSMultiHeadClassifier


MODEL_NAME = "distilbert-base-uncased"


@pytest.fixture(scope="module")
def num_classes():
    return {"group": 3, "family": 6, "subfamily": 12, "segment": 24}


@pytest.fixture(scope="module")
def level_weights():
    return {"group": 0.10, "family": 0.20, "subfamily": 0.30, "segment": 0.40}


@pytest.fixture(scope="module")
def model(num_classes, level_weights):
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.name_or_path = MODEL_NAME
    m = IFLSMultiHeadClassifier(
        config=cfg,
        num_classes=num_classes,
        level_weights=level_weights,
        focal_gamma=2.0,
    )
    m.eval()
    return m


@pytest.fixture(scope="module")
def dummy_batch():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tokenizer(
        ["produit test un", "produit test deux"],
        truncation=True,
        padding="max_length",
        max_length=32,
        return_tensors="pt",
    )
    return enc


def test_forward_returns_segment_logits(model, dummy_batch, num_classes):
    with torch.no_grad():
        out = model(**dummy_batch)
    assert out.logits.shape == (2, num_classes["segment"])


def test_forward_with_labels_returns_loss(model, dummy_batch, num_classes):
    labels = {
        "label_group":     torch.zeros(2, dtype=torch.long),
        "label_family":    torch.zeros(2, dtype=torch.long),
        "label_subfamily": torch.zeros(2, dtype=torch.long),
        "labels":          torch.zeros(2, dtype=torch.long),
    }
    with torch.no_grad():
        out = model(**dummy_batch, **labels)
    assert out.loss is not None
    assert out.loss.item() > 0


def test_forward_all_returns_all_levels(model, dummy_batch, num_classes):
    with torch.no_grad():
        all_logits = model.forward_all(**dummy_batch)
    assert set(all_logits.keys()) == {"group", "family", "subfamily", "segment"}
    for lvl, logits in all_logits.items():
        assert logits.shape == (2, num_classes[lvl]), f"Wrong shape for {lvl}"


def test_no_nan_in_logits(model, dummy_batch):
    with torch.no_grad():
        all_logits = model.forward_all(**dummy_batch)
    for lvl, logits in all_logits.items():
        assert not torch.isnan(logits).any(), f"NaN in {lvl} logits"
