"""Text normalisation for food product names."""

from __future__ import annotations

import re
import unicodedata

# Abbreviations common in French food retail databases.
# Keys are regex patterns (applied after lowercasing), values are expansions.
# Order matters: longer/more specific patterns first to avoid partial conflicts.
_ABBREVIATIONS: list[tuple[str, str]] = [
    # Preservation / temperature
    (r"\bsg\b", "surgelé"),
    # Origin / certification labels
    (r"\bvbf\b", "viande bovine française"),
    (r"\bhve\b", "haute valeur environnementale"),
    # Packaging
    (r"\bbib\b", "bag in box"),
    (r"\bbq\b", "barquette"),
    (r"\bpet\b", "bouteille plastique"),
    (r"\bpac\b", "pack"),
    (r"\bvp\b", "vente par"),
]

# Pre-compile for performance
_COMPILED_ABBREVIATIONS = [
    (re.compile(pattern), replacement) for pattern, replacement in _ABBREVIATIONS
]


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _COMPILED_ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    return text


def normalize(text: str, strip_accents: bool = False) -> str:
    """
    Clean a raw product name for tokenization.

    Steps (order matters):
      1. Lowercase
      2. Expand domain abbreviations ("sg" → "surgelé")
      3. Optionally strip accents (NFD → ASCII) — disable for cased French models (CamemBERT)
      4. Pack normalization "x25" → "bt"
      5. Strip simple volume / weight "1/2 lt" → ""
      6. Strip float volume / weight "1.5 kg" → ""
      7. Collapse whitespace
    """

    # 1. Lowercase
    text = text.lower()
    # 2. Expand abbreviations before accent stripping so expansions keep proper French spelling
    text = _expand_abbreviations(text)
    # 3. Strip accents only when requested (hurts cased French models)
    if strip_accents:
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )
    # 4. Pack normalization
    text = re.sub(r"\bx\s*\d+\b", " bt ", text)
    # 5. Remove simple volume / weight
    text = re.sub(r"\b\d+\s*\/\s*\d+\s*(lt|l|ml|g|kg)\b", "", text)
    # 6. Remove float volume / weight
    text = re.sub(r"\b\d+(\.\d+)?\s*(l|ml|g|kg|lt|m)\b", "", text)
    # 7. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
