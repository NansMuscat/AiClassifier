"""Text normalisation for food product names."""

from __future__ import annotations

import re
import unicodedata


def normalize(text: str) -> str:
    """
    Clean a raw product name for tokenization.

    Steps (order matters):
      1. Lowercase
      2. Strip accents (NFD → ASCII)
      3. Pack normalization "(x25)" → "(Bt)"
      4. Strip simple volume / weight "1 lt" → ""
      5. Strip float volume / weight "1 lt" → ""
      6. Collapse whitespace
    """

    # 1. Lowercase
    text = text.lower()
    # 2. Strip accents (NFD → ASCII)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    # 3. pack normalization
    text = re.sub(r"\bx\s*\d+\b", " bt ", text)
    # 4. Remove simple volume / weight
    text = re.sub(r"\b\d+\s*\/\s*\d+\s*(lt|l|ml|g|kg)\b", "", text)
    # 5. Remove float volume / weight
    text = re.sub(r"\b\d+(\.\d+)?\s*(l|ml|g|kg|lt|m)\b", "", text)
    # 6. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
