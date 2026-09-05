"""Lightweight native sparse-vector utilities for Qdrant hybrid indexing.

The sparse representation is intentionally deterministic and local: tokenize,
hash tokens into a fixed sparse id space, and store normalized term-frequency
weights. It is not a replacement for the SQLite FTS sidecar; it gives new
collections an index-native exact-term signal that Qdrant can search without
requiring a new model download.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_NATIVE_SPARSE_VECTOR_NAME = "text_sparse"
DEFAULT_DENSE_VECTOR_NAME = "dense"
DEFAULT_NATIVE_SPARSE_DIM = 2_000_003

_TOKEN_RE = re.compile(r"(?u)[\w§][\w§.-]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "give", "how", "i", "in", "is", "it", "list", "me", "of", "on", "or",
    "show", "tell", "that", "the", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with",
}


def native_sparse_enabled(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def tokenize_sparse_text(text: str, *, max_tokens: int = 512) -> List[str]:
    tokens: List[str] = []
    for token in _TOKEN_RE.findall(text or ""):
        cleaned = token.strip(".-").lower()
        if not cleaned or cleaned in _STOPWORDS:
            continue
        if len(cleaned) < 2 and not cleaned.isdigit():
            continue
        tokens.append(cleaned)
        if len(tokens) >= max_tokens:
            break
    return tokens


def native_sparse_vector(
    text: str,
    *,
    dimensions: int = DEFAULT_NATIVE_SPARSE_DIM,
    max_tokens: int = 512,
) -> Tuple[List[int], List[float]]:
    tokens = tokenize_sparse_text(text, max_tokens=max_tokens)
    if not tokens:
        return [], []

    counts = Counter(tokens)
    weighted: Dict[int, float] = {}
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % int(dimensions)
        weighted[index] = weighted.get(index, 0.0) + (1.0 + math.log(float(count)))

    norm = math.sqrt(sum(value * value for value in weighted.values())) or 1.0
    items = sorted((index, value / norm) for index, value in weighted.items())
    return [index for index, _ in items], [float(value) for _, value in items]


def sparse_text_from_payload(payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "contextual_retrieval_text",
        "content",
        "source_filename",
        "document_id",
        "section_title",
        "heading_path",
        "email_subject",
    ):
        value = payload.get(key)
        if value:
            parts.append(str(value))

    entity_names = payload.get("entity_names")
    if isinstance(entity_names, Iterable) and not isinstance(entity_names, (str, bytes)):
        parts.extend(str(item) for item in entity_names if item)
    legal_codes = payload.get("legal_codes")
    if isinstance(legal_codes, Iterable) and not isinstance(legal_codes, (str, bytes)):
        parts.extend(str(item) for item in legal_codes if item)

    return "\n".join(parts)
