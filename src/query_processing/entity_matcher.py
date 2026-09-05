"""
Entity matching helpers for structured query handlers.

The goal is high precision for production-facing file/snippet responses:
- Match on normalized full tokens/phrases, not arbitrary substrings.
- Prevent false positives like "elon" matching "elongated".
"""

import re
from typing import List


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize_text(value: str) -> str:
    """Lowercase + collapse spaces + trim edge punctuation."""
    if not value:
        return ""

    normalized = re.sub(r"\s+", " ", value.strip().lower())
    normalized = normalized.strip(".,;:!?()[]{}\"'/\\`")
    return normalized


def _tokenize(value: str) -> List[str]:
    """Tokenize to alphanumeric words for boundary-aware matching."""
    return _TOKEN_PATTERN.findall((value or "").lower())


def _build_ordered_phrase_pattern(tokens: List[str]) -> re.Pattern:
    """
    Build a regex that matches ordered tokens with non-word separators.

    Example: ["elon", "musk"] -> r"\\belon\\W+musk\\b"
    """
    escaped = [re.escape(t) for t in tokens]
    pattern = r"\b" + r"\W+".join(escaped) + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def is_entity_match(query_entity: str, candidate_entity: str) -> bool:
    """
    Return True when two entity strings represent the same entity.

    Rules:
    - Exact normalized equality is a match.
    - Multi-word query requires full-token overlap (all query tokens present
      in candidate, or two-plus-token candidate contained in query).
    - Single-word query matches only full tokens, never substrings.
    """
    query_norm = _normalize_text(query_entity)
    candidate_norm = _normalize_text(candidate_entity)

    if not query_norm or not candidate_norm:
        return False

    if query_norm == candidate_norm:
        return True

    query_tokens = _tokenize(query_norm)
    candidate_tokens = _tokenize(candidate_norm)
    if not query_tokens or not candidate_tokens:
        return False

    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)

    if len(query_tokens) >= 2:
        if query_set.issubset(candidate_set):
            return True
        if len(candidate_tokens) >= 2 and candidate_set.issubset(query_set):
            return True
        return False

    return query_tokens[0] in candidate_set


def count_entity_mentions_in_text(entity: str, text: str) -> int:
    """
    Count entity mentions in raw text using boundary-aware token/phrase matching.

    - Single-token entities: exact whole-word count
    - Multi-token entities: ordered phrase count with punctuation/space tolerance
      (plus reversed order for two-token person-name queries like "Musk, Elon")
    """
    tokens = _tokenize(_normalize_text(entity))
    if not tokens or not text:
        return 0

    if len(tokens) == 1:
        pattern = re.compile(rf"\b{re.escape(tokens[0])}\b", flags=re.IGNORECASE)
        return len(pattern.findall(text))

    total = len(_build_ordered_phrase_pattern(tokens).findall(text))
    if len(tokens) == 2:
        total += len(_build_ordered_phrase_pattern(tokens[::-1]).findall(text))
    return total


def contains_entity_in_text(entity: str, text: str) -> bool:
    """
    Check whether text contains the entity as full word tokens.

    For multi-word entities, all entity tokens must be present in the text.
    """
    return count_entity_mentions_in_text(entity, text) > 0
