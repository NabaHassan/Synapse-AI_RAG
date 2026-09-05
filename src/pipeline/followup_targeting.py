"""Utilities for choosing the factual base for conversation transforms."""

from __future__ import annotations

from typing import Any, Iterable, Optional


_FAILED_TRANSFORM_PREFIXES = (
    "there is no previous answer to explain",
    "there is no previous answer to clarify",
    "there is no previous answer to reformat",
    "i cannot answer this query because it was classified",
)


def find_last_useful_transform_base(turns: Iterable[Any]) -> Optional[Any]:
    """Return the nearest previous turn that can safely be clarified/formatted."""
    for turn in reversed(list(turns or [])):
        if is_useful_transform_base(turn):
            return turn
    return None


def is_useful_transform_base(turn: Any) -> bool:
    """Skip failed placeholders and meta turns, but allow successful transforms."""
    if turn is None:
        return False

    answer = str(getattr(turn, "answer", "") or "").strip()
    if not answer:
        return False

    if answer.lower().startswith(_FAILED_TRANSFORM_PREFIXES):
        return False

    query_type = str(getattr(turn, "query_type", "") or "").lower()
    if query_type == "meta_conversation":
        return False

    metadata = getattr(turn, "metadata", None) or {}
    if metadata.get("answer_state") == "true_out_of_scope":
        return False

    return True
