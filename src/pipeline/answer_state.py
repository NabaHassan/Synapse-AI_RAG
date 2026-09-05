"""Answer-state helpers shared by RAG pipeline entry points."""

from __future__ import annotations

from typing import Tuple


ANSWER_STATE_GROUNDED = "grounded_answer"
ANSWER_STATE_SOURCE_LIMITED = "source_limited_answer"
ANSWER_STATE_TRUE_OUT_OF_SCOPE = "true_out_of_scope"
ANSWER_STATE_CONVERSATION_TRANSFORM = "conversation_transform"
ANSWER_STATE_DIRECT_RESPONSE = "direct_response"

GROUNDING_STATUS_DIRECT = "direct"
GROUNDING_STATUS_MISSING_DETAIL = "missing_detail"
GROUNDING_STATUS_OUT_OF_SCOPE = "out_of_scope"
GROUNDING_STATUS_NOT_APPLICABLE = "not_applicable"


def build_no_results_answer(
        *,
        query_type: str,
        route: str,
        legal_strict_mode: bool = False,
) -> Tuple[str, str, str]:
    """Return answer text plus answer-state metadata for unsupported queries."""
    normalized_query_type = (query_type or "").strip().lower()
    normalized_route = (route or "").strip().lower()

    if normalized_query_type == "out_of_scope" or normalized_route == "reject":
        return (
            "I cannot answer this query because it was classified as out-of-scope or too generic.",
            ANSWER_STATE_TRUE_OUT_OF_SCOPE,
            GROUNDING_STATUS_OUT_OF_SCOPE,
        )

    if legal_strict_mode:
        return (
            "The provided sections do not specify enough information to answer this question.",
            ANSWER_STATE_SOURCE_LIMITED,
            GROUNDING_STATUS_MISSING_DETAIL,
        )

    return (
        "The knowledge base does not contain enough relevant information to answer this question.",
        ANSWER_STATE_SOURCE_LIMITED,
        GROUNDING_STATUS_MISSING_DETAIL,
    )

