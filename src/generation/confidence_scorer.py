"""Grounded answer confidence scoring.

The frontend receives citations separately from the answer text, so production
confidence must not depend on inline citation markers such as ``[1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ConfidenceResult:
    """Confidence score and explainable component scores."""

    confidence: float
    components: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "components": self.components,
        }


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _mean(values: Iterable[float]) -> Optional[float]:
    items = [float(value) for value in values]
    if not items:
        return None
    return sum(items) / len(items)


def _doc_score(doc: Any) -> Optional[float]:
    score = getattr(doc, "score", None)
    if score is None:
        return None
    return _clamp(float(score))


def _doc_nli_score(doc: Any) -> Optional[float]:
    meta = getattr(doc, "meta", None) or {}
    score = meta.get("nli_score")
    if score is None:
        return None
    return _clamp(float(score))


def _citation_count(citations: Optional[List[Dict[str, Any]]]) -> int:
    return len(citations or [])


def _citation_float(citation: Dict[str, Any], key: str) -> Optional[float]:
    value = citation.get(key)
    if value is None:
        return None
    try:
        return _clamp(float(value))
    except (TypeError, ValueError):
        return None


def _citation_quality(citation: Dict[str, Any]) -> float:
    relevance = _citation_float(citation, "relevance")
    nli = _citation_float(citation, "nli_score")
    support = _citation_float(citation, "support_score")

    if nli is not None and nli > 0:
        return _clamp(0.55 * nli + 0.45 * (relevance if relevance is not None else nli))
    if support is not None:
        return support
    if relevance is not None:
        return relevance
    return 0.0


def _citation_support(citations: Optional[List[Dict[str, Any]]], expected_citations: int) -> Tuple[float, float]:
    if not citations:
        return 0.0, 0.0

    qualities = [_citation_quality(citation) for citation in citations]
    avg_quality = _mean(qualities) or 0.0
    coverage = _clamp(len(citations) / max(1, expected_citations))
    # Count still matters as coverage, but it can no longer create support on
    # its own. The score is primarily the average support quality.
    return _clamp(avg_quality * (0.75 + 0.25 * coverage)), _clamp(avg_quality)


def _best_citation_relevance(citations: Optional[List[Dict[str, Any]]]) -> float:
    values = [
        value
        for value in (_citation_float(citation, "relevance") for citation in citations or [])
        if value is not None
    ]
    return max(values) if values else 0.0


def _fallback_used(context_stats: Optional[Dict[str, Any]]) -> bool:
    fallback = (context_stats or {}).get("verification_fallback") or {}
    return bool(fallback.get("used"))


def _all_final_docs_have_zero_nli(final_docs: List[Any]) -> bool:
    if not final_docs:
        return False
    scores = [_doc_nli_score(doc) for doc in final_docs]
    return bool(scores) and all((score or 0.0) <= 0.0 for score in scores)


def _cleanup_removed_material(answer_cleanup: Optional[Dict[str, Any]]) -> bool:
    cleanup = answer_cleanup or {}
    return bool(
        int(cleanup.get("removed_line_count") or 0) > 0
        or int(cleanup.get("trimmed_line_count") or 0) > 0
        or cleanup.get("stripped_trailing")
        or cleanup.get("source_limited")
    )


def _extraction_quality_cap(context_stats: Optional[Dict[str, Any]]) -> Optional[float]:
    quality = (context_stats or {}).get("extraction_quality") or {}
    cap = quality.get("confidence_cap")
    if cap is None:
        return None
    try:
        return _clamp(float(cap))
    except (TypeError, ValueError):
        return None


def _apply_confidence_caps(
        confidence: float,
        *,
        final_docs: List[Any],
        citations: Optional[List[Dict[str, Any]]],
        context_stats: Optional[Dict[str, Any]],
        answer_cleanup: Optional[Dict[str, Any]],
        confidence_policy: Optional[Dict[str, Any]],
) -> Tuple[float, List[Dict[str, Any]]]:
    policy = confidence_policy or {}
    if not bool(policy.get("enabled", True)):
        return confidence, []

    caps: List[Dict[str, Any]] = []

    def apply_cap(reason: str, cap: float, trigger: bool) -> None:
        if trigger and confidence > cap:
            caps.append({"reason": reason, "cap": round(_clamp(cap), 6)})

    low_threshold = float(policy.get("low_relevance_threshold", 0.1))
    apply_cap(
        "low_citation_relevance",
        float(policy.get("low_relevance_cap", 0.3)),
        bool(citations) and _best_citation_relevance(citations) < low_threshold,
    )
    apply_cap(
        "verification_fallback_used",
        float(policy.get("verification_fallback_cap", 0.55)),
        _fallback_used(context_stats),
    )
    apply_cap(
        "fallback_without_nli_support",
        float(policy.get("no_nli_fallback_cap", 0.45)),
        _fallback_used(context_stats) and _all_final_docs_have_zero_nli(final_docs),
    )
    apply_cap(
        "unsupported_material_removed",
        float(policy.get("answer_cleanup_cap", 0.75)),
        _cleanup_removed_material(answer_cleanup),
    )
    quality_cap = _extraction_quality_cap(context_stats)
    apply_cap(
        "low_extraction_quality",
        quality_cap if quality_cap is not None else 1.0,
        quality_cap is not None,
    )

    if not caps:
        return confidence, []

    final_cap = min(item["cap"] for item in caps)
    return min(confidence, final_cap), caps


def calculate_api_citation_confidence(
        final_docs: List[Any],
        citations: Optional[List[Dict[str, Any]]],
        clean_answer: str,
        *,
        is_refusal: bool = False,
        context_stats: Optional[Dict[str, Any]] = None,
        answer_cleanup: Optional[Dict[str, Any]] = None,
        confidence_policy: Optional[Dict[str, Any]] = None,
) -> ConfidenceResult:
    """
    Score grounded confidence using API citations, not inline answer markers.

    This uses only signals already computed by the RAG pipeline:
    reranker scores, optional verification scores, final docs, API citations,
    and answer/refusal state. It intentionally does not invoke any extra model.
    """
    doc_count = len(final_docs or [])
    citation_count = _citation_count(citations)

    avg_rerank = _mean(
        score for score in (_doc_score(doc) for doc in final_docs or []) if score is not None
    )
    retrieval_relevance = _clamp(avg_rerank if avg_rerank is not None else 0.0)

    avg_nli = _mean(
        score for score in (_doc_nli_score(doc) for doc in final_docs or []) if score is not None
    )
    # If a document reached final_docs it passed a pipeline grounding gate. Use a
    # neutral floor rather than punishing older/fallback paths with no NLI score.
    verification_relevance = _clamp(avg_nli if avg_nli is not None else (0.5 if doc_count else 0.0))

    expected_citations = max(1, min(2, doc_count))
    source_support, avg_citation_support = _citation_support(citations, expected_citations)

    answer_state = 1.0 if clean_answer and clean_answer.strip() and not is_refusal else 0.0

    confidence = (
        0.50 * retrieval_relevance
        + 0.25 * verification_relevance
        + 0.15 * source_support
        + 0.10 * answer_state
    )
    uncapped_confidence = _clamp(confidence)
    confidence, caps_applied = _apply_confidence_caps(
        uncapped_confidence,
        final_docs=final_docs or [],
        citations=citations,
        context_stats=context_stats,
        answer_cleanup=answer_cleanup,
        confidence_policy=confidence_policy,
    )

    components = {
        "mode": "api_citations",
        "retrieval_relevance": round(retrieval_relevance, 6),
        "verification_relevance": round(verification_relevance, 6),
        "source_support": round(source_support, 6),
        "avg_citation_support": round(avg_citation_support, 6),
        "best_citation_relevance": round(_best_citation_relevance(citations), 6),
        "answer_state": round(answer_state, 6),
        "doc_count": doc_count,
        "citation_count": citation_count,
        "uncapped_confidence": round(uncapped_confidence, 6),
        "caps_applied": caps_applied,
        "weights": {
            "retrieval_relevance": 0.50,
            "verification_relevance": 0.25,
            "source_support": 0.15,
            "answer_state": 0.10,
        },
    }

    return ConfidenceResult(confidence=_clamp(confidence), components=components)


def calculate_legacy_inline_confidence(
        *,
        coverage_score: float,
        avg_rerank_score: float,
        citation_count: int,
) -> ConfidenceResult:
    """Preserve the previous inline-citation based formula for rollback."""
    confidence = (
        _clamp(coverage_score) * 0.4
        + _clamp(avg_rerank_score) * 0.4
        + _clamp(citation_count / 3) * 0.2
    )
    components = {
        "mode": "legacy_inline_citations",
        "coverage_score": round(_clamp(coverage_score), 6),
        "retrieval_relevance": round(_clamp(avg_rerank_score), 6),
        "source_support": round(_clamp(citation_count / 3), 6),
        "citation_count": citation_count,
        "weights": {
            "coverage_score": 0.40,
            "retrieval_relevance": 0.40,
            "source_support": 0.20,
        },
    }
    return ConfidenceResult(confidence=_clamp(confidence), components=components)
