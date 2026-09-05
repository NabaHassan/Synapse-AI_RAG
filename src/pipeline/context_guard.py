"""Config-driven topic/source admission guard for retrieved context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ContextGuardTopic:
    topic_id: str
    confidence: float
    reason: str
    config: Dict[str, Any]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9§.]+", " ", (text or "").lower())


def _contains_any(haystack: str, patterns: Iterable[str]) -> bool:
    normalized_patterns = [_normalize(pattern) for pattern in patterns or [] if str(pattern).strip()]
    return any(pattern and pattern in haystack for pattern in normalized_patterns)


def _turn_text(turn: Any) -> str:
    pieces = [
        getattr(turn, "query", ""),
        getattr(turn, "reformulated_query", ""),
        getattr(turn, "answer", ""),
        getattr(turn, "response_summary", ""),
    ]
    return " ".join(piece for piece in pieces if piece)


def _recent_history_text(conversation_history: Optional[Iterable[Any]], limit: int) -> str:
    if not conversation_history:
        return ""
    turns = list(conversation_history)[-max(0, int(limit)):]
    return " ".join(_turn_text(turn) for turn in turns)


def context_guard_enabled(config: Optional[Dict[str, Any]]) -> bool:
    return bool((config or {}).get("enabled") and (config or {}).get("topics"))


def infer_context_guard_topic(
        query: str,
        *,
        guard_config: Optional[Dict[str, Any]],
        conversation_history: Optional[Iterable[Any]] = None,
        active_topic: Optional[str] = None,
) -> Optional[ContextGuardTopic]:
    """Infer the best configured topic for source-admission filtering."""
    if not context_guard_enabled(guard_config):
        return None

    history_turns = int((guard_config or {}).get("history_turns", 3) or 0)
    query_norm = _normalize(query)
    context_dependent = _query_looks_context_dependent(query)
    active_norm = _normalize(active_topic or "") if context_dependent else ""
    history_norm = (
        _normalize(_recent_history_text(conversation_history, history_turns))
        if context_dependent
        else ""
    )
    combined_norm = " ".join(part for part in (query_norm, active_norm, history_norm) if part)

    best: Optional[ContextGuardTopic] = None
    best_score = 0.0
    for topic_config in (guard_config or {}).get("topics", []) or []:
        topic_id = str(topic_config.get("topic_id") or "").strip()
        if not topic_id:
            continue

        triggers = topic_config.get("triggers") or []
        attribute_triggers = topic_config.get("attribute_triggers") or []
        trigger_match = _contains_any(combined_norm, triggers)
        attribute_match = _contains_any(query_norm, attribute_triggers)

        if not trigger_match:
            continue

        score = 0.75
        reason = "matched configured topic trigger"
        if attribute_triggers:
            if not attribute_match:
                score = 0.65
            else:
                score = 0.95
                reason = "matched configured topic and attribute trigger"

        if score > best_score:
            best_score = score
            best = ContextGuardTopic(
                topic_id=topic_id,
                confidence=score,
                reason=reason,
                config=dict(topic_config),
            )

    return best


def _query_looks_context_dependent(query: str) -> bool:
    """Return True when prior topic should influence guard-topic inference."""
    q = (query or "").strip().lower()
    if not q:
        return False
    if re.search(
        r"\b(?:this|that|it|these|those|previous|above|same|last answer|your answer|"
        r"what you said|what you meant)\b",
        q,
    ):
        return True
    if re.search(r"^(?:more|continue|go on|elaborate|expand|explain)\b", q):
        return True
    if re.search(r"^(?:what|how|same)\s+(?:about|for)\b", q):
        return True
    return False


def _doc_source_text(doc: Any) -> str:
    meta = getattr(doc, "meta", None) or {}
    values = [
        meta.get("source", ""),
        meta.get("source_filename", ""),
        meta.get("file_name", ""),
        meta.get("filename", ""),
        meta.get("source_filepath", ""),
        meta.get("file_path", ""),
    ]
    return _normalize(" ".join(str(value) for value in values if value))


def _doc_content_text(doc: Any) -> str:
    return _normalize(str(getattr(doc, "content", "") or ""))


def doc_matches_context_topic(doc: Any, topic: ContextGuardTopic) -> bool:
    """Return True when a document matches the configured topic rules."""
    source = _doc_source_text(doc)
    content = _doc_content_text(doc)
    source_patterns = topic.config.get("source_include_patterns") or []
    content_patterns = topic.config.get("content_include_patterns") or []

    source_match = _contains_any(source, source_patterns)
    content_match = _contains_any(content, content_patterns)

    if source_patterns and content_patterns:
        return source_match or content_match
    if source_patterns:
        return source_match
    if content_patterns:
        return content_match
    return True


def filter_context_guard_docs(
        docs: List[Any],
        topic: Optional[ContextGuardTopic],
) -> Tuple[List[Any], Dict[str, Any]]:
    """Filter topic-conflicted docs before verification/generation."""
    metadata: Dict[str, Any] = {
        "enabled": bool(topic),
        "topic": topic.topic_id if topic else None,
        "topic_confidence": topic.confidence if topic else 0.0,
        "reason": topic.reason if topic else "",
        "input_count": len(docs or []),
        "matched_count": 0,
        "removed_count": 0,
        "strict_no_match": False,
    }

    if not topic or not docs:
        metadata["matched_count"] = len(docs or [])
        return docs, metadata

    matched = [doc for doc in docs if doc_matches_context_topic(doc, topic)]
    metadata["matched_count"] = len(matched)
    metadata["removed_count"] = len(docs) - len(matched)

    if matched:
        return matched, metadata

    metadata["strict_no_match"] = True
    return [], metadata


def _doc_key(doc: Any) -> str:
    meta = getattr(doc, "meta", None) or {}
    chunk_id = str(meta.get("chunk_id") or "")
    source = _doc_source_text(doc)
    if chunk_id:
        return f"{source}::{chunk_id}"
    return f"{source}::{hash(str(getattr(doc, 'content', '') or ''))}"


def supplement_context_guard_candidates(
        primary_docs: List[Any],
        candidate_docs: List[Any],
        topic: Optional[ContextGuardTopic],
) -> Tuple[List[Any], Dict[str, Any]]:
    """Append same-topic candidates that reranking may have dropped."""
    max_extra = int((topic.config.get("max_supplemental_docs", 5) if topic else 0) or 0)
    metadata: Dict[str, Any] = {
        "enabled": bool(topic),
        "topic": topic.topic_id if topic else None,
        "primary_count": len(primary_docs or []),
        "candidate_count": len(candidate_docs or []),
        "added_count": 0,
    }

    if not topic or not candidate_docs or max_extra <= 0:
        return primary_docs, metadata

    seen = {_doc_key(doc) for doc in primary_docs or []}
    extras: List[Any] = []
    for doc in candidate_docs:
        key = _doc_key(doc)
        if key in seen:
            continue
        if not doc_matches_context_topic(doc, topic):
            continue
        extras.append(doc)
        seen.add(key)
        if len(extras) >= max_extra:
            break

    metadata["added_count"] = len(extras)
    return list(primary_docs or []) + extras, metadata


def augment_context_guard_search_query(query: str, topic: Optional[ContextGuardTopic]) -> str:
    """Append configured topic anchors to improve retrieval precision."""
    if not topic:
        return query

    anchors = [
        str(anchor).strip()
        for anchor in (topic.config.get("query_anchors") or [])
        if str(anchor).strip()
    ]
    if not anchors:
        return query

    return f"{query} {' '.join(anchors)}"
