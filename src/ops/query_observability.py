"""Lightweight query observability for product operations.

The collector is intentionally CPU-only and dependency-free. It records compact
per-query operational events, keeps a rolling in-memory window for dashboards,
and can append JSONL events for later offline analysis.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


DEFAULT_WINDOW_SIZE = 1000
DEFAULT_EVENT_LOG_NAME = "query_observability_events.jsonl"
DEFAULT_TUNING_MIN_BASELINE = 20
DEFAULT_TUNING_MIN_RECENT = 10
DEFAULT_TUNING_RECENT_WINDOW = 50


STATIC_ALERT_THRESHOLDS = {
    "source_limited_rate": 0.35,
    "no_accepted_evidence_rate": 0.45,
    "citation_failure_rate": 0.05,
    "confidence_cap_rate": 0.25,
    "unsupported_removed_rate": 0.05,
    "p95_latency_seconds": 30.0,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_debug_metric(retrieval_debug: Dict[str, Any], retriever: str, key: str) -> Optional[float]:
    entries = retrieval_debug.get(retriever)
    if isinstance(entries, list) and entries:
        value = entries[0].get(key)
        return _safe_float(value) if value is not None else None
    return None


def _first_debug_value(retrieval_debug: Dict[str, Any], retriever: str, key: str) -> Any:
    entries = retrieval_debug.get(retriever)
    if isinstance(entries, list) and entries:
        return entries[0].get(key)
    return None


def _get_path(data: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    rank = (len(sorted_values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(sorted_values[int(rank)], 3)
    weight = rank - lower
    return round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight, 3)


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 4)


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def _limited_query(query: str, max_chars: int = 240) -> str:
    cleaned = " ".join(str(query or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _useful_label(value: Any) -> str:
    label = str(value or "").strip()
    if label.lower() in {"", "unknown", "none", "null"}:
        return ""
    return label


def _confidence_cap_details(
        *,
        claim_validation: Dict[str, Any],
        confidence_components: Dict[str, Any],
        extraction_quality: Dict[str, Any],
) -> tuple[Optional[float], List[str]]:
    caps: List[tuple[float, str]] = []

    claim_cap = claim_validation.get("confidence_cap")
    if claim_cap is not None:
        caps.append((_safe_float(claim_cap, 1.0), "claim_validation"))

    extraction_cap = extraction_quality.get("confidence_cap")
    if extraction_cap is not None:
        caps.append((_safe_float(extraction_cap, 1.0), "extraction_quality"))

    for item in confidence_components.get("caps_applied") or []:
        if not isinstance(item, dict):
            continue
        cap = item.get("cap")
        reason = str(item.get("reason") or "confidence_scorer").strip() or "confidence_scorer"
        if cap is not None:
            caps.append((_safe_float(cap, 1.0), reason))

    caps = [(max(0.0, min(1.0, cap)), reason) for cap, reason in caps if cap < 1.0]
    if not caps:
        return None, []
    cap = min(value for value, _ in caps)
    reasons = []
    for _, reason in caps:
        if reason not in reasons:
            reasons.append(reason)
    return round(cap, 6), reasons


def _distribution_metrics(events: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    total = len(events)
    return {
        "query_count": float(total),
        "source_limited_rate": _ratio(
            sum(1 for event in events if event.get("answer_state") == "source_limited_answer"),
            total,
        ),
        "no_accepted_evidence_rate": _ratio(
            sum(1 for event in events if _safe_int(event.get("accepted_evidence_count")) <= 0),
            total,
        ),
        "citation_failure_rate": _ratio(
            sum(1 for event in events if _safe_int(event.get("citation_validation_failure_count")) > 0),
            total,
        ),
        "confidence_cap_rate": _ratio(
            sum(1 for event in events if event.get("confidence_cap_applied")),
            total,
        ),
        "unsupported_removed_rate": _ratio(
            sum(1 for event in events if event.get("unsupported_claims_removed")),
            total,
        ),
        "p95_latency_seconds": _percentile(
            [_safe_float(event.get("latency_seconds")) for event in events],
            0.95,
        ),
    }


def _adaptive_rate_threshold(*, baseline_value: float, static_threshold: float, baseline_ready: bool) -> float:
    if not baseline_ready:
        return static_threshold
    return round(max(static_threshold, baseline_value * 2.0 + 0.05), 4)


def _adaptive_latency_threshold(*, baseline_value: float, static_threshold: float, baseline_ready: bool) -> float:
    if not baseline_ready:
        return static_threshold
    return round(max(static_threshold, baseline_value * 1.8), 3)


def _distribution_alert(
        *,
        metric: str,
        current: float,
        threshold: float,
        baseline_value: Optional[float],
        recent_count: int,
        baseline_count: int,
        severity: str,
) -> Dict[str, Any]:
    return {
        "severity": severity,
        "code": f"{metric}_spike",
        "metric": metric,
        "current": round(current, 4) if metric.endswith("_rate") else round(current, 3),
        "threshold": round(threshold, 4) if metric.endswith("_rate") else round(threshold, 3),
        "baseline": None if baseline_value is None else (
            round(baseline_value, 4) if metric.endswith("_rate") else round(baseline_value, 3)
        ),
        "recent_count": recent_count,
        "baseline_count": baseline_count,
        "message": f"{metric} exceeded rolling production threshold",
    }


class QueryObservabilityCollector:
    """Collect rolling query health metrics and alert signals."""

    def __init__(
        self,
        *,
        window_size: int = DEFAULT_WINDOW_SIZE,
        event_log_path: Optional[Path] = None,
    ) -> None:
        self.window_size = max(10, int(window_size or DEFAULT_WINDOW_SIZE))
        self.event_log_path = event_log_path
        self._events: Deque[Dict[str, Any]] = deque(maxlen=self.window_size)
        self._lock = threading.Lock()

        if self.event_log_path:
            try:
                self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.warning("Failed to create observability event directory", exc_info=True)
                self.event_log_path = None

    def record_response(
        self,
        *,
        kb_id: str,
        request_id: str,
        response: Any,
        processing_time: float,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata = dict(getattr(response, "metadata", {}) or {})
        answer = str(getattr(response, "answer", "") or "")
        citations = list(getattr(response, "citations", []) or [])

        evidence_plan = dict(metadata.get("evidence_plan") or {})
        evidence_admission = dict(metadata.get("evidence_admission") or {})
        generation_decision = dict(metadata.get("generation_decision") or {})
        claim_validation = dict(metadata.get("claim_validation") or {})
        retrieval_debug = dict(metadata.get("retrieval_debug") or {})
        retrieval_strategy = dict(retrieval_debug.get("retrieval_strategy") or {})
        confidence_components = dict(metadata.get("confidence_components") or {})
        answer_cleanup = dict(metadata.get("answer_cleanup") or {})
        extraction_quality = dict(metadata.get("extraction_quality") or {})

        answer_state = str(metadata.get("answer_state") or "unknown")
        plan_query_intent = _useful_label(evidence_plan.get("query_intent"))
        metadata_query_type = _useful_label(metadata.get("query_type"))
        query_intent = str(
            plan_query_intent
            or metadata_query_type
            or ("rag_answer" if answer_state in {"grounded_answer", "source_limited_answer"} else answer_state)
            or "unknown"
        )
        accepted_count = _safe_int(evidence_admission.get("accepted_count"))
        rejected_count = _safe_int(evidence_admission.get("rejected_count"))
        citation_failures = len(claim_validation.get("citation_validation_failures") or [])
        unsupported_claims = len(claim_validation.get("unsupported_claims") or [])
        fallback_used = bool(_get_path(retrieval_debug, ["verification_fallback", "used"], False))
        confidence_cap, confidence_cap_reasons = _confidence_cap_details(
            claim_validation=claim_validation,
            confidence_components=confidence_components,
            extraction_quality=extraction_quality,
        )
        confidence_cap_applied = confidence_cap is not None and _safe_float(confidence_cap) < 1.0
        unsupported_removed = bool(answer_cleanup.get("removed_line_count") or answer_cleanup.get("stripped_trailing"))

        event = {
            "timestamp": time.time(),
            "request_id": request_id,
            "kb_id": kb_id,
            "user_id": user_id,
            "session_id": getattr(response, "session_id", None),
            "turn_number": getattr(response, "turn_number", None),
            "query": _limited_query(getattr(response, "query", "")),
            "query_intent": query_intent,
            "answer_state": answer_state,
            "grounding_status": metadata.get("grounding_status"),
            "evidence_plan": {
                "query_intent": query_intent,
                "answer_mode": evidence_plan.get("answer_mode"),
                "rewrite_type": evidence_plan.get("rewrite_type"),
                "retrieval_mode": evidence_plan.get("retrieval_mode"),
                "requested_operation": evidence_plan.get("requested_operation"),
            },
            "accepted_evidence_count": accepted_count,
            "rejected_evidence_count": rejected_count,
            "generation_budget": generation_decision.get("budget"),
            "generation_max_tokens": generation_decision.get("max_tokens"),
            "confidence": _safe_float(metadata.get("confidence")),
            "confidence_cap_applied": confidence_cap_applied,
            "confidence_cap": confidence_cap,
            "confidence_cap_reasons": confidence_cap_reasons,
            "fallback_used": fallback_used,
            "unsupported_claim_count": unsupported_claims,
            "unsupported_claims_removed": unsupported_removed,
            "citation_validation_failure_count": citation_failures,
            "citation_count": len(citations),
            "validated_citation_count": max(0, len(citations) - citation_failures),
            "answer_word_count": len(answer.split()),
            "llm_provider": metadata.get("llm_provider"),
            "llm_model": metadata.get("llm_model"),
            "latency_seconds": round(float(processing_time or 0.0), 3),
            "latency_breakdown": {
                "total": round(float(processing_time or 0.0), 3),
                "pipeline_total": round(_safe_float(metadata.get("total_time")), 3),
                "dense_embed_ms": _first_debug_metric(retrieval_debug, "dense", "embed_ms"),
                "dense_qdrant_search_ms": _first_debug_metric(retrieval_debug, "dense", "qdrant_search_ms"),
                "dense_convert_ms": _first_debug_metric(retrieval_debug, "dense", "convert_ms"),
                "dense_total_ms": _first_debug_metric(retrieval_debug, "dense", "total_ms"),
                "sparse_tokenize_ms": _first_debug_metric(retrieval_debug, "sparse", "tokenize_ms"),
                "sparse_score_ms": _first_debug_metric(retrieval_debug, "sparse", "score_ms"),
                "sparse_sort_ms": _first_debug_metric(retrieval_debug, "sparse", "sort_ms"),
                "sparse_total_ms": _first_debug_metric(retrieval_debug, "sparse", "total_ms"),
                "fusion_ms": _safe_float(_get_path(retrieval_debug, ["fusion", "total_ms"])),
                "rerank_ms": _safe_float(_get_path(retrieval_debug, ["rerank", "total_ms"])),
                "verification_ms": _safe_float(_get_path(retrieval_debug, ["context_processing", "verification_ms"])),
            },
            "retrieval_diagnostics": {
                "qdrant": retrieval_debug.get("qdrant") or {},
                "strategy_mode": retrieval_strategy.get("mode"),
                "strategy_applied": bool(retrieval_strategy.get("applied")),
                "strategy_reason": retrieval_strategy.get("reason"),
                "strategy_budget": retrieval_strategy.get("budget") or {},
                "strategy_shadow_budget": retrieval_strategy.get("shadow_budget") or {},
                "dense_embedding_cache_hit": bool(
                    _first_debug_metric(retrieval_debug, "dense", "embedding_cache_hit")
                ),
                "dense_candidate_cache_hit": bool(
                    _first_debug_metric(retrieval_debug, "dense", "candidate_cache_hit")
                ),
                "sparse_candidate_cache_hit": bool(
                    _first_debug_metric(retrieval_debug, "sparse", "candidate_cache_hit")
                ),
                "sparse_postings_scanned": _safe_int(
                    _first_debug_metric(retrieval_debug, "sparse", "postings_scanned")
                ),
                "sparse_candidate_count": _safe_int(
                    _first_debug_metric(retrieval_debug, "sparse", "candidate_count")
                ),
                "sparse_matched_terms": _safe_int(
                    _first_debug_metric(retrieval_debug, "sparse", "matched_terms")
                ),
                "sparse_lexical_backend": _first_debug_value(retrieval_debug, "sparse", "lexical_backend"),
                "sparse_lexical_backend_used": bool(
                    _first_debug_value(retrieval_debug, "sparse", "lexical_backend_used")
                ),
                "sparse_lexical_fallback_reason": _first_debug_value(
                    retrieval_debug, "sparse", "lexical_fallback_reason"
                ),
                "parallel_execution": bool(retrieval_debug.get("parallel_execution")),
            },
            "retrieval_counts": {
                "dense": _safe_int(retrieval_debug.get("dense_count")),
                "sparse": _safe_int(retrieval_debug.get("sparse_count")),
                "fused": _safe_int(retrieval_debug.get("fused_count")),
                "reranked": _safe_int(retrieval_debug.get("reranked_count")),
                "verified": _safe_int(retrieval_debug.get("verified_count")),
                "final": _safe_int(retrieval_debug.get("final_count")),
            },
            "confidence_components": {
                key: confidence_components.get(key)
                for key in ["mode", "retrieval_relevance", "verification_relevance", "source_support", "answer_state"]
                if key in confidence_components
            },
            "extraction_quality": {
                key: extraction_quality.get(key)
                for key in [
                    "doc_count",
                    "low_quality_count",
                    "low_quality_ratio",
                    "avg_quality",
                    "confidence_cap",
                    "source_limited_recommended",
                    "reasons",
                    "flags",
                ]
                if key in extraction_quality
            },
        }
        event["alerts"] = self._alerts_for_event(event)
        self._store_event(event)
        return event

    def record_error(
        self,
        *,
        kb_id: str,
        request_id: str,
        query: str,
        error_type: str,
        message: str,
        processing_time: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        event = {
            "timestamp": time.time(),
            "request_id": request_id,
            "kb_id": kb_id,
            "user_id": user_id,
            "session_id": session_id,
            "query": _limited_query(query),
            "query_intent": "error",
            "answer_state": "error",
            "error_type": error_type,
            "error_message": _limited_query(message, 500),
            "accepted_evidence_count": 0,
            "rejected_evidence_count": 0,
            "fallback_used": False,
            "unsupported_claim_count": 0,
            "citation_validation_failure_count": 0,
            "citation_count": 0,
            "latency_seconds": round(float(processing_time or 0.0), 3),
            "alerts": [{"severity": "critical", "code": "query_error", "message": error_type}],
        }
        self._store_event(event)
        return event

    def summary(self, *, window: Optional[int] = None, kb_id: Optional[str] = None) -> Dict[str, Any]:
        events = self.recent_events(limit=window or self.window_size, kb_id=kb_id)
        query_events = [event for event in events if event.get("answer_state") != "error"]
        total = len(query_events)
        alert_counts = Counter(alert.get("code") for event in events for alert in event.get("alerts", []))
        answer_states = Counter(event.get("answer_state") or "unknown" for event in query_events)
        intents = Counter(event.get("query_intent") or "unknown" for event in query_events)
        kb_counts = Counter(event.get("kb_id") or "unknown" for event in query_events)

        latencies = [_safe_float(event.get("latency_seconds")) for event in query_events]
        accepted_counts = [_safe_int(event.get("accepted_evidence_count")) for event in query_events]
        by_intent = self._latency_groups(query_events, "query_intent")
        by_kb = self._latency_groups(query_events, "kb_id")

        no_accepted = sum(1 for event in query_events if _safe_int(event.get("accepted_evidence_count")) <= 0)
        source_limited = answer_states.get("source_limited_answer", 0)
        true_out_of_scope = answer_states.get("true_out_of_scope", 0)
        fallback_used = sum(1 for event in query_events if event.get("fallback_used"))
        confidence_capped = sum(1 for event in query_events if event.get("confidence_cap_applied"))
        unsupported_claim_events = sum(1 for event in query_events if _safe_int(event.get("unsupported_claim_count")) > 0)
        citation_failure_events = sum(1 for event in query_events if _safe_int(event.get("citation_validation_failure_count")) > 0)
        transform_events = [event for event in query_events if event.get("answer_state") == "conversation_transform"]
        structured_events = [
            event for event in query_events
            if event.get("query_intent") in {"count", "structured_list"}
            or event.get("evidence_plan", {}).get("retrieval_mode") in {"structured_count", "structured_list"}
        ]
        structured_exact = sum(
            1 for event in structured_events
            if _safe_int(event.get("accepted_evidence_count")) > 0 and _safe_int(event.get("citation_count")) > 0
        )

        return {
            "status": "ok",
            "window_size": self.window_size,
            "window_requested": window or self.window_size,
            "kb_filter": kb_id,
            "event_count": len(events),
            "query_count": total,
            "error_count": len(events) - total,
            "answer_state_distribution": dict(answer_states),
            "query_intent_distribution": dict(intents),
            "kb_distribution": dict(kb_counts),
            "rates": {
                "source_limited": _rate(source_limited, total),
                "true_out_of_scope": _rate(true_out_of_scope, total),
                "no_accepted_evidence": _rate(no_accepted, total),
                "fallback_used": _rate(fallback_used, total),
                "confidence_cap": _rate(confidence_capped, total),
                "unsupported_claim": _rate(unsupported_claim_events, total),
                "citation_validation_failure": _rate(citation_failure_events, total),
                "followup_transform_success": _rate(len(transform_events), max(1, len(transform_events))),
                "structured_search_exact_result": _rate(structured_exact, len(structured_events)),
            },
            "averages": {
                "accepted_evidence_count": round(sum(accepted_counts) / max(1, total), 3),
                "answer_word_count": round(
                    sum(_safe_int(event.get("answer_word_count")) for event in query_events) / max(1, total),
                    3,
                ),
            },
            "latency": {
                "p50_seconds": _percentile(latencies, 0.50),
                "p95_seconds": _percentile(latencies, 0.95),
                "max_seconds": round(max(latencies), 3) if latencies else 0.0,
                "by_intent": by_intent,
                "by_kb": by_kb,
            },
            "alert_counts": dict(alert_counts),
            "active_alerts": self.active_alerts(events),
            "distribution_alerts": self.distribution_alerts(events),
            "generated_at": time.time(),
        }

    def recent_events(self, *, limit: Optional[int] = None, kb_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            events = list(self._events)
        if kb_id:
            events = [event for event in events if event.get("kb_id") == kb_id]
        if limit is not None:
            events = events[-max(0, int(limit)):]
        return events

    def active_alerts(self, events: Optional[Iterable[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        source = list(events) if events is not None else self.recent_events()
        alerts: List[Dict[str, Any]] = []
        for event in source:
            for alert in event.get("alerts") or []:
                alerts.append({
                    **alert,
                    "request_id": event.get("request_id"),
                    "kb_id": event.get("kb_id"),
                    "timestamp": event.get("timestamp"),
                })
        severity_rank = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda item: (severity_rank.get(item.get("severity"), 9), -(item.get("timestamp") or 0)))
        return alerts[:50]

    def distribution_alerts(self, events: Optional[Iterable[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Return rolling alerts tuned from the recent event distribution.

        The newest slice is compared against older events in the same in-memory
        window. If there is not enough baseline data yet, conservative static
        thresholds still catch obvious production risk while the collector warms
        up.
        """
        source = [event for event in (list(events) if events is not None else self.recent_events()) if event.get("answer_state") != "error"]
        if not source:
            return []

        recent_size = min(len(source), max(10, min(DEFAULT_TUNING_RECENT_WINDOW, max(1, len(source) // 3))))
        recent = source[-recent_size:]
        if len(recent) < DEFAULT_TUNING_MIN_RECENT:
            return []
        baseline = source[:-recent_size]
        baseline_ready = len(baseline) >= DEFAULT_TUNING_MIN_BASELINE
        recent_metrics = _distribution_metrics(recent)
        baseline_metrics = _distribution_metrics(baseline) if baseline_ready else {}

        alerts: List[Dict[str, Any]] = []
        for metric, static_threshold in [
            ("source_limited_rate", STATIC_ALERT_THRESHOLDS["source_limited_rate"]),
            ("no_accepted_evidence_rate", STATIC_ALERT_THRESHOLDS["no_accepted_evidence_rate"]),
            ("citation_failure_rate", STATIC_ALERT_THRESHOLDS["citation_failure_rate"]),
            ("confidence_cap_rate", STATIC_ALERT_THRESHOLDS["confidence_cap_rate"]),
            ("unsupported_removed_rate", STATIC_ALERT_THRESHOLDS["unsupported_removed_rate"]),
        ]:
            current = float(recent_metrics.get(metric, 0.0) or 0.0)
            baseline_value = float(baseline_metrics.get(metric, 0.0) or 0.0)
            threshold = _adaptive_rate_threshold(
                baseline_value=baseline_value,
                static_threshold=static_threshold,
                baseline_ready=baseline_ready,
            )
            if current > threshold:
                alerts.append(_distribution_alert(
                    metric=metric,
                    current=current,
                    threshold=threshold,
                    baseline_value=baseline_value if baseline_ready else None,
                    recent_count=len(recent),
                    baseline_count=len(baseline),
                    severity="warning",
                ))

        current_p95 = float(recent_metrics.get("p95_latency_seconds", 0.0) or 0.0)
        baseline_p95 = float(baseline_metrics.get("p95_latency_seconds", 0.0) or 0.0)
        latency_threshold = _adaptive_latency_threshold(
            baseline_value=baseline_p95,
            static_threshold=STATIC_ALERT_THRESHOLDS["p95_latency_seconds"],
            baseline_ready=baseline_ready,
        )
        if current_p95 > latency_threshold:
            alerts.append(_distribution_alert(
                metric="p95_latency_seconds",
                current=current_p95,
                threshold=latency_threshold,
                baseline_value=baseline_p95 if baseline_ready else None,
                recent_count=len(recent),
                baseline_count=len(baseline),
                severity="critical" if current_p95 >= 60.0 else "warning",
            ))

        severity_rank = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda item: (severity_rank.get(item.get("severity"), 9), item.get("metric", "")))
        return alerts

    def _store_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
        self._append_event_log(event)

    def _append_event_log(self, event: Dict[str, Any]) -> None:
        if not self.event_log_path:
            return
        try:
            with self.event_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        except Exception:
            logger.warning("Failed to append query observability event", exc_info=True)

    def _alerts_for_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        kb_id = str(event.get("kb_id") or "")
        query_intent = str(event.get("query_intent") or "")
        answer_state = str(event.get("answer_state") or "")
        no_evidence = _safe_int(event.get("accepted_evidence_count")) <= 0

        if kb_id.startswith("ownify") and no_evidence and answer_state in {"grounded_answer", "direct_response"}:
            alerts.append({
                "severity": "critical",
                "code": "ownify_answer_without_accepted_evidence",
                "message": "Ownify query answered without accepted evidence",
            })
        if event.get("fallback_used") and query_intent in {"count", "structured_list", "requirement", "legal", "archive"}:
            alerts.append({
                "severity": "warning",
                "code": "fallback_used_for_high_risk_query",
                "message": "Verification fallback used for a high-risk query",
            })
        if _safe_int(event.get("unsupported_claim_count")) > 0:
            alerts.append({
                "severity": "warning",
                "code": "unsupported_claims_detected",
                "message": "Claim validator found unsupported claims",
            })
        if _safe_int(event.get("citation_validation_failure_count")) > 0:
            alerts.append({
                "severity": "warning",
                "code": "citation_validation_failures",
                "message": "Citation validator found unsupported or missing citation snippets",
            })
        if event.get("confidence_cap_applied"):
            alerts.append({
                "severity": "info",
                "code": "confidence_cap_applied",
                "message": "Answer confidence was capped by evidence quality controls",
            })
        if query_intent in {"definition", "direct_answer", "factual"} and _safe_float(event.get("latency_seconds")) > 9.0:
            alerts.append({
                "severity": "warning",
                "code": "direct_answer_latency_budget_exceeded",
                "message": "Direct-answer latency exceeded 9 seconds",
            })
        return alerts

    def _latency_groups(self, events: Sequence[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, List[float]] = defaultdict(list)
        for event in events:
            groups[str(event.get(key) or "unknown")].append(_safe_float(event.get("latency_seconds")))
        return {
            group: {
                "count": len(values),
                "p50_seconds": _percentile(values, 0.50),
                "p95_seconds": _percentile(values, 0.95),
            }
            for group, values in sorted(groups.items())
        }
