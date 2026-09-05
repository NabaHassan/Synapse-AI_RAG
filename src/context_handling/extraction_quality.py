"""Extraction-quality risk signals for retrieved evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


LOW_QUALITY_FLAGS = {
    "low_text_quality",
    "low_alpha_ratio",
    "ocr_fragmented_lines",
    "replacement_characters",
    "extraction_error",
    "low_quality_chunk",
}

SOURCE_LIMITING_FLAGS = {
    "low_alpha_ratio",
    "ocr_fragmented_lines",
    "replacement_characters",
    "extraction_error",
}


@dataclass(frozen=True)
class ExtractionQualityResult:
    """Summary of extraction quality across final answer evidence."""

    enabled: bool = True
    doc_count: int = 0
    low_quality_count: int = 0
    low_quality_ratio: float = 0.0
    min_quality: Optional[float] = None
    avg_quality: Optional[float] = None
    all_low_quality: bool = False
    any_low_quality: bool = False
    high_risk_query: bool = False
    source_limited_recommended: bool = False
    confidence_cap: Optional[float] = None
    reasons: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "doc_count": self.doc_count,
            "low_quality_count": self.low_quality_count,
            "low_quality_ratio": round(self.low_quality_ratio, 6),
            "min_quality": None if self.min_quality is None else round(self.min_quality, 6),
            "avg_quality": None if self.avg_quality is None else round(self.avg_quality, 6),
            "all_low_quality": self.all_low_quality,
            "any_low_quality": self.any_low_quality,
            "high_risk_query": self.high_risk_query,
            "source_limited_recommended": self.source_limited_recommended,
            "confidence_cap": None if self.confidence_cap is None else round(self.confidence_cap, 6),
            "reasons": list(self.reasons),
            "flags": list(self.flags),
        }


def evaluate_extraction_quality(
        docs: Sequence[Any],
        evidence_plan: Optional[Any] = None,
        *,
        low_quality_threshold: float = 0.55,
        source_limited_threshold: float = 0.45,
) -> ExtractionQualityResult:
    """Evaluate whether retrieved evidence is extraction-limited.

    The decision is intentionally generic: it only reads document metadata
    produced by indexing/extraction and the evidence plan's risk requirements.
    """
    docs = list(docs or [])
    if not docs:
        return ExtractionQualityResult()

    qualities: List[float] = []
    low_count = 0
    flags_seen: List[str] = []
    reasons: List[str] = []

    for doc in docs:
        meta = dict(getattr(doc, "meta", {}) or {})
        flags = _quality_flags(meta)
        for flag in flags:
            if flag not in flags_seen:
                flags_seen.append(flag)

        quality = _quality_score(meta)
        if quality is not None:
            qualities.append(quality)

        is_low = bool(meta.get("is_low_quality_chunk"))
        is_low = is_low or bool(LOW_QUALITY_FLAGS & set(flags))
        is_low = is_low or (quality is not None and quality < low_quality_threshold)
        if is_low:
            low_count += 1

    doc_count = len(docs)
    ratio = low_count / max(1, doc_count)
    min_quality = min(qualities) if qualities else None
    avg_quality = sum(qualities) / len(qualities) if qualities else None
    any_low = low_count > 0
    all_low = low_count == doc_count
    high_risk = _high_risk_plan(evidence_plan)

    confidence_cap: Optional[float] = None
    if all_low:
        confidence_cap = 0.45
        reasons.append("all_final_evidence_low_extraction_quality")
    elif any_low:
        confidence_cap = 0.65
        reasons.append("some_final_evidence_low_extraction_quality")

    source_limited = bool(
        all_low
        and high_risk
        and (avg_quality is None or avg_quality < source_limited_threshold)
        and bool(SOURCE_LIMITING_FLAGS & set(flags_seen))
    )
    if source_limited:
        reasons.append("high_risk_query_with_only_low_quality_evidence")

    return ExtractionQualityResult(
        doc_count=doc_count,
        low_quality_count=low_count,
        low_quality_ratio=ratio,
        min_quality=min_quality,
        avg_quality=avg_quality,
        all_low_quality=all_low,
        any_low_quality=any_low,
        high_risk_query=high_risk,
        source_limited_recommended=source_limited,
        confidence_cap=confidence_cap,
        reasons=reasons,
        flags=flags_seen[:20],
    )


def _quality_flags(meta: Dict[str, Any]) -> List[str]:
    raw = meta.get("quality_flags") or []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = []
    flags = []
    for value in values:
        flag = str(value).strip()
        if flag and flag not in flags:
            flags.append(flag)
    return flags


def _quality_score(meta: Dict[str, Any]) -> Optional[float]:
    raw = meta.get("ocr_or_extraction_quality")
    if raw is None:
        raw = meta.get("quality_score")
    if raw is None:
        return None
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return None


def _high_risk_plan(evidence_plan: Optional[Any]) -> bool:
    if evidence_plan is None:
        return False
    plan = evidence_plan.to_dict() if hasattr(evidence_plan, "to_dict") else dict(evidence_plan or {})
    intent = str(plan.get("query_intent") or "").strip().lower()
    requirements = dict(plan.get("evidence_requirements") or {})
    return bool(
        intent in {"count", "missing_detail", "requirement", "structured_list"}
        or requirements.get("needs_exact_source")
        or requirements.get("needs_named_entity_support")
        or requirements.get("needs_numeric_support")
        or requirements.get("needs_policy_or_rule_support")
    )
