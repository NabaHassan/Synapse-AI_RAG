"""Evidence admission gate for retrieved RAG context."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "of", "on", "or", "the", "this",
    "that", "to", "was", "what", "when", "where", "which", "who", "why",
    "with", "would", "should", "can", "could", "please", "tell", "me",
}


@dataclass
class EvidencePacket:
    evidence_id: str
    kb_id: str
    document_id: str
    source_file: str
    chunk_id: str
    text: str
    support_type: str
    supports: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    admission_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RejectedEvidence:
    source_file: str
    chunk_id: str
    reason: str
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceAdmissionResult:
    accepted_packets: List[EvidencePacket]
    rejected: List[RejectedEvidence]
    admission_status: str
    mode: str = "shadow"
    input_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "admission_status": self.admission_status,
            "mode": self.mode,
            "input_count": self.input_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "accepted_packets": [packet.to_dict() for packet in self.accepted_packets],
            "rejected": [item.to_dict() for item in self.rejected[:10]],
        }


class EvidenceAdmissionGate:
    """Admit direct/exact/partial evidence and explain rejections."""

    def admit(
        self,
        query: str,
        docs: Sequence[Any],
        evidence_plan: Any,
        *,
        kb_id: str = "",
        mode: str = "shadow",
    ) -> EvidenceAdmissionResult:
        plan_dict = evidence_plan.to_dict() if hasattr(evidence_plan, "to_dict") else dict(evidence_plan or {})
        requirements = plan_dict.get("evidence_requirements") or {}
        query_terms = _content_terms(query)
        exact_required = bool(requirements.get("needs_exact_source"))
        entity_required = bool(requirements.get("needs_named_entity_support"))
        numeric_required = bool(requirements.get("needs_numeric_support"))
        requested_support = _requested_supports(plan_dict)

        accepted: List[EvidencePacket] = []
        rejected: List[RejectedEvidence] = []

        for idx, doc in enumerate(docs or []):
            text = getattr(doc, "content", "") or ""
            meta = dict(getattr(doc, "meta", {}) or {})
            source_file = str(meta.get("source_filename") or meta.get("source") or meta.get("file_name") or "Unknown")
            chunk_id = str(meta.get("chunk_id") or meta.get("id") or idx)
            doc_id = str(meta.get("document_id") or meta.get("file_id") or source_file)
            score = float(getattr(doc, "score", 0.0) or meta.get("rerank_score", 0.0) or 0.0)

            support_type, reason, support_score = self._score_support(
                query=query,
                text=text,
                meta=meta,
                query_terms=query_terms,
                exact_required=exact_required,
                entity_required=entity_required,
                numeric_required=numeric_required,
            )
            scores = {
                "rerank": round(score, 6),
                "metadata_match": round(float(meta.get("metadata_match_score", 0.0) or 0.0), 6),
                "lexical_support": round(support_score, 6),
            }

            if support_type in {"exact", "direct", "partial"}:
                evidence_id = f"{kb_id or meta.get('kb_id', '')}:{chunk_id}:{idx}"
                accepted.append(
                    EvidencePacket(
                        evidence_id=evidence_id,
                        kb_id=kb_id or str(meta.get("kb_id") or ""),
                        document_id=doc_id,
                        source_file=source_file,
                        chunk_id=chunk_id,
                        text=text,
                        support_type=support_type,
                        supports=requested_support,
                        scores=scores,
                        admission_reason=reason,
                        metadata=meta,
                    )
                )
            else:
                rejected.append(
                    RejectedEvidence(
                        source_file=source_file,
                        chunk_id=chunk_id,
                        reason=reason,
                        scores=scores,
                    )
                )

        status = self._status(plan_dict, accepted, rejected, docs)
        return EvidenceAdmissionResult(
            accepted_packets=accepted,
            rejected=rejected,
            admission_status=status,
            mode=mode,
            input_count=len(docs or []),
            accepted_count=len(accepted),
            rejected_count=len(rejected),
        )

    def _score_support(
        self,
        *,
        query: str,
        text: str,
        meta: Dict[str, Any],
        query_terms: List[str],
        exact_required: bool,
        entity_required: bool,
        numeric_required: bool,
    ) -> Tuple[str, str, float]:
        haystack_score = float(meta.get("score", 0.0) or 0.0)
        lowered = text.lower()
        source_lower = _metadata_text(meta)

        exact_phrases = _quoted_phrases(query)
        if exact_phrases and all(phrase.lower() in lowered or phrase.lower() in source_lower for phrase in exact_phrases):
            return "exact", "all quoted/exact phrases found", 1.0

        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", query)
        if numeric_required and numbers and not any(number in lowered or number in source_lower for number in numbers):
            return "rejected", "numeric support required but query number not found", 0.0

        if entity_required:
            entities = _candidate_entities(query)
            if entities and not any(entity.lower() in lowered or entity.lower() in source_lower for entity in entities):
                return "rejected", "named entity support required but entity not found", 0.0

        if not query_terms:
            return "partial", "no strong query terms; preserving verified context", max(haystack_score, 0.25)

        matched = [term for term in query_terms if term in lowered or term in source_lower]
        ratio = len(matched) / max(1, len(query_terms))

        if ratio >= 0.55:
            return "direct", f"matched {len(matched)}/{len(query_terms)} query terms", ratio
        if ratio >= 0.25 and not exact_required:
            return "partial", f"matched {len(matched)}/{len(query_terms)} query terms", ratio
        if haystack_score >= 0.85 and not exact_required:
            return "partial", "high rerank score but weak lexical support", haystack_score
        return "rejected", f"weak lexical support ({len(matched)}/{len(query_terms)} terms)", ratio

    @staticmethod
    def _status(plan_dict: Dict[str, Any], accepted: List[EvidencePacket], rejected: List[RejectedEvidence], docs: Sequence[Any]) -> str:
        if accepted:
            if any(packet.support_type in {"exact", "direct"} for packet in accepted):
                return "direct_support"
            return "partial_support"
        if not docs:
            return "insufficient_evidence"
        if plan_dict.get("query_intent") == "missing_detail":
            return "missing_requested_detail"
        return "background_only"


def filter_docs_to_accepted_packets(docs: Sequence[Any], admission_result: EvidenceAdmissionResult) -> List[Any]:
    """Return docs whose chunk ids/source files were admitted."""
    accepted_keys = {
        (packet.chunk_id, packet.source_file)
        for packet in admission_result.accepted_packets
    }
    if not accepted_keys:
        return []
    selected = []
    for idx, doc in enumerate(docs or []):
        meta = dict(getattr(doc, "meta", {}) or {})
        source_file = str(meta.get("source_filename") or meta.get("source") or meta.get("file_name") or "Unknown")
        chunk_id = str(meta.get("chunk_id") or meta.get("id") or idx)
        if (chunk_id, source_file) in accepted_keys:
            selected.append(doc)
    return selected


def _content_terms(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", text.lower())
    terms = []
    for token in tokens:
        token = token.strip("-_'")
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms[:16]


def _quoted_phrases(text: str) -> List[str]:
    return [match.strip() for match in re.findall(r'"([^"]{2,})"|\'([^\']{2,})\'', text) for match in match if match.strip()]


def _candidate_entities(text: str) -> List[str]:
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text or "")
    return [candidate for candidate in candidates if candidate.lower() not in STOPWORDS]


def _metadata_text(meta: Dict[str, Any]) -> str:
    """Flatten evidence metadata so entity overlays can support admission checks."""
    values: List[str] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (str, int, float, bool)):
            values.append(str(value))
            return
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    visit(meta)
    return " ".join(values).lower()


def _requested_supports(plan_dict: Dict[str, Any]) -> List[str]:
    intent = str(plan_dict.get("query_intent") or "unknown")
    if intent == "requirement":
        return ["requirement"]
    if intent == "definition":
        return ["definition"]
    if intent == "missing_detail":
        return ["missing_detail"]
    if intent == "count":
        return ["count"]
    if intent == "structured_list":
        return ["structured_list"]
    return [intent]
