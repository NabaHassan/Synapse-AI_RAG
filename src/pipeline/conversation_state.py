"""Evidence-aware conversation state for follow-up resolution.

This module keeps the resolver generic: it only uses prior turn evidence,
citations, entities, and source metadata. It does not special-case KB ids,
tenant names, legal domains, or archive domains.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


MAX_STATE_PACKETS = 6
MAX_STATE_CITATIONS = 8
MAX_STATE_ENTITIES = 12
MAX_ANCHOR_CHARS = 180


@dataclass
class EvidenceConversationState:
    turn_id: Optional[int] = None
    query: str = ""
    reformulated_query: str = ""
    answer_summary: str = ""
    answer_state: str = ""
    grounding_status: str = ""
    query_intent: str = ""
    answer_mode: str = ""
    entities: List[str] = field(default_factory=list)
    evidence_packets: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    source_families: List[str] = field(default_factory=list)
    document_ids: List[str] = field(default_factory=list)
    section_ids: List[str] = field(default_factory=list)
    unresolved_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence_packets or self.citations or self.document_ids or self.source_families)

    @property
    def anchor(self) -> str:
        candidates = [
            self.query,
            self.reformulated_query,
            self.answer_summary,
            " ".join(self.entities[:4]),
        ]
        for candidate in candidates:
            cleaned = _clean_anchor(candidate)
            if cleaned:
                return cleaned[:MAX_ANCHOR_CHARS]
        return ""


@dataclass
class FollowupResolution:
    is_follow_up: bool
    method: str = "none"
    follow_up_type: str = "new_query"
    confidence: float = 0.0
    resolved_query: str = ""
    reason: str = ""
    state: Dict[str, Any] = field(default_factory=dict)
    unresolved_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceConversationStateResolver:
    """Resolve follow-ups from prior evidence state before legacy patterns run."""

    def build_from_history(
        self,
        conversation_history: Sequence[Any],
        *,
        session_metadata: Optional[Dict[str, Any]] = None,
    ) -> EvidenceConversationState:
        latest = _state_from_session_metadata(session_metadata or {})
        if latest.has_evidence or latest.anchor:
            return latest

        for turn in reversed(list(conversation_history or [])):
            state = state_from_turn(turn)
            if state.has_evidence or state.anchor:
                return state
        return EvidenceConversationState()

    def resolve(
        self,
        query: str,
        conversation_history: Sequence[Any],
        *,
        session_metadata: Optional[Dict[str, Any]] = None,
    ) -> FollowupResolution:
        state = self.build_from_history(conversation_history, session_metadata=session_metadata)
        query_text = (query or "").strip()
        if not query_text or not (state.has_evidence or state.anchor):
            return FollowupResolution(is_follow_up=False, reason="no_state")

        follow_type = _classify_followup(query_text)
        if follow_type is None:
            return FollowupResolution(
                is_follow_up=False,
                reason="self_contained_or_no_followup_signal",
                state=_compact_state(state),
            )

        unresolved = _unresolved_references(query_text)
        resolved_query = _build_resolved_query(query_text, follow_type, state)
        confidence = _confidence_for(follow_type, state, unresolved)
        return FollowupResolution(
            is_follow_up=True,
            method="evidence_state",
            follow_up_type=follow_type,
            confidence=confidence,
            resolved_query=resolved_query,
            reason=f"{follow_type}_resolved_from_evidence_state",
            state=_compact_state(state),
            unresolved_references=unresolved,
        )


def state_from_turn(turn: Any) -> EvidenceConversationState:
    metadata = _turn_metadata(turn)
    evidence_state = metadata.get("evidence_conversation_state")
    if isinstance(evidence_state, dict):
        return _state_from_dict(evidence_state)

    return _state_from_dict(build_state_snapshot(
        query=str(getattr(turn, "query", "") or _dict_get(turn, "query") or ""),
        reformulated_query=str(getattr(turn, "reformulated_query", "") or _dict_get(turn, "reformulated_query") or ""),
        answer=str(getattr(turn, "answer", "") or _dict_get(turn, "answer") or ""),
        metadata=metadata,
        citations=getattr(turn, "citations", None) or _dict_get(turn, "citations") or [],
        entities=getattr(turn, "entities_mentioned", None) or _dict_get(turn, "entities_mentioned") or [],
        source_documents=getattr(turn, "source_documents", None) or _dict_get(turn, "source_documents") or [],
        turn_id=getattr(turn, "turn_id", None) or _dict_get(turn, "turn_id"),
    ))


def build_state_snapshot(
    *,
    query: str,
    answer: str,
    metadata: Optional[Dict[str, Any]] = None,
    reformulated_query: str = "",
    citations: Optional[Sequence[Dict[str, Any]]] = None,
    entities: Optional[Sequence[str]] = None,
    source_documents: Optional[Sequence[Dict[str, Any]]] = None,
    turn_id: Optional[int] = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    evidence_plan = metadata.get("evidence_plan") or {}
    admission = metadata.get("evidence_admission") or {}
    packets = _limit_dicts(admission.get("accepted_packets") or [], MAX_STATE_PACKETS)
    citations_list = _limit_dicts(citations or [], MAX_STATE_CITATIONS)
    source_docs = _limit_dicts(source_documents or [], MAX_STATE_CITATIONS)

    packet_metadata = [dict(packet.get("metadata") or {}) for packet in packets]
    source_families = _unique(
        _values(packet_metadata, "source_family")
        + _values(source_docs, "source_family")
        + _values(source_docs, "source_file")
        + _values(citations_list, "source")
    )
    document_ids = _unique(_values(packets, "document_id") + _values(packet_metadata, "document_id"))
    section_ids = _unique(_values(packet_metadata, "section_id") + _values(source_docs, "section_id"))
    entity_values = _unique([str(item) for item in entities or []] + _entities_from_packets(packets))

    state = EvidenceConversationState(
        turn_id=turn_id,
        query=_clean_text(query),
        reformulated_query=_clean_text(reformulated_query),
        answer_summary=_answer_summary(answer, metadata),
        answer_state=str(metadata.get("answer_state") or ""),
        grounding_status=str(metadata.get("grounding_status") or ""),
        query_intent=str(evidence_plan.get("query_intent") or evidence_plan.get("intent") or ""),
        answer_mode=str(evidence_plan.get("answer_mode") or ""),
        entities=entity_values[:MAX_STATE_ENTITIES],
        evidence_packets=packets,
        citations=citations_list,
        source_families=source_families[:MAX_STATE_CITATIONS],
        document_ids=document_ids[:MAX_STATE_CITATIONS],
        section_ids=section_ids[:MAX_STATE_CITATIONS],
        unresolved_references=[],
    )
    return state.to_dict()


def _classify_followup(query: str) -> Optional[str]:
    q = query.lower().strip()
    if _looks_like_new_self_contained_query(query):
        return None
    if re.search(r"\b(?:which|what)\s+(?:file|files|document|documents|source|sources|citation|citations)\b", q):
        return "source_request"
    if re.search(r"\b(?:where|in which file|which file|mentioned|found)\b", q) and _has_reference_signal(q):
        return "source_request"
    if re.search(
        r"\b(?:summari[sz]e|short|shorter|concise|brief|simple words|easy words|rephrase|rewrite)\b"
        r"|\b(?:bullet|bullets|bullet points|list|table|format|reformat)\b"
        r"|\b(?:1|one)\s+line\b|\b(?:single|one)\s+sentence\b",
        q,
    ):
        return "transform"
    if re.search(r"^(?:tell|give|show|explain)\s+(?:me\s+)?(?:more|details|more details|more information)\b", q):
        return "expansion"
    if re.search(r"^(?:more|continue|go on|elaborate|expand|explain)\b", q):
        return "expansion"
    if re.search(r"^(?:what|how)\s+about\b", q):
        return "entity_switch"
    if re.search(r"^(?:same for|and for)\b", q):
        return "entity_switch"
    if re.search(r"\b(?:this|that|it|these|those|he|she|they|him|her|them|his|their)\b", q):
        return "coreference"
    if re.search(r"^(?:why|how|who|when|where|which)\??$", q):
        return "implicit_question"
    if re.search(r"\b(?:previous|above|last answer|same)\b", q):
        return "conversation_reference"
    return None


def _build_resolved_query(query: str, follow_type: str, state: EvidenceConversationState) -> str:
    anchor = state.anchor or "the previous answer"
    evidence_terms = _evidence_terms(state)
    if follow_type == "source_request":
        prefix = f"{query} for {anchor}"
    elif follow_type == "transform":
        prefix = f"{query} using the previous answer about {anchor}"
    elif follow_type == "entity_switch":
        prefix = f"{query} in the same evidence context as {anchor}"
    elif follow_type == "implicit_question":
        prefix = f"{query.rstrip('?')} about {anchor}?"
    else:
        prefix = f"{query} about {anchor}"

    if evidence_terms:
        return f"{prefix}. Evidence context: {evidence_terms}"
    return prefix


def _evidence_terms(state: EvidenceConversationState) -> str:
    parts = []
    if state.entities:
        parts.append("entities " + ", ".join(state.entities[:4]))
    if state.section_ids:
        parts.append("sections " + ", ".join(state.section_ids[:3]))
    if state.source_families:
        parts.append("sources " + ", ".join(state.source_families[:3]))
    if state.document_ids:
        parts.append("documents " + ", ".join(state.document_ids[:3]))
    return "; ".join(parts)


def _looks_like_new_self_contained_query(query: str) -> bool:
    q = query.strip()
    q_lower = q.lower()
    if _has_reference_signal(q_lower):
        return False
    if re.search(
        r"^(?:what|who|when|where|why|how|list|count|show|find|can|could|should|"
        r"would|will|is|are|do|does|did|if)\b",
        q_lower,
    ):
        if re.search(r'"[^"]+"|' r"'[^']+'|\b[A-Z]{2,}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", q):
            return True
        if re.search(
            r"\b(?:support|custody|divorce|property|insurance|restraining\s+order|"
            r"spousal|child|family\s+code|law|statute|section|marriage)\b",
            q_lower,
        ):
            return True
    return False


def _has_reference_signal(query_lower: str) -> bool:
    return bool(
        re.search(
            r"\b(?:this|that|it|these|those|he|she|they|him|her|them|his|their|previous|above|same)\b",
            query_lower,
        )
        or re.search(r"^(?:more|continue|go on|elaborate|expand|explain)\b", query_lower)
        or re.search(r"^(?:what|how)\s+about\b", query_lower)
    )


def _unresolved_references(query: str) -> List[str]:
    refs = re.findall(
        r"\b(this|that|it|these|those|he|she|they|him|her|them|his|their|previous|above|same)\b",
        query.lower(),
    )
    return _unique(refs)


def _confidence_for(follow_type: str, state: EvidenceConversationState, unresolved: Sequence[str]) -> float:
    base = {
        "source_request": 0.92,
        "transform": 0.9,
        "expansion": 0.88,
        "entity_switch": 0.84,
        "coreference": 0.82,
        "implicit_question": 0.78,
        "conversation_reference": 0.8,
    }.get(follow_type, 0.7)
    if state.has_evidence:
        base += 0.04
    if unresolved:
        base += 0.02
    return min(0.97, base)


def _compact_state(state: EvidenceConversationState) -> Dict[str, Any]:
    return {
        "turn_id": state.turn_id,
        "query": state.query,
        "reformulated_query": state.reformulated_query,
        "answer_state": state.answer_state,
        "grounding_status": state.grounding_status,
        "query_intent": state.query_intent,
        "answer_mode": state.answer_mode,
        "entities": state.entities[:MAX_STATE_ENTITIES],
        "source_families": state.source_families[:MAX_STATE_CITATIONS],
        "document_ids": state.document_ids[:MAX_STATE_CITATIONS],
        "section_ids": state.section_ids[:MAX_STATE_CITATIONS],
        "evidence_packet_count": len(state.evidence_packets),
        "citation_count": len(state.citations),
        "anchor": state.anchor,
    }


def _state_from_session_metadata(metadata: Dict[str, Any]) -> EvidenceConversationState:
    state = metadata.get("evidence_conversation_state") if isinstance(metadata, dict) else None
    if isinstance(state, dict):
        return _state_from_dict(state)
    return EvidenceConversationState()


def _state_from_dict(data: Dict[str, Any]) -> EvidenceConversationState:
    allowed = set(EvidenceConversationState.__dataclass_fields__.keys())
    return EvidenceConversationState(**{key: value for key, value in dict(data or {}).items() if key in allowed})


def _turn_metadata(turn: Any) -> Dict[str, Any]:
    value = getattr(turn, "metadata", None)
    if isinstance(value, dict):
        return value
    if isinstance(turn, dict) and isinstance(turn.get("metadata"), dict):
        return turn.get("metadata") or {}
    return {}


def _dict_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return None


def _answer_summary(answer: str, metadata: Dict[str, Any]) -> str:
    cleanup = metadata.get("answer_cleanup") or {}
    if cleanup.get("source_limited"):
        prefix = "source-limited answer"
    else:
        prefix = ""
    text = _clean_text(answer)
    if len(text) > MAX_ANCHOR_CHARS:
        text = text[:MAX_ANCHOR_CHARS].rsplit(" ", 1)[0]
    return _clean_text(f"{prefix}: {text}" if prefix else text)


def _entities_from_packets(packets: Sequence[Dict[str, Any]]) -> List[str]:
    entities: List[str] = []
    for packet in packets or []:
        metadata = packet.get("metadata") or {}
        for key in ("entity_names", "entities", "persons", "people"):
            value = metadata.get(key)
            if isinstance(value, str):
                entities.extend([item.strip() for item in value.split(",") if item.strip()])
            elif isinstance(value, Iterable):
                entities.extend([str(item).strip() for item in value if str(item).strip()])
    return entities


def _values(items: Sequence[Dict[str, Any]], key: str) -> List[str]:
    values = []
    for item in items or []:
        value = item.get(key)
        if value is None:
            continue
        values.append(str(value))
    return values


def _limit_dicts(items: Sequence[Any], limit: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for item in list(items or [])[:limit]:
        if isinstance(item, dict):
            results.append(dict(item))
    return results


def _unique(values: Sequence[str]) -> List[str]:
    seen = set()
    results = []
    for value in values or []:
        cleaned = _clean_text(str(value))
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(cleaned)
    return results


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_anchor(value: str) -> str:
    cleaned = _clean_text(value)
    cleaned = re.split(r"\.\s*Evidence context:", cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    return cleaned.strip()
