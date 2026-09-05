"""Evidence-aware retrieval boosts based on overlay metadata and query plans."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from haystack import Document

from src.indexing.metadata_overlay import MetadataOverlayExtractor


@dataclass
class BoostDecision:
    candidate_id: str
    original_rank: int
    original_score: float
    boosted_score: float
    final_score: float
    boost: float
    grouping_penalty: float
    suppression_penalty: float
    suppressed: bool
    group_key: str
    group_rank: int
    reasons: List[str]
    overlay: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceRetrievalBooster:
    """Score boosts that prefer exact source/entity/field support without filtering."""

    def __init__(self):
        self.extractor = MetadataOverlayExtractor()

    def boost(
        self,
        query: str,
        docs: Sequence[Document],
        evidence_plan: Any,
        *,
        domain_profile: Optional[Dict[str, Any]] = None,
        apply_order: bool = False,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        plan = evidence_plan.to_dict() if hasattr(evidence_plan, "to_dict") else dict(evidence_plan or {})
        profile = dict(domain_profile or {})
        decisions: List[Dict[str, Any]] = []
        entries: List[Dict[str, Any]] = []

        for original_rank, doc in enumerate(docs or [], start=1):
            payload = dict(doc.meta or {})
            payload["content"] = doc.content
            overlay = self.extractor.extract(payload).to_payload()
            boost, reasons = self._score(query, overlay, plan, profile)
            original = float(doc.score or 0.0)
            pre_group_score = original + boost
            group_key = self._group_key(overlay)
            suppression_penalty, suppression_reasons = self._suppression_penalty(query, overlay, plan)
            entries.append(
                {
                    "doc": doc,
                    "original_rank": original_rank,
                    "original_score": original,
                    "boost": boost,
                    "pre_group_score": pre_group_score,
                    "group_key": group_key,
                    "overlay": overlay,
                    "reasons": reasons + suppression_reasons,
                    "suppression_penalty": suppression_penalty,
                    "suppressed": suppression_penalty < 0,
                }
            )

        entries.sort(key=lambda item: item["pre_group_score"], reverse=True)
        group_seen: Dict[str, int] = {}
        for entry in entries:
            group_key = entry["group_key"]
            group_seen[group_key] = group_seen.get(group_key, 0) + 1
            group_rank = group_seen[group_key]
            grouping_penalty = self._grouping_penalty(group_rank)
            if grouping_penalty:
                entry["reasons"].append("source_group_crowding_penalty")
            entry["group_rank"] = group_rank
            entry["grouping_penalty"] = grouping_penalty
            entry["final_score"] = entry["pre_group_score"] + grouping_penalty + entry["suppression_penalty"]

        boosted_docs: List[Document] = []
        for entry in sorted(entries, key=lambda item: item["original_rank"]):
            doc = entry["doc"]
            overlay = entry["overlay"]
            meta = dict(doc.meta or {})
            meta.update({key: value for key, value in overlay.items() if key not in meta})
            meta["evidence_boost"] = {
                "boost": round(entry["boost"], 6),
                "grouping_penalty": round(entry["grouping_penalty"], 6),
                "suppression_penalty": round(entry["suppression_penalty"], 6),
                "suppressed": bool(entry["suppressed"]),
                "reasons": entry["reasons"],
                "original_score": round(entry["original_score"], 6),
                "boosted_score": round(entry["pre_group_score"], 6),
                "final_score": round(entry["final_score"], 6),
                "group_key": entry["group_key"],
                "group_rank": entry["group_rank"],
            }
            boosted_docs.append(
                Document(
                    content=doc.content,
                    id=doc.id,
                    meta=meta,
                    score=entry["final_score"] if apply_order else doc.score,
                )
            )
            decisions.append(
                BoostDecision(
                    candidate_id=str(doc.id or meta.get("chunk_id") or entry["original_rank"]),
                    original_rank=entry["original_rank"],
                    original_score=entry["original_score"],
                    boosted_score=entry["pre_group_score"],
                    final_score=entry["final_score"],
                    boost=entry["boost"],
                    grouping_penalty=entry["grouping_penalty"],
                    suppression_penalty=entry["suppression_penalty"],
                    suppressed=entry["suppressed"],
                    group_key=entry["group_key"],
                    group_rank=entry["group_rank"],
                    reasons=entry["reasons"],
                    overlay=overlay,
                ).to_dict()
            )

        if apply_order:
            boosted_docs.sort(key=lambda item: float(item.score or 0.0), reverse=True)

        grouped_counts = self._group_counts(decisions)
        metadata = {
            "enabled": True,
            "applied_to_order": bool(apply_order),
            "input_count": len(docs or []),
            "boosted_count": sum(1 for item in decisions if item["boost"] > 0),
            "suppressed_count": sum(1 for item in decisions if item.get("suppressed")),
            "group_count": len(grouped_counts),
            "crowded_group_count": sum(1 for count in grouped_counts.values() if count > 1),
            "groups": grouped_counts,
            "top_reasons": self._top_reasons(decisions),
            "decisions": sorted(decisions, key=lambda item: item["final_score"], reverse=True)[:20],
            "candidate_debug": self._candidate_debug(decisions),
        }
        return boosted_docs, metadata

    def _score(
        self,
        query: str,
        overlay: Dict[str, Any],
        plan: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        q = (query or "").lower()
        boost = 0.0
        reasons: List[str] = []
        source_family = str(overlay.get("source_family") or "").lower()
        section_id = str(overlay.get("section_id") or "").lower()
        answerable_fields = {str(item).lower() for item in overlay.get("answerable_fields") or []}
        primary_entities = {str(item).lower() for item in overlay.get("primary_entities") or []}

        if source_family and any(term in source_family for term in _important_terms(q)):
            boost += 0.08
            reasons.append("source_family_query_match")

        if section_id and section_id in q:
            boost += 0.18
            reasons.append("section_id_exact_match")

        query_intent = str(plan.get("query_intent") or "")
        if query_intent == "missing_detail" and {"price", "date", "requirement"} & answerable_fields:
            boost += 0.08
            reasons.append("answerable_field_match")
        if query_intent == "requirement" and "requirement" in answerable_fields:
            boost += 0.1
            reasons.append("requirement_field_match")
        if query_intent in {"structured_list", "count"} and primary_entities:
            query_entities = {item.lower() for item in _candidate_entities(query)}
            if query_entities & primary_entities:
                boost += 0.12
                reasons.append("entity_exact_match")

        profile_type = str(profile.get("type") or "")
        if profile_type == "legal_precision" and section_id:
            boost += 0.05
            reasons.append("legal_section_metadata")
        if profile_type == "evidence_archive" and overlay.get("document_id"):
            boost += 0.04
            reasons.append("archive_document_identity")
        if profile_type == "product_ops" and answerable_fields:
            boost += 0.04
            reasons.append("product_answerable_fields")

        quality = overlay.get("ocr_or_extraction_quality")
        try:
            if quality is not None and float(quality) < 0.5:
                boost -= 0.05
                reasons.append("low_extraction_quality_penalty")
        except (TypeError, ValueError):
            pass

        return round(max(-0.1, min(0.35, boost)), 6), reasons

    def _suppression_penalty(
        self,
        query: str,
        overlay: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Softly demote unrelated source families for exact evidence queries."""
        requirements = dict(plan.get("evidence_requirements") or {})
        exactish_query = (
            str(plan.get("query_intent") or "") in {"requirement", "missing_detail"}
            or bool(requirements.get("needs_exact_source"))
            or bool(requirements.get("needs_policy_or_rule_support"))
        )
        if not exactish_query:
            return 0.0, []

        if self._has_exact_match(query, overlay):
            return 0.0, []

        query_facets = set(_important_terms((query or "").lower()))
        query_facets |= self._topic_facets(query)
        candidate_facets = self._candidate_facets(overlay)
        candidate_topics = {str(item).lower() for item in overlay.get("topic_path") or []}
        source_family = str(overlay.get("source_family") or "")

        if not candidate_topics and not candidate_facets and not source_family:
            return 0.0, []

        if query_facets & candidate_facets:
            return 0.0, []

        if candidate_topics:
            return -0.08, ["negative_source_family_suppression"]

        if source_family:
            return -0.05, ["negative_source_family_suppression"]

        return 0.0, []

    @staticmethod
    def _group_key(overlay: Dict[str, Any]) -> str:
        section_id = overlay.get("section_id")
        if section_id:
            return f"section:{section_id}"
        document_id = overlay.get("document_id")
        if document_id:
            return f"document:{document_id}"
        source_family = overlay.get("source_family")
        if source_family:
            return f"source:{source_family}"
        return "unknown"

    @staticmethod
    def _grouping_penalty(group_rank: int) -> float:
        if group_rank <= 1:
            return 0.0
        return round(-min(0.12, 0.035 * (group_rank - 1)), 6)

    @staticmethod
    def _has_exact_match(query: str, overlay: Dict[str, Any]) -> bool:
        q = (query or "").lower()
        section_id = str(overlay.get("section_id") or "").lower()
        if section_id and section_id in q:
            return True
        source_family = str(overlay.get("source_family") or "").lower()
        if source_family and any(term in source_family for term in _important_terms(q)):
            return True
        primary_entities = {str(item).lower() for item in overlay.get("primary_entities") or []}
        query_entities = {item.lower() for item in _candidate_entities(query)}
        return bool(primary_entities & query_entities)

    @staticmethod
    def _candidate_facets(overlay: Dict[str, Any]) -> set:
        facets = set()
        for field in ["source_family", "source_type", "document_type", "section_id"]:
            value = overlay.get(field)
            if value:
                facets.update(_important_terms(str(value).lower()))
        for field in ["topic_path", "primary_entities"]:
            for item in overlay.get(field) or []:
                facets.update(_important_terms(str(item).lower()))
                facets.add(str(item).lower())
        return {item for item in facets if item}

    @staticmethod
    def _topic_facets(query: str) -> set:
        q = (query or "").lower()
        facets = set()
        topic_markers = {
            "pricing": ["price", "pricing", "cost", "fee", "billing", "subscription"],
            "workflow": ["workflow", "process", "steps", "onboarding", "setup"],
            "feature": ["feature", "capability", "dashboard", "agent", "analytics"],
            "policy": ["policy", "terms", "contract", "eligibility", "requirement"],
            "date": ["date", "when", "deadline"],
            "email": ["email", "sender", "recipient"],
        }
        for topic, markers in topic_markers.items():
            if any(marker in q for marker in markers):
                facets.add(topic)
                facets.update(markers)
        return facets

    @staticmethod
    def _group_counts(decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for decision in decisions:
            group_key = str(decision.get("group_key") or "unknown")
            counts[group_key] = counts.get(group_key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:25])

    @staticmethod
    def _candidate_debug(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for decision in sorted(decisions, key=lambda item: item["final_score"], reverse=True)[:20]:
            overlay = decision.get("overlay") or {}
            rows.append({
                "candidate_id": decision.get("candidate_id"),
                "original_rank": decision.get("original_rank"),
                "group_key": decision.get("group_key"),
                "group_rank": decision.get("group_rank"),
                "original_score": round(float(decision.get("original_score") or 0.0), 6),
                "boost": round(float(decision.get("boost") or 0.0), 6),
                "grouping_penalty": round(float(decision.get("grouping_penalty") or 0.0), 6),
                "suppression_penalty": round(float(decision.get("suppression_penalty") or 0.0), 6),
                "final_score": round(float(decision.get("final_score") or 0.0), 6),
                "suppressed": bool(decision.get("suppressed")),
                "reasons": list(decision.get("reasons") or []),
                "source_family": overlay.get("source_family"),
                "document_id": overlay.get("document_id"),
                "section_id": overlay.get("section_id"),
                "source_type": overlay.get("source_type"),
                "topic_path": overlay.get("topic_path") or [],
                "answerable_fields": overlay.get("answerable_fields") or [],
            })
        return rows

    @staticmethod
    def _top_reasons(decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for decision in decisions:
            for reason in decision.get("reasons") or []:
                counts[reason] = counts.get(reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _important_terms(query_lower: str) -> List[str]:
    stop = {"what", "which", "where", "when", "does", "with", "from", "about", "this", "that", "the", "and"}
    return [
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.-]{2,}", query_lower)
        if token not in stop
    ][:12]


def _candidate_entities(text: str) -> List[str]:
    return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text or "")
