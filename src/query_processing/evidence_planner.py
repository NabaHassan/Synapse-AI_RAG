"""Deterministic evidence planning for evidence-first RAG."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


INTENT_DEFINITION = "definition"
INTENT_REQUIREMENT = "requirement"
INTENT_COUNT = "count"
INTENT_STRUCTURED_LIST = "structured_list"
INTENT_MISSING_DETAIL = "missing_detail"
INTENT_ANALYSIS = "analysis"
INTENT_TRANSFORM = "transform"
INTENT_META = "meta"
INTENT_UNKNOWN = "unknown"

ANSWER_DIRECT = "direct"
ANSWER_EXTRACTIVE = "extractive"
ANSWER_SOURCE_LIMITED = "source_limited"
ANSWER_GROUNDED_ANALYSIS = "grounded_analysis"
ANSWER_CONVERSATION_TRANSFORM = "conversation_transform"

RETRIEVAL_ANSWER = "answer"
RETRIEVAL_SEARCH = "search"
RETRIEVAL_STRUCTURED_COUNT = "structured_count"
RETRIEVAL_STRUCTURED_LIST = "structured_list"
RETRIEVAL_EVIDENCE_LOOKUP = "evidence_lookup"

REWRITE_NONE = "none"
REWRITE_LEXICAL = "lexical_variants"
REWRITE_ENTITY = "entity_normalization"
REWRITE_LEGAL_REFERENCE = "legal_reference_normalization"
REWRITE_PRODUCT_TERM = "product_term_normalization"
REWRITE_SEMANTIC = "semantic_expansion"


@dataclass
class EvidenceRequirements:
    needs_exact_source: bool = False
    needs_numeric_support: bool = False
    needs_named_entity_support: bool = False
    needs_policy_or_rule_support: bool = False
    allows_synthesis: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidencePlan:
    query_intent: str = INTENT_UNKNOWN
    answer_mode: str = ANSWER_DIRECT
    expected_answer_shape: str = "sentence"
    answer_budget: str = "short"
    retrieval_mode: str = RETRIEVAL_ANSWER
    rewrite_type: str = REWRITE_SEMANTIC
    evidence_requirements: EvidenceRequirements = field(default_factory=EvidenceRequirements)
    requested_operation: Optional[str] = None
    target_entity_type: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence_requirements"] = self.evidence_requirements.to_dict()
        return data


class EvidencePlanner:
    """Small deterministic planner that turns a query into an evidence contract."""

    _COUNT_PATTERNS = [
        r"\bhow many\b",
        r"\bnumber of\b",
        r"\bcount\b",
        r"\btimes?\s+(?:is|was|are|were)?.*\bmentioned\b",
    ]
    _FILE_LOOKUP_PATTERNS = [
        r"\bwhich files?\b",
        r"\bin which files?\b",
        r"\bdocuments?\s+(?:mention|contain|include)\b",
        r"\bfiles?\s+(?:mention|contain|include)\b",
    ]
    _ENTITY_LIST_PATTERNS = [
        r"\blist\b.*\b(?:people|persons|names|entities|dates|places|locations|emails?)\b",
        r"\bwhich\b.*\b(?:people|persons|names|dates|places|locations|emails?)\b",
        r"\bimportant\s+(?:people|persons|names)\b",
    ]
    _MISSING_DETAIL_PATTERNS = [
        r"\bminimum\s+salary\b",
        r"\bsalary\b",
        r"\bpunishment\b",
        r"\bpenalt(?:y|ies)\b",
        r"\bonline\b",
        r"\bdegree\b",
        r"\bdeadline\b",
        r"\bpricing\b",
        r"\bprice\b",
        r"\bcost\b",
        r"\bfee\b",
    ]
    _ANALYSIS_PATTERNS = [
        r"\banaly[sz]e\b",
        r"\bapply\b",
        r"\bsituation\b",
        r"\bhow should\b",
        r"\bwhat should\b",
        r"\bdetail\b.*\btimeline\b",
        r"\bcomplete\s+timeline\b",
        r"\bwhat can\b.*\bdo\b",
        r"\bcompare\b",
        r"\bpros and cons\b",
    ]
    _TRANSFORM_PATTERNS = [
        r"\bmake (?:it|this|that)\s+(?:short|shorter|brief)\b",
        r"\bmake (?:it|this|that|the answer)\s+(?:a\s+)?(?:single|one|1)\s+sentence\b",
        r"\b(?:in|as)\s+(?:a\s+)?(?:single|one|1)\s+sentence\b",
        r"\bexplain\s+(?:it|this|that)?\s*in\s+(?:1|one)\s+line\b",
        r"\bexplain (?:again|simply|in simple)\b",
        r"\bsummarize (?:it|this|that|the previous)\b",
        r"\brewrite\b",
        r"\brephrase\b",
    ]
    _REQUIREMENT_PATTERNS = [
        r"\brequired\b",
        r"\bmust\b",
        r"\bneed(?:ed)?\b",
        r"\bshall\b",
        r"\bobligation\b",
        r"\brequirement\b",
    ]
    _PROCEDURAL_PATTERNS = [
        r"\bwhat happens\b",
        r"\bwhat is the process\b",
        r"\bprocess for\b",
        r"\bprocedure\b",
        r"\bsteps?\b",
        r"\bhow long\b.*\b(?:respond|file|serve|submit|complete|provide|appeal|object)\b",
        r"\bwhen\b.*\b(?:respond|file|serve|submit|complete|provide|appeal|object)\b",
    ]
    _LEGAL_REFERENCE_PATTERNS = [
        r"\b(?:family|evidence|civil|penal|probate)\s+code\b",
        r"\bsection\s+\d",
        r"\bstatute\b",
        r"\b(?:rule|presumption|privilege)\b",
    ]
    _EMAIL_PATTERNS = [
        r"\bemail\b",
        r"\bsent by\b",
        r"\breceived by\b",
        r"\bsender\b",
        r"\brecipient\b",
    ]

    def plan(self, query: str, *, classification: Optional[Any] = None, kb_id: Optional[str] = None) -> EvidencePlan:
        raw = (query or "").strip()
        q = raw.lower()

        if self._matches(q, self._COUNT_PATTERNS):
            return EvidencePlan(
                query_intent=INTENT_COUNT,
                answer_mode=ANSWER_EXTRACTIVE,
                expected_answer_shape="sentence",
                answer_budget="tiny",
                retrieval_mode=RETRIEVAL_STRUCTURED_COUNT,
                rewrite_type=REWRITE_NONE,
                requested_operation="count",
                target_entity_type=self._infer_entity_type(q),
                evidence_requirements=EvidenceRequirements(
                    needs_exact_source=True,
                    needs_numeric_support=True,
                    needs_named_entity_support=True,
                    allows_synthesis=False,
                ),
                reason="count/search query requires computed structured result",
            )

        if self._matches(q, self._FILE_LOOKUP_PATTERNS):
            return EvidencePlan(
                query_intent=INTENT_STRUCTURED_LIST,
                answer_mode=ANSWER_EXTRACTIVE,
                expected_answer_shape="bullets",
                answer_budget="medium",
                retrieval_mode=RETRIEVAL_STRUCTURED_LIST,
                rewrite_type=REWRITE_ENTITY,
                requested_operation="list_files",
                target_entity_type=self._infer_entity_type(q),
                evidence_requirements=EvidenceRequirements(
                    needs_exact_source=True,
                    needs_named_entity_support=True,
                    allows_synthesis=False,
                ),
                reason="file lookup should return file-backed structured evidence",
            )

        if self._matches(q, self._ENTITY_LIST_PATTERNS):
            return EvidencePlan(
                query_intent=INTENT_STRUCTURED_LIST,
                answer_mode=ANSWER_EXTRACTIVE,
                expected_answer_shape="bullets",
                answer_budget="medium",
                retrieval_mode=RETRIEVAL_SEARCH,
                rewrite_type=REWRITE_ENTITY,
                requested_operation="list_entities",
                target_entity_type=self._infer_entity_type(q),
                evidence_requirements=EvidenceRequirements(
                    needs_exact_source=True,
                    needs_named_entity_support=True,
                    allows_synthesis=False,
                ),
                reason="entity/list query requires evidence-backed search behavior",
            )

        if self._matches(q, self._TRANSFORM_PATTERNS):
            return EvidencePlan(
                query_intent=INTENT_TRANSFORM,
                answer_mode=ANSWER_CONVERSATION_TRANSFORM,
                expected_answer_shape="short_summary",
                answer_budget="short",
                retrieval_mode=RETRIEVAL_ANSWER,
                rewrite_type=REWRITE_NONE,
                evidence_requirements=EvidenceRequirements(allows_synthesis=False),
                reason="conversation transform should preserve previous evidence",
            )

        if self._matches(q, self._ANALYSIS_PATTERNS):
            return EvidencePlan(
                query_intent=INTENT_ANALYSIS,
                answer_mode=ANSWER_GROUNDED_ANALYSIS,
                expected_answer_shape="detailed_analysis",
                answer_budget="long",
                retrieval_mode=RETRIEVAL_EVIDENCE_LOOKUP,
                rewrite_type=self._rewrite_type_for(q),
                evidence_requirements=EvidenceRequirements(
                    needs_exact_source=True,
                    needs_policy_or_rule_support=self._looks_legal(q),
                    allows_synthesis=True,
                ),
                reason="analysis allowed only over accepted evidence",
            )

        if self._matches(q, self._MISSING_DETAIL_PATTERNS):
            return EvidencePlan(
                query_intent=INTENT_MISSING_DETAIL,
                answer_mode=ANSWER_SOURCE_LIMITED,
                expected_answer_shape="sentence",
                answer_budget="tiny",
                retrieval_mode=RETRIEVAL_EVIDENCE_LOOKUP,
                rewrite_type=self._rewrite_type_for(q),
                evidence_requirements=EvidenceRequirements(
                    needs_exact_source=True,
                    needs_numeric_support=bool(re.search(r"\b(?:salary|price|cost|fee|deadline)\b", q)),
                    needs_policy_or_rule_support=self._looks_legal(q),
                    allows_synthesis=False,
                ),
                reason="detail may be absent; answer must stay source-limited unless supported",
            )

        if self._matches(q, self._PROCEDURAL_PATTERNS):
            return EvidencePlan(
                query_intent=INTENT_REQUIREMENT,
                answer_mode=ANSWER_DIRECT,
                expected_answer_shape="bullets",
                answer_budget="medium",
                retrieval_mode=RETRIEVAL_EVIDENCE_LOOKUP,
                rewrite_type=self._rewrite_type_for(q),
                evidence_requirements=EvidenceRequirements(
                    needs_exact_source=True,
                    needs_policy_or_rule_support=self._looks_legal(q),
                    allows_synthesis=False,
                ),
                reason="procedural query needs enough budget to explain source-backed steps",
            )

        if self._matches(q, self._REQUIREMENT_PATTERNS) or self._looks_legal(q):
            return EvidencePlan(
                query_intent=INTENT_REQUIREMENT,
                answer_mode=ANSWER_DIRECT,
                expected_answer_shape="bullets" if re.search(r"\b(?:what are|list)\b", q) else "sentence",
                answer_budget="short",
                retrieval_mode=RETRIEVAL_EVIDENCE_LOOKUP,
                rewrite_type=self._rewrite_type_for(q),
                evidence_requirements=EvidenceRequirements(
                    needs_exact_source=True,
                    needs_policy_or_rule_support=True,
                    allows_synthesis=False,
                ),
                reason="rule/requirement query needs exact source support",
            )

        if re.search(r"\b(?:what is|what are|define|meaning of)\b", q):
            return EvidencePlan(
                query_intent=INTENT_DEFINITION,
                answer_mode=ANSWER_DIRECT,
                expected_answer_shape="sentence",
                answer_budget="short",
                retrieval_mode=RETRIEVAL_ANSWER,
                rewrite_type=self._rewrite_type_for(q),
                evidence_requirements=EvidenceRequirements(
                    needs_exact_source=self._looks_legal(q) or self._looks_archive(q),
                    needs_named_entity_support=self._looks_archive(q),
                    needs_policy_or_rule_support=self._looks_legal(q),
                    allows_synthesis=not (self._looks_legal(q) or self._looks_archive(q)),
                ),
                reason="definition query should be concise and source-grounded",
            )

        return EvidencePlan(
            query_intent=INTENT_UNKNOWN,
            answer_mode=ANSWER_DIRECT,
            expected_answer_shape="sentence",
            answer_budget="short",
            retrieval_mode=RETRIEVAL_ANSWER,
            rewrite_type=self._rewrite_type_for(q),
            evidence_requirements=EvidenceRequirements(
                needs_exact_source=self._looks_legal(q) or self._looks_archive(q),
                needs_named_entity_support=self._looks_archive(q),
                needs_policy_or_rule_support=self._looks_legal(q),
                allows_synthesis=not (self._looks_legal(q) or self._looks_archive(q)),
            ),
            reason="default evidence plan",
        )

    @staticmethod
    def _matches(query_lower: str, patterns: List[str]) -> bool:
        return any(re.search(pattern, query_lower, flags=re.IGNORECASE) for pattern in patterns)

    def _rewrite_type_for(self, query_lower: str) -> str:
        if self._looks_legal(query_lower):
            return REWRITE_LEGAL_REFERENCE
        if self._looks_archive(query_lower):
            return REWRITE_ENTITY
        if re.search(r"\b(?:ownify|workflow|feature|product|pricing|billing)\b", query_lower):
            return REWRITE_PRODUCT_TERM
        return REWRITE_SEMANTIC

    def _looks_legal(self, query_lower: str) -> bool:
        return self._matches(query_lower, self._LEGAL_REFERENCE_PATTERNS)

    def _looks_archive(self, query_lower: str) -> bool:
        return self._matches(query_lower, self._EMAIL_PATTERNS) or bool(
            re.search(r"\b(?:person|people|date|place|location|file|document|mentioned)\b", query_lower)
        )

    @staticmethod
    def _infer_entity_type(query_lower: str) -> Optional[str]:
        if re.search(r"\b(?:email|sender|recipient|sent|received)\b", query_lower):
            return "email"
        if re.search(r"\b(?:person|people|persons|names|who)\b", query_lower):
            return "person"
        if re.search(r"\b(?:date|when|timeline)\b", query_lower):
            return "date"
        if re.search(r"\b(?:place|location|where)\b", query_lower):
            return "place"
        if re.search(r"\b(?:file|document)\b", query_lower):
            return "file"
        return None
