"""Retrieval strategy router and adaptive budget policy.

This module is intentionally KB-agnostic. It uses query/evidence-plan signals,
domain profile risk, and profile/runtime policy to choose retrieval budgets.
The decision can run in shadow mode; only the caller decides whether to apply it.
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RetrievalBudget:
    dense_top_k: int = 50
    sparse_top_k: int = 50
    fusion_top_k: int = 20
    rerank_top_k: int = 7
    per_issue_top_k: int = 12

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]], fallback: "RetrievalBudget") -> "RetrievalBudget":
        raw = raw or {}
        return cls(
            dense_top_k=_positive_int(raw.get("dense_top_k"), fallback.dense_top_k),
            sparse_top_k=_positive_int(raw.get("sparse_top_k"), fallback.sparse_top_k),
            fusion_top_k=_positive_int(raw.get("fusion_top_k"), fallback.fusion_top_k),
            rerank_top_k=_positive_int(raw.get("rerank_top_k"), fallback.rerank_top_k),
            per_issue_top_k=_positive_int(raw.get("per_issue_top_k"), fallback.per_issue_top_k),
        )

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class RetrievalStrategyDecision:
    mode: str
    query_class: str
    applied: bool
    reason: str
    budget: RetrievalBudget
    shadow_budget: RetrievalBudget
    default_budget: RetrievalBudget
    exact_structured_first: bool = False
    issue_decomposition: bool = False
    routing_label: str = "rag_pipeline"
    rollout: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["budget"] = self.budget.to_dict()
        data["shadow_budget"] = self.shadow_budget.to_dict()
        data["default_budget"] = self.default_budget.to_dict()
        return data


class RetrievalStrategyRouter:
    """Choose retrieval budgets from evidence-plan and domain-profile signals."""

    DEFAULT_BUDGETS = {
        "direct_definition_fast": {
            "dense_top_k": 20,
            "sparse_top_k": 20,
            "fusion_top_k": 12,
            "rerank_top_k": 5,
            "per_issue_top_k": 8,
        },
        "legal_code_lookup": {
            "dense_top_k": 30,
            "sparse_top_k": 45,
            "fusion_top_k": 18,
            "rerank_top_k": 7,
            "per_issue_top_k": 10,
        },
        "evidence_entity_lookup": {
            "dense_top_k": 30,
            "sparse_top_k": 50,
            "fusion_top_k": 20,
            "rerank_top_k": 7,
            "per_issue_top_k": 10,
        },
        "email_or_file_lookup": {
            "dense_top_k": 25,
            "sparse_top_k": 50,
            "fusion_top_k": 20,
            "rerank_top_k": 7,
            "per_issue_top_k": 10,
        },
        "relationship_summary": {
            "dense_top_k": 40,
            "sparse_top_k": 40,
            "fusion_top_k": 20,
            "rerank_top_k": 7,
            "per_issue_top_k": 12,
        },
        "long_fact_pattern_analysis": {
            "dense_top_k": 50,
            "sparse_top_k": 50,
            "fusion_top_k": 24,
            "rerank_top_k": 8,
            "per_issue_top_k": 12,
        },
        "default": {
            "dense_top_k": 40,
            "sparse_top_k": 40,
            "fusion_top_k": 20,
            "rerank_top_k": 7,
            "per_issue_top_k": 12,
        },
    }

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        self.policy = dict(policy or {})
        self.enabled = bool(self.policy.get("enabled", True))
        self.mode_overrides = dict(self.policy.get("budgets") or {})
        self.rollout_mode = str(self.policy.get("rollout_mode") or "inherit").strip().lower()
        self.canary_percentage = _percentage(self.policy.get("canary_percentage"), 0.0)
        self.canary_salt = str(self.policy.get("canary_salt") or "adaptive-retrieval-v1")

    def decide(
        self,
        *,
        query: str,
        evidence_plan: Optional[Any],
        classification: Optional[Any],
        domain_profile: Optional[Dict[str, Any]],
        default_budget: RetrievalBudget,
        rollout: Optional[Any] = None,
        feature_enabled: bool = True,
    ) -> RetrievalStrategyDecision:
        query_class, reason = self._classify(query, evidence_plan, classification, domain_profile)
        default_for_class = RetrievalBudget.from_dict(
            self.DEFAULT_BUDGETS.get(query_class) or self.DEFAULT_BUDGETS["default"],
            default_budget,
        )
        shadow_budget = RetrievalBudget.from_dict(
            self.mode_overrides.get(query_class),
            default_for_class,
        )
        affects_response, rollout_reason = self._affects_response(query=query, rollout=rollout)
        applied = bool(self.enabled and feature_enabled and affects_response)
        active_budget = shadow_budget if applied else default_budget
        retrieval_mode = str(_plan_value(evidence_plan, "retrieval_mode") or "")
        routing_label = _route_label(query_class, retrieval_mode)
        rollout_data = rollout.to_dict() if hasattr(rollout, "to_dict") else dict(rollout or {})
        rollout_data["adaptive_retrieval"] = {
            "mode": self.rollout_mode,
            "affects_response": affects_response,
            "reason": rollout_reason,
            "canary_percentage": self.canary_percentage,
        }

        return RetrievalStrategyDecision(
            mode=query_class,
            query_class=query_class,
            applied=applied,
            reason=reason if self.enabled and feature_enabled else "router_disabled",
            budget=active_budget,
            shadow_budget=shadow_budget,
            default_budget=default_budget,
            exact_structured_first=retrieval_mode in {"structured_count", "structured_list"},
            issue_decomposition=query_class == "long_fact_pattern_analysis",
            routing_label=routing_label,
            rollout=rollout_data,
        )

    def _affects_response(self, *, query: str, rollout: Optional[Any]) -> tuple[bool, str]:
        if self.rollout_mode in {"", "inherit"}:
            return bool(getattr(rollout, "affects_response", False)), "inherited_global_rollout"
        if self.rollout_mode in {"off", "shadow", "compare"}:
            return False, f"adaptive_{self.rollout_mode}"
        if self.rollout_mode == "on":
            return True, "adaptive_rollout_on"
        if self.rollout_mode == "canary":
            bucket = _stable_bucket("|".join([self.canary_salt, query or ""]))
            selected = bucket < self.canary_percentage
            return selected, "adaptive_canary_selected" if selected else "adaptive_canary_not_selected"
        return bool(getattr(rollout, "affects_response", False)), "unknown_mode_inherited_global_rollout"

    def _classify(
        self,
        query: str,
        evidence_plan: Optional[Any],
        classification: Optional[Any],
        domain_profile: Optional[Dict[str, Any]],
    ) -> tuple[str, str]:
        q = (query or "").lower()
        words = len(q.split())
        retrieval_mode = str(_plan_value(evidence_plan, "retrieval_mode") or "")
        intent = str(_plan_value(evidence_plan, "query_intent") or "")
        rewrite_type = str(_plan_value(evidence_plan, "rewrite_type") or "")
        domain_type = str((domain_profile or {}).get("type") or "general")
        risk_level = str((domain_profile or {}).get("risk_level") or "standard")

        if retrieval_mode in {"structured_count", "structured_list"}:
            return "email_or_file_lookup", f"structured retrieval mode: {retrieval_mode}"
        if intent == "analysis" or words >= 55 or _has_long_fact_pattern(q):
            return "long_fact_pattern_analysis", "analysis/long fact-pattern query"
        if _looks_like_legal_code(q) or rewrite_type == "legal_reference_normalization":
            return "legal_code_lookup", "legal code/statute signal"
        if _looks_like_file_or_email(q):
            return "email_or_file_lookup", "file/email/date lookup signal"
        if _looks_like_entity_lookup(q) or rewrite_type == "entity_normalization":
            return "evidence_entity_lookup", "entity/evidence lookup signal"
        if re.search(r"\b(?:relation|relationship|associated|connection|linked|between)\b", q):
            return "relationship_summary", "relationship/summary signal"
        if intent == "definition" and risk_level not in {"high", "sensitive"}:
            return "direct_definition_fast", "simple definition query"
        if domain_type in {"legal", "evidence_archive", "client_legal", "client_archive"}:
            return "relationship_summary", f"domain profile keeps recall broad: {domain_type}"
        return "default", "default balanced retrieval"


def _positive_int(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return max(1, value)


def _percentage(raw: Any, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = default
    return max(0.0, min(100.0, value))


def _stable_bucket(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF * 100.0


def _plan_value(evidence_plan: Optional[Any], key: str) -> Any:
    if evidence_plan is None:
        return None
    if isinstance(evidence_plan, dict):
        return evidence_plan.get(key)
    return getattr(evidence_plan, key, None)


def _looks_like_legal_code(query_lower: str) -> bool:
    return bool(
        re.search(
            r"\b(?:family|evidence|civil|penal|probate)\s+code\b|"
            r"\b(?:section|statute|rule|presumption|privilege)\b|"
            r"\b§\s*\d+|\b\d{3,5}\b",
            query_lower,
        )
    )


def _looks_like_file_or_email(query_lower: str) -> bool:
    return bool(
        re.search(
            r"\b(?:which|what|show|list|in)\b.*\b(?:file|files|document|documents|email|emails)\b|"
            r"\b(?:sent|received|sender|recipient|monaco|jet)\b",
            query_lower,
        )
    )


def _looks_like_entity_lookup(query_lower: str) -> bool:
    return bool(
        re.search(
            r"\b(?:mentioned|name|person|people|who is|who were|associated|important persons?)\b",
            query_lower,
        )
    )


def _has_long_fact_pattern(query_lower: str) -> bool:
    signals = [
        "date of marriage", "date of separation", "spousal support", "temporary order",
        "depreciation", "separate property", "community property", "analyze the issues",
    ]
    return sum(1 for signal in signals if signal in query_lower) >= 2


def _route_label(query_class: str, retrieval_mode: str) -> str:
    if retrieval_mode in {"structured_count", "structured_list"}:
        return retrieval_mode
    return query_class
