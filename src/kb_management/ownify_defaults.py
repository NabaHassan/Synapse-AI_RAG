"""Tenant-safe defaults for automated Ownify KBs.

These defaults apply to KBs created through the `/ownify/...` provisioning
surface. They are intentionally tenant-agnostic: no tenant ids, no customer
names, and no domain assumptions beyond "this is an automated uploaded-docs
assistant that must stay grounded in its tenant's sources."
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Tuple, Type


OWNIFY_SOURCE_PREFIX = "ownify_"

OWNIFY_HIGH_RISK_CLAIM_TYPES = [
    "price",
    "pricing",
    "cost",
    "fee",
    "deadline",
    "date",
    "eligibility",
    "requirement",
    "policy",
    "refund",
    "cancellation",
    "contact",
    "email",
    "phone",
    "address",
]

OWNIFY_MISSING_DETAIL_PATTERNS = [
    "price",
    "pricing",
    "cost",
    "fee",
    "monthly",
    "deadline",
    "date",
    "phone",
    "email",
    "address",
    "refund",
    "cancellation",
    "eligibility",
    "requirement",
    "discount",
    "contract",
]

OWNIFY_REMOVE_IF_UNSUPPORTED_PATTERNS = [
    "$",
    "per month",
    "monthly price",
    "annual price",
    "deadline",
    "guarantee",
    "refund",
    "cancellation",
    "phone",
    "email",
    "address",
]


def is_ownify_source(source: str) -> bool:
    return str(source or "").strip().lower().startswith(OWNIFY_SOURCE_PREFIX)


def policy_to_dict(policy: Any) -> Dict[str, Any]:
    if hasattr(policy, "model_dump"):
        return policy.model_dump()
    if hasattr(policy, "dict"):
        return policy.dict()
    if is_dataclass(policy):
        return asdict(policy)
    return dict(policy or {})


def apply_ownify_safe_defaults(
    *,
    generation: Any,
    routing: Any,
    grounding: Any,
    answer_cleanup: Any,
    domain_profile: Any,
) -> Tuple[Any, Any, Any, Any, Any]:
    """Return policy objects with Ownify-safe defaults applied.

    The caller remains responsible for applying explicit tenant overrides after
    this step. A few values are safety floors/caps and are applied again after
    overrides by `enforce_ownify_safety_floor`.
    """

    generation = _with_policy_updates(
        generation,
        {
            "temperature": min(float(getattr(generation, "temperature", 0.4) or 0.4), 0.4),
            "max_tokens": min(int(getattr(generation, "max_tokens", 768) or 768), 1024),
            "enable_min_tokens_strategy": False,
            "min_tokens_long_response": 0,
            "long_response_max_tokens": 0,
        },
    )
    routing = _with_policy_updates(
        routing,
        {
            "enable_enhanced_handlers": True,
            "enable_clarification": True,
            "structured_query_fast_mode": "auto",
            "structured_entity_resolution": True,
            "structured_natural_response_style": True,
        },
    )
    grounding = _with_policy_updates(
        grounding,
        {
            "allow_general_knowledge_fallback": False,
            "enable_collection_query_anchoring": True,
            "min_verification_threshold": max(
                float(getattr(grounding, "min_verification_threshold", 0.1) or 0.1),
                0.1,
            ),
            "confidence_caps_enabled": True,
            "low_relevance_confidence_cap": min(
                float(getattr(grounding, "low_relevance_confidence_cap", 0.3) or 0.3),
                0.3,
            ),
            "verification_fallback_confidence_cap": min(
                float(getattr(grounding, "verification_fallback_confidence_cap", 0.55) or 0.55),
                0.55,
            ),
            "no_nli_fallback_confidence_cap": min(
                float(getattr(grounding, "no_nli_fallback_confidence_cap", 0.45) or 0.45),
                0.45,
            ),
            "answer_cleanup_confidence_cap": min(
                float(getattr(grounding, "answer_cleanup_confidence_cap", 0.75) or 0.75),
                0.75,
            ),
        },
    )
    answer_cleanup = _with_policy_updates(
        answer_cleanup,
        {
            "enabled": True,
            "source_limited_answer": "The uploaded sources do not specify this detail.",
            "source_limited_query_patterns": _merged_list(
                getattr(answer_cleanup, "source_limited_query_patterns", []),
                OWNIFY_MISSING_DETAIL_PATTERNS,
            ),
            "remove_if_unsupported_patterns": _merged_list(
                getattr(answer_cleanup, "remove_if_unsupported_patterns", []),
                OWNIFY_REMOVE_IF_UNSUPPORTED_PATTERNS,
            ),
            "preserve_bold_label_on_removed_detail": True,
        },
    )
    domain_profile = _with_policy_updates(
        domain_profile,
        {
            "type": "automated_tenant",
            "risk_level": "elevated",
            "high_risk_claim_types": _merged_list(
                getattr(domain_profile, "high_risk_claim_types", []),
                OWNIFY_HIGH_RISK_CLAIM_TYPES,
            ),
            "required_metadata_fields": _merged_list(
                getattr(domain_profile, "required_metadata_fields", []),
                ["source_family"],
            ),
            "exact_support_claim_types": _merged_list(
                getattr(domain_profile, "exact_support_claim_types", []),
                OWNIFY_HIGH_RISK_CLAIM_TYPES,
            ),
            "citation_required": True,
            "source_limited_missing_details": True,
        },
    )
    return generation, routing, grounding, answer_cleanup, domain_profile


def enforce_ownify_safety_floor(
    *,
    generation: Any,
    grounding: Any,
    answer_cleanup: Any,
    domain_profile: Any,
) -> Tuple[Any, Any, Any, Any]:
    """Apply non-negotiable automated-tenant safety floors after overrides."""

    generation = _with_policy_updates(
        generation,
        {
            "temperature": min(float(getattr(generation, "temperature", 0.4) or 0.4), 0.4),
        },
    )
    grounding = _with_policy_updates(
        grounding,
        {
            "allow_general_knowledge_fallback": False,
            "confidence_caps_enabled": True,
        },
    )
    answer_cleanup = _with_policy_updates(
        answer_cleanup,
        {
            "enabled": True,
            "source_limited_answer": (
                str(getattr(answer_cleanup, "source_limited_answer", "") or "").strip()
                or "The uploaded sources do not specify this detail."
            ),
            "source_limited_query_patterns": _merged_list(
                getattr(answer_cleanup, "source_limited_query_patterns", []),
                OWNIFY_MISSING_DETAIL_PATTERNS,
            ),
        },
    )
    domain_profile = _with_policy_updates(
        domain_profile,
        {
            "risk_level": (
                "strict"
                if str(getattr(domain_profile, "risk_level", "")).lower() == "strict"
                else "elevated"
            ),
            "citation_required": True,
            "source_limited_missing_details": True,
        },
    )
    return generation, grounding, answer_cleanup, domain_profile


def _with_policy_updates(policy: Any, updates: Dict[str, Any]) -> Any:
    policy_type: Type[Any] = type(policy)
    data = policy_to_dict(policy)
    for key, value in updates.items():
        if key in data:
            data[key] = value
    return policy_type(**data)


def _merged_list(existing: Any, additions: Any) -> list:
    values = []
    seen = set()
    for item in list(existing or []) + list(additions or []):
        value = str(item).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        values.append(value)
    return values
