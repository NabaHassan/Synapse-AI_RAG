"""Feature flags and rollout helpers for the evidence-first RAG pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


ROLLOUT_OFF = "off"
ROLLOUT_SHADOW = "shadow"
ROLLOUT_COMPARE = "compare"
ROLLOUT_CANARY = "canary"
ROLLOUT_ON = "on"

VALID_ROLLOUT_MODES = {
    ROLLOUT_OFF,
    ROLLOUT_SHADOW,
    ROLLOUT_COMPARE,
    ROLLOUT_CANARY,
    ROLLOUT_ON,
}


@dataclass
class RolloutDecision:
    mode: str
    affects_response: bool = False
    would_affect_response: bool = False
    reason: str = ""
    bucket: Optional[float] = None
    canary_percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RAGFeatureFlags:
    """Runtime gates for production-safe RAG changes."""

    evidence_planner: bool = True
    structured_search_router: bool = True
    metadata_overlay: bool = False
    evidence_admission: bool = True
    evidence_retrieval_boosts: bool = True
    evidence_context_builder: bool = True
    evidence_conversation_state: bool = True
    dynamic_generation_controller: bool = True
    adaptive_retrieval_budgets: bool = False
    claim_validation: bool = False
    citation_validation: bool = False
    contextual_indexing_v1: bool = False
    rollout_mode: str = ROLLOUT_SHADOW
    canary_percentage: float = 0.0
    canary_salt: str = "evidence-v1"
    canary_kb_ids: list[str] = field(default_factory=list)
    canary_user_ids: list[str] = field(default_factory=list)
    canary_session_ids: list[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "RAGFeatureFlags":
        if not raw:
            return cls()

        known = {
            "evidence_planner",
            "structured_search_router",
            "metadata_overlay",
            "evidence_admission",
            "evidence_retrieval_boosts",
            "evidence_context_builder",
            "evidence_conversation_state",
            "dynamic_generation_controller",
            "adaptive_retrieval_budgets",
            "claim_validation",
            "citation_validation",
            "contextual_indexing_v1",
            "rollout_mode",
            "canary_percentage",
            "canary_salt",
            "canary_kb_ids",
            "canary_user_ids",
            "canary_session_ids",
        }
        kwargs = {key: raw[key] for key in known if key in raw}
        kwargs["extra"] = {key: value for key, value in raw.items() if key not in known}
        flags = cls(**kwargs)
        flags.rollout_mode = normalize_rollout_mode(flags.rollout_mode)
        flags.canary_percentage = _clamp_percentage(flags.canary_percentage)
        flags.canary_kb_ids = _normalize_list(flags.canary_kb_ids)
        flags.canary_user_ids = _normalize_list(flags.canary_user_ids)
        flags.canary_session_ids = _normalize_list(flags.canary_session_ids)
        return flags

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "evidence_planner": bool(self.evidence_planner),
            "structured_search_router": bool(self.structured_search_router),
            "metadata_overlay": bool(self.metadata_overlay),
            "evidence_admission": bool(self.evidence_admission),
            "evidence_retrieval_boosts": bool(self.evidence_retrieval_boosts),
            "evidence_context_builder": bool(self.evidence_context_builder),
            "evidence_conversation_state": bool(self.evidence_conversation_state),
            "dynamic_generation_controller": bool(self.dynamic_generation_controller),
            "adaptive_retrieval_budgets": bool(self.adaptive_retrieval_budgets),
            "claim_validation": bool(self.claim_validation),
            "citation_validation": bool(self.citation_validation),
            "contextual_indexing_v1": bool(self.contextual_indexing_v1),
            "rollout_mode": normalize_rollout_mode(self.rollout_mode),
            "canary_percentage": _clamp_percentage(self.canary_percentage),
            "canary_salt": str(self.canary_salt or ""),
            "canary_kb_ids": _normalize_list(self.canary_kb_ids),
            "canary_user_ids": _normalize_list(self.canary_user_ids),
            "canary_session_ids": _normalize_list(self.canary_session_ids),
        }
        if self.extra:
            data["extra"] = dict(self.extra)
        return data

    def is_shadow_mode(self) -> bool:
        return normalize_rollout_mode(self.rollout_mode) in {ROLLOUT_SHADOW, ROLLOUT_COMPARE}

    def affects_response(self) -> bool:
        return normalize_rollout_mode(self.rollout_mode) in {ROLLOUT_CANARY, ROLLOUT_ON}

    def decide_rollout(
        self,
        *,
        kb_id: str = "",
        session_id: str = "",
        user_id: str = "",
        query: str = "",
    ) -> RolloutDecision:
        mode = normalize_rollout_mode(self.rollout_mode)
        canary_percentage = _clamp_percentage(self.canary_percentage)
        if mode == ROLLOUT_OFF:
            return RolloutDecision(mode=mode, reason="rollout_off", canary_percentage=canary_percentage)
        if mode == ROLLOUT_SHADOW:
            return RolloutDecision(mode=mode, reason="shadow_telemetry_only", canary_percentage=canary_percentage)
        if mode == ROLLOUT_COMPARE:
            return RolloutDecision(
                mode=mode,
                affects_response=False,
                would_affect_response=True,
                reason="compare_mode_no_user_visible_change",
                canary_percentage=canary_percentage,
            )
        if mode == ROLLOUT_ON:
            return RolloutDecision(
                mode=mode,
                affects_response=True,
                would_affect_response=True,
                reason="rollout_on",
                canary_percentage=canary_percentage,
            )

        if _matches_allowlist(kb_id, self.canary_kb_ids):
            return RolloutDecision(
                mode=mode,
                affects_response=True,
                would_affect_response=True,
                reason="canary_kb_allowlist",
                canary_percentage=canary_percentage,
            )
        if _matches_allowlist(user_id, self.canary_user_ids):
            return RolloutDecision(
                mode=mode,
                affects_response=True,
                would_affect_response=True,
                reason="canary_user_allowlist",
                canary_percentage=canary_percentage,
            )
        if _matches_allowlist(session_id, self.canary_session_ids):
            return RolloutDecision(
                mode=mode,
                affects_response=True,
                would_affect_response=True,
                reason="canary_session_allowlist",
                canary_percentage=canary_percentage,
            )

        bucket_key = "|".join([
            str(self.canary_salt or ""),
            kb_id or "",
            user_id or "",
            session_id or "",
            query or "",
        ])
        bucket = _stable_bucket(bucket_key)
        selected = bucket < canary_percentage
        return RolloutDecision(
            mode=mode,
            affects_response=selected,
            would_affect_response=True,
            reason="canary_bucket_selected" if selected else "canary_bucket_not_selected",
            bucket=round(bucket, 6),
            canary_percentage=canary_percentage,
        )


def normalize_rollout_mode(raw: Optional[str]) -> str:
    mode = (raw or ROLLOUT_SHADOW).strip().lower()
    return mode if mode in VALID_ROLLOUT_MODES else ROLLOUT_SHADOW


def _clamp_percentage(raw: Any) -> float:
    try:
        value = float(raw or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(100.0, value))


def _normalize_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _matches_allowlist(value: str, allowlist: Any) -> bool:
    normalized = {item.lower() for item in _normalize_list(allowlist)}
    return bool(value and value.lower() in normalized)


def _stable_bucket(key: str) -> float:
    digest = hashlib.sha256((key or "").encode("utf-8", errors="ignore")).hexdigest()
    # First 8 hex chars fit comfortably in an int and are stable across processes.
    return (int(digest[:8], 16) / 0xFFFFFFFF) * 100.0
