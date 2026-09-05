"""Intent-driven generation controls for evidence-first RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class GenerationDecision:
    answer_mode: str
    max_tokens: int
    temperature: float
    should_generate: bool = True
    deterministic_reason: Optional[str] = None
    budget: str = "short"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnswerController:
    """Map an evidence plan and admission status to generation parameters."""

    DEFAULT_TOKEN_BUDGETS = {
        "tiny": 80,
        "short": 180,
        "medium": 450,
        "long": 900,
    }

    def __init__(self, token_budgets: Optional[Dict[str, int]] = None):
        self.token_budgets = dict(self.DEFAULT_TOKEN_BUDGETS)
        if token_budgets:
            for key, value in token_budgets.items():
                try:
                    self.token_budgets[key] = max(1, int(value))
                except (TypeError, ValueError):
                    continue

    def decide(
        self,
        evidence_plan: Any,
        *,
        default_max_tokens: int,
        default_temperature: float,
        admission_result: Optional[Any] = None,
    ) -> GenerationDecision:
        plan_dict = evidence_plan.to_dict() if hasattr(evidence_plan, "to_dict") else dict(evidence_plan or {})
        budget = str(plan_dict.get("answer_budget") or "short")
        answer_mode = str(plan_dict.get("answer_mode") or "direct")
        query_intent = str(plan_dict.get("query_intent") or "unknown")
        retrieval_mode = str(plan_dict.get("retrieval_mode") or "answer")

        # Use the intent budget as the primary control. The old behavior took
        # min(default_max_tokens, budget), which meant KBs with a conservative
        # default such as 180 tokens could never produce a real analysis answer
        # even when the planner selected the long budget.
        max_tokens = self.token_budgets.get(budget, self.token_budgets["short"])

        should_generate = True
        deterministic_reason = None
        if query_intent == "missing_detail":
            # Missing detail answers can be deterministic once admission determines the field is unsupported.
            status = getattr(admission_result, "admission_status", None)
            if status in {"missing_requested_detail", "insufficient_evidence"}:
                should_generate = False
                deterministic_reason = status

        temperature = min(float(default_temperature), 0.2 if query_intent in {"count", "requirement", "missing_detail"} else float(default_temperature))

        return GenerationDecision(
            answer_mode=answer_mode,
            max_tokens=max_tokens,
            temperature=temperature,
            should_generate=should_generate,
            deterministic_reason=deterministic_reason,
            budget=budget,
        )
