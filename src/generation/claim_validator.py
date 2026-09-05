"""Deterministic high-risk claim and citation validation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ClaimValidationResult:
    enabled: bool
    domain_profile_type: str = "general"
    high_risk_claims: List[Dict[str, Any]] = field(default_factory=list)
    unsupported_claims: List[Dict[str, Any]] = field(default_factory=list)
    citation_validation_failures: List[Dict[str, Any]] = field(default_factory=list)
    confidence_cap: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClaimValidator:
    """Validate numbers/dates/prices/sections/entities against accepted evidence."""

    MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:usd|dollars?)\b", re.IGNORECASE)
    DATE_RE = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b",
        re.IGNORECASE,
    )
    SECTION_RE = re.compile(r"\b(?:section|code|§)\s*\d{3,5}(?:\.\d+)?\b", re.IGNORECASE)
    REQUIREMENT_RE = re.compile(r"\b(?:must|shall|required|requires|may not|prohibited|eligible|ineligible)\b", re.IGNORECASE)
    _STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "for",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    _NUMBER_WORDS = {
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
    }

    def validate(
        self,
        answer: str,
        *,
        evidence_packets: Iterable[Any],
        citations: List[Dict[str, Any]],
        domain_profile: Optional[Dict[str, Any]] = None,
    ) -> ClaimValidationResult:
        profile = dict(domain_profile or {})
        evidence_text = "\n".join(
            str(packet.get("text") if isinstance(packet, dict) else getattr(packet, "text", ""))
            for packet in evidence_packets or []
        )
        evidence_lower = evidence_text.lower()
        claims = self._extract_high_risk_claims(answer or "", profile)
        unsupported: List[Dict[str, Any]] = []
        for claim in claims:
            if not self._claim_supported(claim, evidence_text, evidence_lower):
                unsupported.append({**claim, "reason": "claim_value_not_found_in_accepted_evidence"})

        citation_failures = self._validate_citations(citations, evidence_lower)
        cap = None
        if unsupported or citation_failures:
            cap = 0.65
        return ClaimValidationResult(
            enabled=True,
            domain_profile_type=str(profile.get("type") or "general"),
            high_risk_claims=claims,
            unsupported_claims=unsupported,
            citation_validation_failures=citation_failures,
            confidence_cap=cap,
        )

    def _extract_high_risk_claims(self, answer: str, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        claims: List[Dict[str, Any]] = []
        for kind, pattern in [
            ("price", self.MONEY_RE),
            ("date", self.DATE_RE),
            ("legal_code_section", self.SECTION_RE),
        ]:
            for match in pattern.finditer(answer):
                claims.append({"type": kind, "value": match.group(0), "start": match.start(), "end": match.end()})

        for sentence in _sentences(answer):
            if self.REQUIREMENT_RE.search(sentence):
                claims.append({"type": "requirement", "value": sentence[:300]})

        profile_types = set(profile.get("high_risk_claim_types") or [])
        if {"person", "place", "email_sender", "email_recipient", "file_name"} & profile_types:
            for entity in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", answer):
                claims.append({"type": "named_entity", "value": entity})

        # Deduplicate while preserving order.
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for claim in claims:
            key = (claim.get("type"), claim.get("value"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(claim)
        return deduped[:50]

    @staticmethod
    def _validate_citations(citations: List[Dict[str, Any]], evidence_lower: str) -> List[Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        for citation in citations or []:
            text = str(citation.get("text") or citation.get("snippet") or "").strip()
            source = str(citation.get("source") or "Unknown")
            if not text:
                failures.append({"source": source, "reason": "missing_citation_snippet"})
                continue
            compact = text[:120].lower()
            if compact and compact not in evidence_lower:
                # Snippets are often previews/truncated. Flag only as telemetry for now.
                failures.append({"source": source, "reason": "citation_snippet_not_in_accepted_evidence"})
        return failures[:25]

    @classmethod
    def _claim_supported(cls, claim: Dict[str, Any], evidence_text: str, evidence_lower: str) -> bool:
        value = str(claim.get("value") or "")
        if not value:
            return True
        if value.lower() in evidence_lower:
            return True

        claim_type = str(claim.get("type") or "")
        normalized_value = cls._normalize_text(value)
        normalized_evidence = cls._normalize_text(evidence_text)
        if normalized_value and normalized_value in normalized_evidence:
            return True

        if claim_type == "requirement":
            return cls._requirement_supported(value, normalized_evidence)

        return False

    @classmethod
    def _requirement_supported(cls, sentence: str, normalized_evidence: str) -> bool:
        tokens = cls._content_tokens(sentence)
        if not tokens:
            return True

        critical_tokens = [
            token for token in tokens
            if token.isdigit() or token in cls._NUMBER_WORDS or token in {"hour", "hours", "day", "days", "deadline"}
        ]
        if critical_tokens and any(token not in normalized_evidence for token in critical_tokens):
            return False

        evidence_tokens = set(normalized_evidence.split())
        overlap = sum(1 for token in tokens if token in evidence_tokens)
        overlap_ratio = overlap / max(1, len(tokens))
        if overlap_ratio >= 0.6 and overlap >= min(4, len(tokens)):
            return True

        # Legal/product requirements are often paraphrased. If the answer carries
        # the same numeric constraint and core nouns, treat it as supported.
        core_tokens = [token for token in tokens if token not in cls._NUMBER_WORDS and not token.isdigit()]
        core_overlap = sum(1 for token in core_tokens if token in evidence_tokens)
        return bool(critical_tokens) and core_overlap >= 3

    @classmethod
    def _content_tokens(cls, text: str) -> List[str]:
        normalized = cls._normalize_text(text)
        tokens = []
        for token in normalized.split():
            if token in cls._STOPWORDS:
                continue
            if len(token) <= 2 and not token.isdigit():
                continue
            tokens.append(token)
        return tokens[:40]

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = re.sub(r"[^a-z0-9$§.]+", " ", (text or "").lower())
        return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> List[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text or "") if item.strip()]
