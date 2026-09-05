"""
Structured entity resolution for typo-tolerant structured queries.

Design goals:
- Keep structured handlers strict and precise.
- Add fuzzy/phonetic name correction before handler execution.
- Support confidence tiers (high/medium/low/ambiguous) for UX routing.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple


logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass
class EntityResolutionResult:
    """Outcome of resolving a user-provided entity string."""

    original_entity: str
    resolved_entity: Optional[str]
    confidence: float
    tier: str  # high | medium | ambiguous | low | unavailable
    was_corrected: bool
    suggestions: List[str]
    reason: str


@dataclass
class _EntityCandidateCache:
    """Cached candidate data for a collection."""

    expires_at: float
    entities: List[str]
    normalized_entities: List[str]
    token_lists: List[List[str]]
    token_sets: List[Set[str]]
    counts: List[int]
    norm_to_best_idx: Dict[str, int]
    token_to_indices: Dict[str, List[int]]
    soundex_to_indices: Dict[str, List[int]]
    top_indices: List[int]


class StructuredEntityResolver:
    """
    Resolve noisy entity mentions against indexed entity metadata.

    Resolver behavior:
    - Exact normalized match -> high confidence
    - Fuzzy + token + phonetic scoring for near matches
    - Confidence tiering to support routing decisions
    """

    _cache_lock = threading.Lock()
    _candidate_cache: Dict[str, _EntityCandidateCache] = {}

    def __init__(
            self,
            qdrant_client,
            collection_name: str,
            cache_ttl_seconds: int = 900,
            candidate_limit: int = 12000,
    ):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.cache_ttl_seconds = cache_ttl_seconds
        self.candidate_limit = candidate_limit

    def resolve(self, raw_entity: str) -> EntityResolutionResult:
        """
        Resolve raw entity text to canonical indexed entity.

        Returns a tiered result suitable for routing:
        - high/medium: safe to auto-resolve (medium with disclosure)
        - ambiguous/low: ask for clarification
        - unavailable: resolver could not load candidates
        """
        cleaned = self._clean_entity(raw_entity)
        if not cleaned:
            return EntityResolutionResult(
                original_entity=raw_entity or "",
                resolved_entity=None,
                confidence=0.0,
                tier="low",
                was_corrected=False,
                suggestions=[],
                reason="empty_entity",
            )

        cache = self._get_or_build_cache()
        if cache is None or not cache.entities:
            return EntityResolutionResult(
                original_entity=cleaned,
                resolved_entity=cleaned,
                confidence=0.0,
                tier="unavailable",
                was_corrected=False,
                suggestions=[],
                reason="candidate_cache_unavailable",
            )

        query_norm = self._normalize(cleaned)
        query_tokens = self._tokenize(cleaned)
        if not query_tokens:
            return EntityResolutionResult(
                original_entity=cleaned,
                resolved_entity=cleaned,
                confidence=0.0,
                tier="low",
                was_corrected=False,
                suggestions=[],
                reason="no_query_tokens",
            )

        # Exact normalized match always wins (prevents fuzzy regressions).
        exact_idx = cache.norm_to_best_idx.get(query_norm)
        if exact_idx is not None:
            resolved = cache.entities[exact_idx]
            return EntityResolutionResult(
                original_entity=cleaned,
                resolved_entity=resolved,
                confidence=1.0,
                tier="high",
                was_corrected=self._normalize(cleaned) != self._normalize(resolved),
                suggestions=[],
                reason="exact_normalized_match",
            )

        candidate_indices = self._select_candidate_indices(query_tokens, cache)
        if not candidate_indices:
            return EntityResolutionResult(
                original_entity=cleaned,
                resolved_entity=None,
                confidence=0.0,
                tier="low",
                was_corrected=False,
                suggestions=[],
                reason="no_candidate_overlap",
            )

        scored: List[Tuple[int, float]] = []
        for idx in candidate_indices:
            score = self._score_candidate(
                query_norm=query_norm,
                query_tokens=query_tokens,
                candidate_norm=cache.normalized_entities[idx],
                candidate_tokens=cache.token_lists[idx],
                candidate_count=cache.counts[idx],
            )
            if score >= 0.42:
                scored.append((idx, score))

        if not scored:
            return EntityResolutionResult(
                original_entity=cleaned,
                resolved_entity=None,
                confidence=0.0,
                tier="low",
                was_corrected=False,
                suggestions=[],
                reason="all_candidates_below_threshold",
            )

        scored.sort(key=lambda pair: pair[1], reverse=True)
        top_idx, top_score = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else 0.0
        gap = top_score - second_score

        top_entity = cache.entities[top_idx]
        suggestions = [cache.entities[idx] for idx, _ in scored[:3]]

        if top_score >= 0.9 and gap >= 0.06:
            tier = "high"
        elif top_score >= 0.78 and gap >= 0.03:
            tier = "medium"
        elif top_score >= 0.72:
            tier = "ambiguous"
        else:
            tier = "low"

        if tier == "high":
            reason = "high_confidence_fuzzy_match"
        elif tier == "medium":
            reason = "medium_confidence_fuzzy_match"
        elif tier == "ambiguous":
            reason = "ambiguous_top_candidates"
        else:
            reason = "low_confidence_match"

        return EntityResolutionResult(
            original_entity=cleaned,
            resolved_entity=top_entity if tier in {"high", "medium"} else None,
            confidence=top_score,
            tier=tier,
            was_corrected=(self._normalize(cleaned) != self._normalize(top_entity))
            if tier in {"high", "medium"}
            else False,
            suggestions=suggestions,
            reason=reason,
        )

    def _get_or_build_cache(self) -> Optional[_EntityCandidateCache]:
        now = time.time()
        with self._cache_lock:
            cached = self._candidate_cache.get(self.collection_name)
            if cached and cached.expires_at > now:
                return cached

        built = self._build_cache()
        if built is None:
            return None

        with self._cache_lock:
            self._candidate_cache[self.collection_name] = built
        return built

    def _build_cache(self) -> Optional[_EntityCandidateCache]:
        """Scan first chunks and build in-memory candidate indexes."""
        counts_by_entity: Dict[str, int] = {}
        offset = None
        scanned_points = 0

        try:
            while True:
                scroll_kwargs = {
                    "collection_name": self.collection_name,
                    "limit": 256,
                    "offset": offset,
                    "with_payload": ["document_entity_counts", "entity_names", "is_first_chunk"],
                    "with_vectors": False,
                }

                # Prefer first-chunk metadata; fallback to full scan if filter models unavailable.
                try:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue

                    scroll_kwargs["scroll_filter"] = Filter(
                        must=[
                            FieldCondition(
                                key="is_first_chunk",
                                match=MatchValue(value=True),
                            )
                        ]
                    )
                except Exception:
                    pass

                results, next_offset = self.client.scroll(**scroll_kwargs)
                if not results:
                    break

                scanned_points += len(results)

                for point in results:
                    payload = getattr(point, "payload", {}) or {}
                    doc_counts = payload.get("document_entity_counts") or {}
                    entity_names = payload.get("entity_names") or []

                    if isinstance(doc_counts, dict):
                        for name, count in doc_counts.items():
                            if not isinstance(name, str):
                                continue
                            cleaned_name = self._clean_entity(name)
                            if not cleaned_name:
                                continue
                            counts_by_entity[cleaned_name] = counts_by_entity.get(cleaned_name, 0) + self._safe_int(count)

                    if isinstance(entity_names, list):
                        for name in entity_names:
                            if not isinstance(name, str):
                                continue
                            cleaned_name = self._clean_entity(name)
                            if not cleaned_name:
                                continue
                            counts_by_entity[cleaned_name] = counts_by_entity.get(cleaned_name, 0) + 1

                offset = next_offset
                if offset is None:
                    break

        except Exception as e:
            logger.warning(f"Entity resolver cache build failed for {self.collection_name}: {e}")
            return None

        if not counts_by_entity:
            logger.warning(
                f"Entity resolver found no candidates in collection {self.collection_name}. "
                f"Structured typo correction will be unavailable."
            )
            return _EntityCandidateCache(
                expires_at=time.time() + self.cache_ttl_seconds,
                entities=[],
                normalized_entities=[],
                token_lists=[],
                token_sets=[],
                counts=[],
                norm_to_best_idx={},
                token_to_indices={},
                soundex_to_indices={},
                top_indices=[],
            )

        sorted_entities = sorted(
            counts_by_entity.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if len(sorted_entities) > self.candidate_limit:
            sorted_entities = sorted_entities[:self.candidate_limit]

        entities: List[str] = []
        normalized_entities: List[str] = []
        token_lists: List[List[str]] = []
        token_sets: List[Set[str]] = []
        counts: List[int] = []
        norm_to_best_idx: Dict[str, int] = {}
        token_to_indices: Dict[str, List[int]] = {}
        soundex_to_indices: Dict[str, List[int]] = {}

        for idx, (entity, count) in enumerate(sorted_entities):
            norm = self._normalize(entity)
            tokens = self._tokenize(entity)
            if not tokens:
                continue

            entities.append(entity)
            normalized_entities.append(norm)
            token_lists.append(tokens)
            token_sets.append(set(tokens))
            counts.append(count)

            actual_idx = len(entities) - 1
            prev_idx = norm_to_best_idx.get(norm)
            if prev_idx is None or counts[actual_idx] > counts[prev_idx]:
                norm_to_best_idx[norm] = actual_idx

            for token in set(tokens):
                token_to_indices.setdefault(token, []).append(actual_idx)
                if len(token) >= 4:
                    sx = self._soundex(token)
                    if sx:
                        soundex_to_indices.setdefault(sx, []).append(actual_idx)

        cache = _EntityCandidateCache(
            expires_at=time.time() + self.cache_ttl_seconds,
            entities=entities,
            normalized_entities=normalized_entities,
            token_lists=token_lists,
            token_sets=token_sets,
            counts=counts,
            norm_to_best_idx=norm_to_best_idx,
            token_to_indices=token_to_indices,
            soundex_to_indices=soundex_to_indices,
            top_indices=list(range(min(len(entities), 800))),
        )

        logger.info(
            f"Entity resolver cache ready for {self.collection_name}: "
            f"{len(entities)} candidates from {scanned_points} points"
        )
        return cache

    def _select_candidate_indices(
            self,
            query_tokens: List[str],
            cache: _EntityCandidateCache,
    ) -> List[int]:
        """Select a focused candidate pool using token and phonetic indexes."""
        selected: Set[int] = set()

        for token in set(query_tokens):
            selected.update(cache.token_to_indices.get(token, []))

        if len(selected) < 80:
            for token in set(query_tokens):
                if len(token) < 4:
                    continue
                sx = self._soundex(token)
                if not sx:
                    continue
                selected.update(cache.soundex_to_indices.get(sx, []))

        if not selected:
            selected.update(cache.top_indices)

        if len(selected) > 5000:
            selected = set(
                sorted(
                    selected,
                    key=lambda idx: cache.counts[idx],
                    reverse=True,
                )[:5000]
            )

        return list(selected)

    def _score_candidate(
            self,
            query_norm: str,
            query_tokens: List[str],
            candidate_norm: str,
            candidate_tokens: List[str],
            candidate_count: int,
    ) -> float:
        """Compute a bounded similarity score for ranking candidates."""
        if query_norm == candidate_norm:
            return 1.0

        query_set = set(query_tokens)
        candidate_set = set(candidate_tokens)

        phrase_ratio = SequenceMatcher(None, query_norm, candidate_norm).ratio()
        token_overlap = len(query_set & candidate_set) / max(1, len(query_set))
        token_edit = self._average_best_token_similarity(query_tokens, candidate_tokens)
        phonetic_score = self._phonetic_overlap_score(query_tokens, candidate_tokens)

        score = (
            (0.44 * phrase_ratio) +
            (0.30 * token_edit) +
            (0.20 * token_overlap) +
            (0.08 * phonetic_score)
        )

        if token_overlap == 0 and phonetic_score == 0:
            score *= 0.65

        score += min(math.log1p(max(0, candidate_count)) / 50.0, 0.03)
        return max(0.0, min(score, 1.0))

    def _average_best_token_similarity(
            self,
            query_tokens: List[str],
            candidate_tokens: List[str],
    ) -> float:
        if not query_tokens or not candidate_tokens:
            return 0.0

        total = 0.0
        for query_token in query_tokens:
            best = 0.0
            for candidate_token in candidate_tokens:
                if query_token == candidate_token:
                    best = 1.0
                    break
                sim = self._token_similarity(query_token, candidate_token)
                if sim > best:
                    best = sim
            total += best
        return total / len(query_tokens)

    def _phonetic_overlap_score(
            self,
            query_tokens: List[str],
            candidate_tokens: List[str],
    ) -> float:
        """
        Phonetic overlap score for misspelled person-name components.

        Applied only to single tokens (>=4 chars) to reduce false positives.
        """
        long_query_tokens = [t for t in query_tokens if len(t) >= 4]
        if not long_query_tokens:
            return 0.0

        candidate_soundex = {self._soundex(token) for token in candidate_tokens if len(token) >= 4}
        candidate_soundex.discard("")

        if not candidate_soundex:
            return 0.0

        hits = 0
        for query_token in long_query_tokens:
            sx = self._soundex(query_token)
            if sx and sx in candidate_soundex:
                hits += 1

        return hits / len(long_query_tokens)

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        dist = StructuredEntityResolver._levenshtein_distance(a, b)
        denom = max(len(a), len(b))
        if denom == 0:
            return 0.0
        return max(0.0, 1.0 - (dist / denom))

    @staticmethod
    def _levenshtein_distance(a: str, b: str) -> int:
        """Compute Levenshtein distance using a memory-efficient DP row."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        if len(a) < len(b):
            a, b = b, a

        previous = list(range(len(b) + 1))
        for i, char_a in enumerate(a, start=1):
            current = [i]
            for j, char_b in enumerate(b, start=1):
                insertions = previous[j] + 1
                deletions = current[j - 1] + 1
                substitutions = previous[j - 1] + (char_a != char_b)
                current.append(min(insertions, deletions, substitutions))
            previous = current
        return previous[-1]

    @staticmethod
    def _soundex(token: str) -> str:
        """Simple Soundex for phonetic approximation."""
        if not token:
            return ""

        token = re.sub(r"[^a-zA-Z]", "", token)
        if not token:
            return ""

        token = token.upper()
        mappings = {
            "B": "1", "F": "1", "P": "1", "V": "1",
            "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
            "D": "3", "T": "3",
            "L": "4",
            "M": "5", "N": "5",
            "R": "6",
        }

        first_letter = token[0]
        digits = []
        last_digit = mappings.get(first_letter, "")

        for char in token[1:]:
            digit = mappings.get(char, "")
            if digit != last_digit:
                if digit:
                    digits.append(digit)
                last_digit = digit

        return (first_letter + "".join(digits) + "000")[:4]

    @staticmethod
    def _clean_entity(value: str) -> str:
        if not value:
            return ""
        cleaned = value.strip().strip("\"'`.,;:!?()[]{}")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    @staticmethod
    def _normalize(value: str) -> str:
        if not value:
            return ""
        normalized = value.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = normalized.strip(".,;:!?()[]{}\"'`")
        return normalized

    @staticmethod
    def _tokenize(value: str) -> List[str]:
        return _TOKEN_PATTERN.findall((value or "").lower())

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
