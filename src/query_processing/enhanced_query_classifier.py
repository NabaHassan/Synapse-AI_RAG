"""
Enhanced Query Classifier for new query types.

Adds pattern-based detection for:
- ENTITY_COUNT: "how many times is X mentioned?"
- FILE_LOCATION: "which files contain X?"
- EXACT_TEXT: "show me where X is written" / "show me the email from X to Y"

Also extracts entity names and email parameters from the query
for use by the specialized handlers.
"""

import re
import logging
from typing import Dict, Tuple, Optional, List
from difflib import SequenceMatcher

from src.query_processing.query_types import QueryType, QueryClassificationResult

logger = logging.getLogger(__name__)


# ============================================================================
# Pattern definitions for new query types
# ============================================================================

ENTITY_COUNT_PATTERNS = [
    r'how\s+many\s+time?s?\s*(?:is|are|was|were|does|do|did|has|have)?\s+(.+?)\s*(?:(?:is|are|was|were|has|have|been)\s+)?(?:mentioned|referenced|cited|named|appear(?:ed)?|occur(?:red)?)\b',
    r'how\s+(?:many|often)\s+(?:times?)?\s*(?:is|are|was|were|does|do|did|has|have)?\s+(.+?)\s*(?:(?:is|are|was|were|has|have|been)\s+)?(?:mentioned|referenced|cited|named|appear(?:ed)?|occur(?:red)?)\b',
    r'(?:count|number|total)\s+(?:of\s+)?(?:mentions?|references?|occurrences?|times?)\s+(?:of|for|to)?\s*(.+?)(?:\s+in\b|\s+from\b|\s+within\b|\s*\?|$)',
    r'how\s+many\s+(?:mentions?|references?|occurrences?)\s+(?:of|for|to)?\s*(.+?)(?:\s+in\b|\s*\?|$)',
    r'(?:mentions?|occurrences?|count)\s+(?:for|of)\s+(.+?)(?:\s+in\b|\s*\?|$)',
]

FILE_LOCATION_PATTERNS = [
    r'(?:which|what)\s+(?:files?|documents?|docs?)\s+(?:contain|have|include|mention|reference)\s+(.+?)(?:\s*\?|$)',
    r'(?:in\s+)?(?:which|what)\s+(?:exact\s+|specific\s+)?(?:files?|documents?|docs?)\s+(?:is|are|was|were)\s+(.+?)\s+(?:mentioned|found|present|written|referenced)',
    r'(?:in\s+)?(?:which|what)\s+(?:exact\s+|specific\s+)?(?:files?|documents?|docs?)\s+(.+?)\s+(?:is|are|was|were)\s+(?:mentioned|found|present|written|referenced)',
    r'(?:in\s+which\s+(?:files?|documents?|docs?))\s+(?:is|are|does|can\s+i\s+find|can\s+we\s+find)\s+(.+?)\s+(?:mentioned|found|appear|referenced|present)',
    r'(?:identify|list|show|find|give|provide)\s+(?:me\s+)?(?:the\s+)?(?:document(?:\s+ids?|s)?|doc(?:ument)?\s+ids?)\s+(?:containing|with)\s+(?:references?\s+to|mentions?\s+of|about)\s+(.+?)(?:\s*\?|$)',
    r'(?:list|show|find|give|tell)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:files?|documents?|docs?)\s+(?:containing|with|mentioning|about|that\s+mentions?|that\s+references?)\s+(.+?)(?:\s*\?|$)',
    r'(?:show|list|find|give|tell)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:files?|documents?|docs?)\s+(?:where|in\s+which)\s+(.+?)\s+(?:is|are|was|were|has\s+been|have\s+been)\s+(?:mentioned|found|present|written|referenced|cited)\b',
    r'(?:show|list|find|give|tell)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?(?:files?|documents?|docs?)\s+(?:where|in\s+which)\s+(.+?)\s+(?:mentioned|found|present|written|referenced|cited)\b',
    r'(?:files?|documents?|docs?)\s+(?:where|in\s+which)\s+(.+?)\s+(?:is|are|appears?|exists?)',
    r'(?:which|what|in\s+which|in\s+what)\s+(?:files?|documents?|docs?)\s+(?:does|do|did)\s+(.+?)\s+(?:appear|occur|show\s+up|exist)\s+in(?:\s*\?|$)',
    r'(?:in\s+)?(?:which|what)\s+(?:files?|documents?|docs?)\s+(?:can\s+i\s+find|could\s+i\s+find|can\s+we\s+find)\s+(.+?)(?:\s*\?|$)',
    r'where\s+(?:is|are|was|were)\s+(.+?)\s+(?:found|located|present)\b',
    r'where\s+(?:can\s+i\s+find|can\s+we\s+find)\s+(.+?)(?:\s*\?|$)',
]

EXACT_TEXT_PATTERNS = [
    # Email-specific
    r'(?:show|give|get)\s+(?:me\s+)?(?:the\s+)?(?:exact\s+)?email\s+(?:from|by)\s+(.+?)\s+(?:to|for)\s+(.+?)(?:\s*\?|$)',
    r'email\s+(?:from|by|between)\s+(.+?)\s+(?:to|and)\s+(.+?)(?:\s*\?|$)',
    r'what\s+(?:did|does|has)\s+(.+?)\s+(?:send|write|email)\s+(?:to)\s+(.+?)(?:\s*\?|$)',
    # General text/quote retrieval
    r'(?:show|give|find|get)\s+(?:me\s+)?(?:the\s+)?(?:exact\s+)?(?:text|words?|wording|content|quote|passage|snippet|paragraph|section|verbatim)\s+(?:where|about|mentioning|regarding|that\s+mentions?)\s+(.+?)(?=\s+(?:(?:is|are|was|were|being|been)\s+)?(?:mentioned|written|stated|said|discussed|referenced|found|present)|\s*\?|$)',
    r'(?:show|give|find|get)\s+(?:me\s+)?(?:the\s+)?(?:exact\s+)?(?:text|words?|wording|content|quote|passage|snippet|paragraph|section|verbatim)\s+(.+?)(?=\s+(?:(?:is|are|was|were|being|been)\s+)?(?:mentioned|written|stated|said|discussed|referenced|found|present)|\s*\?|$)',
    r'where\s+(?:is|are|was|were)\s+(.+?)\s+(?:mentioned|written|stated|said|discussed|referenced|found|present)\b',
    r'(?:tell|show)\s+(?:me\s+)?where\s+(.+?)(?=\s+(?:(?:is|are|was|were|being|been)\s+)?(?:written|mentioned|stated|said|discussed|referenced|found|present))',
    r'(?:where|in\s+what\s+part)\s+(?:is|does)\s+(?:it\s+)?(?:say|write|state|mention)\s+(?:about\s+)?(.+?)(?:\s*\?|$)',
    r'(?:give|show|find)\s+(?:me\s+)?(?:the\s+)?(?:exact\s+)?(?:text|words?|wording|content|quote|passage|snippet|verbatim)\s+(?:of|about|for)\s+(.+?)(?:\s*\?|$)',
    r'what\s+(?:does\s+(?:it|the\s+(?:document|file|text))\s+say\s+about)\s+(.+?)(?:\s*\?|$)',
    r'(?:quote|cite|extract)\s+(?:the\s+)?(?:exact\s+)?(?:part|section|text|words?|wording|content|quote|passage|snippet|verbatim)\s+(?:about|mentioning|regarding|where)\s+(.+?)(?:\s*\?|$)',
    r'(?:extract|show|give|provide|return|get|find)\s+(?:me\s+)?(?:the\s+)?(?:complete\s+)?(?:full\s+)?(?:original\s+)?(?:unmodified\s+)?(?:exact\s+)?(?:text(?:\s+block)?|wording|statement|court\s+statement|transcription(?:\s+wording)?|passage|snippet)\s+where\s+(.+?)\s+(?:is|are|was|were)\s+(?:mentioned|written|stated|said|discussed|referenced|found|present|appears?)\b',
    # Broader production phrasing variants
    r'(?:extract|show|give|provide|return|get|find)\s+(?:me\s+)?(?:the\s+)?(?:complete|full|original|unmodified|exact|verbatim)\s+(?:text(?:\s+block)?|wording|statement|court\s+statement|transcription(?:\s+wording)?|passage|snippet)\s+(?:where|about|mentioning|referencing|regarding)\s+(.+?)(?=\s+(?:(?:is|are|was|were|being|been)\s+)?(?:mentioned|written|stated|said|discussed|referenced|found|present|appears?)|\s*\?|$)',
    r'(?:provide|show|give|return)\s+(?:me\s+)?(?:the\s+)?(?:original|verbatim|exact|unmodified)\s+(?:text|wording|statement|court\s+statement|transcription)\s+(?:as\s+written\s+)?where\s+(.+?)\s+(?:appears?|is\s+mentioned|are\s+mentioned)\b',
    r'(?:show|give|provide|return|extract|find)\s+(?:me\s+)?(?:the\s+)?(?:exact\s+)?(?:scanned\s+)?(?:transcription|transcription\s+wording|wording|text|text\s+block|statement|court\s+statement|verbatim)\s+(?:for\s+(?:any\s+)?(?:entry|record)\s+mentioning|mentioning|referencing|about|regarding|of)\s+(.+?)(?:\s*\?|$)',
]

EXACT_INTENT_PATTERN = re.compile(
    r'\b('
    r'verbatim|wording|quote|snippet|passage|exactly|transcription|unmodified|'
    r'original\s+text|exact\s+(?:text|words?|wording|content|quote|snippet|passage|statement|transcription|text\s+block)'
    r')\b',
    re.IGNORECASE
)

COUNT_INTENT_PATTERN = re.compile(
    r'\b(how\s+many|count|number\s+of|total\s+mentions?)\b',
    re.IGNORECASE
)

BOTH_INTENT_PATTERN = re.compile(
    r'\b(both|the\s+two|two\s+of\s+them|both\s+of\s+them)\b',
    re.IGNORECASE
)

SINGLE_PRONOUN_PLACEHOLDER_PATTERN = re.compile(
    r'^(?:he|she|it|they|him|her|them|his|hers|their|theirs)$',
    re.IGNORECASE
)


class EnhancedQueryClassifier:
    """
    Classifies queries into the new enhanced query types.

    This classifier runs BEFORE the standard QueryClassifier. If a query
    matches one of the new types, it short-circuits the standard classification.
    Otherwise, it returns None to let the standard classifier handle it.
    """

    def __init__(self):
        """Initialize the enhanced query classifier."""
        # Pre-compile patterns
        self._count_patterns = [
            re.compile(p, re.IGNORECASE) for p in ENTITY_COUNT_PATTERNS
        ]
        self._file_patterns = [
            re.compile(p, re.IGNORECASE) for p in FILE_LOCATION_PATTERNS
        ]
        self._text_patterns = [
            re.compile(p, re.IGNORECASE) for p in EXACT_TEXT_PATTERNS
        ]

        logger.info("EnhancedQueryClassifier initialized")

    def classify(
            self,
            query: str,
            conversation_history: Optional[List] = None
    ) -> Optional[Tuple[QueryClassificationResult, Dict]]:
        """
        Classify a query into one of the new enhanced types.

        Args:
            query: User's query text

        Returns:
            Tuple of (QueryClassificationResult, extracted_params) if matched,
            None if the query should be handled by the standard classifier.

            extracted_params contains:
            - For ENTITY_COUNT: {"entity_name": str}
            - For FILE_LOCATION: {"entity_name": str}
            - For EXACT_TEXT: {"entity_name": str, "sender": str, "receiver": str}
        """
        if not query or not query.strip():
            return None

        query_clean = query.strip()
        query_for_match = self._normalize_query_for_matching(query_clean)
        query_lower = query_for_match.lower()
        exact_intent = self._has_exact_text_intent(query_lower)

        # Hybrid request: count + exact text in a single query.
        hybrid_result = self._check_hybrid_count_plus_text(query_for_match, conversation_history)
        if hybrid_result:
            return self._normalize_structured_entity_result(
                hybrid_result,
                query=query_for_match,
                conversation_history=conversation_history,
            )

        # If user explicitly asks for exact wording/text, prefer exact-text routing first.
        if exact_intent:
            result = self._check_text_patterns(query_for_match)
            if result:
                normalized = self._normalize_structured_entity_result(
                    result,
                    query=query_for_match,
                    conversation_history=conversation_history,
                )
                return self._augment_exact_with_email_scope(
                    normalized,
                    query_for_match,
                    conversation_history
                )

            # Fallback: infer entity from "where X mentioned" phrasing.
            inferred = self._extract_entity_from_mentioned_clause(query_for_match)
            if inferred:
                inferred_result = self._build_exact_text_result(
                    inferred,
                    confidence=0.82,
                    pattern_hint="mentioned_clause_fallback"
                )
                return self._augment_exact_with_email_scope(
                    self._normalize_structured_entity_result(
                        inferred_result,
                        query=query_for_match,
                        conversation_history=conversation_history,
                    ),
                    query_for_match,
                    conversation_history
                )

            # Follow-up without explicit entity: recover from conversation context.
            followup_result = self._check_followup_exact_text(query_clean, conversation_history)
            if followup_result:
                return self._augment_exact_with_email_scope(
                    self._normalize_structured_entity_result(
                        followup_result,
                        query=query_for_match,
                        conversation_history=conversation_history,
                    ),
                    query_for_match,
                    conversation_history
                )

        # "where is X mentioned" generally asks for exact mention snippets.
        if self._is_where_mentioned_query(query_for_match):
            inferred = self._extract_entity_from_mentioned_clause(query_for_match)
            if inferred:
                inferred_result = self._build_exact_text_result(
                    inferred,
                    confidence=0.84,
                    pattern_hint="where_mentioned_exact"
                )
                return self._augment_exact_with_email_scope(
                    self._normalize_structured_entity_result(
                        inferred_result,
                        query=query_for_match,
                        conversation_history=conversation_history,
                    ),
                    query_for_match,
                    conversation_history
                )

        # Check ENTITY_COUNT
        result = self._check_count_patterns(query_for_match)
        if result:
            return self._normalize_structured_entity_result(
                result,
                query=query_for_match,
                conversation_history=conversation_history,
            )

        # Check FILE_LOCATION
        result = self._check_file_patterns(query_for_match)
        if result:
            normalized_file_result = self._normalize_structured_entity_result(
                result,
                query=query_for_match,
                conversation_history=conversation_history,
            )
            # If exact intent is present, reinterpret FILE_LOCATION match as EXACT_TEXT.
            if exact_intent:
                _, params = normalized_file_result
                entity_names = params.get("entity_names") or []
                entity_name = params.get("entity_name")
                if entity_names and len(entity_names) > 1:
                    return self._build_exact_text_result_for_entities(
                        entity_names,
                        confidence=0.85,
                        pattern_hint="file_location_to_exact_intent",
                    )
                if entity_name:
                    return self._build_exact_text_result(
                        entity_name,
                        confidence=0.85,
                        pattern_hint="file_location_to_exact_intent"
                    )
            return normalized_file_result

        # Check EXACT_TEXT
        result = self._check_text_patterns(query_for_match)
        if result:
            return self._augment_exact_with_email_scope(
                self._normalize_structured_entity_result(
                    result,
                    query=query_for_match,
                    conversation_history=conversation_history,
                ),
                query_for_match,
                conversation_history
            )

        # Last fallback: exact-text follow-up with no explicit entity.
        if exact_intent:
            followup_result = self._check_followup_exact_text(query_clean, conversation_history)
            if followup_result:
                return self._augment_exact_with_email_scope(
                    self._normalize_structured_entity_result(
                        followup_result,
                        query=query_for_match,
                        conversation_history=conversation_history,
                    ),
                    query_for_match,
                    conversation_history
                )

        return None

    def _augment_exact_with_email_scope(
            self,
            result: Tuple[QueryClassificationResult, Dict],
            query: str,
            conversation_history: Optional[List]
    ) -> Tuple[QueryClassificationResult, Dict]:
        """
        Enrich EXACT_TEXT params when query scopes results to emails.

        Examples:
        - "show exact words in emails of Jeffrey Epstein" -> sender=Jeffrey Epstein
        - If entity is omitted/implicit, reuse last structured entity from memory.
        """
        classification, params = result
        scope = self._extract_email_scope(query)
        if not scope:
            return result

        updated = dict(params)
        sender = scope.get("sender")
        receiver = scope.get("receiver")

        if sender and not updated.get("sender"):
            updated["sender"] = sender
        if receiver and not updated.get("receiver"):
            updated["receiver"] = receiver

        entity = updated.get("entity_name")
        entity_names = updated.get("entity_names") or []
        wants_both = self._has_both_intent(query)
        if self._looks_like_email_scope_entity(entity):
            inferred_entities = self._resolve_entities_from_history(
                conversation_history,
                max_entities=2 if wants_both else 1
            )
            if len(inferred_entities) >= 2:
                updated["entity_name"] = None
                updated["entity_names"] = inferred_entities[:2]
                updated["require_all_entities"] = True
            elif inferred_entities:
                updated["entity_name"] = inferred_entities[0]
        elif not entity and not entity_names:
            inferred_entities = self._resolve_entities_from_history(
                conversation_history,
                max_entities=2 if wants_both else 1
            )
            if len(inferred_entities) >= 2:
                updated["entity_names"] = inferred_entities[:2]
                updated["require_all_entities"] = True
                updated["entity_name"] = None
            elif inferred_entities:
                updated["entity_name"] = inferred_entities[0]

        if updated == params:
            return result

        metadata = dict(classification.metadata or {})
        metadata.update(updated)
        matched_patterns = list(classification.matched_patterns or [])
        if "email_scope_context" not in matched_patterns:
            matched_patterns.append("email_scope_context")

        updated_classification = QueryClassificationResult(
            query_type=classification.query_type,
            confidence=max(classification.confidence, 0.88),
            matched_patterns=matched_patterns,
            reasoning=classification.reasoning,
            metadata=metadata
        )

        return updated_classification, updated

    def _normalize_structured_entity_result(
            self,
            result: Tuple[QueryClassificationResult, Dict],
            query: str,
            conversation_history: Optional[List]
    ) -> Tuple[QueryClassificationResult, Dict]:
        """
        Normalize structured params for single/multi-entity forms.

        Handles:
        - "both X and Y" extraction into entity_names
        - "both were mentioned" follow-ups by recovering entities from history
        - single-pronoun follow-ups ("he/she/they") by recovering entity from history
        """
        classification, params = result
        updated = dict(params or {})
        history_entities = self._resolve_entities_from_history(conversation_history, max_entities=2)

        raw_entity = updated.get("entity_name")
        if isinstance(raw_entity, str):
            raw_entity = self._clean_entity_name(raw_entity)
            if self._is_single_pronoun_placeholder(raw_entity):
                replacement = self._pick_history_entity_for_pronoun(history_entities, [])
                if replacement:
                    raw_entity = replacement
            updated["entity_name"] = raw_entity
        else:
            raw_entity = None

        explicit_entities = []
        existing_entity_names = updated.get("entity_names")
        if isinstance(existing_entity_names, list):
            for e in existing_entity_names:
                if not isinstance(e, str):
                    continue
                cleaned = self._clean_entity_name(e)
                if not cleaned:
                    continue
                if self._is_single_pronoun_placeholder(cleaned):
                    replacement = self._pick_history_entity_for_pronoun(
                        history_entities,
                        explicit_entities
                    )
                    if replacement:
                        cleaned = replacement
                if cleaned:
                    explicit_entities.append(cleaned)
            explicit_entities = self._dedupe_entities(explicit_entities)
        elif raw_entity:
            explicit_entities = self._extract_multi_entities(raw_entity)

        if explicit_entities and len(explicit_entities) >= 2:
            updated["entity_names"] = explicit_entities
            updated["require_all_entities"] = True
            updated["entity_name"] = None
        else:
            if self._is_both_placeholder(raw_entity, query):
                if len(history_entities) >= 2:
                    updated["entity_names"] = history_entities[:2]
                    updated["require_all_entities"] = True
                    updated["entity_name"] = None
                elif history_entities:
                    updated["entity_name"] = history_entities[0]
            elif self._is_single_pronoun_placeholder(raw_entity):
                replacement = self._pick_history_entity_for_pronoun(history_entities, [])
                if replacement:
                    updated["entity_name"] = replacement

        changed = updated != params
        if not changed:
            return result

        metadata = dict(classification.metadata or {})
        metadata.update(updated)
        matched_patterns = list(classification.matched_patterns or [])
        if "multi_entity_normalized" not in matched_patterns:
            matched_patterns.append("multi_entity_normalized")

        updated_classification = QueryClassificationResult(
            query_type=classification.query_type,
            confidence=classification.confidence,
            matched_patterns=matched_patterns,
            reasoning=classification.reasoning,
            metadata=metadata
        )
        return updated_classification, updated

    @staticmethod
    def _has_both_intent(query: str) -> bool:
        """Return True when query asks about both/two entities."""
        return bool(BOTH_INTENT_PATTERN.search(query or ""))

    def _is_both_placeholder(self, raw_entity: Optional[str], query: str) -> bool:
        """Detect placeholder entity text like 'both' that should use history."""
        if raw_entity and raw_entity.strip().lower() in {
            "both",
            "both of them",
            "the two",
            "two of them",
            "them",
        }:
            return True
        return self._has_both_intent(query)

    @staticmethod
    def _is_single_pronoun_placeholder(raw_entity: Optional[str]) -> bool:
        """Detect single pronoun placeholders that should resolve from history."""
        if not raw_entity:
            return False
        return bool(SINGLE_PRONOUN_PLACEHOLDER_PATTERN.match(raw_entity.strip()))

    def _pick_history_entity_for_pronoun(
            self,
            history_entities: List[str],
            already_used: List[str]
    ) -> Optional[str]:
        """Pick best non-pronoun history entity, avoiding duplicates in multi-entity flows."""
        used = {
            e.lower().strip()
            for e in already_used
            if isinstance(e, str) and e.strip()
        }
        for candidate in history_entities or []:
            cleaned = self._clean_entity_name(candidate)
            if not cleaned or self._is_single_pronoun_placeholder(cleaned):
                continue
            if cleaned.lower() in used:
                continue
            return cleaned
        return None

    def _extract_multi_entities(self, entity_text: str) -> List[str]:
        """Extract multi-entity list from text like 'both X and Y'."""
        if not entity_text:
            return []

        value = entity_text.strip()
        has_connector = bool(re.search(r'\s(?:and|&|\+|versus|vs)\s', value, re.IGNORECASE))
        if not (has_connector or self._has_both_intent(value)):
            return []

        value = re.sub(r'^(?:both|the\s+two|two\s+of\s+them|both\s+of\s+them)\s+', '', value, flags=re.IGNORECASE)
        parts = re.split(r'\s+(?:and|&|\+|versus|vs)\s+', value, flags=re.IGNORECASE)

        cleaned_parts = []
        seen = set()
        for part in parts:
            cleaned = self._clean_entity_name(part)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned_parts.append(cleaned)

        if len(cleaned_parts) < 2:
            return []
        return cleaned_parts

    def _check_count_patterns(
            self, query: str
    ) -> Optional[Tuple[QueryClassificationResult, Dict]]:
        """Check if query matches ENTITY_COUNT patterns."""
        for pattern in self._count_patterns:
            match = pattern.search(query)
            if match:
                entity_name = self._clean_entity_name(match.group(1))
                if entity_name and len(entity_name) > 1:
                    logger.info(f"Classified as ENTITY_COUNT for entity: '{entity_name}'")
                    return (
                        QueryClassificationResult(
                            query_type=QueryType.ENTITY_COUNT,
                            confidence=0.9,
                            matched_patterns=[pattern.pattern],
                            reasoning=f"Query asks to count mentions of '{entity_name}'",
                            metadata={"entity_name": entity_name}
                        ),
                        {"entity_name": entity_name}
                    )
        return None

    def _check_file_patterns(
            self, query: str
    ) -> Optional[Tuple[QueryClassificationResult, Dict]]:
        """Check if query matches FILE_LOCATION patterns."""
        for pattern in self._file_patterns:
            match = pattern.search(query)
            if match:
                entity_name = self._clean_entity_name(match.group(1))
                if entity_name and len(entity_name) > 1:
                    logger.info(f"Classified as FILE_LOCATION for entity: '{entity_name}'")
                    return (
                        QueryClassificationResult(
                            query_type=QueryType.FILE_LOCATION,
                            confidence=0.9,
                            matched_patterns=[pattern.pattern],
                            reasoning=f"Query asks for files containing '{entity_name}'",
                            metadata={"entity_name": entity_name}
                        ),
                        {"entity_name": entity_name}
                    )
        return None

    def _check_text_patterns(
            self, query: str
    ) -> Optional[Tuple[QueryClassificationResult, Dict]]:
        """Check if query matches EXACT_TEXT patterns."""
        for pattern in self._text_patterns:
            match = pattern.search(query)
            if match:
                groups = match.groups()
                params = {}

                # Check if this is an email pattern (2 capture groups: sender, receiver)
                if len(groups) >= 2 and groups[1]:
                    params["sender"] = self._clean_entity_name(groups[0])
                    params["receiver"] = self._clean_entity_name(groups[1])
                    params["entity_name"] = None
                    params["include_count"] = False
                    reasoning = f"Query asks for email/text from '{params['sender']}' to '{params['receiver']}'"
                else:
                    # General text pattern (1 capture group: entity/topic)
                    entity = self._clean_entity_name(groups[0])
                    params["entity_name"] = entity
                    params["sender"] = None
                    params["receiver"] = None
                    params["include_count"] = False
                    reasoning = f"Query asks for exact text about '{entity}'"

                if any(v for v in params.values() if v):
                    logger.info(f"Classified as EXACT_TEXT with params: {params}")
                    return (
                        QueryClassificationResult(
                            query_type=QueryType.EXACT_TEXT,
                            confidence=0.85,
                            matched_patterns=[pattern.pattern],
                            reasoning=reasoning,
                            metadata=params
                        ),
                        params
                    )
        return None

    @staticmethod
    def _has_exact_text_intent(query_lower: str) -> bool:
        """Return True if query explicitly asks for exact wording/text snippets."""
        return bool(EXACT_INTENT_PATTERN.search(query_lower))

    @staticmethod
    def _has_count_intent(query_lower: str) -> bool:
        """Return True if query asks for counts/frequencies."""
        return bool(COUNT_INTENT_PATTERN.search(query_lower))

    @staticmethod
    def _normalize_query_for_matching(query: str) -> str:
        """Normalize noisy punctuation/politeness suffixes for regex matching."""
        normalized = query.strip()
        # Intent-phrase typo normalization (safe and entity-agnostic).
        # Keep this strictly limited to fixed intent words to avoid touching
        # entity spans.
        intent_typos = [
            (r'\bwho\s+many\b', 'how many'),
            (r'\bhow\s+mamy\b', 'how many'),
            (r'\bhow\s+mani\b', 'how many'),
            (r'\bhow\s+meny\b', 'how many'),
        ]
        for pattern, replacement in intent_typos:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        normalized = re.sub(
            r'[\s,]*(?:please|pls)\s*[.!?]*$',
            '',
            normalized,
            flags=re.IGNORECASE
        )
        normalized = re.sub(r'[ \t]+', ' ', normalized).strip()
        normalized = re.sub(r'[,;:.!?]+$', '', normalized).strip()
        return normalized

    def _check_hybrid_count_plus_text(
            self,
            query: str,
            conversation_history: Optional[List]
    ) -> Optional[Tuple[QueryClassificationResult, Dict]]:
        """Handle queries asking for both count and exact text in one request."""
        q_lower = query.lower()
        if not (self._has_count_intent(q_lower) and self._has_exact_text_intent(q_lower)):
            return None

        entity = self._extract_entity_from_patterns(query, self._count_patterns)
        if not entity:
            entity = self._extract_entity_from_patterns(query, self._text_patterns)
        if not entity:
            entity = self._resolve_entity_from_history(conversation_history)

        if not entity:
            return None

        logger.info(f"Classified as EXACT_TEXT+COUNT hybrid for entity: '{entity}'")
        params = {
            "entity_name": entity,
            "sender": None,
            "receiver": None,
            "include_count": True
        }
        return (
            QueryClassificationResult(
                query_type=QueryType.EXACT_TEXT,
                confidence=0.92,
                matched_patterns=["hybrid_count_exact"],
                reasoning=f"Query asks for both count and exact text for '{entity}'",
                metadata=params
            ),
            params
        )

    def _check_followup_exact_text(
            self,
            query: str,
            conversation_history: Optional[List]
    ) -> Optional[Tuple[QueryClassificationResult, Dict]]:
        """Resolve exact-text follow-ups that omit the entity."""
        q_lower = query.lower()
        followup_exact = (
            self._has_exact_text_intent(q_lower) or
            bool(re.search(r'\b(i\s+need|show|give|find)\b.*\b(words?|text|content|quote|snippet)\b', q_lower))
        )
        if not followup_exact:
            return None

        wants_both = self._has_both_intent(query)
        history_entities = self._resolve_entities_from_history(
            conversation_history,
            max_entities=2 if wants_both else 1
        )
        if not history_entities:
            return None

        if wants_both and len(history_entities) >= 2:
            return self._build_exact_text_result_for_entities(
                history_entities[:2],
                confidence=0.82,
                pattern_hint="followup_history_multi_entity",
                from_history=True
            )

        return self._build_exact_text_result(
            history_entities[0],
            confidence=0.8,
            pattern_hint="followup_history_entity",
            from_history=True
        )

    @staticmethod
    def _is_where_mentioned_query(query: str) -> bool:
        """Detect direct 'where is X mentioned' forms for exact-text routing."""
        return bool(
            re.search(
                r'^\s*where\s+(?:is|are|was|were)\s+.+?\s+'
                r'(?:mentioned|referenced|written|stated|said|discussed)\b',
                query,
                re.IGNORECASE
            )
        )

    @staticmethod
    def _looks_like_email_scope_entity(entity: Optional[str]) -> bool:
        """Return True when extracted 'entity' is actually an email scope phrase."""
        if not entity:
            return False
        return bool(re.search(r'\bemails?\b', entity, re.IGNORECASE))

    def _extract_email_scope(self, query: str) -> Optional[Dict[str, Optional[str]]]:
        """
        Extract sender/receiver from email-scope phrases.

        Supports forms like:
        - "in the emails of Jeffrey Epstein"
        - "emails from X to Y"
        - "emails by X"
        """
        if not query or "email" not in query.lower():
            return None

        sender = None
        receiver = None

        sender_receiver_patterns = [
            re.compile(
                r'\bemails?\s+(?:from|by)\s+(.+?)\s+to\s+(.+?)'
                r'(?=\s+(?:where|that|which|with|about|mention(?:ed|ing)?|in|on|and|but)\b|[?.!,]|$)',
                re.IGNORECASE
            ),
            re.compile(
                r'\bemails?\s+between\s+(.+?)\s+and\s+(.+?)'
                r'(?=\s+(?:where|that|which|with|about|mention(?:ed|ing)?|in|on|and|but)\b|[?.!,]|$)',
                re.IGNORECASE
            ),
        ]

        for pattern in sender_receiver_patterns:
            match = pattern.search(query)
            if match:
                sender = self._clean_entity_name(match.group(1))
                receiver = self._clean_entity_name(match.group(2))
                if sender or receiver:
                    return {"sender": sender or None, "receiver": receiver or None}

        sender_only_patterns = [
            re.compile(
                r'\bemails?\s+(?:of|from|by)\s+(.+?)'
                r'(?=\s+(?:where|that|which|with|about|mention(?:ed|ing)?|in|on|and|but)\b|[?.!,]|$)',
                re.IGNORECASE
            ),
            re.compile(
                r'\bin\s+(?:the\s+)?emails?\s+(?:of|from|by)\s+(.+?)'
                r'(?=\s+(?:where|that|which|with|about|mention(?:ed|ing)?|in|on|and|but)\b|[?.!,]|$)',
                re.IGNORECASE
            ),
        ]

        for pattern in sender_only_patterns:
            match = pattern.search(query)
            if match:
                sender = self._clean_entity_name(match.group(1))
                if sender:
                    return {"sender": sender, "receiver": None}

        return None

    def _extract_entity_from_patterns(self, query: str, patterns: List[re.Pattern]) -> Optional[str]:
        """Try extracting entity from first capture-group pattern match."""
        for pattern in patterns:
            match = pattern.search(query)
            if not match:
                continue
            groups = match.groups()
            if not groups:
                continue
            entity = self._clean_entity_name(groups[0])
            if entity:
                return entity
        return None

    def _extract_entity_from_mentioned_clause(self, query: str) -> Optional[str]:
        """Extract entity from 'where is X mentioned' forms."""
        patterns = [
            re.compile(
                r'\bwhere\s+(?:is|are|was|were)\s+(.+?)\s+'
                r'(?:mentioned|referenced|found|written|present|appears?)\b',
                re.IGNORECASE
            ),
            re.compile(
                r'\b(?!show\b|give\b|find\b|extract\b|provide\b|return\b|get\b)(.+?)\s+(?:is|are|was|were)\s+'
                r'(?:mentioned|referenced|found|written|present|appears?)\b',
                re.IGNORECASE
            ),
        ]
        for pattern in patterns:
            match = pattern.search(query)
            if match:
                entity = self._clean_entity_name(match.group(1))
                if entity:
                    return entity
        return None

    def _resolve_entity_from_history(self, conversation_history: Optional[List]) -> Optional[str]:
        """Recover the latest relevant single entity from conversation history."""
        entities = self._resolve_entities_from_history(conversation_history, max_entities=1)
        return entities[0] if entities else None

    def _resolve_entities_from_history(
            self,
            conversation_history: Optional[List],
            max_entities: int = 2
    ) -> List[str]:
        """Recover the latest relevant entity/entities from conversation history."""
        if not conversation_history:
            return []

        for turn in reversed(conversation_history):
            entities = []
            if isinstance(turn, dict):
                entities = turn.get("entities_mentioned", []) or []
                metadata = turn.get("metadata", {}) or {}
                turn_query = turn.get("query", "") or ""
            else:
                entities = getattr(turn, "entities_mentioned", []) or []
                metadata = getattr(turn, "metadata", {}) or {}
                turn_query = getattr(turn, "query", "") or ""

            # 1) Prefer structured metadata entities (including multi-entity forms).
            if isinstance(metadata, dict):
                classification = metadata.get("classification", {}) or {}
                if isinstance(classification, dict):
                    cls_meta = classification.get("metadata", {}) or {}
                    entity_names = cls_meta.get("entity_names") or []
                    if isinstance(entity_names, list) and entity_names:
                        normalized_candidates = []
                        for value in entity_names:
                            if not isinstance(value, str):
                                continue
                            cleaned_value = self._clean_entity_name(value)
                            if not cleaned_value:
                                continue
                            if self._is_single_pronoun_placeholder(cleaned_value):
                                continue
                            normalized_candidates.append(cleaned_value)
                        normalized = self._dedupe_entities(normalized_candidates)
                        if normalized:
                            return normalized[:max_entities]
                    entity_name = cls_meta.get("entity_name")
                    if entity_name:
                        cleaned_entity = self._clean_entity_name(str(entity_name))
                        if self._is_single_pronoun_placeholder(cleaned_entity):
                            cleaned_entity = ""
                    else:
                        cleaned_entity = ""
                    if cleaned_entity:
                        if max_entities == 1:
                            return [cleaned_entity]
                        # Keep looking for a second entity in query text or extracted entities.
                        fallback_single = self._dedupe_entities([cleaned_entity])
                    else:
                        fallback_single = []
                else:
                    fallback_single = []
            else:
                fallback_single = []

            # 2) Fallback to turn entities with lightweight noise filtering/ranking.
            filtered = []
            for ent in entities:
                if not isinstance(ent, str):
                    continue
                cleaned = self._clean_entity_name(ent)
                if not cleaned:
                    continue
                if cleaned.lower() in {"in", "he", "she", "it", "during", "about", "public", "statements", "relevant", "mentions", "central"}:
                    continue
                filtered.append(cleaned)

            filtered = self._rank_history_entities(self._dedupe_entities(filtered))

            # 3) Recover explicit pair from last query text and canonicalize against
            # ranked history entities when possible.
            query_pair = self._extract_entity_pair_from_query(turn_query)
            if query_pair:
                if filtered:
                    canonical_pair = []
                    for raw in query_pair:
                        matched = self._best_history_entity_match(raw, filtered)
                        canonical_pair.append(matched or raw)
                    canonical_pair = self._dedupe_entities(canonical_pair)
                    if canonical_pair:
                        return canonical_pair[:max_entities]
                return query_pair[:max_entities]

            if filtered:
                return filtered[:max_entities]

            if fallback_single:
                return fallback_single[:max_entities]

        return []

    def _rank_history_entities(self, entities: List[str]) -> List[str]:
        """
        Rank history entities for follow-up reuse.

        Heuristics:
        - Prefer multi-token person-name entities.
        - Suppress noisy single tokens if a richer entity contains that token.
        - Keep Epstein as valid single-token fallback.
        """
        if not entities:
            return []

        multi_word = [e for e in entities if len(e.split()) >= 2]
        token_cover = set()
        for value in multi_word:
            token_cover.update(t.lower() for t in value.split())

        ranked = []
        for value in entities:
            parts = value.split()
            if len(parts) == 1:
                token = parts[0].lower()
                if token in token_cover and token not in {"epstein"}:
                    continue
                if token in {"prince", "duke", "york", "new", "while", "in", "about", "public"}:
                    continue
            ranked.append(value)

        return ranked

    @staticmethod
    def _best_history_entity_match(raw_entity: str, candidates: List[str]) -> Optional[str]:
        """Find best fuzzy match from history candidates for a raw/misspelled entity."""
        if not raw_entity or not candidates:
            return None

        best = None
        best_score = 0.0
        raw_norm = raw_entity.lower().strip()
        for candidate in candidates:
            cand_norm = candidate.lower().strip()
            score = SequenceMatcher(None, raw_norm, cand_norm).ratio()
            if raw_norm in cand_norm:
                score += 0.08
            if score > best_score:
                best_score = score
                best = candidate

        return best if best_score >= 0.68 else None

    def _extract_entity_pair_from_query(self, query_text: str) -> List[str]:
        """Extract likely pair entities from user phrasing in previous query text."""
        if not query_text:
            return []

        pair_patterns = [
            re.compile(
                r'\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!,]|$)',
                re.IGNORECASE
            ),
            re.compile(
                r'\brelation(?:ship)?\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.!,]|$)',
                re.IGNORECASE
            ),
        ]
        for pattern in pair_patterns:
            match = pattern.search(query_text)
            if not match:
                continue
            first = self._clean_entity_name(match.group(1))
            second = self._clean_entity_name(match.group(2))
            entities = self._dedupe_entities([first, second])
            if len(entities) >= 2:
                return entities[:2]
        return []

    def _build_exact_text_result_for_entities(
            self,
            entities: List[str],
            confidence: float,
            pattern_hint: str,
            from_history: bool = False
    ) -> Tuple[QueryClassificationResult, Dict]:
        """Build EXACT_TEXT classification for multi-entity co-mention queries."""
        normalized_entities = self._dedupe_entities(entities)
        params = {
            "entity_name": None,
            "entity_names": normalized_entities,
            "require_all_entities": True,
            "sender": None,
            "receiver": None,
            "include_count": False,
        }
        metadata = dict(params)
        metadata["from_history"] = from_history

        joined = " and ".join(normalized_entities[:2]) if len(normalized_entities) >= 2 else ", ".join(normalized_entities)
        return (
            QueryClassificationResult(
                query_type=QueryType.EXACT_TEXT,
                confidence=confidence,
                matched_patterns=[pattern_hint],
                reasoning=f"Query asks for exact text where both {joined} are mentioned",
                metadata=metadata
            ),
            params
        )

    def _dedupe_entities(self, entities: List[str]) -> List[str]:
        """Deduplicate entities preserving order."""
        seen = set()
        deduped = []
        for entity in entities:
            if not isinstance(entity, str):
                continue
            cleaned = self._clean_entity_name(entity)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _build_exact_text_result(
            self,
            entity_name: str,
            confidence: float,
            pattern_hint: str,
            from_history: bool = False
    ) -> Tuple[QueryClassificationResult, Dict]:
        """Build a standardized EXACT_TEXT classification result."""
        params = {
            "entity_name": entity_name,
            "sender": None,
            "receiver": None,
            "include_count": False
        }
        metadata = dict(params)
        metadata["from_history"] = from_history

        return (
            QueryClassificationResult(
                query_type=QueryType.EXACT_TEXT,
                confidence=confidence,
                matched_patterns=[pattern_hint],
                reasoning=f"Query asks for exact text about '{entity_name}'",
                metadata=metadata
            ),
            params
        )

    @staticmethod
    def _clean_entity_name(raw_entity: str) -> str:
        """Normalize extracted entity by removing trailing intent/scope phrases."""
        if not raw_entity:
            return ""

        entity = raw_entity.strip()
        entity = entity.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        entity = entity.replace('"', '')
        entity = entity.strip('"\'?.,')
        entity = re.sub(r'^(?:both|the\s+two|two\s+of\s+them|both\s+of\s+them)\s+', '', entity, flags=re.IGNORECASE)
        entity = re.sub(r'^(?:of|for|about)\s+', '', entity, flags=re.IGNORECASE)
        entity = re.sub(
            r'^(?:extract|show|give|provide|return|get|find)\s+(?:me\s+)?(?:the\s+)?'
            r'(?:complete|full|original|unmodified|exact|verbatim)\s+'
            r'(?:text(?:\s+block)?|wording|statement|court\s+statement|transcription(?:\s+wording)?|passage|snippet)\s+'
            r'(?:where|about|mentioning|referencing|regarding)\s+',
            '',
            entity,
            flags=re.IGNORECASE
        )
        entity = re.sub(
            r'^(?:any|the|a|an)\s+(?:entry|entries|record|records)\s+'
            r'(?:mentioning|referencing|about|regarding)\s+',
            '',
            entity,
            flags=re.IGNORECASE
        )
        entity = re.sub(
            r'^(?:any|the|a|an)\s+(?:file|document|doc|files|documents|docs)\s+',
            '',
            entity,
            flags=re.IGNORECASE
        )
        # Run again to handle nested forms like "of any file X"
        entity = re.sub(r'^(?:of|for|about)\s+', '', entity, flags=re.IGNORECASE)

        trailing_cleanup_patterns = [
            r'\band\s+(?:also\s+)?(?:give|show|provide|return|include)\b.*$',
            r',\s*then\s+(?:give|show|provide|return|include|get|share)\b.*$',
            r'\bthen\s+(?:give|show|provide|return|include|get|share)\b.*$',
            r'\band\s+exact\s+(?:text|words?|wording|content|quote|snippet|passage|verbatim)\b.*$',
            r'\b(?:and|then)\s+(?:the\s+)?exact\s+(?:text|words?|wording|content|quote|snippet|passage|verbatim)\b.*$',
            r'\bonly\s+count\b.*$',
            r'\bi\s+need\b.*$',
            r'\bplease\b.*$',
            r'\bfrom\s+(?:the\s+)?(?:epstein\s+)?(?:files?|documents?|docs?|dataset|data)\b.*$',
            r'\bin\s+(?:the\s+)?(?:epstein\s+)?(?:files?|documents?|docs?|dataset|data)\b.*$',
        ]
        for pattern in trailing_cleanup_patterns:
            entity = re.sub(pattern, '', entity, flags=re.IGNORECASE).strip()

        entity = re.sub(r'\s+', ' ', entity).strip().strip('"\'?.,')
        return entity
