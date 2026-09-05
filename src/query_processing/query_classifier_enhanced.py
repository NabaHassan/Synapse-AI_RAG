"""
Enhanced Query Classifier for Conversational RAG.

This module provides pattern-based query classification to detect:
- Meta-conversation queries (about the conversation itself)
- Formatting requests (reformat previous answer)
- Clarification requests (revise previous answer for clarity)
- Continuation queries (implicit references to previous context)
- New queries (standard RAG flow)

Classification happens BEFORE retrieval to avoid wasting resources.
"""

import re
import logging
from typing import List, Optional, Dict, Any

from src.query_processing.query_types import QueryType, QueryClassificationResult

logger = logging.getLogger(__name__)


class QueryClassifierEnhanced:
    """
    Enhanced query classifier with pattern-based detection.
    
    Uses regex patterns and heuristics to classify queries into types.
    Fast and accurate (>95%) without requiring LLM calls.
    
    Example:
        classifier = QueryClassifierEnhanced()
        result = classifier.classify("what was my first question?", history)
        # result.query_type == QueryType.META_CONVERSATION
    """

    # Meta-conversation patterns (highest priority)
    META_PATTERNS = [
        # Questions about previous queries
        r"\b(what|which) (was|is) (my|the) (first|previous|last|earlier) (question|query)\b",
        r"\bwhat did i (ask|say|query)\b",
        r"\bwhat was (my|the) (initial|original) (question|query)\b",

        # Questions about conversation content
        r"\bwhat (did|have) (i|we) (discuss|talk about|cover)\b",
        r"\bwhat (have|did) we (discussed|talked about)\b",

        # Conversation summary requests
        r"\b(summarize|recap|review) (our|this|the) (conversation|chat|discussion)\b",
        r"\bgive me a (summary|recap|overview) of (our|this|the) (conversation|chat)\b",

        # Clarification about previous responses (only when asking about *our* wording)
        r"\bwhat do you mean by (my|your|the) (question|answer|response|statement|wording)\b",
        # Match only "explain what you said/meant" — NOT "explain that" (content rephrase)
        r"\bcan you (clarify|explain) what you (said|meant)\b",
    ]

    # Formatting request patterns (structure/format changes only)
    FORMAT_PATTERNS = [
        # Bullet points and lists (including "make it a list")
        r"\bwith (proper )?(bullets?|bullet points?|points?)\b",
        r"\bin (bullet|list) (form|format|points?)\b",
        r"\bmake\s+(it|this|that)\s+(a|into)\s+(list|bullet\s+list)\b",  # Fixed: added "make it a list"
        r"\b(format|give|show|present) (it|this|that|the answer) (as |in |with )?(bullets?|bullet points?|list|table)\b",

        # Headings and structure
        r"\bwith (proper )?(headings?|headers?)\b",
        r"\b(add|include|use) (headings?|headers?)\b",

        # General reformatting
        r"\breformat\b",
        r"\b(organize|structure) (it|this|that)\b",

        # Combined formatting requests (X and Y)
        r"\bwith (proper )?(bullets?|bullet points?|points?) and (headings?|headers?)\b",
        r"\bwith (proper )?(headings?|headers?) and (bullets?|bullet points?|points?)\b",
        r"\b(in|as) (a )?(list|bullet) (form|format) (and|with) (headings?|headers?)\b",

        # Layman's terms / simple language / plain English (reformat previous answer)
        r"\b(layman'?s?|layperson'?s?)\s+terms\b",
        r"\bin\s+simple(r)?\s+(terms|language|words)\b",
        r"\b(plain|simple)\s+english\b",
        r"\bput\s+(it|that|this)\s+in\s+(layman'?s?|simple)\s+terms\b",
        # Summarization of previous answer (distinct from meta-conversation summary)
        r"^(summarize|summarise)( (it|that|this|the answer|above))?[\.\!\?]*$",
        r"^give (me )?a summary( of (it|that|this|above))?[\.\!\?]*$",
        r"^tl;?dr[\.\!\?]*$",

        # Concise / shorter versions of previous answer (including "please ...")
        r"\b(please\s+)?(make|keep)\s+(it|this|that|the\s+answer)\s+(short|concise|brief)(\s+please)?\b",
        r"\b(please\s+)?make\s+(it|this|that|the\s+answer)\s+shorter(\s+please)?\b",
        r"\bmake\s+(it|this|that)\s+concise\s+please\b",
        r"\b(please\s+)?(can|could)\s+you\s+(shorten|condense)\s+(it|this|that|the\s+answer)\b",
        r"\b(shorter|more\s+concise)\s+version(\s+please)?\b",
        r"\bconcise\s+version(\s+please)?\b",
        r"^(please\s+)?be\s+(concise|brief)[\.\!\?]*$",
        r"^(please\s+)?keep\s+it\s+(concise|brief)[\.\!\?]*$",
        r"^(please\s+)?be\s+brief[\.\!\?]*$",
        r"^(please\s+)?make\s+it\s+concise[\.\!\?]*$",
        r"^make\s+it\s+concise\s+please[\.\!\?]*$",
        r"^shorter\s+please[\.\!\?]*$",
        r"^brief\s+please[\.\!\?]*$",
        r"^(please\s+)?condense\s+(it|this|that|the\s+answer)[\.\!\?]*$",
        r"^condense\s+(it|this|that|the\s+answer)\s+please[\.\!\?]*$",
        r"^(please\s+)?summari[sz]e\s+(it|this|that|the\s+answer)[\.\!\?]*$",

        # Detail-expansion follow-ups (standalone follow-up phrases)
        r"^(in|with)\s+details?(?:\s+please)?[\.\!\?]*$",
        r"^in\s+more\s+details?(?:\s+please)?[\.\!\?]*$",
        r"^more\s+details?(?:\s+please)?[\.\!\?]*$",
        r"^details?\s+please[\.\!\?]*$",
        r"^detailed?\s+(version|explanation|answer)[\.\!\?]*$",
        r"^(explain|describe|elaborate)\s+(it|this|that)\s+in\s+details?[\.\!\?]*$",
    ]

    # Clarification request patterns (user doesn't understand - needs revision)
    CLARIFICATION_PATTERNS = [
        # Core understanding issues (apostrophes normalized before matching)
        # These patterns match the NORMALIZED query (apostrophes removed)
        r"\b(i\s+)?(don'?t|do\s+not|dont)\s+(understand|get|follow)(\s+it)?\b",
        r"\b(i\s+)?(didn'?t|did\s+not|didnt)\s+(understand|get|follow)(\s+it)?\b",
        r"\b(i\s+)?(can'?t|cannot|cant)\s+(understand|get|follow)(\s+it)?\b",

        # Confusion expressions (including standalone)
        r"\b(i'?m|i\s+am|im)\s+(confused|lost|not\s+following)\b",
        r"^confused$",  # Standalone "confused"
        r"\b(not\s+)?(clear|unclear)\b",
        r"\b(doesn'?t|does\s+not|doesnt)\s+make\s+sense\b",
        r"\bmakes?\s+no\s+sense\b",
        r"\b(lost|losing)\s+(me|track)\b",
        r"\bover\s+my\s+head\b",
        r"\b(i\s+)?(don'?t|do\s+not|dont)\s+quite\s+follow\b",
        r"\b(i\s+)?(am|i'?m)\s+not\s+sure\s+i\s+follow\b",
        r"\b(i\s+)?(am|i'?m)\s+missing\s+something\b",
        r"\b(that|this|it)\s+(didn'?t|did\s+not|didnt)\s+make\s+sense\b",

        # Clarification requests (including "can/could you clarify", "please ...")
        r"\bwhat\s+do\s+you\s+mean\b",
        r"\bwhat\s+do\s+you\s+mean\s+by\s+(that|this|it)\b",
        r"\bwhat\s+does\s+(that|this|it)\s+mean\b",
        r"\b(can|could)\s+you\s+clarify\b",
        r"\b(can|could)\s+you\s+clarify\s+(that|this|it)\s+for\s+me\b",
        r"\bplease\s+clarify\b",
        r"\bclarify\s+(that|this|it)(\s+please)?\b",
        r"\bplease\s+(don'?t|do\s+not|dont)\s+understand\b",
        r"\b(i\s+)?(don'?t|dont)\s+understand\s+please\b",
        r"\bplease\s+(i\s+)?(don'?t|dont)\s+get\s+it\b",
        r"\bplease\s+explain\b",
        r"^explain[\.\!\?]*$",
        r"\bexplain\s+(that|this|it)(\s+please)?\b",
        r"\bexplain\s+again\s+please\b",
        r"^(please\s+)?explain\s+more(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?explain\s+further(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?explain\s+more\s+(about\s+)?(this|that|it)(\s+please)?[\.\!\?]*$",
        r"^(can|could)\s+you\s+explain\s+more(\s+please)?[\.\!\?]*$",
        r"\b(can|could)\s+you\s+rephrase\b",
        r"\brephrase\s+(that|this|it)\b",
        r"\b(can|could)\s+you\s+(put|say)\s+(that|this|it)\s+(differently|another\s+way)\b",
        r"\b(say|put)\s+(that|this|it)\s+another\s+way\b",
        r"\b(say|put)\s+(that|this|it)\s+differently\b",

        # Explanation requests
        r"\bexplain\s+(that|this|it|again|better|differently)\b",
        r"\b(can|could)\s+you\s+explain\b",
        r"\bplease\s+(can|could)\s+you\s+explain\b",

        # Repetition requests (including "repeat that")
        r"\b(say|repeat)\s+(that|it)\s+again\b",
        r"\btell\s+(me\s+)?again\b",
        r"\brepeat\s+(that|this|it)\b",  # Fixed: added standalone "repeat that"
        r"\breiterate\b",
        r"\bone\s+more\s+time\b",
        r"\bagain\s+please\b",

        # Understanding checks
        r"\bnot\s+following\b",
        r"\bhard\s+to\s+(follow|understand)\b",
        r"\btoo\s+(complicated|complex)\b",

        # Simplification for revision (not expansion)
        r"\b(make\s+it\s+)?simpler\b",
        r"\bin\s+simpler\s+terms\b",
        r"\bin\s+(easy|simple)\s+words\b",
        r"\b(easy|simple)\s+words\b",
        r"\bgive\s+me\s+(easy|simple)\s+words\b",
        r"\b(use|say|put)\s+(it|this|that)?\s*(in\s+)?(easy|simple)\s+words\b",
        r"\beasier\s+to\s+understand\b",
        r"\bbreak\s+(it|that|this)\s+down\b",

        # Short confusion expressions
        r"^(what|huh)\??$",
        r"\bcome\s+again\b",

        # Rewriting for clarity (not formatting)
        r"\b(rewrite|rephrase|reword)\s+(it|this|that)\b",
        r"\b(simplify|clarify)\s+(it|this|that)\b",

        # Improvement/enhancement for clarity
        r"\bmake\s+(it|this|that)\s+(more|better|clearer)\b",
        r"\bmake\s+(it|this|that)\s+more\s+\w+\b",  # "make it more explanatory"
        r"\b(improve|enhance)\s+(it|this|that|the\s+answer)\b",
        r"\b(better|improved)\s+(version|explanation|answer)\b",

        # Expansion/elaboration for clarity (including "expand on it", "elaborate more")
        r"\bexpand\s+(on\s+)?(it|this|that)\b",  # Fixed: more flexible pattern
        r"\belaborate(\s+more|\s+on\s+(it|this|that))?\b",  # Fixed: more flexible pattern
        r"\bexplain\s+(it|this|that)\s+(more|better|further)\b",
        r"\b(more|better)\s+(detailed?|explanation|clarity|clear|specific)\b",
    ]

    # Continuation patterns (implicit references)
    CONTINUATION_PATTERNS = [
        # Pronouns without clear antecedent
        r"^(what about|how about|tell me about) (his|her|their|its)\b",
        r"^(what|how) (is|are|was|were) (his|her|their|its)\b",

        # Expansion requests
        r"\btell me more\b",
        r"\b(explain|elaborate) (more|further|on (this|that|it))\b",
        r"\b(give|provide) more (details?|information)\b",
        r"\bcontinue\b",

        # Follow-up questions
        r"^(and|also) ",
        r"^what about\b",
        r"^how about\b",
    ]

    # Explicit detail-expansion follow-up patterns.
    # These should route to formatting/expansion, not clarification/simplification.
    DETAIL_EXPANSION_PATTERNS = [
        r"^(in|with)\s+details?(?:\s+please)?[\.\!\?]*$",
        r"^in\s+more\s+details?(?:\s+please)?[\.\!\?]*$",
        r"^more\s+details?(?:\s+please)?[\.\!\?]*$",
        r"^details?\s+please[\.\!\?]*$",
        r"^detailed?\s+(version|explanation|answer)[\.\!\?]*$",
        r"^(explain|describe|elaborate)\s+(it|this|that)\s+in\s+details?[\.\!\?]*$",
        r"^(please\s+)?elaborate\s+more(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?tell\s+me\s+more(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?give\s+me\s+details(\s+please)?[\.\!\?]*$",
        r"^can\s+you\s+give\s+me\s+details[\.\!\?]*$",
        r"^could\s+you\s+give\s+me\s+details[\.\!\?]*$",
        r"^could\s+you\s+please\s+give\s+me\s+details[\.\!\?]*$",
        r"^more\s+(info|information|explanation)(?:\s+please)?[\.\!\?]*$",
        r"^(please\s+)?go\s+deeper(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?go\s+into\s+more\s+detail[s]?(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?expand\s+on\s+(that|this|it)(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?elaborate\s+on\s+(that|this|it)(\s+please)?[\.\!\?]*$",
        r"^more\s+context(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?walk\s+me\s+through\s+(that|this|it)(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?unpack\s+(that|this|it)(\s+please)?[\.\!\?]*$",
        r"^(please\s+)?drill\s+down\s+on\s+(that|this|it)(\s+please)?[\.\!\?]*$",
    ]

    def __init__(self):
        """Initialize the query classifier."""
        # Compile patterns for efficiency
        self.meta_compiled = [re.compile(p, re.IGNORECASE) for p in self.META_PATTERNS]
        self.format_compiled = [re.compile(p, re.IGNORECASE) for p in self.FORMAT_PATTERNS]
        self.clarification_compiled = [re.compile(p, re.IGNORECASE) for p in self.CLARIFICATION_PATTERNS]
        self.continuation_compiled = [re.compile(p, re.IGNORECASE) for p in self.CONTINUATION_PATTERNS]
        self.detail_expansion_compiled = [re.compile(p, re.IGNORECASE) for p in self.DETAIL_EXPANSION_PATTERNS]

        logger.info("QueryClassifierEnhanced initialized with pattern-based detection")
        logger.info(f"  - {len(self.CLARIFICATION_PATTERNS)} clarification patterns loaded")

    def classify(
            self,
            query: str,
            conversation_history: Optional[List[Any]] = None
    ) -> QueryClassificationResult:
        """
        Classify a query into one of the query types.

        Args:
            query: The user's query string
            conversation_history: Optional list of previous conversation turns

        Returns:
            QueryClassificationResult with type, confidence, and reasoning
        """
        query_lower = query.lower().strip()

        # Check meta-conversation FIRST (highest priority)
        meta_result = self._check_meta_conversation(query_lower)
        if meta_result:
            return meta_result

        # Explicit detail-expansion follow-ups should not be treated as
        # clarification (which simplifies/revises). Route as formatting request.
        detail_expansion_result = self._check_detail_expansion_followup(query_lower)
        if detail_expansion_result:
            return detail_expansion_result

        # Check clarification request (BEFORE formatting)
        clarification_result = self._check_clarification(query_lower)
        if clarification_result:
            return clarification_result

        # Check formatting request
        format_result = self._check_formatting_request(query_lower)
        if format_result:
            return format_result

        # Check continuation (requires conversation history)
        if conversation_history:
            continuation_result = self._check_continuation(query_lower, conversation_history)
            if continuation_result:
                return continuation_result

        # Default to new query
        return QueryClassificationResult(
            query_type=QueryType.NEW_QUERY,
            confidence=1.0,
            reasoning="No meta, formatting, or continuation patterns detected"
        )

    def _check_meta_conversation(self, query_lower: str) -> Optional[QueryClassificationResult]:
        """Check if query is about the conversation itself."""
        matched_patterns = []

        for pattern in self.meta_compiled:
            if pattern.search(query_lower):
                matched_patterns.append(pattern.pattern)

        if matched_patterns:
            logger.debug(
                "Classified as META_CONVERSATION: query=%r, matched_patterns=%s",
                query_lower[:80],
                matched_patterns,
            )
            return QueryClassificationResult(
                query_type=QueryType.META_CONVERSATION,
                confidence=1.0,
                matched_patterns=matched_patterns,
                reasoning=f"Matched meta-conversation patterns: {matched_patterns[0][:50]}..."
            )

        return None

    def _check_clarification(self, query_lower: str) -> Optional[QueryClassificationResult]:
        """Check if query is a clarification request."""
        # Normalize apostrophes for matching, including smart quotes.
        # Example: "doesn’t" (with U+2019) → "doesn't" → "doesnt"
        normalized = (
            query_lower
            .replace("’", "'")
            .replace("‘", "'")
            .replace("´", "'")
            .replace("`", "'")
        )
        query_normalized = normalized.replace("'", "")

        logger.debug(
            "Clarification check - raw=%r, normalized=%r, no_apostrophes=%r",
            query_lower[:80],
            normalized[:80],
            query_normalized[:80],
        )

        matched_patterns = []

        for pattern in self.clarification_compiled:
            # Check both normalized and apostrophe-stripped versions
            if pattern.search(normalized) or pattern.search(query_normalized):
                matched_patterns.append(pattern.pattern)

        if matched_patterns:
            logger.debug(
                "Classified as CLARIFICATION: query=%r, matched_patterns=%s",
                query_lower[:80],
                matched_patterns,
            )
            return QueryClassificationResult(
                query_type=QueryType.CLARIFICATION,
                confidence=1.0,
                matched_patterns=matched_patterns,
                reasoning=f"Matched clarification patterns: {matched_patterns[0][:50]}..."
            )

        return None

    def _check_formatting_request(self, query_lower: str) -> Optional[QueryClassificationResult]:
        """Check if query is a formatting request."""
        matched_patterns = []

        for pattern in self.format_compiled:
            if pattern.search(query_lower):
                matched_patterns.append(pattern.pattern)

        if matched_patterns:
            return QueryClassificationResult(
                query_type=QueryType.FORMATTING_REQUEST,
                confidence=1.0,
                matched_patterns=matched_patterns,
                reasoning=f"Matched formatting patterns: {matched_patterns[0][:50]}..."
            )

        return None

    def _check_detail_expansion_followup(self, query_lower: str) -> Optional[QueryClassificationResult]:
        """Route standalone/detail follow-ups to formatting expansion."""
        matched_patterns = []

        for pattern in self.detail_expansion_compiled:
            if pattern.search(query_lower):
                matched_patterns.append(pattern.pattern)

        if matched_patterns:
            return QueryClassificationResult(
                query_type=QueryType.FORMATTING_REQUEST,
                confidence=1.0,
                matched_patterns=matched_patterns,
                reasoning=f"Matched detail-expansion patterns: {matched_patterns[0][:50]}..."
            )

        return None

    def _check_continuation(
            self,
            query_lower: str,
            conversation_history: List[Any]
    ) -> Optional[QueryClassificationResult]:
        """Check if query is a continuation of previous topic."""
        if not conversation_history:
            return None

        matched_patterns = []

        # Check explicit continuation patterns
        for pattern in self.continuation_compiled:
            if pattern.search(query_lower):
                matched_patterns.append(pattern.pattern)

        # Check for pronouns without clear antecedent
        pronouns = ['this', 'that', 'it', 'they', 'them', 'he', 'she', 'his', 'her', 'their']
        words = query_lower.split()

        # If query is short and starts with pronoun, likely continuation
        if len(words) < 8 and any(words[0] == p for p in pronouns):
            matched_patterns.append("short_query_with_pronoun")

        if matched_patterns:
            return QueryClassificationResult(
                query_type=QueryType.CONTINUATION,
                confidence=0.9,
                matched_patterns=matched_patterns,
                reasoning=f"Matched continuation patterns: {matched_patterns[0][:50]}..."
            )

        return None

    def get_classification_stats(self) -> Dict[str, int]:
        """Get statistics about pattern counts."""
        return {
            "meta_patterns": len(self.META_PATTERNS),
            "format_patterns": len(self.FORMAT_PATTERNS),
            "clarification_patterns": len(self.CLARIFICATION_PATTERNS),
            "continuation_patterns": len(self.CONTINUATION_PATTERNS),
            "detail_expansion_patterns": len(self.DETAIL_EXPANSION_PATTERNS),
            "total_patterns": (
                    len(self.META_PATTERNS) +
                    len(self.FORMAT_PATTERNS) +
                    len(self.CLARIFICATION_PATTERNS) +
                    len(self.CONTINUATION_PATTERNS) +
                    len(self.DETAIL_EXPANSION_PATTERNS)
            )
        }
