"""
Query Reformulator for RAG Pipeline.

This module provides query reformulation capabilities to resolve references
from conversation history, enabling the RAG system to understand follow-up
questions that contain pronouns or implicit references.

Features:
- Reference detection (pronouns, temporal, demonstratives, implicit)
- Context-aware reformulation using LLM
- Entity extraction from Q&A pairs
- Rule-based fallback for simple cases
- Configurable thresholds and patterns

Capabilities:
- 2.1 Reference Detection: Detect unresolved references in queries
- 2.2 Query Reformulation: Use LLM to expand queries with context
- 2.3 Entity Extraction: Extract key entities for future reference resolution
"""

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from src.memory.conversation_memory import ConversationTurn

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ReformulatorConfig:
    """
    Configuration for query reformulation behavior.
    
    Attributes:
        enabled: Whether reformulation is enabled (default: True)
        use_llm: Use LLM for reformulation (vs rule-based only) (default: True)
        max_history_turns: Max conversation turns to include in reformulation (default: 5)
        max_answer_chars: Max characters per answer in reformulation prompt (default: 200)
        max_reformulation_tokens: Max tokens for reformulated query (default: 100)
        confidence_threshold: Min confidence for accepting LLM reformulation (default: 0.7)
        fallback_to_rules: Use rule-based if LLM fails (default: True)
        extract_entities_from_answer: Also extract entities from answers (default: True)
        spacy_model: spaCy model for entity extraction (default: "en_core_web_sm")
    """
    enabled: bool = True
    use_llm: bool = True
    max_history_turns: int = 5
    max_answer_chars: int = 200
    max_reformulation_tokens: int = 100
    confidence_threshold: float = 0.7
    fallback_to_rules: bool = True
    extract_entities_from_answer: bool = True
    spacy_model: str = "en_core_web_sm"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


@dataclass
class ReformulationResult:
    """
    Result of query reformulation.
    
    Attributes:
        original_query: The original user query
        reformulated_query: The reformulated query (same as original if no reformulation)
        was_reformulated: Whether reformulation was applied
        detected_patterns: List of detected reference patterns
        resolved_references: Dict mapping references to resolved entities
        confidence: Confidence score for the reformulation
        method: Method used ("llm", "rule_based", "none")
        metadata: Additional metadata
    """
    original_query: str
    reformulated_query: str
    was_reformulated: bool
    detected_patterns: List[str] = field(default_factory=list)
    resolved_references: Dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    method: str = "none"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


# =============================================================================
# Reference Pattern Definitions
# =============================================================================

# Pronouns that may need resolution
# NOTE: include objective-case pronouns like "him" so follow-ups like
# "What were the charges over him?" trigger reformulation.
# IMPORTANT: these are regex patterns; use single backslashes (\b, \s, \w), not double-escaped (\\b).
PRONOUN_PATTERNS = [
    # Demonstratives / inanimate
    r'\b(it|this|that|these|those)\b',

    # Personal + possessive + objective-case
    r'\b(he|she|they|them|him|her|his|hers|their|theirs|its)\b',

    # Reflexives (common in follow-ups)
    r'\b(himself|herself|itself|themselves|themself)\b',
]

# Skip patterns - queries about the conversation itself that should NOT be reformulated
# These are meta-queries where the user is asking about the conversation, not the knowledge base
SKIP_META_QUERY_PATTERNS = [
    r'\bwhat (was|is|were|are) (my|the|your) (previous|last|earlier) (question|query)\b',
    # "What was my previous query"
    r'\bwhat did (i|we) (ask|say|mention)\b',  # "What did I ask"
    r'\bremind me (what|of what) (i|we) (asked|said)\b',  # "Remind me what I asked"
    r'\bwhat (question|query) did (i|we)\b',  # "What question did I ask"
]

# Meta-references to conversation (almost always need context)
META_REFERENCE_PATTERNS = [
    r'\bwhat.*(i|we).*(asked|said|mentioned|talked)\b',  # More flexible: "what I asked previously"
    r'\b(previous|earlier|before|last)\b.*(question|query|topic|answer|thing)\b',
    r'\bthe (same|above|aforementioned)\b',
    r'\bwhat you (said|mentioned|told|answered)\b',
    r'\bmy (previous|last|earlier) (question|query)\b',
    r'\b(as|like) (i|we) (said|mentioned|asked)\b',
    r'\bback to (that|the|my)\b',
    r'\bregarding (that|this|the previous)\b',
    r'\b(asked|said|mentioned|talked).*(previously|before|earlier)\b',  # "asked previously"
    # Formatting requests referencing prior answer/content
    r'\b(give|show|present|format|rewrite|summarize|list|explain).*\b(this|that|it|the (above|previous|same))\b',
    # Formatting requests with various phrasings: "in points", "in bullet points", "as a list", etc.
    r'\b(in|as)\s+(bullet\s+)?points\b',  # "in points", "in bullet points"
    r'\b(in|as)\s+(a\s+)?bullets?\b',  # "as bullets", "as a bullet"
    r'\b(in|as)\s+(a\s+)?list(ed)?(\s+form(at)?)?\b',  # "as a list", "in listed format"
    r'\b(in|as)\s+(a\s+)?summary\b',  # "as a summary", "in summary"
    r'\b(in|as)\s+steps\b',  # "in steps"
    # Explicit conversion/reformatting requests
    r'\b(convert|reformat|rephrase|rewrite|restructure)\s+(this|that|it|the\s+above)\b',
]

# Single-word or very short follow-up patterns
# NOTE: These should trigger reformulation when there's conversation history.
IMPLICIT_FOLLOWUP_PATTERNS = [
    r'^(why|how|who|when|where|which)\??\s*$',  # Single word questions
    r'^and\s+(what|who|why|how|when|where)\b',  # Follow-up conjunctions
    r'^(also|additionally|furthermore)\s*[,]?\s*(what|who|why|how)?\b',
    r'^what about\b',
    r'^how about\b',
    r'^same for\b',
    r'^(tell|give|show|explain)\s*(me\s*)?(more|further)\b',  # "tell me more", "explain more", "give further details"
    r'^(describe|elaborate|expand)\s*(on\s*)?(this|that|it)?\s*(more|further)?\b',
    # "describe more", "elaborate", "expand on this"
    r'^more\b',  # Just "more"
    r'^continue\b',  # "continue"
    r'^go on\b',  # "go on"
    r'^(and|but|so|then)\??\s*$',  # Conjunctions alone
]

# Words to skip when checking for proper nouns
# NOTE: words like "info", "points", "details" are NOT entities, so we skip them.
SKIP_WORDS_FOR_SUBJECT = {
    'i', 'we', 'you', 'why', 'how', 'what', 'who', 'when', 'where', 'which',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'do', 'does', 'did', 'have', 'has', 'had',
    'can', 'could', 'will', 'would', 'should', 'may', 'might',
    'tell', 'show', 'explain', 'describe', 'give', 'list', 'elaborate', 'expand',
    # Formatting/meta words (not entities)
    'info', 'information', 'details', 'points', 'bullets', 'summary',
    'list', 'steps', 'format', 'more', 'continue', 'further',
    # Common capitalized non-entities from answers (reduce entity extraction noise)
    'under', 'both', 'if', 'examples', 'nature', 'insight', 'determining',
    'while', 'according', 'however', 'example', 'exception', 'exceptions',
    'real', 'status', 'ownership', 'marital', 'contributions', 'during',
    'characterization', 'exception', 'exceptions',
}

# Patterns indicating explicit subject (self-contained query)
EXPLICIT_SUBJECT_PATTERNS = [
    r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b',  # Proper nouns (capitalized)
    r'"[^"]+"|\'[^\']+\'',  # Quoted terms
    r'\b\w+\s+(project|system|framework|model|training|data|process|algorithm|method)\b',
    r'\b(project|system|framework|model)\s+\w+\b',
    r'\b[A-Z]{2,}\b',  # Acronyms (2+ uppercase letters)
]


# =============================================================================
# Query Reformulator Class
# =============================================================================

class QueryReformulator:
    """
    Reformulates queries by resolving references from conversation history.
    
    This class detects when a query contains unresolved references (pronouns,
    demonstratives, implicit references) and reformulates it to be self-contained
    using conversation context.
    
    Key capabilities:
    1. Reference Detection: Identify queries that need reformulation
    2. Query Reformulation: Expand queries using LLM or rules
    3. Entity Extraction: Extract entities for future reference resolution
    
    Usage:
        reformulator = QueryReformulator()
        
        # Check if reformulation is needed
        needs_reform, patterns = reformulator.needs_reformulation(query, history)
        
        # Reformulate if needed
        if needs_reform:
            result = reformulator.reformulate(query, history)
            search_query = result.reformulated_query
        
        # Extract entities from Q&A
        entities = reformulator.extract_entities(query, answer)
    """

    def __init__(
            self,
            config: Optional[ReformulatorConfig] = None,
            llm_generator: Optional[Any] = None
    ):
        """
        Initialize the Query Reformulator.
        
        Args:
            config: Reformulator configuration (uses defaults if None)
            llm_generator: LLM generator instance for reformulation (optional)
        """
        self.config = config or ReformulatorConfig()
        self.llm_generator = llm_generator
        self._nlp = None  # Lazy-loaded spaCy model

        # Compile regex patterns for efficiency
        self._compiled_skip_meta_patterns = [re.compile(p, re.IGNORECASE) for p in SKIP_META_QUERY_PATTERNS]
        self._compiled_pronoun_patterns = [re.compile(p, re.IGNORECASE) for p in PRONOUN_PATTERNS]
        self._compiled_meta_patterns = [re.compile(p, re.IGNORECASE) for p in META_REFERENCE_PATTERNS]
        self._compiled_implicit_patterns = [re.compile(p, re.IGNORECASE) for p in IMPLICIT_FOLLOWUP_PATTERNS]
        self._compiled_subject_patterns = [re.compile(p) for p in EXPLICIT_SUBJECT_PATTERNS]

        logger.info("QueryReformulator initialized")
        logger.debug(f"Config: {self.config.to_dict()}")

    # =========================================================================
    # 2.1 Reference Detection
    # =========================================================================

    def needs_reformulation(
            self,
            query: str,
            conversation_history: Optional[List["ConversationTurn"]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Detect if query needs reformulation.
        
        A query needs reformulation ONLY if:
        1. It contains pronouns/references WITHOUT explicit subjects, OR
        2. It contains meta-references to the conversation itself, OR
        3. It's a very short implicit follow-up question
        
        This method avoids false positives by checking if the query is
        already self-contained (has explicit subject/entity).
        
        Args:
            query: The user query to analyze
            conversation_history: List of previous conversation turns
            
        Returns:
            Tuple of (needs_reformulation, list of detected patterns)
            
        Examples:
            >>> reformulator.needs_reformulation("What is PL?", history)
            (False, [])  # Has explicit subject "PL"
            
            >>> reformulator.needs_reformulation("Who created this?", history)
            (True, ["pronoun:this"])  # "this" without explicit subject
            
            >>> reformulator.needs_reformulation("Why was ABC created?", history)
            (False, [])  # Has explicit subject "ABC"
        """
        if not self.config.enabled:
            return False, []

        # No history = nothing to reference
        if not conversation_history:
            return False, []

        query = query.strip()
        if not query:
            return False, []

        query_lower = query.lower()
        detected_patterns = []

        # Step 0: Check for meta-queries about conversation itself (SKIP reformulation)
        # These queries ask about the conversation, not the KB, so reformulating is counterproductive
        for pattern in self._compiled_skip_meta_patterns:
            if pattern.search(query_lower):
                logger.debug(f"Query is a meta-query about conversation itself - skipping reformulation")
                return False, []

        # Step 1: Check for meta-references (almost always need reformulation)
        for pattern in self._compiled_meta_patterns:
            match = pattern.search(query_lower)
            if match:
                detected_patterns.append(f"meta:{match.group()}")

        if detected_patterns:
            logger.debug(f"Query needs reformulation: meta-references detected: {detected_patterns}")
            return True, detected_patterns

        # Step 2: Check for implicit follow-up patterns
        for pattern in self._compiled_implicit_patterns:
            match = pattern.search(query_lower)
            if match:
                detected_patterns.append(f"implicit:{match.group()}")

        if detected_patterns:
            logger.debug(f"Query needs reformulation: implicit follow-up detected: {detected_patterns}")
            return True, detected_patterns

        # Step 3: Check for pronouns WITHOUT explicit subject
        has_pronoun = False
        for pattern in self._compiled_pronoun_patterns:
            match = pattern.search(query_lower)
            if match:
                has_pronoun = True
                detected_patterns.append(f"pronoun:{match.group()}")

        if has_pronoun:
            # Check if query has explicit subject (is self-contained)
            if self._has_explicit_subject(query):
                logger.debug(f"Query has pronouns but also explicit subject - no reformulation needed")
                return False, []
            else:
                logger.debug(f"Query needs reformulation: pronouns without explicit subject: {detected_patterns}")
                return True, detected_patterns

        # No reformulation needed
        return False, []

    @staticmethod
    def _has_explicit_subject(query: str) -> bool:
        """
        Check if query has an explicit subject/entity that makes it self-contained.
        
        Returns True if query contains:
        - Proper nouns (capitalized words not at sentence start)
        - Quoted terms
        - Acronyms
        - Specific noun patterns (e.g., "the X project")
        
        Args:
            query: The query to analyze
            
        Returns:
            True if query has explicit subject, False otherwise
        """
        words = query.split()

        # Very short queries likely need context, unless they have proper nouns
        # Don't immediately reject - still check for acronyms/proper nouns below
        # if len(words) < 3:  # REMOVED - was blocking valid reformulation needs

        # Check for capitalized words (potential proper nouns)
        # Skip first word (sentence start) and common words
        for i, word in enumerate(words):
            if i == 0:
                continue  # Skip first word

            clean_word = word.strip('?.,!:;"\'-')
            if not clean_word:
                continue

            # Check if capitalized and not a skip word
            if clean_word[0].isupper() and clean_word.lower() not in SKIP_WORDS_FOR_SUBJECT:
                # Additional check: make sure it's not just mid-sentence capitalization
                # by checking if it looks like a proper noun (starts with capital, rest lowercase or mixed)
                if len(clean_word) >= 2:
                    return True

        # Check for quoted terms
        if '"' in query or "'" in query:
            # Verify there's actual content in quotes
            if re.search(r'["\'][^"\']+["\']', query):
                return True

        # Check for acronyms (2+ consecutive uppercase letters)
        if re.search(r'\b[A-Z]{2,}\b', query):
            return True

        # Check for specific patterns indicating explicit subject
        specific_patterns = [
            r'\b\w{3,}\s+(project|system|framework|model|training|data|process)\b',
            r'\b(project|system|framework|model)\s+\w{3,}\b',
        ]
        for pattern in specific_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True

        return False

    def get_detected_references(self, query: str) -> Dict[str, List[str]]:
        """
        Get detailed breakdown of detected references in a query.
        
        Args:
            query: The query to analyze
            
        Returns:
            Dict with keys 'pronouns', 'meta', 'implicit' containing matched strings
        """
        query_lower = query.lower()

        results = {
            'pronouns': [],
            'meta': [],
            'implicit': []
        }

        for pattern in self._compiled_pronoun_patterns:
            matches = pattern.findall(query_lower)
            results['pronouns'].extend(matches)

        for pattern in self._compiled_meta_patterns:
            matches = pattern.findall(query_lower)
            if matches:
                results['meta'].extend([m if isinstance(m, str) else m[0] for m in matches])

        for pattern in self._compiled_implicit_patterns:
            match = pattern.search(query_lower)
            if match:
                results['implicit'].append(match.group())

        return results

    # =========================================================================
    # 2.2 Query Reformulation
    # =========================================================================

    def reformulate(
            self,
            query: str,
            conversation_history: List["ConversationTurn"],
            force: bool = False,
            active_topic: Optional[str] = None  # NEW: Active conversation topic
    ) -> ReformulationResult:
        """
        Reformulate query by resolving references from conversation history.
        
        Uses LLM (if available) or rule-based fallback to expand the query
        by replacing pronouns and references with actual entities from context.
        
        Args:
            query: The user query to reformulate
            conversation_history: List of previous conversation turns
            force: Force reformulation even if not detected as needed
            active_topic: Current conversation topic (optional, for better context)
            
        Returns:
            ReformulationResult with reformulated query and metadata
            
        Examples:
            >>> history = [ConversationTurn(query="What is PL?", answer="PL is...")]
            >>> result = reformulator.reformulate("Who created this?", history)
            >>> print(result.reformulated_query)
            "Who created PL?"
        """
        original_query = query.strip()

        # Check if reformulation is needed
        if not force:
            needs_reform, detected_patterns = self.needs_reformulation(query, conversation_history)
            if not needs_reform:
                return ReformulationResult(
                    original_query=original_query,
                    reformulated_query=original_query,
                    was_reformulated=False,
                    detected_patterns=[],
                    method="none"
                )
        else:
            _, detected_patterns = self.needs_reformulation(query, conversation_history)

        # Try LLM reformulation first
        if self.config.use_llm and self.llm_generator is not None:
            try:
                result = self._reformulate_with_llm(
                    original_query,
                    conversation_history,
                    detected_patterns,
                    active_topic  # NEW: Pass active topic
                )

                # If we detected references but the LLM returned the query unchanged,
                # treat it as a non-reformulation and fall back to rules.
                if detected_patterns and not result.was_reformulated:
                    logger.debug(
                        "LLM returned unchanged query despite detected references; falling back to rule-based"
                    )
                elif result.confidence >= self.config.confidence_threshold:
                    logger.info(f"LLM reformulation: '{original_query}' → '{result.reformulated_query}'")
                    return result
                else:
                    logger.debug(f"LLM reformulation confidence too low ({result.confidence}), trying rule-based")
            except Exception as e:
                logger.warning(f"LLM reformulation failed: {e}")

        # Fall back to rule-based reformulation
        if self.config.fallback_to_rules or not self.config.use_llm:
            result = self._reformulate_with_rules(original_query, conversation_history, detected_patterns)
            logger.info(f"Rule-based reformulation: '{original_query}' → '{result.reformulated_query}'")
            return result

        # No reformulation possible
        return ReformulationResult(
            original_query=original_query,
            reformulated_query=original_query,
            was_reformulated=False,
            detected_patterns=detected_patterns,
            method="failed"
        )

    def _reformulate_with_llm(
            self,
            query: str,
            conversation_history: List["ConversationTurn"],
            detected_patterns: List[str],
            active_topic: Optional[str] = None  # NEW
    ) -> ReformulationResult:
        """
        Reformulate query using LLM.
        
        Builds a prompt with conversation history and asks the LLM to
        create a self-contained version of the query.
        """
        # Build reformulation prompt (with active topic if available)
        prompt = self._build_reformulation_prompt(query, conversation_history, active_topic)

        # Generate reformulated query
        reformulated = self.llm_generator.generate(
            prompt,
            max_new_tokens=self.config.max_reformulation_tokens,
            temperature=0.1,  # Low temperature for consistent output
            purpose="query_reformulation",
            stop_sequences=["\n", "Question:", "Q:"]
        )

        # Clean up the output
        reformulated = self._clean_reformulated_query(reformulated, query)

        # Calculate confidence based on output quality
        confidence = self._calculate_reformulation_confidence(query, reformulated, conversation_history)

        # Extract resolved references
        resolved = self._extract_resolved_references(query, reformulated, conversation_history)

        return ReformulationResult(
            original_query=query,
            reformulated_query=reformulated,
            was_reformulated=reformulated != query,
            detected_patterns=detected_patterns,
            resolved_references=resolved,
            confidence=confidence,
            method="llm"
        )

    def _build_reformulation_prompt(
            self,
            query: str,
            conversation_history: List["ConversationTurn"],
            active_topic: Optional[str] = None  # NEW
    ) -> str:
        """Build the prompt for LLM reformulation."""
        # Limit history to recent turns
        recent_history = conversation_history[-self.config.max_history_turns:]

        # Build history string
        history_lines = []
        for turn in recent_history:
            answer_preview = turn.answer[:self.config.max_answer_chars]
            if len(turn.answer) > self.config.max_answer_chars:
                answer_preview = answer_preview.rsplit(' ', 1)[0] + "..."

            history_lines.append(f"User: {turn.query}")
            history_lines.append(f"Assistant: {answer_preview}")

        history_str = "\n".join(history_lines)

        # NEW: Add active topic context if available
        topic_context = ""
        if active_topic:
            topic_context = f"\nCurrent conversation topic: {active_topic}\n"

        prompt = f"""Given the conversation history below, reformulate the user's latest query to be self-contained and specific. Replace pronouns and references with the actual entities they refer to.{topic_context}
Rules:
- Keep the reformulated query concise
- Only replace pronouns/references that are unclear without context
- Preserve the original intent and question type
- If the query is already clear, return it unchanged
- Output ONLY the reformulated query, nothing else

Conversation History:
{history_str}

Latest Query: {query}

Reformulated Query:"""

        return prompt

    def _reformulate_with_rules(
            self,
            query: str,
            conversation_history: List["ConversationTurn"],
            detected_patterns: List[str]
    ) -> ReformulationResult:
        """
        Reformulate query using rule-based approach.
        
        Uses simple substitution rules based on recent conversation entities.
        """
        if not conversation_history:
            return ReformulationResult(
                original_query=query,
                reformulated_query=query,
                was_reformulated=False,
                detected_patterns=detected_patterns,
                method="rule_based"
            )

        reformulated = query
        resolved = {}

        # Get recent entities from conversation
        recent_entities = self._get_recent_entities(conversation_history)
        recent_topics = self._get_recent_topics(conversation_history)

        # Primary entity (most recent main topic)
        primary_entity = recent_entities[0] if recent_entities else None
        primary_topic = recent_topics[0] if recent_topics else None

        # Replace pronouns with primary entity
        if primary_entity or primary_topic:
            replacement = primary_entity or primary_topic

            # Replace demonstrative pronouns (this, that, it)
            for pronoun in ['this', 'that', 'it']:
                pattern = rf'\b{pronoun}\b'
                if re.search(pattern, reformulated, re.IGNORECASE):
                    # Check for common patterns where pronoun refers to the main topic
                    reference_patterns = [
                        rf'\b{pronoun}(?:\s*[?.!])',  # End of sentence
                        rf'\b{pronoun}\s+(?:is|was|are|were|do|does|did|can|could|will|would|should)\b',  # Before verb
                        rf'(?:of|about|for|with|in|on|at|to|from)\s+{pronoun}\b',  # After preposition
                    ]

                    should_replace = any(re.search(p, reformulated, re.IGNORECASE) for p in reference_patterns)

                    if should_replace:
                        reformulated = re.sub(
                            rf'\b{pronoun}\b',
                            replacement,
                            reformulated,
                            count=1,
                            flags=re.IGNORECASE
                        )
                        resolved[pronoun] = replacement

            # Replace possessive pronouns (his, her, their, its)
            # These need possessive form: "his" -> "CARVE's" or "Jeremy Salvador's"
            possessive_replacement = replacement if replacement.endswith("'s") or replacement.endswith(
                "s'") else f"{replacement}'s"
            for pronoun in ['his', 'her', 'their', 'its']:
                pattern = rf'\b{pronoun}\b'
                if re.search(pattern, reformulated, re.IGNORECASE):
                    # Possessive pronouns appear before nouns: "his accomplishments", "her work"
                    if re.search(rf'\b{pronoun}\s+\w+', reformulated, re.IGNORECASE):
                        reformulated = re.sub(
                            rf'\b{pronoun}\b',
                            possessive_replacement,
                            reformulated,
                            count=1,
                            flags=re.IGNORECASE
                        )
                        resolved[pronoun] = possessive_replacement

            # Replace subject/object pronouns (he/she/him/them/they)
            # Example: "What were the charges over him?" -> "... over Jeffrey Epstein?"
            for pronoun in ['he', 'she', 'they', 'them', 'him']:
                pattern = rf'\b{pronoun}\b'
                if re.search(pattern, reformulated, re.IGNORECASE):
                    # Prefer replacing when it appears in common reference positions
                    reference_patterns = [
                        rf'\b{pronoun}(?:\s*[?.!])',
                        rf'\b{pronoun}\s+(?:is|was|are|were|did|does|do|can|could|will|would|should|has|have|had)\b',
                        rf'(?:of|about|for|with|in|on|at|to|from|over|under|against|by)\s+{pronoun}\b',
                    ]
                    should_replace = any(re.search(p, reformulated, re.IGNORECASE) for p in reference_patterns)
                    if should_replace:
                        reformulated = re.sub(
                            pattern,
                            replacement,
                            reformulated,
                            count=1,
                            flags=re.IGNORECASE
                        )
                        resolved[pronoun] = replacement

        # Handle meta-references to conversation
        if any('meta:' in p for p in detected_patterns):
            # For "what I asked" type queries, provide context about the conversation
            if conversation_history:
                last_query = conversation_history[-1].query
                last_topic = primary_topic or primary_entity or "that topic"

                # Build context-rich replacements
                meta_patterns_to_replace = [
                    (r'what (did )?i (just )?ask(ed)?( about)?( previously)?\??', f'the question: "{last_query}"'),
                    (r'(my|the) (last|previous|earlier) (question|query)', f'the question: "{last_query}"'),
                    (r'what (i|we) (said|mentioned|talked about)', f'{last_topic}'),
                    (r'what (was|were) (i|we) (talking|discussing) about', f'{last_topic}'),
                ]

                for pattern, replacement_text in meta_patterns_to_replace:
                    if re.search(pattern, reformulated, re.IGNORECASE):
                        reformulated = re.sub(pattern, replacement_text, reformulated, flags=re.IGNORECASE)
                        resolved[pattern] = replacement_text
                        break

        # Handle implicit follow-ups like "Why?" or "How?"
        if re.match(r'^(why|how|who|when|where)\??\s*$', reformulated, re.IGNORECASE):
            if primary_topic or primary_entity:
                word = reformulated.strip('? ')
                reformulated = f"{word} {primary_topic or primary_entity}?"
                resolved['implicit'] = primary_topic or primary_entity

        # Handle "tell me more" / "explain more" / "describe more" / "elaborate" / "continue" etc.
        if (re.search(r'^(tell|give|show|explain)\s*(me\s*)?(more|further)\b', reformulated, re.IGNORECASE) or
                re.search(r'^(describe|elaborate|expand)\s*(on\s*)?(this|that|it)?\s*(more|further)?\b', reformulated,
                          re.IGNORECASE) or
                re.match(r'^more\b', reformulated, re.IGNORECASE) or
                re.match(r'^continue\b', reformulated, re.IGNORECASE) or
                re.match(r'^go on\b', reformulated, re.IGNORECASE)):
            if primary_topic or primary_entity:
                # Use a natural reformulation
                reformulated = f"Tell me more about {primary_topic or primary_entity}"
                resolved['implicit_continuation'] = primary_topic or primary_entity

        # Handle formatting requests: "Give me this info in points"
        if re.search(r'\b(in|as)\s+(points|bullets|list|steps|summary)\b', reformulated, re.IGNORECASE):
            # Get the last answer's topic
            if conversation_history:
                last_turn = conversation_history[-1]
                last_answer_topic = primary_topic or primary_entity or "the information"
                # Rewrite to make it explicit
                reformulated = re.sub(
                    r'\b(this|that|it|the)\s+(info|information|answer|content)\b',
                    f"{last_answer_topic}",
                    reformulated,
                    flags=re.IGNORECASE
                )
                # If still ambiguous, prepend context
                if re.search(r'^(give|show|present|format|list)', reformulated, re.IGNORECASE):
                    reformulated = f"Provide information about {last_answer_topic} in bullet points"
                    resolved['formatting_request'] = last_answer_topic

        was_reformulated = reformulated != query

        return ReformulationResult(
            original_query=query,
            reformulated_query=reformulated,
            was_reformulated=was_reformulated,
            detected_patterns=detected_patterns,
            resolved_references=resolved,
            confidence=0.7 if was_reformulated else 1.0,
            method="rule_based"
        )

    def _get_recent_entities(
            self,
            conversation_history: List["ConversationTurn"],
            max_entities: int = 5
    ) -> List[str]:
        """Get entities from recent conversation, most recent first."""
        entities = []
        for turn in reversed(conversation_history):
            for entity in turn.entities_mentioned:
                if entity not in entities:
                    entities.append(entity)
                    if len(entities) >= max_entities:
                        return entities
        return entities

    @staticmethod
    def _get_recent_topics(
            conversation_history: List["ConversationTurn"],
            max_topics: int = 3
    ) -> List[str]:
        """Extract main topics from recent queries."""
        topics = []

        for turn in reversed(conversation_history):
            # Try to extract main noun phrase from query
            query = turn.query

            # Look for "What is X" pattern
            match = re.search(r'what (?:is|are|was|were) (?:the |a |an )?(.+?)(?:\?|$)', query, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                if topic and topic not in topics:
                    topics.append(topic)
                    continue

            # Look for capitalized terms
            caps = re.findall(r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b', query)
            for cap in caps:
                if cap.lower() not in SKIP_WORDS_FOR_SUBJECT and cap not in topics:
                    topics.append(cap)
                    break

            if len(topics) >= max_topics:
                break

        return topics

    @staticmethod
    def _clean_reformulated_query(reformulated: str, original: str) -> str:
        """Clean up LLM output to get just the reformulated query."""
        # Remove common prefixes
        reformulated = reformulated.strip()

        prefixes_to_remove = [
            "Reformulated Query:",
            "Reformulated:",
            "Query:",
            "Here is the reformulated query:",
            "The reformulated query is:",
        ]

        for prefix in prefixes_to_remove:
            if reformulated.lower().startswith(prefix.lower()):
                reformulated = reformulated[len(prefix):].strip()

        # Remove quotes if wrapped
        if reformulated.startswith('"') and reformulated.endswith('"'):
            reformulated = reformulated[1:-1]
        if reformulated.startswith("'") and reformulated.endswith("'"):
            reformulated = reformulated[1:-1]

        # If empty or too different, return original
        if not reformulated or len(reformulated) > len(original) * 3:
            return original

        return reformulated.strip()

    def _calculate_reformulation_confidence(
            self,
            original: str,
            reformulated: str,
            history: List["ConversationTurn"]
    ) -> float:
        """Calculate confidence score for reformulation quality."""
        if reformulated == original:
            return 1.0

        confidence = 0.8  # Base confidence

        # Boost if reformulated query contains entities from history
        history_entities = set()
        for turn in history:
            history_entities.update(e.lower() for e in turn.entities_mentioned)

        reformulated_lower = reformulated.lower()
        for entity in history_entities:
            if entity in reformulated_lower and entity not in original.lower():
                confidence += 0.1

        # Penalize if too long or too short
        length_ratio = len(reformulated) / len(original) if original else 1
        if length_ratio > 2.0 or length_ratio < 0.5:
            confidence -= 0.2

        return min(max(confidence, 0.0), 1.0)

    @staticmethod
    def _extract_resolved_references(
            original: str,
            reformulated: str,
            history: List["ConversationTurn"]
    ) -> Dict[str, str]:
        """Extract which references were resolved to which entities."""
        resolved = {}

        # Find pronouns in original that are not in reformulated
        original_lower = original.lower()
        reformulated_lower = reformulated.lower()

        pronouns = ['this', 'that', 'it', 'these', 'those', 'he', 'she', 'they', 'them', 'him', 'his', 'her', 'their',
                    'its']
        for pronoun in pronouns:
            if pronoun in original_lower and pronoun not in reformulated_lower:
                # Find what replaced it (approximate)
                history_entities = []
                for turn in history:
                    history_entities.extend(turn.entities_mentioned)

                for entity in history_entities:
                    if entity.lower() in reformulated_lower:
                        resolved[pronoun] = entity
                        break

        return resolved

    # =========================================================================
    # 2.3 Entity Extraction
    # =========================================================================

    def extract_entities(
            self,
            query: str,
            answer: Optional[str] = None,
            use_spacy: bool = True
    ) -> List[str]:
        """
        Extract key entities from query and optionally answer.
        
        Uses multiple methods:
        1. spaCy NER for named entities (if available)
        2. Pattern matching for proper nouns and acronyms
        3. Noun phrase extraction

        Args:
            query: The user query
            answer: Optional model answer to also extract from
            use_spacy: Whether to use spaCy for extraction (default: True)
            
        Returns:
            List of unique entity strings, ordered by importance
            
        Examples:
            >>> entities = reformulator.extract_entities(
            ...     "What is PL?",
            ...     "PL is a framework developed by ABC..."
            ... )
            >>> print(entities)
            ["PL", "ABC", "framework"]
        """
        entities = []
        seen = set()

        def add_entity(entity: str):
            """Add entity if not seen."""
            entity = entity.strip()
            if entity and entity.lower() not in seen and len(entity) >= 2:
                entities.append(entity)
                seen.add(entity.lower())

        # Process query first (higher priority)
        query_entities = self._extract_entities_from_text(query, use_spacy)
        for entity in query_entities:
            add_entity(entity)

        # Process answer if provided
        if answer and self.config.extract_entities_from_answer:
            answer_entities = self._extract_entities_from_text(answer, use_spacy)
            for entity in answer_entities:
                add_entity(entity)

        return entities

    def _extract_entities_from_text(self, text: str, use_spacy: bool = True) -> List[str]:
        """Extract entities from a single text."""
        entities = []

        # Method 1: spaCy NER (if available and enabled)
        if use_spacy:
            spacy_entities = self._extract_with_spacy(text)
            entities.extend(spacy_entities)

        # Method 2: Pattern matching for proper nouns
        pattern_entities = self._extract_with_patterns(text)
        entities.extend(pattern_entities)

        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                unique_entities.append(entity)
                seen.add(entity.lower())

        return unique_entities

    def _extract_with_spacy(self, text: str) -> List[str]:
        """Extract entities using spaCy NER."""
        try:
            # Lazy load spaCy model
            if self._nlp is None:
                import spacy
                try:
                    self._nlp = spacy.load(self.config.spacy_model)
                    logger.debug(f"Loaded spaCy model: {self.config.spacy_model}")
                except OSError:
                    logger.warning(
                        f"spaCy model '{self.config.spacy_model}' not found. "
                        "Install with: python -m spacy download en_core_web_sm"
                    )
                    return []

            doc = self._nlp(text)
            entities = []

            # Extract named entities
            for ent in doc.ents:
                # Skip certain entity types that are less useful for reference resolution
                if ent.label_ not in ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    entities.append(ent.text)

            return entities

        except ImportError:
            logger.debug("spaCy not available for entity extraction")
            return []
        except Exception as e:
            logger.warning(f"spaCy entity extraction failed: {e}")
            return []

    def _extract_with_patterns(self, text: str) -> List[str]:
        """Extract entities using regex patterns."""
        entities = []
        # Extract acronyms (2+ uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        entities.extend(acronyms)

        # Extract camelCase technical terms (e.g., ConversationalPromptBuilder)
        camel_case_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
        entities.extend(camel_case_terms)

        # Extract proper nouns (capitalized words, excluding sentence starts)
        words = text.split()
        for i, word in enumerate(words):
            clean_word = word.strip('?.,!:;"\'-()[]')
            if not clean_word:
                continue

            # Skip first word unless it's special
            if i == 0:
                # Include if it's an acronym, all caps, or camelCase
                if (clean_word.isupper() and len(clean_word) >= 2) or \
                        re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]+)+$', clean_word):
                    entities.append(clean_word)
                continue

            # Include capitalized words that aren't common words
            if clean_word[0].isupper() and clean_word.lower() not in SKIP_WORDS_FOR_SUBJECT:
                entities.append(clean_word)

        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
        for match in quoted:
            term = match[0] or match[1]
            if term:
                entities.append(term)

        # Extract "X project/system/framework" patterns
        pattern_matches = re.findall(
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s+(?:project|system|framework|model)\b',
            text
        )
        entities.extend(pattern_matches)

        return entities

    def extract_entities_batch(
            self,
            query_answer_pairs: List[Tuple[str, str]]
    ) -> List[List[str]]:
        """
        Extract entities from multiple query-answer pairs.
        
        Args:
            query_answer_pairs: List of (query, answer) tuples
            
        Returns:
            List of entity lists, one per input pair
        """
        return [
            self.extract_entities(query, answer)
            for query, answer in query_answer_pairs
        ]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def set_llm_generator(self, llm_generator: Any) -> None:
        """
        Set the LLM generator for reformulation.
        
        Args:
            llm_generator: LLM generator instance with generate() method
        """
        self.llm_generator = llm_generator
        logger.info("LLM generator set for query reformulation")

    def get_config(self) -> Dict[str, Any]:
        """Get reformulator configuration."""
        return self.config.to_dict()

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config: {key}={value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"QueryReformulator(enabled={self.config.enabled}, "
            f"use_llm={self.config.use_llm}, "
            f"has_llm={self.llm_generator is not None})"
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_query_reformulator(
        use_llm: bool = True,
        llm_generator: Optional[Any] = None,
        max_history_turns: int = 5,
        **kwargs
) -> QueryReformulator:
    """
    Factory function to create a QueryReformulator instance.
    
    Args:
        use_llm: Use LLM for reformulation (default: True)
        llm_generator: LLM generator instance (optional)
        max_history_turns: Max conversation turns for context
        **kwargs: Additional ReformulatorConfig parameters
        
    Returns:
        Configured QueryReformulator instance
    """
    config = ReformulatorConfig(
        use_llm=use_llm,
        max_history_turns=max_history_turns,
        **kwargs
    )

    return QueryReformulator(config=config, llm_generator=llm_generator)
