"""
Follow-Up Query Detector for Conversational RAG.

This module detects and classifies follow-up queries in conversation,
enabling the system to handle affirmative continuations like "Yes please!",
clarification requests, and coreference queries.

Features:
- Pattern-based detection for affirmative/negative responses
- Pending offer awareness for continuation handling
- Clarification and coreference detection
- Lightweight and fast (no LLM calls)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FollowUpResult:
    """
    Result of follow-up query detection.
    
    Attributes:
        is_follow_up: Whether the query is a follow-up
        follow_up_type: Type of follow-up detected
        expansion_topic: Topic to expand on (from pending offer)
        expansion_entities: Entities to focus on (from pending offer)
        confidence: Confidence score (0.0-1.0)
        detected_patterns: List of patterns that matched
    """
    is_follow_up: bool = False
    follow_up_type: str = "NEW_QUERY"  # NEW_QUERY, AFFIRMATIVE_CONTINUATION, NEGATIVE, CLARIFICATION, COREFERENCE
    expansion_topic: Optional[str] = None
    expansion_entities: List[str] = field(default_factory=list)
    confidence: float = 0.0
    detected_patterns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_follow_up": self.is_follow_up,
            "follow_up_type": self.follow_up_type,
            "expansion_topic": self.expansion_topic,
            "expansion_entities": self.expansion_entities,
            "confidence": self.confidence,
            "detected_patterns": self.detected_patterns
        }


# =============================================================================
# Main FollowUpDetector Class
# =============================================================================

class FollowUpDetector:
    """
    Detects and classifies follow-up queries in conversation.
    
    This class uses pattern matching to identify when a user query is:
    - An affirmative continuation ("yes please", "tell me more")
    - A negative response ("no thanks", "skip")
    - A clarification request ("what do you mean", "explain that")
    - A coreference query (contains pronouns like "it", "that", "this")
    
    Example:
        ```python
        detector = FollowUpDetector()
        
        # Last turn had pending offer: "Would you like to know more about X?"
        result = detector.detect("Yes please!", last_turn)
        
        if result.is_follow_up and result.follow_up_type == "AFFIRMATIVE_CONTINUATION":
            # Expand on the pending offer topic
            search_query = f"{result.expansion_topic} detailed information"
        ```
    """

    def __init__(self):
        """Initialize the follow-up detector with pattern definitions."""

        # Affirmative continuation patterns
        self.affirmative_patterns = [
            'yes', 'yes please', 'yes!', 'yeah', 'yep', 'yup',
            'sure', 'sure!', 'of course', 'absolutely',
            'go ahead', 'please do', 'please',
            'tell me more', 'tell me', 'continue', 'go on',
            "i'd like that", "i would", "i'd love to",
            'sounds good', 'that would be great', 'that would be helpful',
            'more details', 'more info', 'more information',
            'give me details', 'please give me details', 'can you give me details',
            'could you please give me details',
            'tell again'
        ]

        # Negative continuation patterns
        self.negative_patterns = [
            'no', 'no thanks', 'no thank you', 'nope', 'nah',
            'not now', 'not really', 'skip', 'skip it',
            "that's enough", "that's all", "i'm good",
            'maybe later', 'not interested'
        ]

        # Clarification request patterns
        self.clarification_patterns = [
            'what do you mean', 'what does that mean',
            'clarify', 'can you clarify', 'please clarify',
            'explain', 'explain that', 'can you explain',
            'what about', 'how about', 'what else',
            "i don't understand", "i can't understand", "i cant understand", "i'm confused",
            "i don't quite follow", "i am not sure i follow", "i'm not sure i follow",
            "i'm lost", "im lost", "i am lost",
            "that didn't make sense", "that did not make sense",
            "can you rephrase", "rephrase that",
            "put that differently", "say that another way",
            "can you clarify that for me", "what do you mean by that",
            "i'm missing something", "im missing something", "i am missing something",
            'could you elaborate', 'elaborate',
            'give me details', 'please give me details', 'can you give me details',
            'could you please give me details',
            'give me easy words', 'easy words', 'simple words',
            'tell again',
            'please be concise', 'be concise', 'be brief',
            'keep it concise', 'keep it brief',
            'make it concise', 'make it shorter',
            'shorter please', 'brief please',
            'please make it shorter', 'please make it concise',
            'please keep it concise', 'please keep it brief',
            'short version', 'concise version', 'brief version',
            'condense it', 'please condense it',
            'summarize it', 'summarise it'
        ]

        # Pronouns that indicate coreference
        self.coreference_pronouns = [
            'it', 'this', 'that', 'these', 'those',
            'he', 'she', 'they', 'him', 'her', 'them',
            'his', 'hers', 'their', 'theirs'
        ]

        logger.info("FollowUpDetector initialized")

    def detect(
            self,
            query: str,
            last_turn: Optional[Any] = None
    ) -> FollowUpResult:
        """
        Detect if query is a follow-up and classify its type.
        
        Args:
            query: Current user query
            last_turn: Last conversation turn (ConversationTurn object or None)
            
        Returns:
            FollowUpResult with detection results
        """
        if not query or not query.strip():
            return FollowUpResult(is_follow_up=False, follow_up_type="NEW_QUERY")

        query_lower = query.lower().strip()
        query_words = query_lower.split()

        # Check for affirmative continuation (highest priority if pending offer exists)
        if last_turn and self._has_pending_offer(last_turn):
            affirmative_match = self._check_affirmative(query_lower)
            if affirmative_match:
                pending_offer = self._get_pending_offer(last_turn)
                return FollowUpResult(
                    is_follow_up=True,
                    follow_up_type="AFFIRMATIVE_CONTINUATION",
                    expansion_topic=pending_offer.get('topic', 'the previous topic'),
                    expansion_entities=pending_offer.get('entities', []),
                    confidence=0.95,
                    detected_patterns=affirmative_match
                )

            # Check for negative response
            negative_match = self._check_negative(query_lower)
            if negative_match:
                return FollowUpResult(
                    is_follow_up=True,
                    follow_up_type="NEGATIVE",
                    confidence=0.90,
                    detected_patterns=negative_match
                )

        # Check for clarification request (requires conversation history)
        if last_turn:
            clarification_match = self._check_clarification(query_lower)
            if clarification_match:
                return FollowUpResult(
                    is_follow_up=True,
                    follow_up_type="CLARIFICATION",
                    confidence=0.85,
                    detected_patterns=clarification_match
                )

        # Check for coreference (pronouns without explicit subjects)
        if last_turn and len(query_words) <= 15:  # Short queries more likely to be coreference
            coreference_match = self._check_coreference(query_lower, query_words)
            if coreference_match:
                return FollowUpResult(
                    is_follow_up=True,
                    follow_up_type="COREFERENCE",
                    confidence=0.75,
                    detected_patterns=coreference_match
                )

        # Not a follow-up
        return FollowUpResult(
            is_follow_up=False,
            follow_up_type="NEW_QUERY",
            confidence=1.0
        )

    def _has_pending_offer(self, last_turn: Any) -> bool:
        """Check if last turn has a pending offer."""
        if hasattr(last_turn, 'pending_offer'):
            return last_turn.pending_offer is not None
        return False

    def _get_pending_offer(self, last_turn: Any) -> Dict[str, Any]:
        """Get pending offer from last turn."""
        if hasattr(last_turn, 'pending_offer') and last_turn.pending_offer:
            return last_turn.pending_offer
        return {}

    def _check_affirmative(self, query_lower: str) -> List[str]:
        """Check for affirmative patterns."""
        matched = []
        for pattern in self.affirmative_patterns:
            if pattern in query_lower:
                matched.append(pattern)
        return matched

    def _check_negative(self, query_lower: str) -> List[str]:
        """Check for negative patterns."""
        matched = []
        for pattern in self.negative_patterns:
            if pattern in query_lower:
                matched.append(pattern)
        return matched

    def _check_clarification(self, query_lower: str) -> List[str]:
        """Check for clarification request patterns."""
        matched = []
        for pattern in self.clarification_patterns:
            if pattern in query_lower:
                matched.append(pattern)
        return matched

    def _check_coreference(self, query_lower: str, query_words: List[str]) -> List[str]:
        """Check for coreference pronouns."""
        matched = []
        for pronoun in self.coreference_pronouns:
            if pronoun in query_words:
                matched.append(pronoun)
        return matched

    def is_standalone_query(
            self,
            query: str,
            conversation_history: Optional[List[Any]] = None
    ) -> bool:
        """
        Determine if a query is standalone (cacheable) or a follow-up (not cacheable).
        
        A query is considered STANDALONE (cacheable) if:
        - It's a self-contained question with a clear subject
        - It doesn't contain follow-up phrases ("tell me more", "what else")
        - It doesn't rely on pronouns without clear antecedents
        - It's not a continuation request
        - No conversation history exists, OR it's a completely new topic
        
        A query is considered FOLLOW-UP (not cacheable) if:
        - Contains follow-up phrases ("tell me more", "what else", "continue")
        - Contains pronouns that reference previous context ("it", "that", "this")
        - Is an affirmative/negative response to a pending offer
        - Is a clarification request
        
        Args:
            query: Query text to classify
            conversation_history: Optional conversation history (list of ConversationTurn)
            
        Returns:
            True if standalone (cacheable), False if follow-up (not cacheable)
            
        Examples:
            >>> detector.is_standalone_query("What is CARVE?", [])
            True  # Self-contained, no history
            
            >>> detector.is_standalone_query("Tell me more", history)
            False  # Follow-up phrase
            
            >>> detector.is_standalone_query("What else?", history)
            False  # Follow-up phrase
            
            >>> detector.is_standalone_query("What is it?", history)
            False  # Pronoun reference
            
            >>> detector.is_standalone_query("What is the definition of custody?", history)
            True  # Self-contained even with history
        """
        if not query or not query.strip():
            return False

        query_lower = query.lower().strip()
        # Remove punctuation before splitting to properly detect pronouns
        import string
        query_lower = query_lower.translate(str.maketrans('', '', string.punctuation))
        query_words = query_lower.split()

        # Get last turn if available
        last_turn = conversation_history[-1] if conversation_history else None

        # RULE 1: Check for explicit follow-up phrases (NOT standalone)
        follow_up_phrases = [
            'tell me more', 'tell me', 'more about', 'what else',
            'continue', 'go on', 'keep going', 'elaborate',
            'explain more', 'more details', 'more info', 'more information',
            'go deeper', 'go into more detail', 'expand on', 'elaborate on',
            'more context', 'walk me through', 'unpack', 'drill down on',
            'give me details', 'please give me details', 'can you give me details',
            'could you please give me details',
            'give me easy words', 'easy words', 'simple words',
            'tell again',
            'please be concise', 'be concise', 'be brief',
            'keep it concise', 'keep it brief',
            'make it concise', 'make it shorter',
            'shorter please', 'brief please',
            'please make it shorter', 'please make it concise',
            'please keep it concise', 'please keep it brief',
            'short version', 'concise version', 'brief version',
            'condense it', 'please condense it',
            'summarize it', 'summarise it',
            'what about', 'how about', 'and what', 'what then',
            'anything else', 'something else'
        ]

        for phrase in follow_up_phrases:
            if phrase in query_lower:
                logger.debug(f"Query contains follow-up phrase '{phrase}': NOT standalone")
                return False

        # RULE 2: Check for affirmative/negative responses (NOT standalone)
        if last_turn and self._has_pending_offer(last_turn):
            affirmative_match = self._check_affirmative(query_lower)
            negative_match = self._check_negative(query_lower)

            if affirmative_match or negative_match:
                logger.debug(f"Query is affirmative/negative response: NOT standalone")
                return False

        # RULE 3: Check for clarification requests (NOT standalone)
        # Only check for clarification phrases that indicate reference to previous content
        clarification_phrases = [
            'what do you mean', 'what does that mean',
            'clarify', 'can you clarify', 'please clarify',
            'explain that', 'can you explain that', 'explain it',
            "i don't understand", "i can't understand", "i cant understand", "i'm confused",
            "i don't quite follow", "i am not sure i follow", "i'm not sure i follow",
            "i'm lost", "im lost", "i am lost",
            "that didn't make sense", "that did not make sense",
            "can you rephrase", "rephrase that",
            "put that differently", "say that another way",
            "can you clarify that for me", "what do you mean by that",
            "i'm missing something", "im missing something", "i am missing something",
            'could you elaborate', 'elaborate on',
            'give me details', 'please give me details', 'can you give me details',
            'could you please give me details',
            'give me easy words', 'easy words', 'simple words',
            'tell again',
            'please be concise', 'be concise', 'be brief',
            'keep it concise', 'keep it brief',
            'make it concise', 'make it shorter',
            'shorter please', 'brief please',
            'please make it shorter', 'please make it concise',
            'please keep it concise', 'please keep it brief',
            'short version', 'concise version', 'brief version',
            'condense it', 'please condense it',
            'summarize it', 'summarise it'
        ]

        for phrase in clarification_phrases:
            if phrase in query_lower:
                logger.debug(f"Query is clarification request: NOT standalone")
                return False

        # RULE 4: Check for pronouns without clear antecedents (NOT standalone if conversation exists)
        # Only check if conversation history exists
        if conversation_history and len(conversation_history) > 0:
            # Check for pronouns at the start of the query (strong indicator of reference)
            start_pronouns = ['it', 'this', 'that', 'these', 'those', 'he', 'she', 'they']
            first_word = query_words[0] if query_words else ""

            if first_word in start_pronouns:
                logger.debug(f"Query starts with pronoun '{first_word}': NOT standalone")
                return False

            # Check for "what is it" pattern specifically
            if len(query_words) >= 3:
                # Pattern: "what is it", "what is that", "how does it work", etc.
                if query_words[0] in ['what', 'how', 'where', 'when', 'why']:
                    # Check if there's a pronoun in positions 2-3 without a clear noun
                    for i in range(1, min(4, len(query_words))):
                        if query_words[i] in ['it', 'this', 'that', 'these', 'those']:
                            # Check if there's a noun after the pronoun
                            has_noun_after = False
                            for j in range(i + 1, len(query_words)):
                                if self._is_likely_noun(query_words[j]):
                                    has_noun_after = True
                                    break

                            # If no noun after OR pronoun is at the end, it's a reference
                            if not has_noun_after or i == len(query_words) - 1:
                                logger.debug(
                                    f"Query has pronoun '{query_words[i]}' without clear subject: NOT standalone")
                                return False

            # Also check for pronouns at the end of short queries
            # e.g., "Tell me about this", "Explain those"
            if len(query_words) <= 6:
                last_word = query_words[-1] if query_words else ""
                if last_word in ['it', 'this', 'that', 'these', 'those', 'them']:
                    logger.debug(f"Query ends with pronoun '{last_word}': NOT standalone")
                    return False

            # Check for pronouns in short queries (likely referencing previous context)
            if len(query_words) <= 10:
                coreference_match = self._check_coreference(query_lower, query_words)
                if coreference_match:
                    # Check if the query has a clear subject (noun) to anchor the pronoun
                    # If no clear subject, it's likely referencing previous context
                    has_clear_subject = self._has_clear_subject(query_lower, query_words)
                    if not has_clear_subject:
                        logger.debug(f"Query has pronouns without clear subject: NOT standalone")
                        return False

        # RULE 5: Check for continuation words at the start (NOT standalone)
        continuation_starters = ['also', 'additionally', 'furthermore', 'moreover', 'besides']
        if query_words and query_words[0] in continuation_starters:
            logger.debug(f"Query starts with continuation word: NOT standalone")
            return False

        # If none of the follow-up indicators matched, it's a standalone query
        logger.debug(f"Query is standalone (cacheable)")
        return True

    def _is_likely_noun(self, word: str) -> bool:
        """
        Check if a word is likely a noun (simple heuristic).
        
        Args:
            word: Word to check
            
        Returns:
            True if likely a noun
        """
        # Common question words and verbs to exclude
        non_nouns = [
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'do', 'does', 'did', 'done', 'doing',
            'have', 'has', 'had', 'having',
            'can', 'could', 'will', 'would', 'should', 'shall',
            'may', 'might', 'must',
            'the', 'a', 'an', 'and', 'or', 'but',
            'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'what', 'how', 'where', 'when', 'why', 'which', 'who'
        ]

        return word not in non_nouns and len(word) > 2

    def _has_clear_subject(self, query_lower: str, query_words: List[str]) -> bool:
        """
        Check if query has a clear subject (noun/entity) that anchors pronouns.
        
        This is a simple heuristic: if the query contains question words followed
        by nouns/entities, it likely has a clear subject.
        
        Args:
            query_lower: Lowercase query
            query_words: Query words
            
        Returns:
            True if query has clear subject, False otherwise
        """
        # Question words that typically introduce subjects
        question_words = ['who', 'what', 'where', 'when', 'why', 'how', 'which']

        # Common nouns/entities that indicate clear subjects
        subject_indicators = [
            'definition', 'meaning', 'purpose', 'law', 'rule', 'statute',
            'court', 'judge', 'case', 'document', 'file', 'person',
            'child', 'parent', 'custody', 'support', 'divorce', 'property',
            'rights', 'obligations', 'process', 'procedure', 'requirement'
        ]

        # Check if query has question word + subject indicator
        has_question_word = any(word in query_words for word in question_words)
        has_subject_indicator = any(indicator in query_lower for indicator in subject_indicators)

        return has_question_word and has_subject_indicator

    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "affirmative_patterns": len(self.affirmative_patterns),
            "negative_patterns": len(self.negative_patterns),
            "clarification_patterns": len(self.clarification_patterns),
            "coreference_pronouns": len(self.coreference_pronouns)
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_follow_up_detector() -> FollowUpDetector:
    """
    Factory function to create a FollowUpDetector instance.
    
    Returns:
        Configured FollowUpDetector instance
    """
    return FollowUpDetector()
