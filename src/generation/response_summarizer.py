"""
Response Summarizer for Conversational RAG.

This module provides response summarization and pending offer detection
to enable efficient conversation context management.

Features:
- Token-efficient response summarization (1-2 sentences)
- Pending offer detection ("Would you like to know more?")
- Entity extraction from responses
- Configurable summarization thresholds
"""

import re
import logging
from typing import Optional, List, Dict, Any

from .answer_sanitizer import sanitize_generated_answer

logger = logging.getLogger(__name__)


# =============================================================================
# Main ResponseSummarizer Class
# =============================================================================

class ResponseSummarizer:
    """
    Summarizes assistant responses and detects pending offers.
    
    This class creates compact summaries of assistant responses for efficient
    context inclusion in future prompts, and detects when the assistant offers
    to provide more information.
    
    Example:
        ```python
        summarizer = ResponseSummarizer(llm_generator=llm)
        
        response = "X was CEO when Y was CTO in 2015. Would you like to know more?"
        summary = summarizer.summarize(response)
        # Returns: "X was CEO when Y was CTO in 2015."
        
        offer = summarizer.extract_pending_offer(response, entities=["X", "Y"])
        # Returns: {"topic": "X and Y meeting", "entities": ["X", "Y"]}
        ```
    """
    
    def __init__(
        self,
        llm_generator: Optional[Any] = None,
        min_length_for_summary: int = 150,
        max_summary_tokens: int = 100
    ):
        """
        Initialize the response summarizer.
        
        Args:
            llm_generator: LLM generator instance for summarization (optional)
            min_length_for_summary: Minimum response length to trigger summarization
            max_summary_tokens: Maximum tokens for summary generation
        """
        self.llm_generator = llm_generator
        self.min_length_for_summary = min_length_for_summary
        self.max_summary_tokens = max_summary_tokens
        
        # Patterns for detecting pending offers
        self.offer_patterns = [
            # Explicit offers
            r'would you like to know more',
            r'would you like me to',
            r'would you like details',
            r'would you like to hear',
            r'shall i explain',
            r'shall i provide',
            r'want me to elaborate',
            r'want to know more',
            r'interested in',
            r'let me know if you',
            r'if you.*d like.*more',
            r'feel free to ask',
            r'happy to provide',
            r'can provide more',
            r'more information.*available',
            # Subtle/conditional offers
            r'if needed.*clarification',
            r'if needed.*further',
            r'could assist',
            r'could help',
            r'further clarification',
            r'further details',
            r'further information',
            r'additional information',
            r'additional details',
            r'more details',
            r'if you need',
            r'if you require',
            r'should you need',
            r'should you require',
            r'for more information',
            r'to learn more',
            r'for further',
        ]
        
        logger.info(f"ResponseSummarizer initialized (llm_available={llm_generator is not None})")
    
    def summarize(self, response: str) -> str:
        """
        Create a compact summary of the assistant response.
        
        For short responses (<150 chars), returns the response as-is.
        For longer responses, creates a 1-2 sentence summary using LLM if available,
        or falls back to extractive summarization.
        
        Args:
            response: The assistant's response text
            
        Returns:
            Summarized response (or original if short)
        """
        if not response or not response.strip():
            return ""

        response = sanitize_generated_answer(response)
        
        # Short responses don't need summarization
        if len(response) < self.min_length_for_summary:
            return response.strip()
        
        # Use LLM summarization if available
        if self.llm_generator:
            try:
                summary = self._summarize_with_llm(response)
                if summary and len(summary) < len(response):
                    logger.debug(f"LLM summary: {len(response)} chars -> {len(summary)} chars")
                    return summary
            except Exception as e:
                logger.warning(f"LLM summarization failed: {e}, falling back to extractive")
        
        # Fallback to extractive summarization
        summary = self._extractive_summarize(response)
        logger.debug(f"Extractive summary: {len(response)} chars -> {len(summary)} chars")
        return summary
    
    def _summarize_with_llm(self, response: str) -> str:
        """Summarize using LLM."""
        prompt = f"""Summarize this assistant response in 1-2 concise sentences.
Focus on FACTS, ENTITIES, and KEY POINTS. Omit pleasantries and offers for more information.

Response:
{response}

Concise summary (1-2 sentences, facts only):"""
        
        summary = self.llm_generator.generate(
            prompt,
            max_new_tokens=self.max_summary_tokens,
            temperature=0.1,
            purpose="answer_summarization",
            stop_sequences=["\\n\\n", "Response:", "Summary:"]
        )
        
        return summary.strip()
    
    def _extractive_summarize(self, response: str, max_sentences: int = 2) -> str:
        """
        Fallback extractive summarization.
        
        Takes the first N sentences that don't contain offer patterns.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+\s+', response)
        
        # Filter out sentences with offer patterns
        factual_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Skip if contains offer pattern
            has_offer = any(
                re.search(pattern, sentence.lower())
                for pattern in self.offer_patterns
            )
            
            if not has_offer and len(factual_sentences) < max_sentences:
                factual_sentences.append(sentence)
        
        if factual_sentences:
            summary = '. '.join(factual_sentences)
            # Ensure it ends with punctuation
            if summary and summary[-1] not in '.!?':
                summary += '.'
            return summary
        
        # Fallback: take first sentence
        if sentences:
            first = sentences[0].strip()
            if first and first[-1] not in '.!?':
                first += '.'
            return first
        
        return response[:200] + "..." if len(response) > 200 else response
    
    def extract_pending_offer(
        self,
        response: str,
        entities: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if response contains an offer to provide more information.
        
        Args:
            response: The assistant's response text
            entities: List of entities mentioned in the response
            
        Returns:
            Dict with 'topic' and 'entities' if offer detected, None otherwise
        """
        if not response:
            return None

        response = sanitize_generated_answer(response)
        
        response_lower = response.lower()
        
        # Check for offer patterns
        for pattern in self.offer_patterns:
            match = re.search(pattern, response_lower)
            if match:
                # Extract topic from context
                topic = self._extract_offer_topic(response, match, entities)
                
                return {
                    'offered': True,
                    'topic': topic,
                    'entities': entities[:5] if entities else [],
                    'pattern_matched': pattern
                }
        
        return None
    
    def _extract_offer_topic(
        self,
        response: str,
        match: re.Match,
        entities: Optional[List[str]]
    ) -> str:
        """
        Extract the topic being offered from the response.
        
        Uses heuristics to identify what the assistant is offering to elaborate on.
        """
        # Get the sentence containing the offer
        match_pos = match.start()
        
        # Find sentence boundaries
        before_text = response[:match_pos]
        sentences = re.split(r'[.!?]+', before_text)
        
        # Get the last sentence before the offer (likely contains the topic)
        if len(sentences) >= 2:
            last_sentence = sentences[-2].strip()
        else:
            last_sentence = sentences[-1].strip() if sentences else ""
        
        # If we have entities, use them to construct topic
        if entities and len(entities) > 0:
            if len(entities) == 1:
                return f"{entities[0]}"
            elif len(entities) == 2:
                return f"{entities[0]} and {entities[1]}"
            else:
                return f"{entities[0]}, {entities[1]}, and others"
        
        # Fallback: use last sentence or generic topic
        if last_sentence and len(last_sentence) > 10:
            # Truncate to reasonable length
            if len(last_sentence) > 100:
                last_sentence = last_sentence[:100] + "..."
            return last_sentence
        
        return "the previous topic"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summarizer statistics."""
        return {
            "llm_available": self.llm_generator is not None,
            "min_length_for_summary": self.min_length_for_summary,
            "max_summary_tokens": self.max_summary_tokens,
            "offer_patterns_count": len(self.offer_patterns)
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_response_summarizer(
    llm_generator: Optional[Any] = None,
    **kwargs
) -> ResponseSummarizer:
    """
    Factory function to create a ResponseSummarizer instance.
    
    Args:
        llm_generator: LLM generator instance for summarization
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ResponseSummarizer instance
    """
    return ResponseSummarizer(llm_generator=llm_generator, **kwargs)
