"""
Query Type Definitions for Conversational RAG.

This module defines the different types of queries that can be handled
by the conversational RAG system, along with classification results.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


class QueryType(Enum):
    """
    Types of queries in conversational RAG.
    
    Each type requires different handling:
    - META_CONVERSATION: Answer from memory, no retrieval
    - FORMATTING_REQUEST: Reformat previous answer, reuse documents
    - CLARIFICATION: Revise previous answer for clarity, no new sources
    - CONTINUATION: Enrich with context, then retrieve
    - NEW_QUERY: Standard RAG pipeline
    - ENTITY_COUNT: Count entity mentions across documents (Qdrant scroll)
    - FILE_LOCATION: Find files containing an entity (Qdrant filter)
    - EXACT_TEXT: Retrieve exact text snippets/quotes/emails (Qdrant filter)
    """
    META_CONVERSATION = "meta_conversation"
    FORMATTING_REQUEST = "formatting_request"
    CLARIFICATION = "clarification"
    CONTINUATION = "continuation"
    NEW_QUERY = "new_query"
    ENTITY_COUNT = "entity_count"
    FILE_LOCATION = "file_location"
    EXACT_TEXT = "exact_text"


@dataclass
class QueryClassificationResult:
    """
    Result of query classification.
    
    Attributes:
        query_type: The classified type of the query
        confidence: Confidence score (0-1) for the classification
        matched_patterns: List of patterns that matched (for debugging)
        reasoning: Human-readable explanation of why this type was chosen
        metadata: Additional classification metadata
    """
    query_type: QueryType
    confidence: float
    matched_patterns: List[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
            "reasoning": self.reasoning,
            "metadata": self.metadata
        }
