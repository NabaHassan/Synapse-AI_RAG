"""
Topic extraction and tracking utilities for conversation memory.

This module provides functions to extract topics from responses and
track topic continuity across conversation turns.
"""

import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def extract_topic_from_response(answer: str, entities: List[str]) -> str:
    """
    Extract the main topic from a response.
    
    Uses entities and answer content to determine the topic.
    
    Args:
        answer: The response text
        entities: List of entities mentioned
        
    Returns:
        Topic string
    """
    if not entities:
        # Fallback: use first few words of answer
        words = answer.split()[:5]
        return " ".join(words) if words else "General discussion"
    
    # Use top 3 entities as topic
    top_entities = entities[:3]
    return ", ".join(top_entities)


def update_active_topic(
    new_turn,
    current_topic: str,
    current_entities: List[str],
    query_type: str
) -> Tuple[str, List[str]]:
    """
    Update the active topic based on a new turn.
    
    CRITICAL: Ignores meta-conversation and formatting queries for topic tracking.
    
    Args:
        new_turn: The new ConversationTurn
        current_topic: Current active topic
        current_entities: Current topic entities
        query_type: Type of query (meta_conversation, formatting_request, continuation, new_query)
        
    Returns:
        Tuple of (new_topic, new_entities)
    """
    # Don't update topic for meta or formatting queries
    if query_type in ["meta_conversation", "formatting_request"]:
        logger.debug(f"Keeping current topic (query type: {query_type})")
        return current_topic, current_entities
    
    # Get entities from new turn
    new_entities = new_turn.entities_mentioned if hasattr(new_turn, 'entities_mentioned') else []
    
    if not new_entities:
        # No entities in new turn, keep current topic
        return current_topic, current_entities
    
    if not current_entities:
        # First real query, set initial topic
        new_topic = extract_topic_from_response(new_turn.answer, new_entities)
        return new_topic, new_entities[:5]
    
    # Calculate entity overlap
    overlap = set(new_entities) & set(current_entities)
    overlap_ratio = len(overlap) / len(current_entities)
    
    if overlap_ratio > 0.3:  # 30% overlap = same topic
        # Topic continued, merge entities
        merged = list(set(current_entities + new_entities[:3]))[:5]
        logger.debug(f"Topic continued (overlap: {overlap_ratio:.2f})")
        return current_topic, merged
    else:
        # Topic switched
        new_topic = extract_topic_from_response(new_turn.answer, new_entities)
        logger.debug(f"Topic switched: '{current_topic}' -> '{new_topic}'")
        return new_topic, new_entities[:5]
