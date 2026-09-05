"""
Conversation Memory Module for RAG Pipeline.

This module provides conversation memory capabilities for maintaining
context across multiple query turns in the RAG system.

Components:
- ConversationTurn: Single turn in a conversation (query + answer + metadata)
- ConversationSession: Full conversation session with multiple turns
- ConversationMemory: Main class for managing conversation memory
- MemoryConfig: Configuration for memory behavior

Usage:
    from src.memory import ConversationMemory, MemoryConfig
    
    config = MemoryConfig(max_turns=10, persistence_enabled=True)
    memory = ConversationMemory(session_id="user_123", config=config)
    
    # Add a turn
    memory.add_turn(
        query="What is PL?",
        answer="PL is a framework...",
        citations=[{"source": "doc.pdf", "page": 1}],
        entities=["PL"]
    )
    
    # Get recent turns for context
    recent = memory.get_recent_turns(n=3)
"""

from src.memory.conversation_memory import (
    ConversationTurn,
    ConversationSession,
    ConversationMemory,
    MemoryConfig,
)

__all__ = [
    "ConversationTurn",
    "ConversationSession",
    "ConversationMemory",
    "MemoryConfig",
]
