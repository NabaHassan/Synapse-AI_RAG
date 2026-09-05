"""
Conversation Memory for RAG Pipeline.

This module provides conversation memory management for maintaining context
across multiple query turns. It enables the RAG system to understand follow-up
questions and resolve references from previous conversation.

Features:
- Store and retrieve conversation turns with metadata
- Configurable memory window (max turns, token limits)
- Persistence to JSON files
- Conversation summary generation
- History search for relevant past turns
- Thread-safe operations
"""

import json
import uuid
import logging
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple

from src.utils.source_normalization import normalize_citations_sources

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MemoryConfig:
    """
    Configuration for conversation memory behavior.
    
    Attributes:
        max_turns: Maximum number of turns to keep in memory (default: 10)
        max_tokens_per_turn: Maximum tokens to store per turn for prompt budget (default: 200)
        history_in_prompt_turns: Number of turns to include in prompt (default: 3)
        persistence_enabled: Whether to auto-save to disk (default: True)
        persistence_directory: Directory for saving conversation files
        auto_save_interval: Save every N turns (default: 1, save after each turn)
        truncate_answers: Whether to truncate long answers when storing (default: True)
        answer_truncate_length: Max characters for stored answers (default: 500)
    """
    max_turns: int = 10
    max_tokens_per_turn: int = 200
    history_in_prompt_turns: int = 3
    persistence_enabled: bool = True
    persistence_directory: str = "./data/conversations"
    auto_save_interval: int = 1
    truncate_answers: bool = True
    answer_truncate_length: int = 500

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_tokens_per_turn < 50:
            raise ValueError("max_tokens_per_turn must be at least 50")
        if self.history_in_prompt_turns < 1:
            raise ValueError("history_in_prompt_turns must be at least 1")
        if self.history_in_prompt_turns > self.max_turns:
            logger.warning(
                f"history_in_prompt_turns ({self.history_in_prompt_turns}) > max_turns ({self.max_turns}), "
                f"capping to max_turns"
            )
            self.history_in_prompt_turns = self.max_turns

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryConfig":
        """Create config from dictionary."""
        return cls(**data)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ConversationTurn:
    """
    Single turn in a conversation.
    
    A turn represents one query-response pair with associated metadata.
    
    Attributes:
        turn_id: Unique identifier for this turn
        timestamp: ISO format timestamp when turn was created
        query: Original user query
        reformulated_query: Query after reformulation (same as query if no reformulation)
        answer: Model's response
        response_summary: Compact summary of answer for context
        pending_offer: Dict with topic/entities if assistant offered more info
        citations: List of citation dictionaries with source info
        entities_mentioned: Key entities extracted for reference resolution
        confidence: Model's confidence score for the answer
        metadata: Additional metadata (retrieval stats, timing, etc.)
        query_type: Type of query (meta_conversation, formatting_request, continuation, new_query)
        used_retrieval: Whether retrieval was performed for this turn
        source_documents: List of source documents used (for reuse in formatting requests)
        reformatted_from_turn: Turn ID if this is a reformatted version of a previous answer
    """
    turn_id: int
    timestamp: str
    query: str
    reformulated_query: str
    answer: str
    response_summary: str = ""  # Compact summary for efficient context
    pending_offer: Optional[Dict[str, Any]] = None  # What assistant offered to elaborate on
    citations: List[Dict[str, Any]] = field(default_factory=list)
    entities_mentioned: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # NEW FIELDS for enhanced query tracking
    query_type: str = "new_query"  # meta_conversation, formatting_request, continuation, new_query
    used_retrieval: bool = True  # Did we retrieve docs for this turn?
    source_documents: List[Dict[str, Any]] = field(default_factory=list)  # Docs used (for reuse)
    reformatted_from_turn: Optional[int] = None  # If formatting request, which turn?

    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary for serialization."""
        data = asdict(self)
        data["citations"] = normalize_citations_sources(data.get("citations", []))
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create turn from dictionary."""
        normalized = dict(data)
        normalized["citations"] = normalize_citations_sources(normalized.get("citations", []))
        return cls(**normalized)

    def get_truncated_answer(self, max_length: int = 150) -> str:
        """Get truncated answer for display/prompt inclusion."""
        if len(self.answer) <= max_length:
            return self.answer
        return self.answer[:max_length].rsplit(' ', 1)[0] + "..."

    def get_token_estimate(self) -> int:
        """Estimate token count for this turn (rough: 1 token ≈ 4 chars)."""
        total_chars = len(self.query) + len(self.answer)
        return int(total_chars / 4)


@dataclass
class ConversationSession:
    """
    Full conversation session containing multiple turns.
    
    A session represents a continuous conversation with a user, identified
    by a unique session_id. Sessions can be persisted and restored.
    
    Attributes:
        session_id: Unique identifier for this session
        created_at: ISO timestamp when session was created
        updated_at: ISO timestamp of last update
        turns: List of conversation turns in chronological order
        active_topic: Current topic being discussed (NEW)
        topic_entities: Key entities in current topic (NEW)
        metadata: Session-level metadata (user_id, context, etc.)
    """
    session_id: str
    created_at: str
    updated_at: str
    turns: List[ConversationTurn] = field(default_factory=list)
    active_topic: str = ""  # NEW: Current conversation topic
    topic_entities: List[str] = field(default_factory=list)  # NEW: Entities in current topic
    topic_start_turn: int = 0  # NEW: Turn number when current topic started
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turns": [turn.to_dict() for turn in self.turns],
            "active_topic": self.active_topic,
            "topic_entities": self.topic_entities,
            "topic_start_turn": self.topic_start_turn,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Create session from dictionary."""
        turns = [ConversationTurn.from_dict(t) for t in data.get("turns", [])]
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            turns=turns,
            active_topic=data.get("active_topic", ""),
            topic_entities=data.get("topic_entities", []),
            topic_start_turn=data.get("topic_start_turn", 0),
            metadata=data.get("metadata", {})
        )

    @property
    def turn_count(self) -> int:
        """Get number of turns in this session."""
        return len(self.turns)

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        if not self.turns:
            return 0.0
        try:
            start = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
            end = datetime.fromisoformat(self.updated_at.replace('Z', '+00:00'))
            return (end - start).total_seconds()
        except (ValueError, AttributeError):
            return 0.0


# =============================================================================
# Main ConversationMemory Class
# =============================================================================

class ConversationMemory:
    """
    Main class for managing conversation memory.
    
    Provides methods to store, retrieve, and manage conversation history
    for a single session. Supports persistence, search, and summary generation.
    
    Thread-safe: All mutating operations are protected by a lock.
    
    Usage:
        config = MemoryConfig(max_turns=10)
        memory = ConversationMemory(session_id="user_123", config=config)
        
        # Add turns
        memory.add_turn(query="What is X?", answer="X is...", ...)
        
        # Get recent context
        recent = memory.get_recent_turns(n=3)
        
        # Save/load
        memory.save_to_file()
        memory.load_from_file("path/to/session.json")
    """

    def __init__(
            self,
            session_id: Optional[str] = None,
            config: Optional[MemoryConfig] = None,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize conversation memory.
        
        Args:
            session_id: Unique session identifier (auto-generated if None)
            config: Memory configuration (uses defaults if None)
            metadata: Optional session-level metadata
        """
        self.config = config or MemoryConfig()
        self._lock = threading.RLock()
        self._turns_since_save = 0

        # Generate session ID if not provided
        if session_id is None:
            session_id = self._generate_session_id()

        # Create session
        now = datetime.utcnow().isoformat() + "Z"
        self._session = ConversationSession(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            turns=[],
            metadata=metadata or {}
        )

        # Ensure persistence directory exists
        if self.config.persistence_enabled:
            Path(self.config.persistence_directory).mkdir(parents=True, exist_ok=True)

        logger.info(f"ConversationMemory initialized: session_id={session_id}")
        logger.debug(f"Config: {self.config.to_dict()}")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self._session.session_id

    @property
    def session(self) -> ConversationSession:
        """Get the full session object."""
        return self._session

    @property
    def turn_count(self) -> int:
        """Get number of turns in current session."""
        return len(self._session.turns)

    @property
    def is_empty(self) -> bool:
        """Check if memory has no turns."""
        return len(self._session.turns) == 0

    # =========================================================================
    # Core Methods
    # =========================================================================

    def add_turn(
            self,
            query: str,
            answer: str,
            reformulated_query: Optional[str] = None,
            response_summary: Optional[str] = None,
            pending_offer: Optional[Dict[str, Any]] = None,
            citations: Optional[List[Dict[str, Any]]] = None,
            entities: Optional[List[str]] = None,
            confidence: float = 0.0,
            metadata: Optional[Dict[str, Any]] = None,
            query_type: str = "new_query",
            used_retrieval: bool = True,
            source_documents: Optional[List[Dict[str, Any]]] = None,
            reformatted_from_turn: Optional[int] = None
    ) -> ConversationTurn:
        """
        Add a new conversation turn.
        
        Args:
            query: Original user query
            answer: Model's response
            reformulated_query: Query after reformulation (defaults to query)
            response_summary: Compact summary of answer (defaults to answer if short)
            pending_offer: Dict with topic/entities if assistant offered more info
            citations: List of citation dictionaries
            entities: Key entities mentioned (for reference resolution)
            confidence: Model's confidence score
            metadata: Additional turn metadata
            query_type: Type of query (meta_conversation, formatting_request, continuation, new_query)
            used_retrieval: Whether retrieval was performed
            source_documents: List of source documents used
            reformatted_from_turn: Turn ID if this is a reformatted answer
            
        Returns:
            The created ConversationTurn object
        """
        with self._lock:
            # Create turn
            turn_id = len(self._session.turns) + 1
            timestamp = datetime.utcnow().isoformat() + "Z"

            # Truncate answer if configured
            stored_answer = answer
            if self.config.truncate_answers and len(answer) > self.config.answer_truncate_length:
                stored_answer = answer[:self.config.answer_truncate_length].rsplit(' ', 1)[0] + "..."

            # Use answer as summary if not provided and answer is short
            if response_summary is None:
                if len(answer) < 150:
                    response_summary = answer
                else:
                    response_summary = ""

            turn = ConversationTurn(
                turn_id=turn_id,
                timestamp=timestamp,
                query=query,
                reformulated_query=reformulated_query or query,
                answer=stored_answer,
                response_summary=response_summary,
                pending_offer=pending_offer,
                citations=normalize_citations_sources(citations or []),
                entities_mentioned=entities or [],
                confidence=confidence,
                metadata=metadata or {},
                query_type=query_type,
                used_retrieval=used_retrieval,
                source_documents=source_documents or [],
                reformatted_from_turn=reformatted_from_turn
            )

            # Add to session
            self._session.turns.append(turn)
            self._session.updated_at = timestamp

            # Enforce max_turns limit (remove oldest)
            while len(self._session.turns) > self.config.max_turns:
                removed = self._session.turns.pop(0)
                logger.debug(f"Removed oldest turn (id={removed.turn_id}) to maintain max_turns limit")

            # Auto-save if configured
            self._turns_since_save += 1
            if (self.config.persistence_enabled and
                    self._turns_since_save >= self.config.auto_save_interval):
                self._auto_save()

            logger.info(
                f"Added turn {turn_id}: query='{query[:50]}...', "
                f"entities={entities}, confidence={confidence:.3f}"
            )

            return turn

    def get_recent_turns(self, n: Optional[int] = None) -> List[ConversationTurn]:
        """
        Get the most recent N turns.
        
        Args:
            n: Number of turns to retrieve (defaults to history_in_prompt_turns)
            
        Returns:
            List of recent ConversationTurn objects (oldest first)
        """
        with self._lock:
            if n is None:
                n = self.config.history_in_prompt_turns

            n = min(n, len(self._session.turns))
            return self._session.turns[-n:] if n > 0 else []

    def get_turn_by_id(self, turn_id: int) -> Optional[ConversationTurn]:
        """
        Get a specific turn by its ID.
        
        Args:
            turn_id: The turn ID to look up
            
        Returns:
            ConversationTurn if found, None otherwise
        """
        with self._lock:
            for turn in self._session.turns:
                if turn.turn_id == turn_id:
                    return turn
            return None

    def get_all_turns(self) -> List[ConversationTurn]:
        """
        Get all turns in the session.
        
        Returns:
            List of all ConversationTurn objects (oldest first)
        """
        with self._lock:
            return list(self._session.turns)

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """
        Get the most recent turn.
        
        Returns:
            Most recent ConversationTurn, or None if empty
        """
        with self._lock:
            if self._session.turns:
                return self._session.turns[-1]
            return None

    def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Load a persisted session by session_id without mutating current memory.
        """
        if not self.config.persistence_enabled:
            logger.warning(
                "Persistence is disabled, cannot load session %s",
                session_id
            )
            return None

        filepath = Path(self.config.persistence_directory) / f"{session_id}.json"
        if not filepath.exists():
            logger.warning("Persisted session file not found: %s", filepath)
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ConversationSession.from_dict(data["session"])
        except Exception as e:
            logger.warning("Failed to load session %s: %s", session_id, e)
            return None

    def get_history(self, session_id: Optional[str] = None, limit: Optional[int] = None) -> List[ConversationTurn]:
        """
        Thread-safe retrieval of historical conversation turns for a given session.
        Ensures context stickiness can calculate previous targets without crashing.
        """
        with self._lock:
            session: ConversationSession
            if session_id is not None and session_id != self.session_id:
                session = self.load_session(session_id)
            else:
                session = self._session

            if not session or not session.turns:
                return []

            sorted_turns = sorted(session.turns, key=lambda turn: turn.turn_id)
            if limit is not None and limit > 0:
                return sorted_turns[-limit:]
            return sorted_turns

    def clear(self) -> None:
        """
        Clear all turns from memory (reset conversation).
        
        Note: This does not delete persisted files.
        """
        with self._lock:
            self._session.turns.clear()
            self._session.updated_at = datetime.utcnow().isoformat() + "Z"
            self._turns_since_save = 0
            logger.info(f"Cleared conversation memory for session {self.session_id}")

    # =========================================================================
    # Search & Summary Methods
    # =========================================================================

    def search_history(
            self,
            query: str,
            max_results: int = 3
    ) -> List[Tuple[ConversationTurn, float]]:
        """
        Search conversation history for turns relevant to a query.
        
        Uses simple keyword matching. For more sophisticated search,
        consider using embeddings (can be extended).
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of (ConversationTurn, relevance_score) tuples, sorted by relevance
        """
        with self._lock:
            if not self._session.turns:
                return []

            query_words = set(query.lower().split())
            results = []

            for turn in self._session.turns:
                # Calculate simple relevance score based on word overlap
                turn_text = f"{turn.query} {turn.answer}".lower()
                turn_words = set(turn_text.split())

                # Word overlap score
                overlap = len(query_words & turn_words)
                if overlap > 0:
                    score = overlap / len(query_words)

                    # Boost for entity matches
                    for entity in turn.entities_mentioned:
                        if entity.lower() in query.lower():
                            score += 0.3

                    results.append((turn, min(score, 1.0)))

            # Sort by relevance (descending) and return top results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]

    def get_conversation_summary(self, max_length: int = 500) -> str:
        """
        Generate a condensed summary of the conversation.
        
        Creates a brief summary suitable for including in prompts
        when full history is too long.
        
        Args:
            max_length: Maximum length of summary in characters
            
        Returns:
            Condensed summary string
        """
        with self._lock:
            if not self._session.turns:
                return "No conversation history."

            # Collect all entities mentioned
            all_entities = set()
            topics = []

            for turn in self._session.turns:
                all_entities.update(turn.entities_mentioned)
                # Extract main topic from query (first few words)
                query_words = turn.query.split()[:5]
                topics.append(' '.join(query_words))

            # Build summary
            summary_parts = []

            # Entities mentioned
            if all_entities:
                entities_str = ", ".join(sorted(all_entities)[:10])
                summary_parts.append(f"Topics discussed: {entities_str}")

            # Number of turns
            summary_parts.append(f"Conversation length: {len(self._session.turns)} turns")

            # Recent questions
            recent_queries = [t.query for t in self._session.turns[-3:]]
            if recent_queries:
                summary_parts.append(f"Recent questions: {'; '.join(recent_queries)}")

            summary = ". ".join(summary_parts)

            # Truncate if needed
            if len(summary) > max_length:
                summary = summary[:max_length].rsplit(' ', 1)[0] + "..."

            return summary

    def get_mentioned_entities(self) -> List[str]:
        """
        Get all entities mentioned across the conversation.
        
        Returns:
            List of unique entity strings
        """
        with self._lock:
            entities = set()
            for turn in self._session.turns:
                entities.update(turn.entities_mentioned)
            return sorted(entities)

    def get_context_for_prompt(
            self,
            n_turns: Optional[int] = None,
            max_chars_per_turn: int = 200
    ) -> List[Dict[str, str]]:
        """
        Get conversation context formatted for prompt inclusion.
        
        Returns a list of simplified turn dictionaries suitable for
        including in the LLM prompt.
        
        Args:
            n_turns: Number of turns to include (defaults to config)
            max_chars_per_turn: Max characters per answer
            
        Returns:
            List of dicts with 'query' and 'answer' keys
        """
        with self._lock:
            turns = self.get_recent_turns(n_turns)

            context = []
            for turn in turns:
                truncated_answer = turn.get_truncated_answer(max_chars_per_turn)
                context.append({
                    "query": turn.query,
                    "answer": truncated_answer
                })

            return context

    # =========================================================================
    # Persistence Methods
    # =========================================================================

    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """
        Save conversation to JSON file.
        
        Args:
            filepath: Custom filepath (uses default if None)
            
        Returns:
            Path to saved file
        """
        with self._lock:
            if filepath is None:
                filepath = self._get_default_filepath()

            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "config": self.config.to_dict(),
                "session": self._session.to_dict()
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._turns_since_save = 0
            logger.info(f"Saved conversation to {filepath}")

            return str(filepath)

    def load_from_file(self, filepath: str) -> bool:
        """
        Load conversation from JSON file.
        
        Args:
            filepath: Path to the conversation file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        with self._lock:
            filepath = Path(filepath)

            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return False

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load session
                self._session = ConversationSession.from_dict(data["session"])

                # Optionally update config
                if "config" in data:
                    # Keep current config but log if different
                    saved_config = data["config"]
                    if saved_config.get("max_turns") != self.config.max_turns:
                        logger.info(
                            f"Note: Loaded session had max_turns={saved_config.get('max_turns')}, "
                            f"current config has max_turns={self.config.max_turns}"
                        )

                self._turns_since_save = 0
                logger.info(
                    f"Loaded conversation from {filepath}: "
                    f"{len(self._session.turns)} turns"
                )

                return True

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return False
            except KeyError as e:
                logger.error(f"Missing required field in file: {e}")
                return False
            except Exception as e:
                logger.error(f"Failed to load conversation: {e}")
                return False

    def _auto_save(self) -> None:
        """Internal method for auto-saving."""
        try:
            filepath = self._get_default_filepath()
            self.save_to_file(filepath)
        except Exception as e:
            logger.warning(f"Auto-save failed: {e}")

    def _get_default_filepath(self) -> str:
        """Get default filepath for this session."""
        filename = f"{self.session_id}.json"
        return str(Path(self.config.persistence_directory) / filename)

    def delete_persisted_file(self) -> bool:
        """
        Delete the persisted file for this session.
        
        Returns:
            True if deleted, False if file didn't exist
        """
        filepath = Path(self._get_default_filepath())
        if filepath.exists():
            filepath.unlink()
            logger.info(f"Deleted persisted file: {filepath}")
            return True
        return False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_part = uuid.uuid4().hex[:8]
        return f"session_{timestamp}_{unique_part}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current conversation.
        
        Returns:
            Dictionary with conversation statistics
        """
        with self._lock:
            if not self._session.turns:
                return {
                    "session_id": self.session_id,
                    "turn_count": 0,
                    "total_queries": 0,
                    "unique_entities": 0,
                    "avg_confidence": 0.0,
                    "duration_seconds": 0.0
                }

            confidences = [t.confidence for t in self._session.turns]
            all_entities = self.get_mentioned_entities()

            return {
                "session_id": self.session_id,
                "turn_count": len(self._session.turns),
                "total_queries": len(self._session.turns),
                "unique_entities": len(all_entities),
                "entities": all_entities[:10],  # Top 10
                "avg_confidence": sum(confidences) / len(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences),
                "duration_seconds": self._session.duration_seconds,
                "created_at": self._session.created_at,
                "updated_at": self._session.updated_at
            }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entire memory state to dictionary.
        
        Returns:
            Dictionary representation of memory state
        """
        with self._lock:
            return {
                "config": self.config.to_dict(),
                "session": self._session.to_dict(),
                "statistics": self.get_statistics()
            }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ConversationMemory(session_id='{self.session_id}', "
            f"turns={self.turn_count}, "
            f"max_turns={self.config.max_turns})"
        )

    def __len__(self) -> int:
        """Return number of turns."""
        return self.turn_count


# =============================================================================
# Factory Function
# =============================================================================

def create_conversation_memory(
        session_id: Optional[str] = None,
        max_turns: int = 10,
        persistence_enabled: bool = True,
        persistence_directory: str = "./data/conversations",
        **kwargs
) -> ConversationMemory:
    """
    Factory function to create a ConversationMemory instance.
    
    Args:
        session_id: Unique session identifier (auto-generated if None)
        max_turns: Maximum turns to keep in memory
        persistence_enabled: Whether to auto-save to disk
        persistence_directory: Directory for saving conversations
        **kwargs: Additional MemoryConfig parameters
        
    Returns:
        Configured ConversationMemory instance
    """
    config = MemoryConfig(
        max_turns=max_turns,
        persistence_enabled=persistence_enabled,
        persistence_directory=persistence_directory,
        **kwargs
    )

    return ConversationMemory(session_id=session_id, config=config)


# =============================================================================
# Topic Extraction Helper
# =============================================================================

def extract_topic_from_response(
        response: str,
        entities: Optional[List[str]] = None,
        max_length: int = 100
) -> str:
    """
    Extract conversation topic from response (lightweight, no LLM).
    
    Uses simple heuristics:
    1. If entities provided, use top 2-3 entities as topic
    2. Otherwise, use first sentence (truncated to max_length)
    
    This is intentionally simple and fast (<1ms) to avoid overhead.
    
    Args:
        response: The assistant's response text
        entities: List of entities mentioned in response (optional)
        max_length: Maximum length of extracted topic
        
    Returns:
        Topic string (e.g., "Jeffrey Epstein, Ghislaine Maxwell")
        
    Examples:
        >>> extract_topic_from_response("...", ["Jeffrey Epstein", "Ghislaine Maxwell"])
        "Jeffrey Epstein, Ghislaine Maxwell"
    """
    # Strategy 1: Use entities if available
    if entities and len(entities) > 0:
        # Take top 3 entities
        top_entities = entities[:3]
        topic = ", ".join(top_entities)

        # Truncate if too long
        if len(topic) > max_length:
            topic = topic[:max_length - 3] + "..."

        return topic

    # Strategy 2: Use first sentence
    sentences = response.split('.')
    if sentences and sentences[0].strip():
        first_sentence = sentences[0].strip()

        # Truncate if too long
        if len(first_sentence) > max_length:
            first_sentence = first_sentence[:max_length - 3] + "..."

        return first_sentence

    # Fallback: truncated response
    if len(response) > max_length:
        return response[:max_length - 3] + "..."

    return response.strip()
