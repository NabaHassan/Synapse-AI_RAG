"""
Conversational RAG Pipeline with Memory Integration.

This module extends the base RAGPipeline to support conversation memory,
enabling multi-turn conversations with reference resolution and contextual
understanding while maintaining strict grounding in the knowledge base.

Features:
- Session-based conversation memory management
- Automatic query reformulation for reference resolution
- Conversation-aware prompt building
- Entity extraction and turn storage
- Configurable memory behavior

Flow:
1. Get or create conversation session
2. Check if query needs reformulation (pronouns, references)
3. Reformulate query using conversation history if needed
4. Run retrieval with (possibly reformulated) query
5. Build prompt with conversation context + KB context
6. Generate answer
7. Extract entities and store turn in memory
8. Return result with session info
"""

import time
import uuid
import threading
import logging
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from src.retrieval.web_retriever import WebRetriever
from concurrent.futures import ThreadPoolExecutor, as_completed

# Memory components
from src.memory import ConversationMemory, MemoryConfig, ConversationTurn
from src.memory.conversation_memory import ConversationSession
from src.concurrency import (
    RedisBackendUnavailable,
    RedisConnection,
    RedisRuntimeConfig,
    RedisSessionStore,
)

# Import base pipeline
from src.pipeline.rag_pipeline import RAGPipeline, PipelineConfig, PipelineResult

# Query reformulation
from src.query_processing.follow_up_detector import FollowUpDetector
from src.query_processing import QueryReformulator, ReformulatorConfig

# Memory and topic tracking
from src.pipeline.query_routing import classify_and_route_query
from src.generation.response_summarizer import ResponseSummarizer
from src.query_processing.meta_handler import MetaConversationHandler
from src.memory.conversation_memory import extract_topic_from_response
from src.query_processing.formatting_handler import FormattingRequestHandler
from src.query_processing.query_classifier_enhanced import QueryClassifierEnhanced
from src.generation.answer_sanitizer import sanitize_generated_answer
from src.generation.llm_generator import LLMOverloadedError
from src.utils.source_normalization import normalize_citations_sources
from src.pipeline.session_doc_store import (
    get_session_docs,
    cosine_similarity_search,
    session_has_uploaded_docs,
)
from haystack import Document

# Conversational prompt building
from src.generation.conversational_prompt_builder import (
    ConversationalPromptBuilder,
    ConversationalPromptConfig,
    create_conversational_prompt_builder,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP display-name helper
# ---------------------------------------------------------------------------
_MCP_DISPLAY_NAMES = {
    "slack": "Slack",
    "notion": "Notion",
    "gmail": "Google Gmail",
    "outlook": "Outlook",
    "onedrive": "OneDrive",
    "drive": "Google Drive",
    "calendar": "Google Calendar",
    "sheets": "Google Sheets",
    "docs": "Google Docs",
    "presentation": "Google Slides",
    "outlook": "Outlook",
    "onedrive": "OneDrive",
}

_MAIL_CONNECTORS = frozenset({"outlook", "email", "gmail"})


def _is_mail_connector(connector: Optional[str]) -> bool:
    if not connector:
        return False
    return connector.lower().strip() in _MAIL_CONNECTORS

_LOW_VALUE_MCP_RESPONSES = (
    "no upcoming events found",
    "no events found",
    "no results found",
    "no messages found",
    "no threads found",
    "no files found",
    "no pages found",
    "no databases found",
)


def _is_low_value_mcp_response(text: Optional[str]) -> bool:
    if not text or not str(text).strip():
        return True
    lowered = str(text).lower().strip()
    if len(lowered) < 24:
        return True
    return any(marker in lowered for marker in _LOW_VALUE_MCP_RESPONSES)


def _query_explicitly_requests_external_connector(query: str, connector: Optional[str]) -> bool:
    """True when the user clearly wants workspace data instead of session-uploaded docs."""
    q = (query or "").strip().lower()
    if not q:
        return False
    if q.startswith("/"):
        return True
    if "@" in query or "#" in query:
        return True
    workspace_signals = (
        "my calendar",
        "my email",
        "my inbox",
        "google drive",
        "slack channel",
        "notion page",
        "upcoming events",
        "schedule for",
        "meetings this week",
        "check my email",
        "search drive",
        "search gmail",
    )
    if any(signal in q for signal in workspace_signals):
        return True
    if connector and connector.lower().strip() in ("slack", "notion") and any(
        token in q for token in ("slack", "notion", "channel", "dm", "workspace")
    ):
        return True
    return False


def _mcp_display_name(service: Optional[str]) -> str:
    """Return a human-readable label for an MCP service string."""
    if not service:
        return "MCP"
    return _MCP_DISPLAY_NAMES.get(service.lower(), f"Google {service.capitalize()}")



class PipelineCancelledError(RuntimeError):
    """Raised when cooperative cancellation is requested for an in-flight pipeline run."""


class PipelineOverloadedError(RuntimeError):
    """Raised when pipeline capacity is temporarily saturated."""

    def __init__(
            self,
            message: str,
            retry_after_seconds: int = 2,
            reason: str = "pipeline_overloaded",
            details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        self.reason = reason
        self.details = details or {}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ConversationalPipelineConfig(PipelineConfig):
    """
    Extended configuration for conversational RAG pipeline.

    Inherits all settings from PipelineConfig and adds memory-specific options.

    Memory Settings:
        enable_memory: Whether to enable conversation memory (default: True)
        memory_directory: Directory for storing conversation files
        max_turns: Maximum conversation turns to keep in memory
        history_in_prompt_turns: Number of turns to include in LLM prompt
        auto_save: Whether to auto-save conversation after each turn

    Reformulation Settings:
        enable_reformulation: Whether to enable query reformulation
        use_llm_reformulation: Use LLM for reformulation (vs rule-based)
        reformulation_confidence_threshold: Minimum confidence for reformulation

    Prompt Settings:
        prompt_template_name: Template to use ("default", "detailed", "concise")
        history_answer_max_length: Max chars for historical answers in prompt
    """
    # Memory settings
    enable_memory: bool = True
    memory_directory: str = "./data/conversations"
    max_turns: int = 50
    history_in_prompt_turns: int = 10
    auto_save: bool = True

    # Reformulation settings
    enable_reformulation: bool = True
    use_llm_reformulation: bool = True  # Rule-based by default (faster)
    reformulation_confidence_threshold: float = 0.6

    # Prompt settings
    prompt_template_name: str = "default"  # "default", "detailed", "concise", "box"
    prompt_template: Optional[str] = None
    history_answer_max_length: int = 150

    # Generation tuning
    top_p: float = 0.8
    repetition_penalty: float = 1.2
    presence_penalty: float = 1.0
    frequency_penalty: float = 0.3
    enable_min_tokens_strategy: bool = False
    min_tokens_long_response: int = 0
    long_response_max_tokens: int = 0

    # Cache settings
    enable_cache: bool = False
    cache_file: str = "./data/query_cache.json"
    max_cache_size: int = 1000
    cache_ttl_hours: int = 168  # around 7 days
    cache_similarity_threshold: float = 0.95
    cache_redis_enabled: bool = False
    cache_redis_ttl_seconds: int = 0

    # Redis safety layer settings (Phase 1)
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "synapse"
    redis_socket_timeout_seconds: float = 0.2
    redis_connect_timeout_seconds: float = 0.2
    session_store_ttl_seconds: int = 7 * 24 * 3600
    session_read_through_enabled: bool = True
    redis_session_namespace: Optional[str] = None

    # Grounding settings
    # When False, pipeline will return "no results" instead of using general knowledge
    # if retrieval/context is insufficiently grounded in the KB.
    allow_general_knowledge_fallback: bool = True
    min_verification_threshold: float = 0.1
    enable_collection_query_anchoring: bool = True
    collection_anchor_terms: List[str] = field(default_factory=list)

    # Structured query performance settings
    # When True, structured handlers prefer metadata-only fast paths (best with reindexed data).
    # When False, preserve current precision-first behavior with runtime text verification.
    structured_query_fast_mode: Optional[bool] = None
    # When True, typo-tolerant entity resolution is applied before structured handlers.
    structured_entity_resolution: bool = True
    # When True, structured responses use natural disclosure/messaging improvements.
    structured_natural_response_style: bool = True

    # Query handler behavior
    canned_responses: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConversationalPipelineResult(PipelineResult):
    """
    Extended result from conversational RAG pipeline.

    Adds conversation-specific metadata to the base PipelineResult.
    """
    # Session info
    session_id: str = ""
    turn_number: int = 0

    # Reformulation info
    was_reformulated: bool = False
    reformulated_query: str = ""
    reformulation_method: str = ""
    detected_references: List[str] = field(default_factory=list)

    # Entity info
    extracted_entities: List[str] = field(default_factory=list)

    # Memory stats
    memory_stats: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Session Manager
# =============================================================================

class SessionManager:
    """
    Manages conversation sessions for the pipeline.

    Handles session creation, retrieval, persistence, and cleanup.
    Each session has its own ConversationMemory instance.
    """

    def __init__(
            self,
            memory_directory: str = "./data/conversations",
            max_turns: int = 10,
            auto_save: bool = True,
            redis_enabled: bool = False,
            redis_url: str = "redis://localhost:6379/0",
            redis_key_prefix: str = "synapse",
            redis_socket_timeout_seconds: float = 0.2,
            redis_connect_timeout_seconds: float = 0.2,
            redis_session_ttl_seconds: int = 7 * 24 * 3600,
            session_read_through_enabled: bool = True,
            redis_session_namespace: Optional[str] = None,
    ):
        """
        Initialize session manager.

        Args:
            memory_directory: Directory for storing session files
            max_turns: Maximum turns per session
            auto_save: Whether to auto-save after each turn
        """
        self.memory_directory = Path(memory_directory)
        self.max_turns = max_turns
        self.auto_save = auto_save
        self.session_read_through_enabled = session_read_through_enabled

        # Active sessions in memory
        self._sessions: Dict[str, ConversationMemory] = {}

        self._redis_connection: Optional[RedisConnection] = None
        self._redis_session_store: Optional[RedisSessionStore] = None
        self._redis_enabled = bool(redis_enabled)
        self._redis_failures = 0
        self._redis_session_namespace = str(redis_session_namespace).strip() if redis_session_namespace else None
        if self._redis_enabled:
            redis_runtime = RedisRuntimeConfig(
                enabled=True,
                url=redis_url,
                key_prefix=redis_key_prefix,
                socket_timeout_seconds=redis_socket_timeout_seconds,
                connect_timeout_seconds=redis_connect_timeout_seconds,
            )
            self._redis_connection = RedisConnection(redis_runtime)
            self._redis_session_store = RedisSessionStore(
                connection=self._redis_connection,
                session_ttl_seconds=redis_session_ttl_seconds,
                session_namespace=self._redis_session_namespace,
            )
            if self._redis_connection.init_error:
                logger.warning(
                    "Session Redis store initialization issue (%s). "
                    "Session reads will fallback to JSON read-through.",
                    self._redis_connection.init_error,
                )
            else:
                logger.info("SessionManager Redis store enabled")

        # Ensure directory exists
        self.memory_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"SessionManager initialized (directory: {self.memory_directory})")

    def _record_redis_failure(self, message: str) -> None:
        self._redis_failures += 1
        logger.warning("%s (redis_failures=%s)", message, self._redis_failures)

    def _build_memory_config(self) -> MemoryConfig:
        return MemoryConfig(
            max_turns=self.max_turns,
            persistence_enabled=self.auto_save,
            persistence_directory=str(self.memory_directory),
            auto_save_interval=1 if self.auto_save else 0,
            truncate_answers=False
        )

    def _serialize_memory_payload(self, memory: ConversationMemory) -> Dict[str, Any]:
        from datetime import timezone
        return {
            "version": "1.0",
            "saved_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "config": memory.config.to_dict(),
            "session": memory.session.to_dict(),
            "session_namespace": self._redis_session_namespace,
        }

    def _memory_from_payload(self, session_id: str, payload: Dict[str, Any]) -> Optional[ConversationMemory]:
        session_dict = payload.get("session")
        if not isinstance(session_dict, dict):
            return None
        payload_namespace = payload.get("session_namespace")
        if payload_namespace is not None and payload_namespace != self._redis_session_namespace:
            logger.warning(
                "Ignoring Redis session payload for %s due to namespace mismatch (%s != %s)",
                session_id,
                payload_namespace,
                self._redis_session_namespace,
            )
            return None
        try:
            memory = ConversationMemory(config=self._build_memory_config(), session_id=session_id)
            memory._session = ConversationSession.from_dict(session_dict)
            memory._turns_since_save = 0
            return memory
        except Exception as exc:
            logger.warning("Failed to hydrate memory from payload for session %s: %s", session_id, exc)
            return None

    def _load_session_from_redis(self, session_id: str) -> Optional[ConversationMemory]:
        if not self._redis_enabled or self._redis_session_store is None:
            return None
        try:
            payload = self._redis_session_store.get_session_payload(session_id)
        except RedisBackendUnavailable as exc:
            self._record_redis_failure(f"Session Redis read failed: {exc}")
            return None

        if payload is None:
            return None

        memory = self._memory_from_payload(session_id, payload)
        if memory is None:
            return None
        return memory

    def _save_session_to_redis(self, session_id: str, memory: ConversationMemory) -> bool:
        if not self._redis_enabled or self._redis_session_store is None:
            return False
        payload = self._serialize_memory_payload(memory)
        try:
            self._redis_session_store.set_session_payload(session_id, payload)
            return True
        except RedisBackendUnavailable as exc:
            self._record_redis_failure(f"Session Redis write failed: {exc}")
            return False

    def _load_session_from_disk(self, session_id: str) -> Optional[ConversationMemory]:
        session_file = self.memory_directory / f"{session_id}.json"
        if not session_file.exists():
            return None

        logger.info(f"Loading session from disk: {session_id}")
        memory = self._load_session(session_id)
        return memory

    def get_or_create_session(self, session_id: Optional[str] = None) -> Tuple[str, ConversationMemory]:
        """
        Get existing session or create new one.

        Args:
            session_id: Optional session ID. If None, creates new session.

        Returns:
            Tuple of (session_id, ConversationMemory)
        """
        # Generate new session ID if not provided
        if session_id is None:
            session_id = self._generate_session_id()
            logger.info(f"Creating new session: {session_id}")

        # In Redis mode, always reload session hot state to avoid stale per-worker copies.
        if not self._redis_enabled and session_id in self._sessions:
            logger.debug(f"Returning active session: {session_id}")
            return session_id, self._sessions[session_id]

        # Redis-first read path (Phase 1)
        memory = self._load_session_from_redis(session_id)
        if memory is not None:
            self._sessions[session_id] = memory
            return session_id, memory

        # JSON read-through during migration window.
        disk_memory = self._load_session_from_disk(session_id)
        if disk_memory is not None:
            self._sessions[session_id] = disk_memory
            if self.session_read_through_enabled:
                self._save_session_to_redis(session_id, disk_memory)
            return session_id, disk_memory

        # Create new session
        logger.info(f"Creating new session: {session_id}")
        memory = ConversationMemory(config=self._build_memory_config(), session_id=session_id)
        self._sessions[session_id] = memory
        self._save_session_to_redis(session_id, memory)

        return session_id, memory

    def get_session(self, session_id: str) -> Optional[ConversationMemory]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ConversationMemory or None if not found
        """
        if not self._redis_enabled and session_id in self._sessions:
            return self._sessions[session_id]

        # Redis-first read path
        memory = self._load_session_from_redis(session_id)
        if memory is not None:
            self._sessions[session_id] = memory
            return memory

        # JSON read-through fallback
        disk_memory = self._load_session_from_disk(session_id)
        if disk_memory is not None:
            self._sessions[session_id] = disk_memory
            if self.session_read_through_enabled:
                self._save_session_to_redis(session_id, disk_memory)
            return disk_memory

        return None

    def _load_session(self, session_id: str) -> ConversationMemory:
        """Load session from disk."""
        memory = ConversationMemory(config=self._build_memory_config(), session_id=session_id)

        session_file = self.memory_directory / f"{session_id}.json"
        if session_file.exists():
            memory.load_from_file(str(session_file))

        return memory

    def save_session(self, session_id: str) -> bool:
        """
        Save session to disk.

        Args:
            session_id: Session to save

        Returns:
            True if saved successfully
        """
        if session_id not in self._sessions:
            logger.warning(f"Session not found: {session_id}")
            return False

        memory = self._sessions[session_id]
        redis_ok = self._save_session_to_redis(session_id, memory)
        disk_ok = True
        try:
            filepath = self.memory_directory / f"{session_id}.json"
            memory.save_to_file(str(filepath))
        except Exception:
            disk_ok = False
            logger.warning("Failed to persist session to disk: %s", session_id, exc_info=True)

        if self._redis_enabled:
            return bool(redis_ok or disk_ok)
        return disk_ok

    def clear_session(self, session_id: str) -> bool:
        """
        Clear session history (keeps session active).

        Args:
            session_id: Session to clear

        Returns:
            True if cleared successfully
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            self.save_session(session_id)
            logger.info(f"Session cleared: {session_id}")
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session completely.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted successfully
        """
        # Remove from memory
        if session_id in self._sessions:
            del self._sessions[session_id]

        redis_deleted = False
        if self._redis_enabled and self._redis_session_store is not None:
            try:
                redis_deleted = self._redis_session_store.delete_session(session_id)
            except RedisBackendUnavailable as exc:
                self._record_redis_failure(f"Session Redis delete failed: {exc}")

        # Remove from disk
        session_file = self.memory_directory / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            logger.info(f"Session deleted: {session_id}")
            return True

        return bool(redis_deleted)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions (active and saved).

        Returns:
            List of session info dictionaries
        """
        sessions = []

        # Active sessions (in-process)
        for session_id, memory in self._sessions.items():
            sessions.append({
                "session_id": session_id,
                "turn_count": len(memory.get_all_turns()),
                "active": True,
                "file_exists": (self.memory_directory / f"{session_id}.json").exists()
            })

        known_ids = {entry["session_id"] for entry in sessions}

        # Redis-backed session IDs (cross-worker visibility)
        if self._redis_enabled and self._redis_session_store is not None:
            try:
                redis_ids = self._redis_session_store.list_session_ids()
                for session_id in redis_ids:
                    if session_id in known_ids:
                        continue
                    sessions.append({
                        "session_id": session_id,
                        "turn_count": None,
                        "active": False,
                        "file_exists": (self.memory_directory / f"{session_id}.json").exists(),
                        "source": "redis",
                    })
                    known_ids.add(session_id)
            except RedisBackendUnavailable as exc:
                self._record_redis_failure(f"Session Redis list failed: {exc}")

        # Saved sessions not currently active
        for session_file in self.memory_directory.glob("*.json"):
            session_id = session_file.stem
            if session_id not in known_ids:
                sessions.append({
                    "session_id": session_id,
                    "turn_count": None,  # Would need to load to check
                    "active": False,
                    "file_exists": True
                })
                known_ids.add(session_id)

        return sessions

    def get_session_history(
            self,
            session_id: str,
            n: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session identifier
            n: Number of recent turns (None for all)

        Returns:
            List of ConversationTurn objects
        """
        memory = self.get_session(session_id)
        if memory is None:
            return []

        if n is not None:
            return memory.get_recent_turns(n=n)
        return memory.get_all_turns()

    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID."""
        return f"session_{uuid.uuid4().hex[:12]}"


# =============================================================================
# Main ConversationalRAGPipeline Class
# =============================================================================

class ConversationalRAGPipeline(RAGPipeline):
    """
    RAG Pipeline with Conversation Memory Integration.

    Extends the base RAGPipeline to support multi-turn conversations with:
    - Session-based conversation memory
    - Automatic query reformulation for reference resolution
    - Conversation-aware prompt building
    - Entity extraction and turn storage

    Example:
        ```python
        from src.pipeline import ConversationalRAGPipeline, ConversationalPipelineConfig

        # Initialize with memory enabled
        config = ConversationalPipelineConfig(
            collection_name="knowledge_base",
            qdrant_url="http://localhost:6333",
            enable_memory=True
        )
        pipeline = ConversationalRAGPipeline(
            collection_name="knowledge_base",
            config=config
        )

        # First query
        result1 = pipeline.run("What is PL?")
        session_id = result1.session_id

        # Follow-up query (references resolved automatically)
        result2 = pipeline.run("Who created this?", session_id=session_id)
        # "this" is resolved to "PL" based on conversation history
        ```
    """

    def __init__(
            self,
            collection_name: str = "knowledge_base",
            config: Optional[ConversationalPipelineConfig] = None
    ):
        """
        Initialize Conversational RAG Pipeline.

        Args:
            collection_name: Qdrant collection name
            config: Pipeline configuration (use ConversationalPipelineConfig)
        """
        # Use default config if not provided
        if config is None:
            config = ConversationalPipelineConfig()

        # Store conversational config before calling parent init
        self.conv_config = config if isinstance(config,
                                                ConversationalPipelineConfig) else ConversationalPipelineConfig()

        # Initialize base pipeline
        super().__init__(collection_name=collection_name, config=config)

        # Auto-configure structured query fast mode based on collection metadata.
        self._auto_configure_structured_fast_mode()

        # Initialize web retriever
        self.web_retriever = WebRetriever()
        # Initialize memory components if enabled
        if self.conv_config.enable_memory:
            self._init_memory_components()
        else:
            self.session_manager = None
            self.query_reformulator = None
            self.conversational_prompt_builder = None
            self.follow_up_detector = None
            self.response_summarizer = None
            self.query_classifier_enhanced = None
            self.meta_handler = None
            self.formatting_handler = None
            self.embedder = None
            self.cache_manager = None
            logger.info("Memory disabled - running in stateless mode")

    def _init_memory_components(self):
        """Initialize memory-related components."""
        logger.info("\nInitializing Memory Components...")
        logger.info("-" * 80)

        # Session Manager
        self.session_manager = SessionManager(
            memory_directory=self.conv_config.memory_directory,
            max_turns=self.conv_config.max_turns,
            auto_save=self.conv_config.auto_save,
            redis_enabled=self.conv_config.redis_enabled,
            redis_url=self.conv_config.redis_url,
            redis_key_prefix=self.conv_config.redis_key_prefix,
            redis_socket_timeout_seconds=self.conv_config.redis_socket_timeout_seconds,
            redis_connect_timeout_seconds=self.conv_config.redis_connect_timeout_seconds,
            redis_session_ttl_seconds=self.conv_config.session_store_ttl_seconds,
            session_read_through_enabled=self.conv_config.session_read_through_enabled,
            redis_session_namespace=self.conv_config.redis_session_namespace,
        )
        logger.info("SessionManager initialized")

        # Query Reformulator
        if self.conv_config.enable_reformulation:
            reformulator_config = ReformulatorConfig(
                use_llm=self.conv_config.use_llm_reformulation,
                confidence_threshold=self.conv_config.reformulation_confidence_threshold
            )
            # Pass LLM generator from parent pipeline for LLM-based reformulation
            llm_gen = self.llm_generator if self.conv_config.use_llm_reformulation else None
            self.query_reformulator = QueryReformulator(
                config=reformulator_config,
                llm_generator=llm_gen
            )
            logger.info(
                f"QueryReformulator initialized (use_llm={self.conv_config.use_llm_reformulation}, has_llm={llm_gen is not None})")
        else:
            self.query_reformulator = None
            logger.info("QueryReformulator disabled")

        # Ensure the prompt template contains Google Workspace context template block dynamically
        template_str = self.conv_config.prompt_template
        if template_str and "mcp_context" not in template_str:
            # Let's insert the Google Workspace / Slack context rendering block
            mcp_block = (
                "\n\n"
                "{% if mcp_context %}\n"
                "{% if mcp_service == 'slack' %}\n"
                "=== Slack Context ===\n"
                "{% elif mcp_service == 'notion' %}\n"
                "=== Notion Context ===\n"
                "{% else %}\n"
                "=== Google Workspace Context ({{ mcp_service | capitalize }}) ===\n"
                "{% endif %}\n"
                "Tool used: {{ mcp_tool }}\n"
                "Retrieved data:\n"
                "{{ mcp_context }}\n"
                "{% if mcp_service == 'slack' %}\n"
                "=== End Slack Context ===\n"
                "{% elif mcp_service == 'notion' %}\n"
                "=== End Notion Context ===\n"
                "{% else %}\n"
                "=== End Google Workspace Context ===\n"
                "{% endif %}\n"
                "{% endif %}\n\n"
            )
            # Find insertion point
            if "Current Question:" in template_str:
                template_str = template_str.replace("Current Question:", f"{mcp_block}Current Question:", 1)
            else:
                template_str = template_str + mcp_block
            
            # Ensure response guidelines contain Google Workspace context instructions
            if "Google Workspace Context" not in template_str:
                mcp_guideline = (
                    "\n- If Google Workspace Context is present, synthesize the retrieved email, calendar, or file "
                    "information into a natural language response. Avoid presenting raw JSON/metadata fields, "
                    "explain them in plain English."
                    "\n- If Notion Context is present, summarize the retrieved pages, databases, or page content "
                    "clearly and naturally. Reference page titles and relevant details."
                )
                if "### Response Guidelines" in template_str:
                    template_str = template_str.replace(
                        "### Response Guidelines",
                        f"### Response Guidelines{mcp_guideline}",
                        1
                    )
            
            self.conv_config.prompt_template = template_str
            logger.info("Dynamically injected mcp_context rendering block and response guidelines to inline prompt template")

        # Conversational Prompt Builder
        prompt_config = ConversationalPromptConfig(
            max_history_turns=self.conv_config.history_in_prompt_turns,
            history_answer_max_length=self.conv_config.history_answer_max_length
        )
        self.conversational_prompt_builder = create_conversational_prompt_builder(
            template=self.conv_config.prompt_template,
            template_name=self.conv_config.prompt_template_name,
            config=prompt_config
        )
        template_label = (
            "inline" if self.conv_config.prompt_template else self.conv_config.prompt_template_name
        )
        logger.info(f"ConversationalPromptBuilder initialized (template={template_label})")

        # Follow-up Detector
        self.follow_up_detector = FollowUpDetector()
        logger.info("FollowUpDetector initialized")

        # Response Summarizer (reuses LLM generator)
        self.response_summarizer = ResponseSummarizer(llm_generator=self.llm_generator)
        logger.info("ResponseSummarizer initialized")

        # Query Classifier (NEW)
        self.query_classifier_enhanced = QueryClassifierEnhanced()
        logger.info("QueryClassifierEnhanced initialized")

        # Meta-Conversation Handler (NEW)
        self.meta_handler = MetaConversationHandler()
        logger.info("MetaConversationHandler initialized")

        # Formatting Request Handler (NEW)
        self.formatting_handler = FormattingRequestHandler(llm_generator=self.llm_generator)
        logger.info("FormattingRequestHandler initialized")

        # Store reference to dense retriever's embedder (for continuation detection)
        # This reuses the already-loaded BGE model for semantic similarity
        self.embedder = self.dense_retriever.embedder if hasattr(self.dense_retriever, 'embedder') else None
        if self.embedder:
            logger.info("Embedder reference stored for continuation detection (reusing BGE model)")
        else:
            logger.warning("No embedder found in dense retriever - continuation detection will use patterns only")

        # Query Cache Manager (NEW)
        if self.conv_config.enable_cache:
            from src.caching import QueryCacheManager, CacheConfig

            cache_config = CacheConfig(
                cache_file=self.conv_config.cache_file,
                max_cache_size=self.conv_config.max_cache_size,
                ttl_hours=self.conv_config.cache_ttl_hours,
                similarity_threshold=self.conv_config.cache_similarity_threshold,
                enable_semantic_matching=True,  # Use embedder for semantic matching
                redis_enabled=self.conv_config.cache_redis_enabled,
                redis_url=self.conv_config.redis_url,
                redis_key_prefix=self.conv_config.redis_key_prefix,
                redis_socket_timeout_seconds=self.conv_config.redis_socket_timeout_seconds,
                redis_connect_timeout_seconds=self.conv_config.redis_connect_timeout_seconds,
                redis_ttl_seconds=self.conv_config.cache_redis_ttl_seconds,
            )
            self.cache_manager = QueryCacheManager(
                config=cache_config,
                embedder=self.embedder  # Reuse BGE embedder for semantic similarity!
            )
            logger.info("QueryCacheManager initialized (cache enabled)")
        else:
            self.cache_manager = None
            logger.info("QueryCacheManager disabled")

    def _auto_configure_structured_fast_mode(self) -> None:
        """
        Enable structured fast mode only when required payload metadata exists.

        If `structured_query_fast_mode` is explicitly set (True/False), respect it.
        Otherwise, sample payloads to determine whether fast paths are safe.
        """
        if self.conv_config.structured_query_fast_mode is not None:
            logger.info(
                "Structured fast mode explicitly set to %s",
                self.conv_config.structured_query_fast_mode,
            )
            return

        required_keys = ["entity_names", "document_entity_counts", "is_first_chunk"]

        try:
            from src.indexing.vector_store import VectorStore

            vector_store = VectorStore(
                collection_name=self.collection_name,
                qdrant_url=self.config.qdrant_url,
            )
            missing_key = any(
                vector_store.has_missing_payload_key(payload_key)
                for payload_key in required_keys
            )

            if missing_key:
                self.conv_config.structured_query_fast_mode = False
                self.conv_config.structured_natural_response_style = True
                logger.info(
                    "Structured fast mode disabled (missing payload metadata in %s)",
                    self.collection_name,
                )
            else:
                self.conv_config.structured_query_fast_mode = True
                logger.info(
                    "Structured fast mode enabled (metadata present) for %s",
                    self.collection_name,
                )
        except Exception as exc:
            logger.warning(
                "Structured fast-mode auto-detection failed for %s: %s",
                self.collection_name,
                exc,
            )
            self.conv_config.structured_query_fast_mode = False
            self.conv_config.structured_natural_response_style = True

        logger.info("-" * 80)

    def _get_session_redis_client(self):
        redis_conn = None
        if self.session_manager is not None:
            redis_conn = getattr(self.session_manager, "_redis_connection", None)
        if redis_conn is None:
            shared = getattr(self, "_shared_resources", None)
            if shared is not None:
                redis_conn = getattr(shared, "redis_connection", None)
        if redis_conn is None:
            return None
        try:
            return redis_conn.client()
        except Exception as exc:
            logger.warning("Session Redis client unavailable: %s", exc)
            return None

    def _session_has_uploaded_docs(self, session_id: Optional[str]) -> bool:
        if not session_id:
            return False
        redis_client = self._get_session_redis_client()
        if redis_client is None:
            return False
        return session_has_uploaded_docs(redis_client, session_id)

    def _retrieve_session_documents(
        self,
        session_id: Optional[str],
        search_query: str,
        top_k: int = 5,
    ) -> list:
        if not session_id:
            return []
        q_lower = (search_query or "").lower()
        if any(token in q_lower for token in ("chapter", "table of contents", "sections", "outline")):
            top_k = max(top_k, 8)
            logger.info("Expanded session doc top_k to %s for structure/outline query", top_k)
        redis_client = self._get_session_redis_client()
        if redis_client is None:
            logger.warning("Cannot retrieve session documents — Redis unavailable")
            return []
        try:
            all_session_chunks = get_session_docs(redis_client, session_id)
            if not all_session_chunks:
                logger.info("No session document chunks stored for session_id=%s", session_id)
                return []

            embedding_res = self.dense_retriever.embedder.run(text=search_query)
            query_vec = embedding_res["embedding"]
            top_session_chunks = cosine_similarity_search(
                query_vec,
                all_session_chunks,
                top_k=top_k,
                session_id=session_id,
            )

            session_docs = []
            for chunk in top_session_chunks:
                session_docs.append(Document(
                    content=chunk["text"],
                    meta={
                        "source": chunk["metadata"].get("filename", "session_doc"),
                        "source_filename": chunk["metadata"].get("filename", "session_doc"),
                        "chunk_id": chunk["id"],
                        "session_id": session_id,
                        "is_session_doc": True,
                        "relevance": chunk["score"],
                    },
                    score=chunk["score"],
                ))
            logger.info(
                "Retrieved %d session document chunks for session_id=%s (query=%r)",
                len(session_docs),
                session_id,
                search_query[:80],
            )
            return session_docs
        except Exception as exc:
            logger.error("Session document retrieval failed: %s", exc, exc_info=True)
            return []

    async def resolve_mcp_context_async(
        self,
        query: str,
        user_id: Optional[str],
        connector: Optional[str],
        session_id: Optional[str] = None,
        google_file_id: Optional[str] = None,
        google_file_name: Optional[str] = None,
        google_calendar_id: Optional[str] = None,
        google_calendar_name: Optional[str] = None,
        gmail_location: Optional[str] = None,
        gmail_category: Optional[str] = None,
        outlook_folder: Optional[str] = None,
        outlook_location: Optional[str] = None,
        outlook_message_id: Optional[str] = None,
        microsoft_file_id: Optional[str] = None,
        microsoft_file_name: Optional[str] = None,
        microsoft_drive_path: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Resolve MCP context (Google Workspace / Microsoft 365 / Slack) asynchronously.
        Returns a tuple of (mcp_context, mcp_service, mcp_tool).
        """
        try:
            from src.api.multi_kb_server import _mcp_client as _active_mcp_client
        except ImportError:
            _active_mcp_client = None

        mcp_context, mcp_service, mcp_tool = None, None, None
        used_mcp = False

        if session_id and self._session_has_uploaded_docs(session_id):
            if not _query_explicitly_requests_external_connector(query, connector):
                logger.info(
                    "Session %s has uploaded document(s) — skipping connector MCP pre-resolution (connector=%r)",
                    session_id,
                    connector,
                )
                return mcp_context, mcp_service, mcp_tool

        # Load session memory and conversation history if available
        memory = None
        conversation_history = []
        if self.session_manager and session_id:
            try:
                _, memory = self.session_manager.get_or_create_session(session_id)
                if memory:
                    conversation_history = memory.get_history() or []
            except Exception as _mem_exc:
                logger.warning("Failed to load session memory in resolve_mcp_context_async: %s", _mem_exc)

        if google_file_id:
            logger.info(
                "resolve_mcp_context_async: @-mention file_id=%s file_name=%r connector=%r",
                google_file_id,
                google_file_name,
                connector,
            )
        if google_calendar_id:
            logger.info(
                "resolve_mcp_context_async: @-mention calendar_id=%s calendar_name=%r connector=%r",
                google_calendar_id,
                google_calendar_name,
                connector,
            )
        if gmail_location or gmail_category:
            logger.info(
                "resolve_mcp_context_async: #folder location=%r category=%r connector=%r",
                gmail_location,
                gmail_category,
                connector,
            )
        if microsoft_file_id:
            logger.info(
                "resolve_mcp_context_async: @-mention microsoft_file_id=%s file_name=%r drive_path=%r connector=%r",
                microsoft_file_id,
                microsoft_file_name,
                microsoft_drive_path,
                connector,
            )
        if outlook_folder or outlook_location:
            logger.info(
                "resolve_mcp_context_async: #folder outlook_folder=%r outlook_location=%r connector=%r",
                outlook_folder,
                outlook_location,
                connector,
            )

        if _active_mcp_client is not None and user_id and connector:
            _mcp_service = None
            _mcp_tool = None
            _mcp_params = {}
            c_lower = connector.lower().strip()
            if c_lower == "email":
                _mcp_service = "gmail"
                _mcp_tool = "search_threads"
                _mcp_params = {"query": query}

                from src.mcp.gmail_search import apply_gmail_follow_up_routing

                _follow_up = None
                if memory and memory.session.metadata:
                    _follow_up = apply_gmail_follow_up_routing(
                        query,
                        memory.session.metadata,
                        embedder=self.embedder,
                    )

                if _follow_up:
                    _mcp_tool, _mcp_params = _follow_up
                    logger.info("Gmail follow-up routed to get_thread from session memory")
                else:
                    try:
                        from src.mcp.intent_detector import MCPIntentDetector

                        _gmail_detector = MCPIntentDetector.get_instance()
                        _mcp_tool, _mcp_params = await _gmail_detector.detect_gmail_intent(query)
                    except Exception as _gmail_exc:
                        logger.warning("Gmail intent detection failed in resolve_mcp_context_async: %s", _gmail_exc)
                        from src.mcp.gmail_search import prepare_gmail_call_params

                        _mcp_tool, _mcp_params = prepare_gmail_call_params(
                            query,
                            gmail_location=gmail_location,
                            gmail_category=gmail_category,
                        )
                    if gmail_location or gmail_category:
                        from src.mcp.gmail_search import patch_gmail_call_params

                        _mcp_tool, _mcp_params = patch_gmail_call_params(
                            _mcp_tool,
                            _mcp_params,
                            gmail_location=gmail_location,
                            gmail_category=gmail_category,
                        )
                    if self.embedder:
                        _mcp_params["_embedder"] = self.embedder
            elif c_lower in ("drive", "google", "google_workspace"):
                _mcp_service = "drive"
                _mcp_tool = "search_files"
                _mcp_params = {"query": query}
                if google_file_id:
                    _mcp_params["file_id"] = google_file_id
                elif google_file_name:
                    _mcp_params["query"] = google_file_name
            elif c_lower == "calendar":
                _mcp_service = "calendar"
                _mcp_tool = "list_events"
                from src.mcp.calendar_search import prepare_calendar_call_params

                _mcp_params = prepare_calendar_call_params(
                    query,
                    calendar_id=google_calendar_id,
                    calendar_name=google_calendar_name,
                )
            elif c_lower == "sheets":
                _mcp_service = "sheets"
                _mcp_tool = "search_files"
                _mcp_params = {"query": query, "mime_type": "application/vnd.google-apps.spreadsheet,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"}
                if google_file_id:
                    _mcp_params["file_id"] = google_file_id
                elif google_file_name:
                    _mcp_params["query"] = google_file_name
            elif c_lower == "docs":
                _mcp_service = "docs"
                _mcp_tool = "search_files"
                _mcp_params = {"query": query, "mime_type": "application/vnd.google-apps.document,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword"}
                if google_file_id:
                    _mcp_params["file_id"] = google_file_id
                elif google_file_name:
                    _mcp_params["query"] = google_file_name
            elif c_lower == "presentation":
                _mcp_service = "presentation"
                _mcp_tool = "search_files"
                _mcp_params = {"query": query, "mime_type": "application/vnd.google-apps.presentation,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/vnd.ms-powerpoint"}
                if google_file_id:
                    _mcp_params["file_id"] = google_file_id
                elif google_file_name:
                    _mcp_params["query"] = google_file_name

            if _mcp_service and _mcp_tool and _active_mcp_client.is_authenticated(user_id, _mcp_service):
                try:
                    _mcp_result = _active_mcp_client.call_tool(
                        user_id=user_id,
                        service=_mcp_service,
                        tool=_mcp_tool,
                        params=_mcp_params,
                    )
                    _mcp_content = _mcp_result.get("content") or []
                    _mcp_text = "\n".join(
                        item.get("text", "") for item in _mcp_content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ).strip()
                    if _mcp_text and _is_low_value_mcp_response(_mcp_text):
                        logger.info(
                            "MCP low-value response for %s/%s — falling back to RAG/session docs",
                            _mcp_service,
                            _mcp_tool,
                        )
                        _mcp_text = ""
                    if _mcp_text:
                        logger.info("MCP tool executed: service=%s tool=%s user_id=%s", _mcp_service, _mcp_tool, user_id)
                        mcp_context = _mcp_text
                        mcp_service = _mcp_service
                        mcp_tool = _mcp_tool
                        used_mcp = True
                        if _mcp_service == "gmail":
                            _gmail_threads = _mcp_result.get("gmail_threads") or []
                            if _gmail_threads and memory and self.session_manager and session_id:
                                from src.mcp.gmail_search import persist_gmail_thread_memory

                                persist_gmail_thread_memory(memory, _gmail_threads)
                                try:
                                    self.session_manager.save_session(session_id)
                                except Exception as _save_exc:
                                    logger.warning("Failed to persist Gmail thread memory: %s", _save_exc)
                except Exception as _mcp_exc:
                    logger.warning("MCP tool call failed, falling back to RAG (service=%s tool=%s): %s", _mcp_service, _mcp_tool, _mcp_exc)

        # ── Microsoft 365 connector (Outlook + OneDrive) ─────────────────
        try:
            from src.api.multi_kb_server import _microsoft_mcp_client as _active_ms_client
        except ImportError:
            _active_ms_client = None

        if not used_mcp and _active_ms_client is not None and user_id and connector:
            _ms_service = None
            _ms_tool = None
            _ms_params: Dict[str, Any] = {}
            c_lower = connector.lower().strip()
            if c_lower == "outlook":
                _ms_service = "outlook"
                from src.mcp.outlook_search import (
                    apply_outlook_follow_up_routing,
                    patch_outlook_call_params,
                    prepare_outlook_call_params,
                )

                _follow_up = None
                if memory and memory.session.metadata:
                    _follow_up = apply_outlook_follow_up_routing(query, memory.session.metadata)
                if _follow_up:
                    _ms_tool, _ms_params = _follow_up
                else:
                    _ms_tool, _ms_params = prepare_outlook_call_params(
                        query,
                        message_id=outlook_message_id,
                        outlook_folder=outlook_folder,
                        outlook_location=outlook_location,
                    )
                if outlook_folder or outlook_location:
                    _ms_tool, _ms_params = patch_outlook_call_params(
                        _ms_tool,
                        _ms_params,
                        outlook_folder=outlook_folder,
                        outlook_location=outlook_location,
                    )
            elif c_lower in ("onedrive", "sharepoint"):
                _ms_service = "onedrive"
                from src.mcp.onedrive_search import prepare_onedrive_call_params

                _ms_tool, _ms_params = prepare_onedrive_call_params(
                    query,
                    item_id=microsoft_file_id,
                    file_name=microsoft_file_name,
                    drive_path=microsoft_drive_path,
                )

            if _ms_service and _ms_tool and _active_ms_client.is_authenticated(user_id, _ms_service):
                try:
                    _ms_result = _active_ms_client.call_tool(
                        user_id=user_id,
                        service=_ms_service,
                        tool=_ms_tool,
                        params=_ms_params,
                    )
                    _ms_content = _ms_result.get("content") or []
                    _ms_text = "\n".join(
                        item.get("text", "") for item in _ms_content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ).strip()
                    if _ms_text and _is_low_value_mcp_response(_ms_text):
                        logger.info(
                            "Microsoft MCP low-value response for %s/%s — falling back to RAG",
                            _ms_service,
                            _ms_tool,
                        )
                        _ms_text = ""
                    if _ms_text:
                        logger.info(
                            "Microsoft MCP tool executed: service=%s tool=%s user_id=%s",
                            _ms_service,
                            _ms_tool,
                            user_id,
                        )
                        mcp_context = _ms_text
                        mcp_service = _ms_service
                        mcp_tool = _ms_tool
                        used_mcp = True
                except Exception as _ms_exc:
                    logger.warning(
                        "Microsoft MCP tool call failed, falling back to RAG (service=%s tool=%s): %s",
                        _ms_service,
                        _ms_tool,
                        _ms_exc,
                    )

        # ── Slack connector ───────────────────────────────────────────
        try:
            from src.api.multi_kb_server import _slack_mcp_client as _active_slack_client
            from src.mcp.intent_detector import MCPIntentDetector
        except ImportError:
            _active_slack_client = None
            MCPIntentDetector = None

        _is_slack_req = False
        _slack_tool_detected = None
        _slack_params_detected = {}
        if not used_mcp and user_id:
            c_lower = connector.lower().strip() if connector else ""
            if c_lower == "slack":
                _is_slack_req = True
            elif not connector and session_id and session_id.startswith("slack_session_"):
                # Dynamically run intent detection to see if it's a specific Slack action (not just search)
                if MCPIntentDetector is not None:
                    try:
                        _intent_detector = MCPIntentDetector.get_instance()
                        _slack_tool_detected, _slack_params_detected = await _intent_detector.detect_slack_intent(query)
                        if _slack_tool_detected and _slack_tool_detected != "search_messages":
                            _is_slack_req = True
                    except Exception as _det_exc:
                        logger.warning("Slack intent detection failed during auto-routing check: %s", _det_exc)

        if not used_mcp and user_id and _is_slack_req:
            if _active_slack_client is not None and _active_slack_client.is_authenticated(user_id):
                try:
                    from src.mcp.slack_client import parse_slack_mentions

                    _slack_tool = "search_messages"
                    _slack_params = {"query": query}
                    _mentions = parse_slack_mentions(query)

                    if _mentions.get("channel"):
                        _slack_tool = "get_channel_history"
                        _slack_params = {
                            "query": query,
                            "channel_name": _mentions["channel"],
                        }
                        logger.info(
                            "Slack routing via #mention -> get_channel_history(channel=%r)",
                            _mentions["channel"],
                        )
                    elif _mentions.get("dm_user"):
                        _slack_tool = "get_channel_history"
                        _slack_params = {
                            "query": query,
                            "channel_name": _mentions["dm_user"],
                        }
                        logger.info(
                            "Slack routing via @mention -> get_channel_history(dm_user=%r)",
                            _mentions["dm_user"],
                        )
                    elif _slack_tool_detected is not None:
                        _slack_tool = _slack_tool_detected
                        _slack_params = _slack_params_detected
                    elif MCPIntentDetector is not None:
                        try:
                            _intent_detector = MCPIntentDetector.get_instance()
                            _slack_tool, _slack_params = await _intent_detector.detect_slack_intent(query)
                        except Exception as _det_exc:
                            logger.warning("Slack intent detection failed: %s, defaulting to search_messages", _det_exc)

                    if _slack_tool == "list_channels":
                        _slack_result = await _active_slack_client.list_channels(user_id=user_id)
                    elif _slack_tool == "list_dms":
                        _slack_result = await _active_slack_client._list_dms(user_id=user_id)
                    elif _slack_tool == "get_channel_history":
                        _target_ch = _slack_params.get("channel_name")
                        if not _target_ch or _target_ch in ("current", "this"):
                            if session_id and session_id.startswith("slack_session_"):
                                _target_ch = session_id.replace("slack_session_", "")
                        _slack_result = await _active_slack_client.get_channel_history(user_id=user_id, channel_name=_target_ch or "")
                    elif _slack_tool == "get_channel_members":
                        _target_ch = _slack_params.get("channel_name")
                        if not _target_ch or _target_ch in ("current", "this"):
                            if session_id and session_id.startswith("slack_session_"):
                                _target_ch = session_id.replace("slack_session_", "")
                        _slack_result = await _active_slack_client.get_channel_members(user_id=user_id, channel_name=_target_ch or "")
                    else:
                        _slack_result = await _active_slack_client.search_messages(user_id=user_id, query=_slack_params.get("query") or query)

                    _slack_content = _slack_result.get("content") or []
                    _slack_text = "\n".join(
                        item.get("text", "") for item in _slack_content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ).strip()
                    if _slack_text:
                        logger.info("Slack MCP executed: %s user_id=%s", _slack_tool, user_id)
                        mcp_context = _slack_text
                        mcp_service = "slack"
                        mcp_tool = _slack_tool
                except Exception as _slack_exc:
                    logger.warning("Slack MCP call failed, falling back to RAG: %s", _slack_exc)

        # ── Notion connector ──────────────────────────────────────────────────
        if not used_mcp and user_id and connector and connector.lower().strip() == "notion":
            try:
                from src.api.multi_kb_server import _notion_mcp_client as _active_notion_client
                from src.mcp.intent_detector import MCPIntentDetector
                from src.mcp.notion_query_parser import parse_database_hint
                from src.mcp.notion_query_planner import plan_notion_query
                from src.mcp.notion_task_memory import (
                    apply_notion_follow_up_routing,
                    extract_tasks_from_mcp_text,
                    persist_notion_task_memory,
                    pick_active_task_from_query,
                )
            except ImportError:
                _active_notion_client = None
                MCPIntentDetector = None

            if _active_notion_client is not None and _active_notion_client.is_authenticated(user_id):
                try:
                    _notion_tool = "search_pages"
                    _notion_params: Dict[str, Any] = {"query": query, "page_name": None}
                    _session_meta = memory.session.metadata if memory else {}

                    _followup_patterns = [
                        "is there more", "show me more", "anything else", "show more", "more results",
                        "what else", "any more", "next", "continue",
                    ]
                    _is_pagination_followup = any(
                        query.lower().strip() == p or query.lower().strip().startswith(p)
                        for p in _followup_patterns
                    )

                    _notion_result: Optional[Dict[str, Any]] = None

                    if _is_pagination_followup and conversation_history:
                        for _past_turn in reversed(conversation_history):
                            _past_meta = _past_turn.get("metadata", {})
                            if _past_meta.get("mcp_service") == "notion":
                                _last_q = _past_turn.get("reformulated_query", "")
                                if _last_q:
                                    logger.info(
                                        "Notion pagination follow-up — re-running prior query: %r",
                                        _last_q,
                                    )
                                    _notion_tool = "search_pages"
                                    _notion_result = await _active_notion_client.search_pages(
                                        user_id=user_id, query=_last_q, limit=100
                                    )
                                break
                    else:
                        _notion_follow_up = apply_notion_follow_up_routing(query, _session_meta)

                        if MCPIntentDetector is not None:
                            try:
                                _intent_detector = MCPIntentDetector.get_instance()
                                _notion_tool, _notion_params = await _intent_detector.detect_notion_intent(query)
                            except Exception as _det_exc:
                                logger.warning(
                                    "Notion intent detection failed: %s, defaulting to search_pages",
                                    _det_exc,
                                    exc_info=True,
                                )

                        _db_hint = parse_database_hint(query)
                        if not _db_hint:
                            _active_db = _session_meta.get("active_notion_database") or {}
                            _db_hint = _active_db.get("title")

                        _plan = plan_notion_query(
                            query,
                            _session_meta,
                            database_hint=_db_hint,
                            follow_up=_notion_follow_up,
                            intent_tool=_notion_tool,
                            intent_params=_notion_params,
                        )
                        _notion_tool = _plan.tool
                        logger.info(
                            "Notion query plan: tool=%s reason=%s uses_active_task=%s",
                            _plan.tool,
                            _plan.reason,
                            _plan.uses_active_task,
                        )

                        if _plan.tool == "query_database":
                            _notion_result = await _active_notion_client.query_database(
                                user_id=user_id,
                                filter_spec=_plan.params.get("filter_spec") or {},
                                database_hint=_plan.params.get("database_hint"),
                                limit=100,
                            )
                        elif _plan.tool == "get_page_details":
                            _notion_result = await _active_notion_client.get_page_details(
                                user_id=user_id,
                                page_id=_plan.params.get("page_id") or query,
                                focus_property=_plan.params.get("focus_property"),
                            )
                        elif _plan.tool == "search_databases":
                            _notion_result = await _active_notion_client.search_databases(
                                user_id=user_id,
                                query=_plan.params.get("query") or query,
                                limit=100,
                            )
                        else:
                            _notion_result = await _active_notion_client.search_pages(
                                user_id=user_id,
                                query=_plan.params.get("query") or query,
                                limit=100,
                            )

                    if _notion_result:
                        _notion_content = _notion_result.get("content") or []
                        _notion_text = "\n".join(
                            item.get("text", "") for item in _notion_content
                            if isinstance(item, dict) and item.get("type") == "text"
                        ).strip()
                        if _notion_text:
                            logger.info("Notion MCP executed: %s user_id=%s", _notion_tool, user_id)
                            mcp_context = _notion_text
                            mcp_service = "notion"
                            mcp_tool = _notion_tool

                            if memory and self.session_manager and session_id:
                                _tasks = extract_tasks_from_mcp_text(_notion_text)
                                _active_task = None
                                if _notion_tool == "get_page_details":
                                    _props = _notion_result.get("properties") or {}
                                    _active_task = {
                                        "page_id": _notion_result.get("page_id"),
                                        "title": _notion_result.get("title"),
                                        "url": _notion_result.get("url"),
                                        "properties": _props,
                                        "status": _props.get("Status", ""),
                                        "priority": _props.get("Priority", ""),
                                        "category": _props.get("Category", ""),
                                        "assignee": _props.get("Assigned To", ""),
                                        "due_date": _props.get("Due Date", _props.get("Due date", "")),
                                        "description": _props.get("Description", ""),
                                    }
                                elif _tasks:
                                    _active_task = pick_active_task_from_query(query, _tasks) or _tasks[0]

                                _db_info = None
                                if _notion_result.get("database_id"):
                                    _db_info = {
                                        "database_id": _notion_result.get("database_id"),
                                        "title": _notion_result.get("database_title"),
                                    }

                                persist_notion_task_memory(
                                    memory,
                                    _tasks,
                                    active_task=_active_task,
                                    database_info=_db_info,
                                )
                                try:
                                    self.session_manager.save_session(session_id)
                                except Exception as _save_exc:
                                    logger.warning(
                                        "Failed to persist Notion task memory: %s",
                                        _save_exc,
                                    )
                except Exception as _notion_exc:
                    logger.warning("Notion MCP call failed, falling back to RAG: %s", _notion_exc, exc_info=True)

        return mcp_context, mcp_service, mcp_tool

    def _execute_direct_mcp_tool(
            self,
            tool_name: str,
            params: Dict[str, Any],
            session: ConversationMemory,
            connector: Optional[str],
            user_id: Optional[str] = None,
            original_query: str = "",
            slash_command: Optional[str] = "read",
    ) -> ConversationalPipelineResult:
        """
        Execute a deterministic direct MCP tool call for explicit slash commands.
        """
        import asyncio

        async def _run_direct_tool():
            tool_output = ""
            annotations = {
                "tool": tool_name,
                "params": params,
                "connector": connector,
            }

            try:
                if connector and connector.lower().strip() == "slack":
                    from src.api.multi_kb_server import _slack_mcp_client as _active_slack_client
                    if _active_slack_client is None or not user_id or not _active_slack_client.is_authenticated(user_id):
                        raise RuntimeError("Slack MCP client unavailable or user unauthenticated")

                    if tool_name == "get_channel_history":
                        target = params.get("channel_name") or ""
                        if target in ("current", "this", "") and session.session_id.startswith("slack_session_"):
                            target = session.session_id.replace("slack_session_", "")
                        result = await _active_slack_client.get_channel_history(user_id=user_id, channel_name=target)
                    else:
                        result = await _active_slack_client.search_messages(user_id=user_id, query=params.get("query") or "")

                    content = result.get("content") or []
                    tool_output = "\n".join(
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ).strip()

                elif connector and connector.lower().strip() == "notion":
                    from src.api.multi_kb_server import _notion_mcp_client as _active_notion_client
                    if _active_notion_client is None or not user_id or not _active_notion_client.is_authenticated(user_id):
                        raise RuntimeError("Notion MCP client unavailable or user unauthenticated")

                    if tool_name in ("get_page_content", "get_page_details"):
                        page_name = params.get("page_name") or params.get("page_id") or ""
                        result = await _active_notion_client.get_page_details(
                            user_id=user_id,
                            page_id=page_name,
                        )
                    elif tool_name == "search_databases":
                        result = await _active_notion_client.search_databases(
                            user_id=user_id,
                            query=params.get("query") or "",
                            limit=25,
                        )
                    else:
                        result = await _active_notion_client.search_pages(
                            user_id=user_id, query=params.get("query") or "", limit=100
                        )

                    content = result.get("content") or []
                    tool_output = "\n".join(
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ).strip()

                elif connector and connector.lower().strip() in ["google", "google_workspace", "drive"]:
                    from src.api.multi_kb_server import _mcp_client as _active_mcp_client
                    if _active_mcp_client is None or not user_id or not _active_mcp_client.is_authenticated(user_id, "drive"):
                        raise RuntimeError("Google Workspace MCP client unavailable or user unauthenticated")

                    result = _active_mcp_client.call_tool(
                        user_id=user_id,
                        service="drive",
                        tool=tool_name,
                        params=params,
                    )
                    content = result.get("content") or []
                    tool_output = "\n".join(
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ).strip()
                elif connector and connector.lower().strip() in ("onedrive", "sharepoint"):
                    from src.api.multi_kb_server import _microsoft_mcp_client as _active_ms_client
                    if _active_ms_client is None or not user_id or not _active_ms_client.is_authenticated(user_id, "onedrive"):
                        raise RuntimeError("Microsoft 365 MCP client unavailable or user unauthenticated")

                    from src.mcp.onedrive_search import prepare_onedrive_call_params

                    ms_tool, ms_params = prepare_onedrive_call_params(
                        params.get("query") or "",
                        item_id=params.get("item_id") or params.get("file_id"),
                        file_name=params.get("file_name"),
                        drive_path=params.get("drive_path"),
                    )
                    if tool_name in ("search_files", "read_file", "read_drive_file"):
                        ms_tool = tool_name if tool_name in ("search_files", "read_file") else ms_tool
                    result = _active_ms_client.call_tool(
                        user_id=user_id,
                        service="onedrive",
                        tool=ms_tool,
                        params=ms_params,
                    )
                    content = result.get("content") or []
                    tool_output = "\n".join(
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ).strip()
                elif connector and connector.lower().strip() == "outlook":
                    from src.api.multi_kb_server import _microsoft_mcp_client as _active_ms_client
                    if _active_ms_client is None or not user_id or not _active_ms_client.is_authenticated(user_id, "outlook"):
                        raise RuntimeError("Microsoft Outlook MCP client unavailable or user unauthenticated")

                    result = _active_ms_client.call_tool(
                        user_id=user_id,
                        service="outlook",
                        tool=tool_name or "search_messages",
                        params=params,
                    )
                    content = result.get("content") or []
                    tool_output = "\n".join(
                        item.get("text", "") for item in content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ).strip()
                else:
                    raise RuntimeError("Unsupported connector for direct MCP tool execution")
            except Exception as exc:
                logger.warning("Direct MCP tool execution failed: %s", exc, exc_info=True)

            return tool_output, annotations

        try:
            tool_output, annotations = asyncio.run(_run_direct_tool())
        except RuntimeError as exc:
            # In case there is already a running event loop, fall back to synchronous placeholder.
            logger.warning("Unable to run direct MCP tool asynchronously: %s", exc)
            tool_output = ""
            annotations = {"tool": tool_name, "params": params, "connector": connector}

        return ConversationalPipelineResult(
            query=original_query,
            answer=tool_output or "",
            citations=[],
            metadata={"direct_tool_execution": True, **annotations},
            query_classification={"query_type": "direct_command"},
            routing_decision={
                "handler": "direct_mcp",
                "reason": f"explicit /{slash_command or 'read'} command",
            },
            retrieval_stats={"dense_count": 0, "sparse_count": 0, "web_count": 0, "session_count": 0, "fused_count": 0, "reranked_count": 0, "search_query": "", "parallel_execution": False, "web_search_enabled": False, "kb_search_enabled": False},
            context_stats={"verified": 0, "unique": 0, "final": 0},
            generation_stats={"token_count": 0},
            total_time=0.0,
            stage_times={},
            session_id=session.session_id,
            turn_number=len(session.get_all_turns()) + 1,
            was_reformulated=False,
            reformulated_query=original_query,
            reformulation_method="fast_path_direct_command",
            detected_references=[],
            extracted_entities=[],
            memory_stats={"active_mcp_tool": session.session.metadata.get("active_mcp_tool")}
        )


    def run(
            self,
            query: str,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
            cancel_event: Optional[threading.Event] = None,
            web: str = "off",
            kb: str = "on",
            connector: Optional[str] = None,
            mcp_context: Optional[str] = None,
            mcp_service: Optional[str] = None,
            mcp_tool: Optional[str] = None,
            google_file_id: Optional[str] = None,
            google_file_name: Optional[str] = None,
            google_calendar_id: Optional[str] = None,
            google_calendar_name: Optional[str] = None,
            gmail_location: Optional[str] = None,
            gmail_category: Optional[str] = None,
            outlook_folder: Optional[str] = None,
            outlook_location: Optional[str] = None,
            outlook_message_id: Optional[str] = None,
            microsoft_file_id: Optional[str] = None,
            microsoft_file_name: Optional[str] = None,
            microsoft_drive_path: Optional[str] = None,
    ) -> ConversationalPipelineResult:
        """
        Run conversational RAG pipeline.

        Args:
            query: User's question
            session_id: Session identifier (creates new if None)
            user_id: Optional user identifier
            cancel_event: Optional cooperative cancellation token

        Returns:
            ConversationalPipelineResult with answer, citations, and session info
        """
        pipeline_start = time.time()
        stage_times = {}

        logger.info("\n" + "=" * 80)
        logger.info("Starting Conversational RAG Pipeline")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")

        if _is_mail_connector(connector) and kb == "on":
            logger.info(
                "Mail connector %r — KB disabled; using MCP mail fetch + LLM only",
                connector,
            )
            kb = "off"

        # Variables for tracking
        was_reformulated = False
        reformulated_query = query
        reformulation_method = ""
        detected_references = []
        extracted_entities = []
        conversation_history = []
        turn_number = 0
        memory = None  # Initialize memory variable at function scope

        # Dummy classification and routing classes for MCP flow
        class DummyClassification:
            query_type = "workspace"
            complexity = 0.5
            requires_long_response = False
            def to_dict(self):
                return {"query_type": "workspace", "complexity": 0.5}

        class DummyRouting:
            def to_dict(self):
                return {"handler": "mcp", "reason": "workspace intent"}

        # Initialize defaults
        classification = DummyClassification()
        routing_decision = DummyRouting()
        is_continuation = False
        final_docs = []
        session_docs = []
        retrieval_stats = {
            "dense_count": 0,
            "sparse_count": 0,
            "web_count": 0,
            "session_count": 0,
            "fused_count": 0,
            "reranked_count": 0,
            "search_query": query,
            "parallel_execution": False,
            "web_search_enabled": False,
            "kb_search_enabled": False
        }
        context_stats = {
            "verified": 0,
            "unique": 0,
            "final": 0
        }
        citation_map = {}
        enhancement_info = None

        # Google Workspace MCP context variables
        used_mcp = False
        if mcp_context is not None:
            used_mcp = True
            logger.info("Using pre-resolved MCP context (service=%s tool=%s)", mcp_service, mcp_tool)
            if _is_low_value_mcp_response(mcp_context):
                logger.info("Pre-resolved MCP context is low-value — falling back to RAG/session docs")
                used_mcp = False
                mcp_context = None
                mcp_service = None
                mcp_tool = None

        session_has_uploaded_doc = False

        def _check_cancel(stage_name: str) -> None:
            if cancel_event is not None and cancel_event.is_set():
                logger.warning("Pipeline cancellation requested at stage: %s", stage_name)
                raise PipelineCancelledError(f"Pipeline cancelled during {stage_name}")

        try:
            _check_cancel("start")

            slash_result = None
            try:
                from src.mcp.slash_commands import parse_slash_command

                slash_result = parse_slash_command(query, connector)
            except Exception as _slash_exc:
                logger.warning("Slash command parse failed: %s", _slash_exc)
                slash_result = None

            if slash_result and slash_result.kind == "rewrite" and slash_result.rewritten_query:
                logger.info(
                    "Slash /%s rewritten query: %r -> %r",
                    slash_result.command,
                    query,
                    slash_result.rewritten_query,
                )
                query = slash_result.rewritten_query

            if slash_result and slash_result.kind == "help":
                help_answer = slash_result.help_text or ""
                logger.info("Slash /help for connector=%r", connector)
                return ConversationalPipelineResult(
                    query=query,
                    answer=help_answer,
                    citations=[],
                    metadata={"slash_command": slash_result.command},
                    query_classification={"query_type": "slash_help"},
                    routing_decision={
                        "handler": "slash_help",
                        "reason": f"explicit /{slash_result.command} command",
                    },
                    retrieval_stats={
                        "dense_count": 0,
                        "sparse_count": 0,
                        "web_count": 0,
                        "session_count": 0,
                        "fused_count": 0,
                        "reranked_count": 0,
                        "search_query": "",
                        "parallel_execution": False,
                        "web_search_enabled": False,
                        "kb_search_enabled": False,
                    },
                    context_stats={"verified": 0, "unique": 0, "final": 0},
                    generation_stats={"token_count": 0},
                    total_time=0.0,
                    stage_times={},
                    session_id=session_id or "stateless",
                    turn_number=1,
                    was_reformulated=False,
                    reformulated_query=query,
                    reformulation_method="slash_help",
                    detected_references=[],
                    extracted_entities=[],
                    memory_stats={},
                )

            # =================================================================
            # STAGE 0: Session Management (if memory enabled)
            # =================================================================
            if self.conv_config.enable_memory and self.session_manager:
                stage_start = time.time()
                logger.info("\nSTAGE 0: Session Management")
                logger.info("-" * 80)

                # Get or create session
                session_id, memory = self.session_manager.get_or_create_session(session_id)
                conversation_history = memory.get_recent_turns(
                    n=self.conv_config.history_in_prompt_turns
                )
                turn_number = len(memory.get_all_turns()) + 1

                logger.info(f"Session: {session_id}")
                logger.info(f"Turn: {turn_number}")
                logger.info(f"History turns available: {len(conversation_history)}")
                session_has_uploaded_doc = self._session_has_uploaded_docs(session_id)
                if session_has_uploaded_doc:
                    logger.info("Session has uploaded document(s) available for retrieval")

                stage_times["session_management"] = time.time() - stage_start

                # =================================================================
                # STAGE 0.1: Fast-path direct MCP slash commands (/read, /page, …)
                # =================================================================
                if slash_result and slash_result.kind == "direct_mcp":
                    mcp_tool = slash_result.tool
                    parsed_params = slash_result.params or {}
                    clean_target_name = (
                        parsed_params.get("channel_name")
                        or parsed_params.get("page_name")
                        or parsed_params.get("file_name")
                        or parsed_params.get("query")
                    )

                    if connector and connector.lower().strip() == "slack":
                        memory.session.metadata["active_slack_target"] = clean_target_name
                        memory.session.metadata["active_mcp_tool"] = mcp_tool
                    elif connector and connector.lower().strip() == "notion":
                        memory.session.metadata["active_notion_target"] = clean_target_name
                        memory.session.metadata["active_mcp_tool"] = mcp_tool
                    elif connector and connector.lower().strip() in [
                        "google",
                        "google_workspace",
                        "drive",
                    ]:
                        memory.session.metadata["active_google_target"] = clean_target_name
                        memory.session.metadata["active_mcp_tool"] = mcp_tool
                    elif connector and connector.lower().strip() in ("onedrive", "sharepoint"):
                        memory.session.metadata["active_microsoft_target"] = clean_target_name
                        memory.session.metadata["active_mcp_tool"] = mcp_tool

                    if self.session_manager:
                        self.session_manager.save_session(session_id)

                    return self._execute_direct_mcp_tool(
                        tool_name=mcp_tool,
                        params=parsed_params,
                        session=memory,
                        connector=connector,
                        user_id=user_id,
                        original_query=query,
                        slash_command=slash_result.command,
                    )
            else:
                session_id = session_id or "stateless"
                memory = None
                session_has_uploaded_doc = (
                    self._session_has_uploaded_docs(session_id)
                    if session_id != "stateless" else False
                )

            _check_cancel("session_management")

            # End MCP variables mapping

            # =================================================================
            # STAGE 0.3: Query Classification and Routing (NEW)
            # =================================================================
            classified_query_type = "new_query"
            classification_metadata = None

            # CRITICAL: Run enhanced classification even on first queries (no history needed)
            # Enhanced classifier handles ENTITY_COUNT, FILE_LOCATION, EXACT_TEXT
            if not used_mcp and self.conv_config.enable_memory:
                result, classified_query_type, classification_metadata = classify_and_route_query(
                    query=query,
                    conversation_history=conversation_history,
                    memory=memory,
                    query_classifier=self.query_classifier_enhanced,
                    meta_handler=self.meta_handler,
                    formatting_handler=self.formatting_handler,
                    llm_generator=self.llm_generator,
                    session_id=session_id,
                    turn_number=turn_number,
                    pipeline_start=pipeline_start,
                    stage_times=stage_times,
                    ConversationalPipelineResult=ConversationalPipelineResult,
                    collection_name=self.collection_name,  # Pass collection name from pipeline
                    structured_query_fast_mode=self.conv_config.structured_query_fast_mode,
                    structured_entity_resolution=self.conv_config.structured_entity_resolution,
                    structured_natural_response_style=self.conv_config.structured_natural_response_style,
                )

                # If query was handled (meta or formatting), return immediately
                if result is not None:
                    if self.conv_config.auto_save and self.session_manager:
                        self.session_manager.save_session(session_id)
                    return result

            _check_cancel("query_classification_and_routing")

            # =================================================================
            # STAGE 0.5: Enhanced Continuation Detection
            # =================================================================
            follow_up_result = None
            is_continuation = False
            continuation_confidence = 0.0

            if not used_mcp and (self.conv_config.enable_memory and
                    conversation_history):

                stage_start = time.time()
                logger.info("\\nSTAGE 0.5: Enhanced Continuation Detection")
                logger.info("-" * 80)

                last_turn = conversation_history[-1]

                # Use QueryClassifier's continuation detection (pattern + semantic similarity)
                is_continuation, continuation_confidence, cont_metadata = self.query_classifier.detect_continuation(
                    query=query,
                    last_turn=last_turn,
                    embedder=self.embedder  # Reuse BGE embeddings!
                )

                if is_continuation:
                    logger.info(f"✓ Continuation detected (confidence: {continuation_confidence:.2f})")
                    logger.info(f"  Reason: {cont_metadata.get('reason', 'unknown')}")
                    if 'semantic_similarity' in cont_metadata:
                        logger.info(f"  Semantic similarity: {cont_metadata['semantic_similarity']:.3f}")
                    if 'last_query' in cont_metadata:
                        logger.info(f"  Previous query: {cont_metadata['last_query']}")

                    # Log active topic if available
                    if memory and memory.session.active_topic:
                        logger.info(f"  Active topic: {memory.session.active_topic}")
                else:
                    logger.info(f"✗ New topic detected")
                    logger.info(f"  Reason: {cont_metadata.get('reason', 'unknown')}")

                stage_times["continuation_detection"] = time.time() - stage_start

            _check_cancel("continuation_detection")

            # =================================================================
            # STAGE 1: Context-Aware Query Reformulation
            # =================================================================
            search_query = query

            if not used_mcp and (self.conv_config.enable_memory and
                    self.conv_config.enable_reformulation and
                    self.query_reformulator and
                    conversation_history):

                stage_start = time.time()
                logger.info("\nSTAGE 1: Context-Aware Query Reformulation")
                logger.info("-" * 80)

                # Check if reformulation needed
                needs_reform, patterns = self.query_reformulator.needs_reformulation(
                    query, conversation_history
                )

                if needs_reform or is_continuation:  # Also reformulate continuations
                    if needs_reform:
                        logger.info(f"Query needs reformulation (patterns: {patterns})")
                    if is_continuation:
                        logger.info(f"Continuation query - adding topic context")

                    # Get detected references for result
                    refs = self.query_reformulator.get_detected_references(query)
                    detected_references = refs.get("pronouns", []) + refs.get("meta", [])

                    # Reformulate with active topic context
                    reform_result = self.query_reformulator.reformulate(
                        query,
                        conversation_history,
                        active_topic=(
                            None
                            if session_has_uploaded_doc
                            else (memory.session.active_topic if memory else None)
                        ),
                    )

                    if reform_result.was_reformulated:
                        was_reformulated = True
                        reformulated_query = reform_result.reformulated_query
                        reformulation_method = reform_result.method
                        search_query = reformulated_query

                        logger.info(f"Reformulated: '{query}' -> '{reformulated_query}'")
                        logger.info(f"Method: {reformulation_method}")
                        if memory and memory.session.active_topic:
                            logger.info(f"Used active topic: {memory.session.active_topic}")
                    else:
                        logger.info("Reformulation not applied (low confidence)")
                else:
                    logger.info("Query is self-contained, no reformulation needed")

                stage_times["reformulation"] = time.time() - stage_start

            _check_cancel("reformulation")

            # =================================================================
            # STAGE 1.5: Cache Lookup (NEW)
            # =================================================================
            # Check if query is cacheable and if we have a cached response
            # This happens AFTER reformulation but BEFORE retrieval to save costs

            if not used_mcp and self.cache_manager and self.follow_up_detector:
                stage_start = time.time()
                logger.info("\\nSTAGE 1.5: Cache Lookup")
                logger.info("-" * 80)

                # Determine if query is standalone (cacheable)
                is_standalone = self.follow_up_detector.is_standalone_query(
                    query, conversation_history
                )

                if is_standalone:
                    logger.info(f"Query is standalone (cacheable)")

                    # Try to get cached response
                    cached_response = self.cache_manager.get_cached_response(query)

                    if cached_response:
                        logger.info(f"✓ Cache HIT for query: '{query}'")
                        logger.info(f"  Original cached query: '{cached_response.original_query}'")
                        logger.info(f"  Hit count: {cached_response.hit_count}")
                        logger.info(f"  Cached at: {cached_response.timestamp}")

                        stage_times["cache_lookup"] = time.time() - stage_start

                        # Return cached result (will be added to memory)
                        return self._create_cached_result(
                            query=query,
                            cached_response=cached_response,
                            session_id=session_id,
                            turn_number=turn_number,
                            memory=memory,
                            stage_times=stage_times,
                            pipeline_start=pipeline_start
                        )
                    else:
                        logger.info(f"✗ Cache MISS for query: '{query}'")
                else:
                    logger.info(f"Query is follow-up (not cacheable)")

                stage_times["cache_lookup"] = time.time() - stage_start

            _check_cancel("cache_lookup")

            # =================================================================
            # STAGE 2-4: Run Base Pipeline (Query Processing, Retrieval, Context)
            # =================================================================
            # Call parent's run method for core RAG logic
            # Note: We use search_query (possibly reformulated) for retrieval

            if not used_mcp:
                stage_start = time.time()
                logger.info("\nSTAGE 2-4: Core RAG Pipeline")
                logger.info("-" * 80)

                # We need to run the pipeline stages manually to use reformulated query
                # and conversational prompt builder

                # Stage 2: Query Processing
                query_obj = self.query_handler.handle_query(
                    query_text=search_query,
                    user_id=user_id,
                    session_id=session_id
                )
                _check_cancel("query_processing")

                # Check for chitchat FIRST (fast path - no retrieval needed)
                if query_obj.metadata.is_chitchat:
                    logger.info(f"Chitchat detected ({query_obj.metadata.chitchat_type}), returning canned response")
                    stage_times["query_processing"] = time.time() - stage_start

                    # Return canned response immediately without retrieval
                    return self._create_chitchat_result(
                        query=query,
                        session_id=session_id,
                        turn_number=turn_number,
                        query_obj=query_obj,
                        stage_times=stage_times,
                        pipeline_start=pipeline_start,
                        memory=memory
                    )

                if not query_obj.is_valid:
                    raise ValueError(f"Invalid query: {query_obj.metadata.validation_message}")

                # Strip greeting words from search_query for better retrieval
                # This happens AFTER chitchat detection (so pure greetings are already handled)
                # but BEFORE classification and retrieval
                # Example: "hi who is Jeremy Salvador?" → "who is Jeremy Salvador?"
                cleaned_search_query = self.query_handler.validator.strip_greeting_words(search_query)
                if cleaned_search_query != search_query:
                    logger.info(f"Stripped greeting from search query: '{search_query}' → '{cleaned_search_query}'")
                    search_query = cleaned_search_query

                classification = self.query_classifier.classify(search_query)
                routing_decision = self.query_router.route(classification, connector=connector)

                logger.info(f"Query classified: {classification.query_type}")
                logger.info(f"Routing: {routing_decision.route}")

                if routing_decision.route == "reject":
                    logger.warning("Query rejected by router")
                    return self._create_conversational_rejection_result(
                        query=query,
                        session_id=session_id,
                        turn_number=turn_number,
                        classification=classification,
                        routing_decision=routing_decision,
                        stage_times=stage_times,
                        pipeline_start=pipeline_start,
                        was_reformulated=was_reformulated,
                        reformulated_query=reformulated_query
                    )

                # Handle Google Workspace MCP route - call MCP tool instead of standard retrieval
                if routing_decision.route == "google_workspace_mcp":
                    logger.info("Google Workspace MCP route detected - executing MCP tool call instead of standard retrieval")
                    try:
                        from src.api.multi_kb_server import _mcp_client as _active_mcp_client
                        if _active_mcp_client is None:
                            logger.error("Google Workspace MCP client not available")
                            return self._create_conversational_no_results(
                                query=query,
                                session_id=session_id,
                                turn_number=turn_number,
                                classification=classification,
                                routing_decision=routing_decision,
                                stage_times=stage_times,
                                pipeline_start=pipeline_start,
                                was_reformulated=was_reformulated,
                                reformulated_query=reformulated_query
                            )

                        # Determine service and tool based on connector
                        service = "drive"  # default
                        tool = "search_files"
                        params = {"query": search_query}
                        if google_file_id:
                            params["file_id"] = google_file_id
                            logger.info(
                                "Drive @-mention file_id=%s name=%r — direct fetch, skipping name search",
                                google_file_id,
                                google_file_name,
                            )
                        elif google_file_name:
                            params["query"] = google_file_name
                            logger.info(
                                "Drive @-mention file_name=%r — targeted name search",
                                google_file_name,
                            )

                        if connector:
                            connector_lower = connector.lower().strip()
                            if connector_lower in ["email", "gmail"]:
                                service = "gmail"
                                from src.mcp.gmail_search import (
                                    apply_gmail_follow_up_routing,
                                    persist_gmail_thread_memory,
                                    prepare_gmail_call_params,
                                )

                                _follow_up = None
                                if memory and memory.session.metadata:
                                    _follow_up = apply_gmail_follow_up_routing(
                                        search_query,
                                        memory.session.metadata,
                                        embedder=self.embedder,
                                    )
                                if _follow_up:
                                    tool, params = _follow_up
                                else:
                                    tool, params = prepare_gmail_call_params(
                                        search_query,
                                        embedder=self.embedder,
                                        gmail_location=gmail_location,
                                        gmail_category=gmail_category,
                                    )
                                if gmail_location or gmail_category:
                                    from src.mcp.gmail_search import patch_gmail_call_params

                                    tool, params = patch_gmail_call_params(
                                        tool,
                                        params,
                                        gmail_location=gmail_location,
                                        gmail_category=gmail_category,
                                    )
                            elif connector_lower in ["calendar"]:
                                service = "calendar"
                                tool = "list_events"
                                from src.mcp.calendar_search import prepare_calendar_call_params

                                params = prepare_calendar_call_params(
                                    search_query,
                                    calendar_id=google_calendar_id,
                                    calendar_name=google_calendar_name,
                                )
                                if google_calendar_id:
                                    logger.info(
                                        "Calendar @-mention calendar_id=%s name=%r — scoped search",
                                        google_calendar_id,
                                        google_calendar_name,
                                    )
                            elif connector_lower in ["sheets", "spreadsheet"]:
                                service = "drive"
                                tool = "search_files"
                                params["mime_type"] = "application/vnd.google-apps.spreadsheet"
                            elif connector_lower in ["docs", "document"]:
                                service = "drive"
                                tool = "search_files"
                                params["mime_type"] = "application/vnd.google-apps.document"
                            elif connector_lower in ["presentation", "slides"]:
                                service = "drive"
                                tool = "search_files"
                                params["mime_type"] = "application/vnd.google-apps.presentation"

                        logger.info(f"Calling Google Workspace MCP: service={service}, tool={tool}, query={search_query}")

                        # Call the MCP tool
                        mcp_result = _active_mcp_client.call_tool(
                            user_id=user_id,
                            service=service,
                            tool=tool,
                            params=params,
                        )

                        # Extract content from MCP response
                        content = mcp_result.get("content") or []
                        mcp_text = "\n".join(
                            item.get("text", "") for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        ).strip()

                        if not mcp_text:
                            logger.warning("Google Workspace MCP returned empty content")
                            return self._create_conversational_no_results(
                                query=query,
                                session_id=session_id,
                                turn_number=turn_number,
                                classification=classification,
                                routing_decision=routing_decision,
                                stage_times=stage_times,
                                pipeline_start=pipeline_start,
                                was_reformulated=was_reformulated,
                                reformulated_query=reformulated_query
                            )

                        logger.info(f"Google Workspace MCP returned content: {len(mcp_text)} chars")

                        if service == "gmail" and memory and self.session_manager and session_id:
                            _gmail_threads = mcp_result.get("gmail_threads") or []
                            if _gmail_threads:
                                from src.mcp.gmail_search import persist_gmail_thread_memory

                                persist_gmail_thread_memory(memory, _gmail_threads)
                                try:
                                    self.session_manager.save_session(session_id)
                                except Exception as _save_exc:
                                    logger.warning("Failed to persist Gmail thread memory: %s", _save_exc)

                        # Create documents from MCP content for the pipeline
                        from src.indexing.document_loader import Document as LoaderDocument, DocumentMetadata
                        mcp_doc = LoaderDocument(
                            content=mcp_text,
                            metadata=DocumentMetadata(
                                filename=f"google_workspace_{service}_{tool}",
                                filepath=f"mcp://{service}/{tool}",
                                file_type="text",
                                source=f"google_workspace_{service}",
                                tool=tool,
                                connector=connector or "google",
                                query=search_query
                            )
                        )

                        # Skip standard retrieval and proceed directly to generation with MCP content
                        final_docs = [mcp_doc]
                        logger.info(f"Proceeding to generation with {len(final_docs)} MCP document(s)")

                        # Stage 4: Generation (with MCP content)
                        stage_start = time.time()
                        logger.info("\nSTAGE 4: Generation (with MCP content)")
                        logger.info("-" * 80)

                        _check_cancel("generation")

                        # Build prompt with MCP content
                        from src.generation.conversational_prompt_builder import ConversationalPromptBuilder
                        prompt_builder = ConversationalPromptBuilder()

                        prompt = prompt_builder.build_prompt_with_memory(
                            query=query,
                            documents=final_docs,
                            conversation_history=conversation_history,
                            detected_references=detected_references,
                            extracted_entities=extracted_entities
                        )

                        # Generate answer
                        answer = self.llm_generator.generate(prompt)
                        generation_time = time.time() - stage_start
                        stage_times["generation"] = generation_time

                        # Extract citations from MCP content
                        citations = []
                        if mcp_text:
                            citations.append({
                                "source": f"Google Workspace {service}",
                                "content": mcp_text[:500] + "..." if len(mcp_text) > 500 else mcp_text
                            })

                        total_time = time.time() - pipeline_start
                        stage_times["total"] = total_time

                        return ConversationalPipelineResult(
                            query=query,
                            answer=answer,
                            citations=citations,
                            metadata={
                                "confidence": 0.8,
                                "mcp_execution": True,
                                "service": service,
                                "tool": tool
                            },
                            query_classification=classification.to_dict(),
                            routing_decision=routing_decision.to_dict(),
                            retrieval_stats={"mcp_count": 1, "dense_count": 0, "sparse_count": 0, "fused_count": 0, "reranked_count": 0},
                            context_stats={"verified": 1, "unique": 1, "final": 1},
                            generation_stats={"prompt_tokens": len(prompt), "completion_tokens": len(answer)},
                            total_time=total_time,
                            stage_times=stage_times,
                            session_id=session_id,
                            turn_number=turn_number,
                            was_reformulated=was_reformulated,
                            reformulated_query=reformulated_query,
                            reformulation_method=reformulation_method if was_reformulated else None,
                            detected_references=detected_references,
                            extracted_entities=extracted_entities,
                            memory_stats={}
                        )

                    except Exception as exc:
                        logger.error(f"Google Workspace MCP execution failed: {exc}", exc_info=True)
                        return self._create_conversational_no_results(
                            query=query,
                            session_id=session_id,
                            turn_number=turn_number,
                            classification=classification,
                            routing_decision=routing_decision,
                            stage_times=stage_times,
                            pipeline_start=pipeline_start,
                            was_reformulated=was_reformulated,
                            reformulated_query=reformulated_query
                        )

                if routing_decision.route == "microsoft365_mcp":
                    logger.info("Microsoft 365 MCP route detected - executing MCP tool call")
                    try:
                        from src.api.multi_kb_server import _microsoft_mcp_client as _active_ms_client
                        if _active_ms_client is None:
                            raise RuntimeError("Microsoft 365 MCP client not available")

                        service = "onedrive"
                        tool = "search_files"
                        params: Dict[str, Any] = {"query": search_query}

                        if connector:
                            connector_lower = connector.lower().strip()
                            if connector_lower == "outlook":
                                service = "outlook"
                                from src.mcp.outlook_search import (
                                    apply_outlook_follow_up_routing,
                                    patch_outlook_call_params,
                                    persist_outlook_thread_memory,
                                    prepare_outlook_call_params,
                                )

                                _follow_up = None
                                if memory and memory.session.metadata:
                                    _follow_up = apply_outlook_follow_up_routing(
                                        search_query,
                                        memory.session.metadata,
                                    )
                                if _follow_up:
                                    tool, params = _follow_up
                                else:
                                    tool, params = prepare_outlook_call_params(
                                        search_query,
                                        message_id=outlook_message_id,
                                        outlook_folder=outlook_folder,
                                        outlook_location=outlook_location,
                                    )
                                if outlook_folder or outlook_location:
                                    tool, params = patch_outlook_call_params(
                                        tool,
                                        params,
                                        outlook_folder=outlook_folder,
                                        outlook_location=outlook_location,
                                    )
                            elif connector_lower in ("onedrive", "sharepoint"):
                                service = "onedrive"
                                from src.mcp.onedrive_search import prepare_onedrive_call_params

                                tool, params = prepare_onedrive_call_params(
                                    search_query,
                                    item_id=microsoft_file_id,
                                    file_name=microsoft_file_name,
                                    drive_path=microsoft_drive_path,
                                )
                                if microsoft_file_id:
                                    logger.info(
                                        "OneDrive @-mention item_id=%s name=%r drive_path=%r — direct fetch",
                                        microsoft_file_id,
                                        microsoft_file_name,
                                        microsoft_drive_path or "me",
                                    )
                        else:
                            from src.mcp.onedrive_search import prepare_onedrive_call_params

                            tool, params = prepare_onedrive_call_params(
                                search_query,
                                item_id=microsoft_file_id,
                                file_name=microsoft_file_name,
                                drive_path=microsoft_drive_path,
                            )

                        mcp_result = _active_ms_client.call_tool(
                            user_id=user_id,
                            service=service,
                            tool=tool,
                            params=params,
                        )
                        content = mcp_result.get("content") or []
                        mcp_text = "\n".join(
                            item.get("text", "") for item in content
                            if isinstance(item, dict) and item.get("type") == "text"
                        ).strip()
                        if not mcp_text:
                            raise RuntimeError("Microsoft 365 MCP returned empty content")

                        if service == "outlook" and memory and self.session_manager and session_id:
                            _outlook_messages = mcp_result.get("outlook_messages") or []
                            if _outlook_messages:
                                from src.mcp.outlook_search import persist_outlook_thread_memory

                                persist_outlook_thread_memory(memory, _outlook_messages)
                                try:
                                    self.session_manager.save_session(session_id)
                                except Exception as _save_exc:
                                    logger.warning("Failed to persist Outlook message memory: %s", _save_exc)

                        from src.indexing.document_loader import Document as LoaderDocument, DocumentMetadata
                        mcp_doc = LoaderDocument(
                            content=mcp_text,
                            metadata=DocumentMetadata(
                                filename=f"microsoft365_{service}_{tool}",
                                filepath=f"mcp://microsoft/{service}/{tool}",
                                file_type="text",
                                source=f"microsoft365_{service}",
                                tool=tool,
                                connector=connector or service,
                                query=search_query,
                            ),
                        )
                        final_docs = [mcp_doc]
                        stage_start = time.time()
                        logger.info("Proceeding to generation with Microsoft MCP content")
                        _check_cancel("generation")

                        from src.generation.conversational_prompt_builder import ConversationalPromptBuilder
                        prompt_builder = ConversationalPromptBuilder()
                        prompt = prompt_builder.build_prompt_with_memory(
                            query=query,
                            documents=final_docs,
                            conversation_history=conversation_history,
                            detected_references=detected_references,
                            extracted_entities=extracted_entities,
                        )
                        answer = self.llm_generator.generate(prompt)
                        stage_times["generation"] = time.time() - stage_start

                        citations = []
                        if mcp_text:
                            citations.append({
                                "source": f"Microsoft 365 {service}",
                                "content": mcp_text[:500] + "..." if len(mcp_text) > 500 else mcp_text,
                            })

                        total_time = time.time() - pipeline_start
                        stage_times["total"] = total_time

                        return ConversationalPipelineResult(
                            query=query,
                            answer=answer,
                            citations=citations,
                            metadata={
                                "confidence": 0.8,
                                "mcp_execution": True,
                                "service": service,
                                "tool": tool,
                            },
                            query_classification=classification.to_dict(),
                            routing_decision=routing_decision.to_dict(),
                            retrieval_stats={
                                "mcp_count": 1,
                                "dense_count": 0,
                                "sparse_count": 0,
                                "fused_count": 0,
                                "reranked_count": 0,
                            },
                            context_stats={"verified": 1, "unique": 1, "final": 1},
                            generation_stats={"prompt_tokens": len(prompt), "completion_tokens": len(answer)},
                            total_time=total_time,
                            stage_times=stage_times,
                            session_id=session_id,
                            turn_number=turn_number,
                            was_reformulated=was_reformulated,
                            reformulated_query=reformulated_query,
                        )
                    except Exception as exc:
                        logger.error("Microsoft 365 MCP execution failed: %s", exc, exc_info=True)
                        return self._create_conversational_no_results(
                            query=query,
                            session_id=session_id,
                            turn_number=turn_number,
                            classification=classification,
                            routing_decision=routing_decision,
                            stage_times=stage_times,
                            pipeline_start=pipeline_start,
                            was_reformulated=was_reformulated,
                            reformulated_query=reformulated_query,
                        )

                # Query enhancement (if enabled)
                final_search_query = search_query
                enhancement_info = None
                if self.query_enhancer and routing_decision.route == "rag_pipeline":
                    enhanced = self.query_enhancer.enhance(
                        search_query,
                        query_type=classification.query_type,
                        complexity=classification.complexity
                    )
                    if enhanced.expanded_queries:
                        final_search_query = f"{search_query} {enhanced.expanded_queries[0]}"
                        enhancement_info = {"num_expansions": len(enhanced.expanded_queries)}

                # Collection-anchored retrieval for ambiguous biography-style queries.
                # Anchoring ONLY applies to KB (dense/sparse) retrieval — NOT to web or session docs.
                anchor_terms = self._extract_collection_anchor_terms(
                    self.collection_name,
                    self.conv_config.collection_anchor_terms
                )
                kb_search_query = final_search_query  # default: same as the base query
                if (self.conv_config.enable_collection_query_anchoring and
                        self._should_anchor_query_to_collection(search_query, anchor_terms)):
                    kb_search_query = self._anchor_query_to_collection(final_search_query, anchor_terms)
                    logger.info(f"Collection-anchored KB query: '{final_search_query}' -> '{kb_search_query}'")
                # final_search_query remains unanchored for web + session doc retrieval

                stage_times["query_processing"] = time.time() - stage_start
                _check_cancel("query_processing_post_routing")

                # Stage 3: Retrieval (Parallel Execution)
                stage_start = time.time()
                logger.info("\nSTAGE 3: Parallel Retrieval")
                logger.info("-" * 80)

                # Run dense, sparse, and optionally web retrieval in parallel
                dense_docs = []
                sparse_docs = []
                web_docs = []
                session_docs = []
                parallel_success = False

                try:
                    retrieval_msg = "Running dense and sparse retrieval in parallel...!"
                    if web == "on":
                        retrieval_msg = "Running dense, sparse, and web retrieval in parallel...!"
                    logger.info(retrieval_msg)

                    _check_cancel("retrieval_parallel_submit")

                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = {}
                        # Submit base retrieval tasks if KB is enabled
                        if kb == "on":
                            dense_future = executor.submit(
                                self.dense_retriever.retrieve,
                                kb_search_query  # anchored query for KB only
                            )
                            sparse_future = executor.submit(
                                self.sparse_retriever.retrieve,
                                kb_search_query  # anchored query for KB only
                            )
                            futures[dense_future] = "dense"
                            futures[sparse_future] = "sparse"

                        # Optional web search — uses unanchored query
                        if web == "on":
                            web_future = executor.submit(
                                self.web_retriever.retrieve,
                                final_search_query  # NOT anchored
                            )
                            futures[web_future] = "web"

                        # Wait for all to complete and collect results
                        for future in as_completed(futures):
                            _check_cancel("retrieval_parallel_wait")
                            retriever_type = futures[future]
                            try:
                                result = future.result(timeout=30)
                                if retriever_type == "dense":
                                    dense_docs = result
                                elif retriever_type == "sparse":
                                    sparse_docs = result
                                elif retriever_type == "web":
                                    web_docs = result
                            except Exception as e:
                                logger.error(f" {retriever_type.capitalize()} retrieval failed: {e}")
                                if retriever_type == "dense":
                                    dense_docs = []
                                elif retriever_type == "sparse":
                                    sparse_docs = []
                                elif retriever_type == "web":
                                    web_docs = []

                        # 3.2.2: Session-specific retrieval (always runs if session_id is provided)
                        if session_id:
                            session_docs = self._retrieve_session_documents(
                                session_id,
                                final_search_query,
                                top_k=5,
                            )

                    parallel_success = True
                    logger.info("Parallel retrieval completed successfully")

                except Exception as e:
                    # Fallback to sequential retrieval if parallel execution fails
                    logger.warning(f"Parallel retrieval failed: {e}")
                    logger.info("Falling back to sequential retrieval...!")

                    try:
                        _check_cancel("retrieval_fallback_dense")
                        dense_docs = self.dense_retriever.retrieve(final_search_query)
                    except Exception as dense_error:
                        logger.error(f" Dense retrieval failed: {dense_error}")
                        dense_docs = []

                    try:
                        _check_cancel("retrieval_fallback_sparse")
                        sparse_docs = self.sparse_retriever.retrieve(final_search_query)
                    except Exception as sparse_error:
                        logger.error(f" Sparse retrieval failed: {sparse_error}")
                        sparse_docs = []

                # Fusion and reranking
                _check_cancel("retrieval_fusion")
                logger.info("Fusing and reranking results...!")

                retrieval_results = [dense_docs, sparse_docs]
                if web == "on" and web_docs:
                    logger.info(f"Adding {len(web_docs)} web documents to fusion")
                    retrieval_results.append(web_docs)

                if session_docs:
                    logger.info(f"Retrieved {len(session_docs)} session documents (keeping separate from KB fusion)")
                    # DO NOT append to retrieval_results, they are passed directly to prompt builder now

                fused_docs = self.result_fusion.fuse(retrieval_results)
                # CRITICAL: Use final_search_query (same as retrieval) for consistent scoring
                reranked_docs = self.reranker.rerank(final_search_query, fused_docs)

                if not reranked_docs and not self.conv_config.allow_general_knowledge_fallback:
                    logger.warning("No documents retrieved - returning no-results response")
                    stage_times["retrieval"] = time.time() - stage_start
                    return self._create_conversational_no_results(
                        query=query,
                        session_id=session_id,
                        turn_number=turn_number,
                        classification=classification,
                        routing_decision=routing_decision,
                        stage_times=stage_times,
                        pipeline_start=pipeline_start,
                        was_reformulated=was_reformulated,
                        reformulated_query=reformulated_query,
                        memory=memory
                    )

                retrieval_stats = {
                    "dense_count": len(dense_docs),
                    "sparse_count": len(sparse_docs),
                    "web_count": len(web_docs),
                    "session_count": len(session_docs) if session_docs else 0,
                    "fused_count": len(fused_docs),
                    "reranked_count": len(reranked_docs),
                    "search_query": final_search_query,
                    "parallel_execution": parallel_success,
                    "web_search_enabled": web == "on",
                    "kb_search_enabled": kb == "on"
                }

                stage_times["retrieval"] = time.time() - stage_start
                _check_cancel("retrieval_complete")

                # Stage 4: Context Processing
                stage_start = time.time()

                # # Don't early exit even if no docs verified - allow general knowledge answers
                # if reranked_docs:
                #     # CRITICAL FIX: Skip NLI verification if reranker scores are very high
                #     # Reranker scores > 0.85 are highly reliable, no need for additional verification
                #     high_quality_docs = [doc for doc in reranked_docs if doc.score >= 0.85]
                #
                #     if high_quality_docs:
                #         # Trust the reranker for high-quality documents
                #         logger.info(
                #             f"Skipping NLI verification for {len(high_quality_docs)} high-quality documents "
                #             f"(rerank scores >= 0.85)"
                #         )
                #         verified_docs = high_quality_docs
                #     else:
                #         # Use NLI verification for lower-scored documents
                #         # Lowered threshold from 0.1 to 0.05 to be less aggressive
                #         use_keyword_filter = len(search_query.split()) <= 4
                #         verified_docs = self.context_verifier.verify_relevance(
                #             search_query, reranked_docs,
                #             threshold=0.05,  # Lowered from 0.1
                #             require_keyword_match=use_keyword_filter
                #         )
                # else:
                #     verified_docs = []

                if reranked_docs:
                    use_keyword_filter = len(final_search_query.split()) <= 4
                    verification_threshold = max(
                        self.config.nli_threshold,
                        self.conv_config.min_verification_threshold
                    )
                    verified_docs = self.context_verifier.verify_relevance(
                        final_search_query,
                        reranked_docs,
                        threshold=verification_threshold,
                        require_keyword_match=use_keyword_filter
                    )
                else:
                    verified_docs = []

                _check_cancel("context_processing_verification")

                if not verified_docs and not self.conv_config.allow_general_knowledge_fallback:
                    logger.warning("No documents passed relevance verification - returning no-results response")
                    stage_times["context_processing"] = time.time() - stage_start
                    return self._create_conversational_no_results(
                        query=query,
                        session_id=session_id,
                        turn_number=turn_number,
                        classification=classification,
                        routing_decision=routing_decision,
                        stage_times=stage_times,
                        pipeline_start=pipeline_start,
                        was_reformulated=was_reformulated,
                        reformulated_query=reformulated_query,
                        memory=memory
                    )

                if verified_docs:
                    unique_docs = self.context_verifier.deduplicate_context(verified_docs)
                    ordered_docs = self.context_verifier.order_context(
                        unique_docs, strategy=self.config.ordering_strategy
                    )
                    final_docs, citation_map = self.context_verifier.prepare_citations(ordered_docs)
                else:
                    # Empty docs only allowed when general-knowledge fallback is enabled.
                    final_docs = []
                    citation_map = {}

                context_stats = {
                    "verified": len(verified_docs) if verified_docs else 0,
                    "unique": len(unique_docs) if verified_docs else 0,
                    "final": len(final_docs) + (len(session_docs) if session_docs else 0)
                }

                stage_times["context_processing"] = time.time() - stage_start
                _check_cancel("context_processing_complete")

            if session_id and not session_docs:
                session_docs = self._retrieve_session_documents(
                    session_id,
                    reformulated_query or search_query or query,
                    top_k=5,
                )
                if session_docs:
                    retrieval_stats["session_count"] = len(session_docs)
                    logger.info(
                        "Loaded %d session docs on MCP bypass path for session_id=%s",
                        len(session_docs),
                        session_id,
                    )
                    if used_mcp and _is_low_value_mcp_response(mcp_context):
                        logger.info("Session docs override low-value MCP context")
                        used_mcp = False
                        mcp_context = None
                        mcp_service = None
                        mcp_tool = None

            # =================================================================
            # STAGE 5: Generation with Conversation Context
            # =================================================================
            stage_start = time.time()
            logger.info("\nSTAGE 5: Conversational Generation")
            logger.info("-" * 80)
            _check_cancel("generation_start")

            # Build prompt with conversation history
            if self.conv_config.enable_memory and self.conversational_prompt_builder:
                # CRITICAL: Only pass active_topic if this is a continuation
                # Otherwise, passing old topic confuses the LLM on new queries
                topic_to_pass = (
                    memory.session.active_topic
                    if (memory and is_continuation and not session_has_uploaded_doc)
                    else None
                )
                entities_to_pass = (
                    memory.session.topic_entities
                    if (memory and is_continuation and not session_has_uploaded_doc)
                    else None
                )

                prompt = self.conversational_prompt_builder.build_prompt_with_memory(
                    query=query,  # Use original query for display
                    documents=final_docs,
                    conversation_history=conversation_history,
                    active_topic=topic_to_pass,  # Only pass if continuation
                    topic_entities=entities_to_pass,  # Only pass if continuation
                    session_docs=session_docs,
                    mcp_context=mcp_context,
                    mcp_service=mcp_service,
                    mcp_tool=mcp_tool,
                )
                logger.info(
                    f"Built conversational prompt ({len(prompt)} chars, {len(conversation_history)} history turns)"
                )
            else:
                # Fall back to regular prompt builder
                prompt = self.prompt_builder.build_prompt(
                    query,
                    final_docs,
                    session_docs=session_docs,
                    mcp_context=mcp_context,
                    mcp_service=mcp_service,
                    mcp_tool=mcp_tool,
                )
                logger.info(f"Built standard prompt ({len(prompt)} chars)")

            # Generate the answer
            # "---", "\n---", "\n\n\n" as LLM uses these in formatted responses (md format)
            # Pass requires_long_response flag from query classification for min_tokens decision
            try:
                requires_long_response = (
                    classification.requires_long_response and self.conv_config.enable_min_tokens_strategy
                )
                min_tokens = self.conv_config.min_tokens_long_response if requires_long_response else 0
                long_response_max_tokens = (
                    self.conv_config.long_response_max_tokens if requires_long_response else None
                )

                answer = self.llm_generator.generate(
                    prompt,
                    max_new_tokens=self.conv_config.max_tokens,
                    temperature=self.conv_config.temperature,
                    top_p=self.conv_config.top_p,
                    repetition_penalty=self.conv_config.repetition_penalty,
                    presence_penalty=self.conv_config.presence_penalty,
                    frequency_penalty=self.conv_config.frequency_penalty,
                    requires_long_response=requires_long_response,
                    min_tokens=min_tokens,
                    long_response_max_tokens=long_response_max_tokens,
                    purpose="answer_generation",
                    stop_sequences=[
                        "Question:", "Context:", "\n\nQuestion:", "\n\nContext:",
                        "\n\nNote:", "Note:", "> Note", "\n\nAnswer:",
                        "Instructions:", "\nUser:", "System:", "<|im_end|>", "\nQ:",
                        "[End of message]", "<|endoftext|>", "[End response]",
                        "\n\nKnowledge Base Information:", "\n\nCurrent Question:",
                        "\n\nPrevious Context:", "[End response.]",
                        "\nEnd of message.", "End of message.", "\nEnd of message",
                        "End of message", "\n\nEnd of message", "\n\nEnd of message.",
                        # PHASE 1: Meta-commentary stoppers
                        "\n\nEnd", "\nEnd of", "End response", "\n\nThe response",
                        # "\n\nPlease note", "\n\nFor precise", "\n\nIf further",
                        "\n\nDo you", "\n\nIt sounds",
                        # PHASE 1: Self-critique stoppers
                        # "\n\nHowever, ", "\n\nBut ", "\n\nAlso,", "\n\nTherefore,",
                        # PHASE 1: Parenthetical stoppers
                        "\n(End", "\n(The", "\n(Please"
                    ]
                )
            except LLMOverloadedError as e:
                raise PipelineOverloadedError(
                    message="LLM generation capacity is saturated",
                    retry_after_seconds=e.retry_after_seconds,
                    reason="llm_overloaded",
                    details={
                        "inflight": e.inflight,
                        "max_concurrency": e.max_concurrency,
                        "waiters": e.waiters,
                    },
                ) from e
            logger.info(f"Generated answer ({len(answer)} chars)")
            _check_cancel("generation_complete")

            # Extract citations
            source_docs_dicts = []
            if used_mcp and mcp_context and str(mcp_context).strip():
                source_docs_dicts.append({
                    'content': mcp_context,
                    'source_file': f"{_mcp_display_name(mcp_service)} ({mcp_tool})",
                    'page': 0,
                    'chunk_id': 'mcp_result',
                    'rerank_score': 1.0,
                })
                logger.info(
                    "MCP-only turn: using MCP context as citation source (%d chars)",
                    len(mcp_context),
                )

            all_citation_sources = list(final_docs)
            if session_docs:
                all_citation_sources.extend(session_docs)

            for doc in all_citation_sources:
                doc_dict = {
                    'content': doc.content,
                    'source_file': doc.meta.get('url') or doc.meta.get('source_filename') or doc.meta.get('source') or 'Unknown',
                    'page': doc.meta.get('page_number', doc.meta.get('page', 0)),
                    'chunk_id': doc.meta.get('chunk_id', 'unknown'),
                    'rerank_score': doc.score if hasattr(doc, 'score') else doc.meta.get('relevance', 0.0)
                }
                source_docs_dicts.append(doc_dict)

            sanitized_answer = sanitize_generated_answer(answer, current_query=query)
            citations_list, valid_ids, invalid_ids = self.citation_extractor.extract_citations(
                sanitized_answer, source_docs_dicts
            )
            clean_answer = self.citation_extractor.remove_invalid_citations(sanitized_answer, invalid_ids)
            clean_answer = sanitize_generated_answer(clean_answer, current_query=query)

            # If LLM indicates insufficient grounded context, return no-results response
            refusal_patterns = [
                "provided context does not contain",
                "context does not include",
                "cannot find",
                "no information",
                "don't have information",
                "not mentioned in the context",
                "context doesn't mention",
                "unable to answer based on",
                "cannot answer based on",
                "i don't have specific information about this in the available epstein files",
            ]
            answer_lower = clean_answer.lower()
            is_refusal = any(pattern in answer_lower for pattern in refusal_patterns) if not used_mcp else False
            if is_refusal and not self.conv_config.allow_general_knowledge_fallback:
                logger.warning("LLM indicated insufficient grounded context - returning no-results response")
                return self._create_conversational_no_results(
                    query=query,
                    session_id=session_id,
                    turn_number=turn_number,
                    classification=classification,
                    routing_decision=routing_decision,
                    stage_times=stage_times,
                    pipeline_start=pipeline_start,
                    was_reformulated=was_reformulated,
                    reformulated_query=reformulated_query,
                    memory=memory
                )

            # Convert citations
            if used_mcp:
                # Populate source_documents so it's not empty in the session log
                source_docs_dicts = [{
                    'content': mcp_context,
                    'source_file': f"{_mcp_display_name(mcp_service)} ({mcp_tool})",
                    'page': 0,
                    'chunk_id': 'mcp_result',
                    'rerank_score': 1.0
                }]

                # Build a specific source label from the MCP context.
                # For Slack: extract channel/DM names and show e.g. "Slack #general, #random".
                # For Google services: extract file names, subjects, or events.
                specific_source = _mcp_display_name(mcp_service)
                specific_text = f"Retrieved via {_mcp_display_name(mcp_service)}"

                if mcp_context:
                    import re

                    if mcp_service == "slack":
                        # ── Slack: extract channel / DM names ─────────────────────────
                        # Pattern 1: get_channel_history header → "Recent messages in #general:"
                        channel_header = re.findall(
                            r'Recent messages in (#?\S+):', mcp_context
                        )
                        # Pattern 2: search_messages / history scan lines → "**#general**" or "**DM**"
                        channel_bold = re.findall(
                            r'\*\*(#[\w\-]+|DM(?:\s+with\s+[^\*]+)?)\*\*', mcp_context
                        )
                        # Pattern 3: list_channels lines → "**#general** (Public)"
                        channel_list = re.findall(
                            r'\*\*(#[\w\-]+)\*\*\s+\(', mcp_context
                        )

                        all_channels = list(dict.fromkeys(
                            ch.strip() for ch in (channel_header + channel_bold + channel_list)
                            if ch.strip()
                        ))

                        if all_channels:
                            # Prefix bare names with # if not already prefixed
                            labelled = [
                                ch if ch.startswith("#") or ch.startswith("DM") else f"#{ch}"
                                for ch in all_channels
                            ]
                            specific_source = f"Slack {', '.join(labelled)}"
                        else:
                            specific_source = "Slack"

                        # Use a snippet from the context as the citation text
                        specific_text = mcp_context[:200].replace('\n', ' ').strip()
                        if len(mcp_context) > 200:
                            specific_text += "..."

                    elif mcp_service in ("gmail", "drive", "calendar", "sheets", "docs", "presentation"):
                        # ── Google Workspace: extract file names / subjects / events ──
                        files = re.findall(r'- \*\*Name\*\*: \[(.*?)\]', mcp_context)
                        subjects = re.findall(r'  \*\*Subject\*\*: (.*?)\n', mcp_context)
                        events = re.findall(r'- \*\*Event\*\*: (.*?)\n', mcp_context)

                        all_items = list(dict.fromkeys(
                            item for item in files + subjects + events
                            if item and len(item) > 3
                        ))
                        mentioned_items = [
                            item for item in all_items
                            if item.lower() in clean_answer.lower()
                        ]
                        source_items = mentioned_items if mentioned_items else all_items

                        if source_items:
                            specific_source = (
                                f"{_mcp_display_name(mcp_service)}: "
                                + ", ".join(source_items)
                            )
                            first_item = source_items[0]
                            idx = mcp_context.find(first_item)
                            if idx != -1:
                                start = max(0, idx - 20)
                                end = min(len(mcp_context), idx + len(first_item) + 80)
                                specific_text = mcp_context[start:end].replace('\n', ' ').strip()
                                if start > 0:
                                    specific_text = "..." + specific_text
                                if end < len(mcp_context):
                                    specific_text = specific_text + "..."
                        else:
                            specific_text = (clean_answer[:100] + "...") if clean_answer else specific_text

                    elif mcp_service == "notion":
                        # ── Notion: extract page names ──
                        pages = re.findall(r'- \*\*(.*?)\*\*', mcp_context)
                        all_items = list(dict.fromkeys(
                            item for item in pages
                            if item and len(item) > 1 and item != "Untitled"
                        ))
                        mentioned_items = [
                            item for item in all_items
                            if item.lower() in clean_answer.lower()
                        ]
                        source_items = mentioned_items if mentioned_items else all_items

                        if source_items:
                            specific_source = "Notion Pages: " + ", ".join(source_items[:3])
                            if len(source_items) > 3:
                                specific_source += "..."
                                
                            first_item = source_items[0]
                            idx = mcp_context.find(first_item)
                            if idx != -1:
                                start = max(0, idx - 20)
                                end = min(len(mcp_context), idx + len(first_item) + 120)
                                specific_text = mcp_context[start:end].replace('\n', ' ').strip()
                                if start > 0:
                                    specific_text = "..." + specific_text
                                if end < len(mcp_context):
                                    specific_text = specific_text + "..."
                        else:
                            specific_text = (clean_answer[:100] + "...") if clean_answer else specific_text

                citations_dicts = [{
                    "id": 1,
                    "source": specific_source,
                    "tool": f"{_mcp_display_name(mcp_service)} {mcp_tool}",
                    "text": specific_text,
                    "relevance": 1.0
                }]
            else:
                citations_dicts = [cit.to_dict() for cit in citations_list]

                # Fallback to citation map if no citations found
                if not citations_dicts and citation_map and "citations" in citation_map:
                    for cit_entry in citation_map["citations"]:
                        citations_dicts.append({
                            "id": cit_entry.get('citation_number', 0),
                            "source": cit_entry.get('source_file', 'Unknown'),
                            "page": cit_entry.get('page', 0),
                            "chunk_id": cit_entry.get('chunk_id', 'unknown'),
                            "text": cit_entry.get('content_preview', ''),
                            "relevance": round(cit_entry.get('rerank_score', 0.0), 4)
                        })

                # Filter out citations with very low relevance scores
                # This prevents showing irrelevant sources when KB wasn't actually used
                MIN_CITATION_RELEVANCE = 0.1
                if citations_dicts:
                    original_count = len(citations_dicts)
                    citations_dicts = [
                        cit for cit in citations_dicts
                        if cit.get('relevance', 0) >= MIN_CITATION_RELEVANCE
                    ]
                    if len(citations_dicts) < original_count:
                        logger.info(
                            f"Filtered {original_count - len(citations_dicts)} low-relevance citations "
                            f"(threshold: {MIN_CITATION_RELEVANCE})"
                        )

            citations_dicts = normalize_citations_sources(citations_dicts)

            # Calculate confidence
            coverage = self.citation_extractor.validate_citation_coverage(clean_answer)
            coverage_score = coverage.get('coverage', 0.0)
            all_scores = [doc.score if (hasattr(doc, 'score') and doc.score is not None) else doc.meta.get('relevance', 0.0) for doc in all_citation_sources]
            avg_rerank_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            confidence = 1.0 if used_mcp else (coverage_score * 0.4 + avg_rerank_score * 0.4 + min(1.0, len(citations_dicts) / 3) * 0.2)

            generation_stats = {
                "prompt_length": len(prompt),
                "answer_length": len(clean_answer),
                "citations_found": len(citations_dicts),
                "confidence": confidence,
                "used_conversation_history": len(conversation_history) > 0
            }

            stage_times["generation"] = time.time() - stage_start
            _check_cancel("post_generation_processing")

            # =================================================================
            # STAGE 6: Entity Extraction and Turn Storage
            # =================================================================

            if self.conv_config.enable_memory and memory is not None:
                stage_start = time.time()
                logger.info("\nSTAGE 6: Memory Update")
                logger.info("-" * 80)
                _check_cancel("memory_update_start")

                # Extract entities
                if self.query_reformulator:
                    extracted_entities = self.query_reformulator.extract_entities(
                        query, clean_answer, use_spacy=True
                    )
                    logger.info(f"Extracted entities: {extracted_entities}")

                # Create response summary (for token efficiency in future prompts)
                response_summary = ""
                if self.response_summarizer:
                    try:
                        response_summary = self.response_summarizer.summarize(clean_answer)
                        logger.info(f"Response summary created ({len(response_summary)} chars)")
                    except Exception as e:
                        logger.warning(f"Response summarization failed: {e}")
                        # Fallback: use answer if short, empty otherwise
                        response_summary = clean_answer if len(clean_answer) < 150 else ""

                # Detect pending offers (e.g., "Would you like to know more?")
                pending_offer = None
                if self.response_summarizer:
                    try:
                        pending_offer = self.response_summarizer.extract_pending_offer(
                            clean_answer, extracted_entities
                        )
                        if pending_offer:
                            logger.info(f"Pending offer detected: {pending_offer.get('topic', 'N/A')}")
                    except Exception as e:
                        logger.warning(f"Pending offer extraction failed: {e}")

                turn_metadata = {
                    "retrieval_stats": retrieval_stats,
                    "was_reformulated": was_reformulated,
                    "reformulation_method": reformulation_method
                }
                if used_mcp:
                    turn_metadata["mcp_service"] = mcp_service
                    turn_metadata["mcp_tool"] = mcp_tool

                # Store turn in memory
                memory.add_turn(
                    query=query,
                    reformulated_query=reformulated_query,
                    answer=clean_answer,
                    response_summary=response_summary,
                    pending_offer=pending_offer,
                    citations=citations_dicts,
                    entities=extracted_entities,
                    confidence=confidence,
                    source_documents=source_docs_dicts,
                    metadata=turn_metadata,
                    query_type="mcp_tool" if used_mcp else "rag_query",
                    used_retrieval=not used_mcp
                )
                logger.info(f"Turn {turn_number} stored in memory")

                # NEW: Topic Tracking for Conversational Continuity
                # Extract and update active topic (lightweight, <1ms)
                if not is_continuation:  # New topic started
                    new_topic = extract_topic_from_response(clean_answer, extracted_entities)
                    memory.session.active_topic = new_topic
                    memory.session.topic_entities = extracted_entities[:5] if extracted_entities else []
                    memory.session.topic_start_turn = turn_number
                    logger.info(f"New topic: {new_topic}")
                    logger.info(f"Topic entities: {memory.session.topic_entities}")
                else:  # Continuation - topic persists
                    logger.info(f"Topic continues: {memory.session.active_topic}")
                    # Optionally update entities with new ones
                    if extracted_entities:
                        # Merge new entities with existing (keep unique)
                        existing = set(memory.session.topic_entities)
                        for entity in extracted_entities:
                            if entity not in existing and len(memory.session.topic_entities) < 10:
                                memory.session.topic_entities.append(entity)

                # Auto-save if enabled
                if self.conv_config.auto_save:
                    self.session_manager.save_session(session_id)

                stage_times["memory_update"] = time.time() - stage_start
                _check_cancel("memory_update_complete")

            # =================================================================
            # Create Final Result
            # =================================================================
            _check_cancel("final_result")
            total_time = time.time() - pipeline_start

            memory_stats = {}
            if memory is not None:
                memory_stats = {
                    "session_turn_count": len(memory.get_all_turns()),
                    "history_turns_used": len(conversation_history),
                    "auto_save_enabled": self.conv_config.auto_save
                }

            result = ConversationalPipelineResult(
                # Base result fields
                query=query,
                answer=clean_answer,
                citations=citations_dicts,
                metadata={
                    "confidence": confidence,
                    "generation_time": stage_times.get("generation", 0),
                    "model": self.config.llm_model,
                    "temperature": self.config.temperature,
                    "query_enhancement": enhancement_info
                },
                query_classification=classification.to_dict(),
                routing_decision=routing_decision.to_dict(),
                retrieval_stats=retrieval_stats,
                context_stats=context_stats,
                generation_stats=generation_stats,
                total_time=total_time,
                stage_times=stage_times,

                # Conversational fields
                session_id=session_id,
                turn_number=turn_number,
                was_reformulated=was_reformulated,
                reformulated_query=reformulated_query,
                reformulation_method=reformulation_method,
                detected_references=detected_references,
                extracted_entities=extracted_entities,
                memory_stats=memory_stats
            )

            logger.info("\n" + "=" * 80)
            logger.info("Conversational Pipeline Complete")
            logger.info("=" * 80)
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"Session: {session_id}, Turn: {turn_number}")
            if was_reformulated:
                logger.info(f"Reformulated: '{query}' -> '{reformulated_query}'")
            logger.info(f"Citations: {len(result.citations)}, Confidence: {confidence:.3f}")
            logger.info("=" * 80 + "\n")

            # =================================================================
            # STAGE 7: Cache Storage (NEW)
            # =================================================================
            # Store response in cache if it's a standalone query
            if self.cache_manager and self.follow_up_detector:
                _check_cancel("cache_store")
                is_standalone = self.follow_up_detector.is_standalone_query(
                    query, conversation_history
                )

                if is_standalone:
                    try:
                        # Extract response summary for cache
                        response_summary = ""
                        if memory:
                            last_turn = memory.get_all_turns()[-1] if memory.get_all_turns() else None
                            if last_turn and hasattr(last_turn, 'response_summary'):
                                response_summary = last_turn.response_summary

                        # Store in cache
                        self.cache_manager.store_response(
                            query=query,
                            answer=clean_answer,
                            citations=citations_dicts,
                            entities=extracted_entities,
                            response_summary=response_summary,
                            confidence=confidence,
                            source_documents=[],  # Could add source docs if needed
                            metadata={
                                "was_reformulated": was_reformulated,
                                "reformulated_query": reformulated_query if was_reformulated else None,
                                "retrieval_stats": retrieval_stats
                            }
                        )
                        logger.info(f"✓ Response cached for future queries")
                    except Exception as e:
                        logger.warning(f"Failed to cache response: {e}")

            return result

        except PipelineCancelledError:
            raise
        except PipelineOverloadedError:
            raise
        except Exception as e:
            logger.error(f"Conversational pipeline failed: {e}", exc_info=True)
            raise

    def _create_conversational_rejection_result(
            self,
            query: str,
            session_id: str,
            turn_number: int,
            classification: Any,
            routing_decision: Any,
            stage_times: Dict[str, float],
            pipeline_start: float,
            was_reformulated: bool,
            reformulated_query: str
    ) -> ConversationalPipelineResult:
        """Create result for rejected queries."""
        total_time = time.time() - pipeline_start

        return ConversationalPipelineResult(
            query=query,
            answer="I cannot answer this query because it was classified as out-of-scope or too generic.",
            citations=[],
            metadata={"confidence": 0.0, "rejection_reason": routing_decision.reason},
            query_classification=classification.to_dict(),
            routing_decision=routing_decision.to_dict(),
            retrieval_stats={},
            context_stats={},
            generation_stats={},
            total_time=total_time,
            stage_times=stage_times,
            session_id=session_id,
            turn_number=turn_number,
            was_reformulated=was_reformulated,
            reformulated_query=reformulated_query,
            reformulation_method="",
            detected_references=[],
            extracted_entities=[],
            memory_stats={}
        )

    def _create_chitchat_result(
            self,
            query: str,
            session_id: str,
            turn_number: int,
            query_obj: Any,
            stage_times: Dict[str, float],
            pipeline_start: float,
            memory: Optional[Any] = None
    ) -> ConversationalPipelineResult:
        """Create result for chitchat/greeting queries with canned response."""
        total_time = time.time() - pipeline_start

        # Get canned response from query metadata
        canned_response = query_obj.metadata.canned_response
        chitchat_type = query_obj.metadata.chitchat_type

        logger.info(f"Returning canned response for {chitchat_type}: '{canned_response[:50]}...'")

        # Store turn in memory if enabled
        if memory is not None:
            memory.add_turn(
                query=query,
                reformulated_query=query,
                answer=canned_response,
                citations=[],
                entities=[],
                confidence=1.0,  # High confidence for canned responses
                metadata={
                    "chitchat_type": chitchat_type,
                    "fast_path": True,
                    "skipped_retrieval": True
                }
            )
            logger.info(f"Turn {turn_number} stored in memory (chitchat)")

            # Auto-save if enabled
            if self.conv_config.auto_save and self.session_manager:
                self.session_manager.save_session(session_id)

        memory_stats = {}
        if memory is not None:
            memory_stats = {
                "session_turn_count": len(memory.get_all_turns()),
                "history_turns_used": 0,
                "auto_save_enabled": self.conv_config.auto_save
            }

        return ConversationalPipelineResult(
            query=query,
            answer=canned_response,
            citations=[],
            metadata={
                "confidence": 1.0,
                "query_type": "conversational",
                "documents_used": 0,
                "total_time": total_time,
                "knowledge_base": self.config.collection_name,
                "extracted_entities": [],
                "reformulation_method": None,
                "memory_stats": memory_stats,
                "chitchat_type": chitchat_type,
                "fast_path": True
            },
            query_classification={"query_type": "conversational", "confidence": 1.0},
            routing_decision={"route": "chitchat", "reason": f"Detected as {chitchat_type}"},
            retrieval_stats={"skipped": True},
            context_stats={"skipped": True},
            generation_stats={"skipped": True, "used_canned_response": True},
            total_time=total_time,
            stage_times=stage_times,
            session_id=session_id,
            turn_number=turn_number,
            was_reformulated=False,
            reformulated_query=query,
            reformulation_method="",
            detected_references=[],
            extracted_entities=[],
            memory_stats=memory_stats
        )

    def _create_cached_result(
            self,
            query: str,
            cached_response: Any,  # CachedResponse object
            session_id: str,
            turn_number: int,
            memory: Optional[Any],
            stage_times: Dict[str, float],
            pipeline_start: float
    ) -> ConversationalPipelineResult:
        """
        Create result from cached response.
        
        CRITICAL: Cached responses are added to conversation memory to enable
        proper context for follow-up queries. Each session gets its own context
        even when reusing the same cached response.
        
        Args:
            query: Original query
            cached_response: CachedResponse object from cache
            session_id: Session ID
            turn_number: Turn number
            memory: ConversationMemory instance
            stage_times: Stage timing dict
            pipeline_start: Pipeline start time
            
        Returns:
            ConversationalPipelineResult with cached data
        """
        total_time = time.time() - pipeline_start

        logger.info(f"Creating result from cached response")
        logger.info(f"  Answer length: {len(cached_response.answer)} chars")
        logger.info(f"  Citations: {len(cached_response.citations)}")
        logger.info(f"  Entities: {cached_response.entities}")

        # Store turn in memory (CRITICAL for conversation continuity)
        # This ensures follow-up queries can reference this cached response
        if memory is not None:
            memory.add_turn(
                query=query,
                reformulated_query=query,  # Cached responses are not reformulated
                answer=cached_response.answer,
                response_summary=cached_response.response_summary,
                citations=cached_response.citations,
                entities=cached_response.entities,
                confidence=cached_response.confidence,
                source_documents=cached_response.source_documents,
                metadata={
                    "from_cache": True,
                    "cache_hit_timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                    "original_cached_query": cached_response.original_query,
                    "cache_hit_count": cached_response.hit_count,
                    "fast_path": True,
                    "skipped_retrieval": True,
                    "skipped_generation": True
                }
            )
            logger.info(f"Turn {turn_number} stored in memory (cached response)")

            # Update active topic from cached entities
            if cached_response.entities:
                from src.memory.topic_tracking import extract_topic_from_response
                new_topic = extract_topic_from_response(
                    cached_response.answer,
                    cached_response.entities
                )
                memory.session.active_topic = new_topic
                memory.session.topic_entities = cached_response.entities[:5]
                memory.session.topic_start_turn = turn_number
                logger.info(f"Topic set from cache: {new_topic}")

            # Auto-save if enabled
            if self.conv_config.auto_save and self.session_manager:
                self.session_manager.save_session(session_id)

        memory_stats = {}
        if memory is not None:
            memory_stats = {
                "session_turn_count": len(memory.get_all_turns()),
                "history_turns_used": 0,  # Cache hit doesn't use history for retrieval
                "auto_save_enabled": self.conv_config.auto_save
            }

        return ConversationalPipelineResult(
            query=query,
            answer=cached_response.answer,
            citations=cached_response.citations,
            metadata={
                "confidence": cached_response.confidence,
                "query_type": "factual",
                "documents_used": len(cached_response.source_documents),
                "total_time": total_time,
                "knowledge_base": self.config.collection_name,
                "extracted_entities": cached_response.entities,
                "reformulation_method": None,
                "memory_stats": memory_stats,
                "from_cache": True,
                "cache_hit_count": cached_response.hit_count,
                "fast_path": True
            },
            query_classification={"query_type": "factual", "confidence": 1.0},
            routing_decision={
                "route": "cached_response",
                "reason": "Exact match found in cache"
            },
            retrieval_stats={
                "skipped": True,
                "reason": "Cache hit - retrieval not needed"
            },
            context_stats={
                "skipped": True,
                "reason": "Cache hit - context building not needed"
            },
            generation_stats={
                "skipped": True,
                "used_cached_response": True,
                "cache_hit_count": cached_response.hit_count
            },
            total_time=total_time,
            stage_times=stage_times,
            session_id=session_id,
            turn_number=turn_number,
            was_reformulated=False,
            reformulated_query=query,
            reformulation_method="",
            detected_references=[],
            extracted_entities=cached_response.entities,
            memory_stats=memory_stats
        )

    def _create_conversational_no_results(
            self,
            query: str,
            session_id: str,
            turn_number: int,
            classification: Any,
            routing_decision: Any,
            stage_times: Dict[str, float],
            pipeline_start: float,
            was_reformulated: bool,
            reformulated_query: str,
            memory: Optional[ConversationMemory] = None
    ) -> ConversationalPipelineResult:
        """Create result when no relevant documents found."""
        total_time = time.time() - pipeline_start

        # Still store the turn with "no results" answer if memory enabled
        if memory is not None:
            no_results_answer = "I couldn't find relevant information in the knowledge base to answer this question."
            memory.add_turn(
                query=query,
                reformulated_query=reformulated_query,
                answer=no_results_answer,
                citations=[],
                entities=[],
                confidence=0.0,
                metadata={"failure_reason": "no_relevant_documents"}
            )

        return ConversationalPipelineResult(
            query=query,
            answer="I couldn't find relevant information in the knowledge base to answer this question.",
            citations=[],
            metadata={"confidence": 0.0, "failure_reason": "no_relevant_documents"},
            query_classification=classification.to_dict(),
            routing_decision=routing_decision.to_dict(),
            retrieval_stats=stage_times.get("retrieval_stats", {}),
            context_stats={"verified": 0, "final": 0},
            generation_stats={},
            total_time=total_time,
            stage_times=stage_times,
            session_id=session_id,
            turn_number=turn_number,
            was_reformulated=was_reformulated,
            reformulated_query=reformulated_query,
            reformulation_method="",
            detected_references=[],
            extracted_entities=[],
            memory_stats={}
        )

    @staticmethod
    def _extract_collection_anchor_terms(
            collection_name: str,
            extra_terms: Optional[List[str]] = None
    ) -> List[str]:
        """
        Build anchor terms from collection name (plus optional configured terms).

        This keeps query-anchoring generic and KB-agnostic.
        """
        import re

        stopwords = {
            "data", "indexed", "index", "collection", "kb", "knowledge", "base",
            "docs", "documents", "dataset", "rag"
        }

        terms: List[str] = []
        for token in re.split(r"[^a-z0-9]+", (collection_name or "").lower()):
            if token and len(token) > 2 and token not in stopwords and token not in terms:
                terms.append(token)

        for token in (extra_terms or []):
            cleaned = (token or "").strip().lower()
            if cleaned and len(cleaned) > 2 and cleaned not in terms:
                terms.append(cleaned)

        return terms[:6]

    @staticmethod
    def _should_anchor_query_to_collection(query: str, anchor_terms: List[str]) -> bool:
        """
        Detect ambiguous queries that should be anchored to collection context.

        Anchor only when:
        - Query matches ambiguous biography-style patterns
        - Query does not already contain any collection anchor term
        - We actually have anchor terms for this collection
        """
        if not query or not anchor_terms:
            return False

        query_lower = query.lower().strip()
        if not query_lower:
            return False

        if any(term in query_lower for term in anchor_terms):
            return False

        import re
        ambiguous_patterns = [
            r"^\s*who\s+(?:is|was)\s+.+\??\s*$",
            r"^\s*what\s+(?:is|was)\s+.+\??\s*$",
            r"^\s*tell\s+me\s+about\s+.+\??\s*$",
            r"^\s*give\s+me\s+info(?:rmation)?\s+about\s+.+\??\s*$",
        ]
        return any(re.match(pattern, query_lower) for pattern in ambiguous_patterns)

    @staticmethod
    def _anchor_query_to_collection(query: str, anchor_terms: List[str]) -> str:
        """Append collection-derived anchor terms for retrieval."""
        if not anchor_terms:
            return query
        return f"{query} in {' '.join(anchor_terms)} records"

    # =========================================================================
    # Session Management API
    # =========================================================================

    def get_session(self, session_id: str) -> Optional[ConversationMemory]:
        """
        Get a conversation session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationMemory or None
        """
        if self.session_manager:
            return self.session_manager.get_session(session_id)
        return None

    def get_session_history(
            self,
            session_id: str,
            n: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            n: Number of recent turns (None for all)
            
        Returns:
            List of ConversationTurn objects
        """
        if self.session_manager:
            return self.session_manager.get_session_history(session_id, n)
        return []

    def clear_session(self, session_id: str) -> bool:
        """Clear a session's history."""
        if self.session_manager:
            return self.session_manager.clear_session(session_id)
        return False

    def delete_session(self, session_id: str) -> bool:
        """Delete a session completely."""
        if self.session_manager:
            return self.session_manager.delete_session(session_id)
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions."""
        if self.session_manager:
            return self.session_manager.list_sessions()
        return []

    def save_session(self, session_id: str) -> bool:
        """Manually save a session."""
        if self.session_manager:
            return self.session_manager.save_session(session_id)
        return False

    def get_memory_config(self) -> Dict[str, Any]:
        """Get current memory configuration."""
        return {
            "enable_memory": self.conv_config.enable_memory,
            "enable_reformulation": self.conv_config.enable_reformulation,
            "max_turns": self.conv_config.max_turns,
            "history_in_prompt_turns": self.conv_config.history_in_prompt_turns,
            "auto_save": self.conv_config.auto_save,
            "memory_directory": self.conv_config.memory_directory,
            "redis_enabled": self.conv_config.redis_enabled,
            "session_read_through_enabled": self.conv_config.session_read_through_enabled,
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_conversational_pipeline(
        collection_name: str = "knowledge_base",
        storage_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        enable_memory: bool = True,
        memory_directory: str = "./data/conversations",
        **kwargs
) -> ConversationalRAGPipeline:
    """
    Factory function to create a ConversationalRAGPipeline.
    
    Args:
        collection_name: Qdrant collection name
        storage_path: Path to local vector DB storage
        qdrant_url: URL for Qdrant server
        enable_memory: Whether to enable conversation memory
        memory_directory: Directory for conversation files
        **kwargs: Additional config options
        
    Returns:
        Configured ConversationalRAGPipeline instance
        
    Example:
        # Local storage with memory
        pipeline = create_conversational_pipeline(
            collection_name="my_kb",
            storage_path="./data/vector_db",
            enable_memory=True
        )
        
        # Qdrant server without memory
        pipeline = create_conversational_pipeline(
            collection_name="my_kb",
            qdrant_url="http://localhost:6333",
            enable_memory=False
        )
    """
    config = ConversationalPipelineConfig(
        collection_name=collection_name,
        storage_path=storage_path,
        qdrant_url=qdrant_url,
        enable_memory=enable_memory,
        memory_directory=memory_directory,
        **kwargs
    )

    return ConversationalRAGPipeline(
        collection_name=collection_name,
        config=config
    )
