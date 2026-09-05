"""
Conversational Prompt Builder for RAG Pipeline.

This module extends the base RAGPromptBuilder to support conversation history,
enabling contextual understanding across multiple query turns while maintaining
strict grounding in the knowledge base.

Features:
- Build prompts with conversation history context
- Configurable history window (number of turns, token budget)
- Clear separation between KB context and conversation context
- Memory-aware prompt templates with KB grounding instructions
- Support for ConversationTurn objects from memory module
- Token budget management for prompt optimization
"""

import logging
from haystack import Document
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from haystack.components.builders import PromptBuilder as HaystackPromptBuilder

from src.config import load_prompt_catalog, load_prompt_text

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ConversationalPromptConfig:
    """
    Configuration for conversational prompt building.

    Attributes:
        max_history_turns: Maximum conversation turns to include in prompt (default: 3)
        max_history_tokens: Maximum total tokens for history section (default: 500)
        truncate_history_answers: Whether to truncate long answers in history (default: True)
        history_answer_max_length: Max characters per historical answer (default: 150)
        include_entities: Whether to include entity hints from history (default: True)
        history_format: Format style for history ("condensed" or "detailed")
        separator_style: Style of section separators ("simple", "box", "none")
    """
    max_history_turns: int = 3
    max_history_tokens: int = 500
    truncate_history_answers: bool = True
    history_answer_max_length: int = 150
    include_entities: bool = True
    history_format: str = "condensed"  # "condensed" or "detailed"
    separator_style: str = "simple"  # "simple", "box", or "none"

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_history_turns < 0:
            raise ValueError("max_history_turns cannot be negative")
        if self.max_history_tokens < 100:
            raise ValueError("max_history_tokens must be at least 100")
        if self.history_format not in ("condensed", "detailed"):
            raise ValueError("history_format must be 'condensed' or 'detailed'")
        if self.separator_style not in ("simple", "box", "none"):
            raise ValueError("separator_style must be 'simple', 'box', or 'none'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "max_history_turns": self.max_history_turns,
            "max_history_tokens": self.max_history_tokens,
            "truncate_history_answers": self.truncate_history_answers,
            "history_answer_max_length": self.history_answer_max_length,
            "include_entities": self.include_entities,
            "history_format": self.history_format,
            "separator_style": self.separator_style,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationalPromptConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


CONVERSATIONAL_RAG_TEMPLATE, _CONVERSATIONAL_RAG_TEMPLATE_SOURCE = load_prompt_text(template_name="default")
CONVERSATIONAL_DETAILED_TEMPLATE, _CONVERSATIONAL_DETAILED_TEMPLATE_SOURCE = load_prompt_text(template_name="detailed")
CONVERSATIONAL_CONCISE_TEMPLATE, _CONVERSATIONAL_CONCISE_TEMPLATE_SOURCE = load_prompt_text(template_name="concise")
CONVERSATIONAL_BOX_TEMPLATE, _CONVERSATIONAL_BOX_TEMPLATE_SOURCE = load_prompt_text(template_name="box")


# =============================================================================
# History Formatting Utilities
# =============================================================================

@dataclass
class FormattedTurn:
    """
    A formatted conversation turn ready for template insertion.

    This is a simplified representation of ConversationTurn for template use.
    """
    query: str
    answer: str
    response_summary: str = ""  # NEW: Compact summary for context
    pending_offer: Optional[Dict[str, Any]] = None  # NEW: Pending offer info
    entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for template rendering."""
        return {
            "query": self.query,
            "answer": self.answer,
            "response_summary": self.response_summary,
            "pending_offer": self.pending_offer,
            "entities": self.entities,
        }


class HistoryFormatter:
    """
    Utility class for formatting conversation history for prompt inclusion.

    Handles truncation, token budgeting, and format conversion.
    """

    def __init__(self, config: Optional[ConversationalPromptConfig] = None):
        """
        Initialize the history formatter.

        Args:
            config: Configuration for formatting behavior
        """
        self.config = config or ConversationalPromptConfig()

    def format_history(
            self,
            turns: List[Any],
            max_turns: Optional[int] = None,
            max_tokens: Optional[int] = None
    ) -> List[FormattedTurn]:
        """
        Format conversation turns for prompt inclusion.

        Args:
            turns: List of ConversationTurn objects or dicts
            max_turns: Override max turns from config
            max_tokens: Override max tokens from config

        Returns:
            List of FormattedTurn objects ready for template
        """
        if not turns:
            return []

        max_turns = max_turns or self.config.max_history_turns
        max_tokens = max_tokens or self.config.max_history_tokens

        # Get most recent turns
        recent_turns = turns[-max_turns:] if len(turns) > max_turns else turns

        formatted = []
        total_tokens = 0

        for turn in recent_turns:
            # Handle both ConversationTurn objects and dicts
            if hasattr(turn, 'query'):
                query = turn.query
                # Use response_summary if available, otherwise use answer
                if hasattr(turn, 'response_summary') and turn.response_summary:
                    answer = turn.response_summary
                else:
                    answer = turn.answer if hasattr(turn, 'answer') else ""
                response_summary = turn.response_summary if hasattr(turn, 'response_summary') else ""
                pending_offer = turn.pending_offer if hasattr(turn, 'pending_offer') else None
                entities = turn.entities_mentioned if hasattr(turn, 'entities_mentioned') else []
            elif isinstance(turn, dict):
                query = turn.get('query', '')
                # Use response_summary if available, otherwise use answer
                if turn.get('response_summary'):
                    answer = turn.get('response_summary')
                else:
                    answer = turn.get('answer', '')
                response_summary = turn.get('response_summary', '')
                pending_offer = turn.get('pending_offer')
                entities = turn.get('entities_mentioned', turn.get('entities', []))
            else:
                logger.warning(f"Skipping invalid turn format: {type(turn)}")
                continue

            # Truncate answer if configured (only if not already a summary)
            if not response_summary and self.config.truncate_history_answers:
                answer = self._truncate_text(answer, self.config.history_answer_max_length)

            # Estimate tokens for this turn
            turn_tokens = self._estimate_tokens(query + answer)

            # Check token budget
            if total_tokens + turn_tokens > max_tokens:
                logger.debug(f"Token budget exceeded at turn {len(formatted)}")
                break

            total_tokens += turn_tokens

            formatted.append(FormattedTurn(
                query=query,
                answer=answer,
                response_summary=response_summary,
                pending_offer=pending_offer,
                entities=entities if self.config.include_entities else []
            ))

        logger.debug(f"Formatted {len(formatted)} turns (~{total_tokens} tokens)")
        return formatted

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text at word boundary with ellipsis."""
        if len(text) <= max_length:
            return text
        truncated = text[:max_length].rsplit(' ', 1)[0]
        return truncated + "..."

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough: 1 token ≈ 4 characters)."""
        return len(text) // 4


# Main ConversationalPromptBuilder Class
class ConversationalPromptBuilder:
    """
    Enhanced prompt builder with conversation history support.

    Extends the functionality of RAGPromptBuilder to include conversation
    context while maintaining strict KB grounding. Supports multiple template
    styles and configurable history windows.

    Example:
        ```python
        from src.memory import ConversationMemory
        from src.generation import ConversationalPromptBuilder

        # Initialize
        builder = ConversationalPromptBuilder()
        memory = ConversationMemory()

        # Add some conversation history
        memory.add_turn(
            query="What is PL?",
            answer="PL is a framework for...",
            entities=["PL", "framework"]
        )

        # Build prompt with history
        prompt = builder.build_prompt_with_memory(
            query="Who created this?",
            documents=retrieved_docs,
            conversation_history=memory.get_recent_turns(n=3)
        )
        ```
    """

    def __init__(
            self,
            template: Optional[str] = None,
            template_name: Optional[str] = None,
            config: Optional[ConversationalPromptConfig] = None,
            required_variables: Optional[List[str]] = None
    ):
        """
        Initialize the conversational prompt builder.

        Args:
            template: Custom Jinja2 template
            template_name: Named template from prompt catalog
            config: Configuration for history formatting and behavior
            required_variables: Required template variables
        """
        self.config = config or ConversationalPromptConfig()
        if template is None:
            resolved_template, _ = load_prompt_text(template_name=template_name or "default")
            self.template = resolved_template
            self.template_name = template_name or "default"
        else:
            self.template = template
            self.template_name = template_name
        self.required_variables = required_variables or ["query", "documents"]

        # History formatter
        self.history_formatter = HistoryFormatter(self.config)

        logger.info("Initializing ConversationalPromptBuilder")

        # Initialize Haystack PromptBuilder
        # Note: conversation_history is optional, so not in required_variables
        self._init_haystack_builder()

        logger.info("ConversationalPromptBuilder initialized successfully")

    def _init_haystack_builder(self):
        """Initialize or reinitialize the Haystack PromptBuilder."""
        self.haystack_builder = HaystackPromptBuilder(
            template=self.template,
            required_variables=self.required_variables
        )

    def build_prompt(
            self,
            query: str,
            documents: List[Document],
            **kwargs
    ) -> str:
        """
        Build a basic prompt without conversation history.

        This method provides backward compatibility with RAGPromptBuilder.

        Args:
            query: User's question
            documents: List of retrieved documents
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string
        """
        return self.build_prompt_with_memory(
            query=query,
            documents=documents,
            conversation_history=None,
            **kwargs
        )

    def build_prompt_with_memory(
            self,
            query: str,
            documents: List[Document],
            conversation_history: Optional[List[Any]] = None,
            max_history_turns: Optional[int] = None,
            session_docs: Optional[List[Document]] = None,
            **kwargs
    ) -> str:
        """
        Build a prompt with conversation history context.

        This is the main method for building conversational prompts. It formats
        the conversation history and combines it with KB context for the LLM.

        Args:
            query: User's current question
            documents: List of retrieved documents from knowledge base
            conversation_history: List of ConversationTurn objects or dicts
            max_history_turns: Override config's max_history_turns
            **kwargs: Additional template variables

        Returns:
            Formatted prompt string with conversation context

        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        mcp_context = kwargs.get("mcp_context")
        if not documents:
            if mcp_context and str(mcp_context).strip():
                logger.info(
                    "No KB documents; building prompt with MCP context (%d chars)",
                    len(str(mcp_context)),
                )
            else:
                logger.warning("No documents provided, building prompt with empty KB context")

        # Format conversation history
        formatted_history = None
        pending_follow_up = None

        if conversation_history:
            formatted_turns = self.history_formatter.format_history(
                turns=conversation_history,
                max_turns=max_history_turns or self.config.max_history_turns
            )
            if formatted_turns:
                # Convert to dict format for template
                formatted_history = [turn.to_dict() for turn in formatted_turns]
                logger.info(f"Including {len(formatted_history)} conversation turns in prompt")

                # Extract pending_follow_up from last turn if it exists
                last_turn = formatted_turns[-1]
                if last_turn.pending_offer:
                    pending_follow_up = last_turn.pending_offer
                    logger.info(f"Pending follow-up detected: {pending_follow_up.get('topic', 'N/A')}")

        # Extract active_topic and topic_entities from kwargs (passed from pipeline)
        active_topic = kwargs.pop('active_topic', None)
        topic_entities = kwargs.pop('topic_entities', None)

        if active_topic:
            logger.info(f"Including active topic in prompt: {active_topic}")

        # Log prompt building
        history_info = f"with {len(formatted_history)} history turns" if formatted_history else "without history"
        logger.info(f"Building prompt for query: '{query[:50]}...' {history_info}, {len(documents)} documents")

        try:
            session_context = None
            session_filename = None
            if session_docs:
                session_context = "\n\n".join([doc.content for doc in session_docs if hasattr(doc, 'content') and doc.content])
                session_filename = session_docs[0].meta.get("source_filename", "Uploaded Document") if hasattr(session_docs[0], 'meta') else "Uploaded Document"

            # Build prompt using Haystack
            result = self.haystack_builder.run(
                query=query,
                documents=documents,
                conversation_history=formatted_history,
                pending_follow_up=pending_follow_up,
                active_topic=active_topic,  # NEW
                topic_entities=topic_entities,  # NEW
                session_context=session_context,
                session_filename=session_filename,
                **kwargs
            )

            prompt = result["prompt"]

            logger.info(f"Prompt built successfully ({len(prompt)} chars)")

            return prompt

        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            raise

    def build_prompt_with_metadata(
            self,
            query: str,
            documents: List[Document],
            conversation_history: Optional[List[Any]] = None,
            citation_map: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Build prompt and return with metadata for reference.

        Args:
            query: User's question
            documents: List of retrieved documents
            conversation_history: Optional conversation history
            citation_map: Optional citation map from retrieval
            **kwargs: Additional template variables

        Returns:
            Dictionary with prompt, metadata, and citation info
        """
        prompt = self.build_prompt_with_memory(
            query=query,
            documents=documents,
            conversation_history=conversation_history,
            **kwargs
        )

        history_count = len(conversation_history) if conversation_history else 0

        return {
            "prompt": prompt,
            "query": query,
            "num_documents": len(documents),
            "num_history_turns": history_count,
            "citation_map": citation_map or {},
            "config": self.config.to_dict(),
            "template_type": self._get_template_type()
        }

    def _get_template_type(self) -> str:
        """Identify the current template type."""
        if self.template_name:
            return f"conversational_{self.template_name}"
        return "custom"

    def get_template(self) -> str:
        """Get the current template string."""
        return self.template

    def set_template(
            self,
            template: str,
            required_variables: Optional[List[str]] = None
    ):
        """
        Update the template.

        Args:
            template: New Jinja2 template string
            required_variables: Updated required variables list
        """
        self.template = template
        self.template_name = None
        if required_variables:
            self.required_variables = required_variables

        # Reinitialize Haystack builder
        self._init_haystack_builder()

        logger.info("Template updated successfully")

    def get_config(self) -> ConversationalPromptConfig:
        """Get the current configuration."""
        return self.config

    def update_config(self, **kwargs):
        """
        Update configuration options.

        Args:
            **kwargs: Configuration fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")

        # Reinitialize formatter with new config
        self.history_formatter = HistoryFormatter(self.config)

        logger.info(f"Config updated: {kwargs}")

    def format_history_preview(
            self,
            conversation_history: List[Any],
            max_turns: Optional[int] = None
    ) -> str:
        """
        Get a preview of how history will be formatted in the prompt.

        Useful for debugging and understanding prompt construction.

        Args:
            conversation_history: List of conversation turns
            max_turns: Override max turns

        Returns:
            Formatted history string as it would appear in prompt
        """
        if not conversation_history:
            return "(No conversation history)"

        formatted = self.history_formatter.format_history(
            turns=conversation_history,
            max_turns=max_turns or self.config.max_history_turns
        )

        if not formatted:
            return "(No history after formatting/filtering)"

        lines = ["=== Recent Conversation ==="]
        for turn in formatted:
            lines.append(f"User: {turn.query}")
            lines.append(f"Assistant: {turn.answer}")
            if turn.entities:
                lines.append(f"  [Entities: {', '.join(turn.entities)}]")
        lines.append("=== End Conversation ===")

        return "\n".join(lines)

    def estimate_prompt_tokens(
            self,
            query: str,
            documents: List[Document],
            conversation_history: Optional[List[Any]] = None
    ) -> Dict[str, int]:
        """
        Estimate token counts for prompt components.

        Args:
            query: User's question
            documents: Retrieved documents
            conversation_history: Conversation history

        Returns:
            Dictionary with token estimates per component
        """
        estimates = {
            "query": len(query) // 4,
            "documents": 0,
            "history": 0,
            "template_overhead": 200,  # Approximate
        }

        # Estimate document tokens
        for doc in documents:
            content = doc.content if hasattr(doc, 'content') else str(doc)
            estimates["documents"] += len(content) // 4

        # Estimate history tokens
        if conversation_history:
            formatted = self.history_formatter.format_history(conversation_history)
            for turn in formatted:
                estimates["history"] += len(turn.query + turn.answer) // 4

        estimates["total"] = sum(estimates.values())

        return estimates


# =============================================================================
# Template Collection
# =============================================================================

class ConversationalPromptTemplates:
    """Prompt catalog-backed conversational template helper."""

    DEFAULT = CONVERSATIONAL_RAG_TEMPLATE
    DETAILED = CONVERSATIONAL_DETAILED_TEMPLATE
    CONCISE = CONVERSATIONAL_CONCISE_TEMPLATE
    BOX = CONVERSATIONAL_BOX_TEMPLATE

    @staticmethod
    def get_template(name: str) -> str:
        template, _ = load_prompt_text(template_name=name)
        return template

    @staticmethod
    def list_templates() -> List[str]:
        """List available template names from prompt catalog."""
        return sorted(load_prompt_catalog().keys())


# =============================================================================
# Factory Function
# =============================================================================

def create_conversational_prompt_builder(
        template: Optional[str] = None,
        template_name: Optional[str] = None,
        config: Optional[ConversationalPromptConfig] = None,
        **config_kwargs
) -> ConversationalPromptBuilder:
    """
    Factory function to create a ConversationalPromptBuilder instance.

    Args:
        template: Custom template string (takes precedence over template_name)
        template_name: Name of pre-defined template ("default", "detailed", "concise", "box")
        config: Configuration object
        **config_kwargs: Configuration overrides (passed to ConversationalPromptConfig)

    Returns:
        Configured ConversationalPromptBuilder instance

    Example:
        ```python
        # Using default template
        builder = create_conversational_prompt_builder()

        # Using detailed template with custom config
        builder = create_conversational_prompt_builder(
            template_name="detailed",
            max_history_turns=5,
            history_answer_max_length=200
        )

        # Using custom template
        builder = create_conversational_prompt_builder(
            template=my_custom_template
        )
        ```
    """
    # Build config
    if config is None:
        config = ConversationalPromptConfig(**config_kwargs)
    elif config_kwargs:
        # Merge overrides into existing config
        for key, value in config_kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return ConversationalPromptBuilder(
        template=template,
        template_name=template_name,
        config=config
    )
