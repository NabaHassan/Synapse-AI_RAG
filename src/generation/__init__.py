"""
Generation module for RAG pipeline.

This module provides:
- Prompt construction with Jinja2 templates
- Conversational prompt building with history support
- LLM integration
- Citation extraction
- Post-processing
"""

from .prompt_builder import (
    RAGPromptBuilder,
    create_prompt_builder,
    PromptTemplates,
    DEFAULT_RAG_TEMPLATE,
    CONCISE_TEMPLATE,
    DETAILED_TEMPLATE,
    STEP_BY_STEP_TEMPLATE
)

from .conversational_prompt_builder import (
    ConversationalPromptBuilder,
    ConversationalPromptConfig,
    ConversationalPromptTemplates,
    create_conversational_prompt_builder,
    HistoryFormatter,
    FormattedTurn,
    CONVERSATIONAL_RAG_TEMPLATE,
    CONVERSATIONAL_DETAILED_TEMPLATE,
    CONVERSATIONAL_CONCISE_TEMPLATE,
    CONVERSATIONAL_BOX_TEMPLATE,
)

from .llm_generator import (
    LLMGenerator,
    create_llm_generator,
    GenerationConfig,
)

from .response_summarizer import (
    ResponseSummarizer,
    create_response_summarizer,
)

__all__ = [
    # Prompt building (basic)
    "RAGPromptBuilder",
    "create_prompt_builder",
    "PromptTemplates",
    "DEFAULT_RAG_TEMPLATE",
    "CONCISE_TEMPLATE",
    "DETAILED_TEMPLATE",
    "STEP_BY_STEP_TEMPLATE",
    # Conversational prompt building
    "ConversationalPromptBuilder",
    "ConversationalPromptConfig",
    "ConversationalPromptTemplates",
    "create_conversational_prompt_builder",
    "HistoryFormatter",
    "FormattedTurn",
    "CONVERSATIONAL_RAG_TEMPLATE",
    "CONVERSATIONAL_DETAILED_TEMPLATE",
    "CONVERSATIONAL_CONCISE_TEMPLATE",
    "CONVERSATIONAL_BOX_TEMPLATE",
    # LLM generation
    "LLMGenerator",
    "create_llm_generator",
    "GenerationConfig",
    # Response summarization
    "ResponseSummarizer",
    "create_response_summarizer",
]
