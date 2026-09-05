"""
Context Handling module for RAG pipeline.

This module provides:
- Relevance verification using NLI models
- Semantic deduplication using embedding-based similarity
- Context ordering strategies (relevance, chronological, lost-in-middle)
- Citation preparation with [N] notation for LLM prompts
"""
 
from .context_verifier import ContextVerifier, create_context_verifier

__all__ = ["ContextVerifier", "create_context_verifier"]
