"""
Pipeline Module

This module contains:
- RAGPipeline: Base RAG pipeline for question answering
- ConversationalRAGPipeline: Memory-enabled pipeline for multi-turn conversations
- Query Routing: Query classification and routing logic (NEW)
"""

from src.pipeline.rag_pipeline import RAGPipeline, PipelineConfig, PipelineResult
from src.pipeline.conversational_rag_pipeline import (
    ConversationalRAGPipeline,
    ConversationalPipelineConfig,
    ConversationalPipelineResult,
    SessionManager,
    create_conversational_pipeline,
)

# Query Routing (NEW)
from src.pipeline.query_routing import (
    classify_and_route_query
)

__all__ = [
    # Base pipeline
    "RAGPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Conversational pipeline
    "ConversationalRAGPipeline",
    "ConversationalPipelineConfig",
    "ConversationalPipelineResult",
    "SessionManager",
    "create_conversational_pipeline",
    # Query Routing (NEW)
    "classify_and_route_query",
]
