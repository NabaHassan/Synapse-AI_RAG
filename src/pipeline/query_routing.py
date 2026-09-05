"""
Query classification and routing logic for conversational RAG pipeline.

This module contains the logic to classify queries and route them to
appropriate handlers before retrieval.
"""

import time
import logging
import re
from typing import Tuple, Optional, Dict, Any, List

from src.query_processing.query_types import QueryType, QueryClassificationResult
from src.memory.topic_tracking import update_active_topic
from src.query_processing.enhanced_query_classifier import EnhancedQueryClassifier
from src.query_processing.entity_query_handler import EntityQueryHandler
from src.query_processing.exact_text_handler import ExactTextHandler
from src.generation.structured_response_formatter import (
    format_count_response,
    format_file_location_response,
    format_exact_text_response
)
from src.utils.source_normalization import normalize_citations_sources

logger = logging.getLogger(__name__)

# Module-level enhanced classifier instance (singleton)
_enhanced_classifier = None

_AFFIRMATIVE_FOLLOWUP_RE = re.compile(
    r"^\s*(?:yes|y|yeah|yep|correct|right|sure|ok|okay|go\s+ahead|please\s+do)\s*[.!?]*\s*$",
    re.IGNORECASE
)
_NEGATIVE_FOLLOWUP_RE = re.compile(
    r"^\s*(?:no|n|nope|nah|not\s+really|don't|do\s+not)\s*[.!?]*\s*$",
    re.IGNORECASE
)
_NUMERIC_CHOICE_RE = re.compile(r"^\s*([1-9])\s*[.)]?\s*$")
_DID_YOU_MEAN_RE = re.compile(
    r"did\s+you\s+mean\s*:?\s*(.+?)(?:\?|$)",
    re.IGNORECASE | re.DOTALL
)


def _get_enhanced_classifier():
    """Lazily initialize the enhanced query classifier."""
    global _enhanced_classifier
    if _enhanced_classifier is None:
        _enhanced_classifier = EnhancedQueryClassifier()
    return _enhanced_classifier


def classify_and_route_query(
        query: str,
        conversation_history: list,
        memory,
        query_classifier,
        meta_handler,
        formatting_handler,
        llm_generator,
        session_id: str,
        turn_number: int,
        pipeline_start: float,
        stage_times: dict,
        ConversationalPipelineResult,
        collection_name: str = "CAFL_data",  # NEW: collection name parameter
        structured_query_fast_mode: bool = False,
        structured_entity_resolution: bool = True,
        structured_natural_response_style: bool = True,
) -> Tuple[Optional[Any], str, Optional[Dict]]:
    """
    Classify query and route to appropriate handler.
    
    Returns:
        Tuple of (result_if_handled, query_type, classification_dict)
        - result_if_handled: ConversationalPipelineResult if query was handled, None otherwise
        - query_type: String query type
        - classification_dict: Classification metadata
    """
    stage_start = time.time()
    logger.info("\nSTAGE 0.3: Query Classification")
    logger.info("-" * 80)

    # Follow-up confirmation for prior structured entity disambiguation.
    followup_resolution = _handle_structured_resolution_followup(
        query=query,
        conversation_history=conversation_history,
        memory=memory,
        session_id=session_id,
        turn_number=turn_number,
        pipeline_start=pipeline_start,
        stage_times=stage_times,
        stage_start=stage_start,
        ConversationalPipelineResult=ConversationalPipelineResult,
        collection_name=collection_name,
        structured_query_fast_mode=structured_query_fast_mode,
        structured_natural_response_style=structured_natural_response_style,
    )
    if followup_resolution is not None:
        return followup_resolution

    # PRE-CHECK: Enhanced classifier for new query types (ENTITY_COUNT, FILE_LOCATION, EXACT_TEXT)
    # This runs before the standard classifier and short-circuits if matched
    try:
        enhanced = _get_enhanced_classifier()
        enhanced_result = enhanced.classify(query, conversation_history=conversation_history)

        if enhanced_result is not None:
            enhanced_classification, extracted_params = enhanced_result
            classified_query_type = enhanced_classification.query_type.value

            logger.info(f"Enhanced Query Type: {classified_query_type}")
            logger.info(f"Confidence: {enhanced_classification.confidence:.2f}")
            logger.info(f"Extracted Params: {extracted_params}")

            # Route to specialized handler
            handler_result = _handle_enhanced_query(
                query=query,
                collection_name=collection_name,  # Pass collection name
                classification=enhanced_classification,
                params=extracted_params,
                memory=memory,
                session_id=session_id,
                turn_number=turn_number,
                pipeline_start=pipeline_start,
                stage_times=stage_times,
                stage_start=stage_start,
                ConversationalPipelineResult=ConversationalPipelineResult,
                structured_query_fast_mode=structured_query_fast_mode,
                structured_entity_resolution=structured_entity_resolution,
                structured_natural_response_style=structured_natural_response_style,
            )

            if handler_result is not None:
                return handler_result, classified_query_type, enhanced_classification.to_dict()
    except Exception as e:
        logger.warning(f"Enhanced classifier failed, falling back to standard: {e}")

    # Standard classification (original behavior)
    query_classification = query_classifier.classify(query, conversation_history)
    classified_query_type = query_classification.query_type.value

    logger.info(f"Query Type: {classified_query_type}")
    logger.info(f"Confidence: {query_classification.confidence:.2f}")
    if query_classification.matched_patterns:
        logger.info(f"Matched Patterns: {query_classification.matched_patterns[0][:60]}...")
    logger.info(f"Reasoning: {query_classification.reasoning}")

    # Handle meta-conversation queries immediately (NO retrieval!)
    if query_classification.query_type == QueryType.META_CONVERSATION:
        logger.info("→ Routing to MetaConversationHandler (NO retrieval)")

        answer = meta_handler.handle(query, conversation_history)

        # Store turn with metadata
        turn = memory.add_turn(
            query=query,
            answer=answer,
            reformulated_query=query,
            entities=[],
            confidence=1.0,  # High confidence for meta-questions
            metadata={"classification": query_classification.to_dict()},
            query_type="meta_conversation",
            used_retrieval=False,
            source_documents=[]
        )

        # Topic tracking (will ignore meta queries)
        memory.session.active_topic, memory.session.topic_entities = update_active_topic(
            turn,
            memory.session.active_topic,
            memory.session.topic_entities,
            "meta_conversation"
        )

        # Return result immediately
        stage_times["query_classification"] = time.time() - stage_start
        stage_times["total"] = time.time() - pipeline_start

        result = ConversationalPipelineResult(
            answer=answer,
            query=query,
            citations=[],
            metadata={
                "query_classification": query_classification.to_dict(),
                "used_retrieval": False,
                "confidence": 1.0
            },
            query_classification=query_classification.to_dict(),
            routing_decision={"handler": "meta_conversation", "reason": "Meta-question detected"},
            retrieval_stats={"dense_count": 0, "sparse_count": 0, "fused_count": 0, "reranked_count": 0},
            context_stats={"total_chunks": 0, "total_tokens": 0},
            generation_stats={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            total_time=time.time() - pipeline_start,
            stage_times=stage_times,
            session_id=session_id,
            turn_number=turn_number,
            was_reformulated=False,
            reformulated_query=query,
            reformulation_method="",
            detected_references=[],
            extracted_entities=[],
            memory_stats=memory.get_statistics() if memory is not None else {}
        )

        return result, classified_query_type, query_classification.to_dict()

    # Handle clarification requests (revise previous answer, NO new sources!)
    elif query_classification.query_type == QueryType.CLARIFICATION:
        logger.info("→ Routing to ClarificationHandler (NO new retrieval, NO sources)")

        last_turn = _get_last_turn_for_followup(conversation_history, memory)

        # Import clarification handler here to avoid circular imports
        from src.query_processing.clarification_handler import ClarificationHandler
        clarification_handler = ClarificationHandler(llm_generator=llm_generator)

        if not clarification_handler.can_revise(last_turn):
            answer = "There is no previous answer to explain."
            citations = []
            clarification_metadata = {
                "query_type": "clarification",
                "reason": "no_revisable_previous_turn",
            }
        else:
            # Handle clarification
            clarification_result = clarification_handler.handle(query, last_turn, llm_generator)

            answer = clarification_result["answer"]
            citations = clarification_result.get("citations", [])
            clarification_metadata = clarification_result.get("metadata", {})

        # Store turn with metadata
        turn = memory.add_turn(
            query=query,
            answer=answer,
            reformulated_query=query,
            entities=last_turn.entities_mentioned if last_turn else [],
            confidence=0.95,  # High confidence for clarification
            metadata={
                "classification": query_classification.to_dict(),
                "clarification_metadata": clarification_metadata
            },
            query_type="clarification",
            used_retrieval=False,
            source_documents=[]  # CRITICAL: No sources for clarifications
        )

        # Topic tracking (will ignore clarification queries)
        memory.session.active_topic, memory.session.topic_entities = update_active_topic(
            turn,
            memory.session.active_topic,
            memory.session.topic_entities,
            "clarification"
        )

        # Return result immediately
        stage_times["query_classification"] = time.time() - stage_start
        stage_times["total"] = time.time() - pipeline_start

        result = ConversationalPipelineResult(
            answer=answer,
            query=query,
            citations=citations,  # Should be empty
            metadata={
                "query_classification": query_classification.to_dict(),
                "used_retrieval": False,
                "clarification_metadata": clarification_metadata,
                "confidence": 0.95
            },
            query_classification=query_classification.to_dict(),
            routing_decision={"handler": "clarification", "reason": "Clarification request detected"},
            retrieval_stats={"dense_count": 0, "sparse_count": 0, "fused_count": 0, "reranked_count": 0},
            context_stats={"total_chunks": 0, "total_tokens": 0},
            generation_stats={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            total_time=time.time() - pipeline_start,
            stage_times=stage_times,
            session_id=session_id,
            turn_number=turn_number,
            was_reformulated=False,
            reformulated_query=query,
            reformulation_method="",
            detected_references=[],
            extracted_entities=last_turn.entities_mentioned if last_turn else [],
            memory_stats=memory.get_statistics() if memory is not None else {}
        )

        return result, classified_query_type, query_classification.to_dict()

    # Handle formatting requests (reuse docs, NO new retrieval!)
    elif query_classification.query_type == QueryType.FORMATTING_REQUEST:
        logger.info("→ Routing to FormattingRequestHandler (NO new retrieval)")

        last_turn = _get_last_turn_for_followup(conversation_history, memory)

        if not last_turn or not formatting_handler.can_reformat(last_turn):
            answer = "There is no previous answer to reformat."
            source_docs = []
            reformatted_from = None
        else:
            answer = formatting_handler.handle(query, last_turn, llm_generator)
            source_docs = last_turn.source_documents if hasattr(last_turn, 'source_documents') else []
            reformatted_from = last_turn.turn_id

        # Store turn with metadata
        turn = memory.add_turn(
            query=query,
            answer=answer,
            reformulated_query=query,
            entities=last_turn.entities_mentioned if last_turn else [],
            confidence=0.95,  # High confidence for formatting
            metadata={"classification": query_classification.to_dict()},
            query_type="formatting_request",
            used_retrieval=False,
            source_documents=source_docs,
            reformatted_from_turn=reformatted_from
        )

        # Topic tracking (will ignore formatting queries)
        memory.session.active_topic, memory.session.topic_entities = update_active_topic(
            turn,
            memory.session.active_topic,
            memory.session.topic_entities,
            "formatting_request"
        )

        # Return result immediately
        stage_times["query_classification"] = time.time() - stage_start
        stage_times["total"] = time.time() - pipeline_start

        result = ConversationalPipelineResult(
            answer=answer,
            query=query,
            citations=[],
            metadata={
                "query_classification": query_classification.to_dict(),
                "used_retrieval": False,
                "reformatted_from_turn": reformatted_from,
                "confidence": 0.95
            },
            query_classification=query_classification.to_dict(),
            routing_decision={"handler": "formatting_request", "reason": "Formatting request detected"},
            retrieval_stats={"dense_count": 0, "sparse_count": 0, "fused_count": 0, "reranked_count": 0},
            context_stats={"total_chunks": 0, "total_tokens": 0},
            generation_stats={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            total_time=time.time() - pipeline_start,
            stage_times=stage_times,
            session_id=session_id,
            turn_number=turn_number,
            was_reformulated=False,
            reformulated_query=query,
            reformulation_method="",
            detected_references=[],
            extracted_entities=last_turn.entities_mentioned if last_turn else [],
            memory_stats=memory.get_statistics() if memory is not None else {}
        )

        return result, classified_query_type, query_classification.to_dict()

    # For continuation and new queries, continue to retrieval
    logger.info(f"Continuing to retrieval pipeline (type: {classified_query_type})")
    stage_times["query_classification"] = time.time() - stage_start

    return None, classified_query_type, query_classification.to_dict()


def _handle_enhanced_query(
        query: str,
        classification: QueryClassificationResult,
        params: Dict[str, Any],
        memory,
        session_id: str,
        turn_number: int,
        pipeline_start: float,
        stage_times: dict,
        stage_start: float,
        ConversationalPipelineResult,
        collection_name: str = "CAFL_data",  # NEW: collection name parameter
        structured_query_fast_mode: bool = False,
        structured_entity_resolution: bool = True,
        structured_natural_response_style: bool = True,
) -> Optional[Any]:
    """
    Handle enhanced query types (ENTITY_COUNT, FILE_LOCATION, EXACT_TEXT).

    These bypass the standard retrieval pipeline and operate directly
    on Qdrant payloads.

    Returns:
        ConversationalPipelineResult if handled, None if handler failed
    """
    try:
        # Get Qdrant client from memory's pipeline context if available
        qdrant_url = None
        # REMOVED: collection_name = None (was overwriting the function parameter!)

        # Try to get Qdrant connection info from environment or defaults
        import os
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        # collection_name is now passed as parameter, no need to get from env
        # But allow override from memory/session if available AND not None
        if hasattr(memory, 'session') and hasattr(memory.session, 'collection_name'):
            if memory.session.collection_name is not None:
                collection_name = memory.session.collection_name
                logger.info(f"Using collection_name from memory.session: {collection_name}")

        logger.info(f"Using collection_name: {collection_name}")

        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(
            url=qdrant_url,
            timeout=600,  # 10 minutes timeout for large operations
            prefer_grpc=False,
        )

        query_type = classification.query_type
        answer = ""
        citations: List[Dict[str, Any]] = []
        extracted_entities = []  # Track entities for memory
        resolution_metadata: Dict[str, Any] = {}

        if query_type == QueryType.ENTITY_COUNT:
            original_entities = _extract_requested_entities_from_params(params)
            require_all_entities = bool(params.get("require_all_entities", len(original_entities) > 1))
            resolution_ctx = _resolve_entities_for_structured_query(
                entity_names=original_entities,
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                enable_resolution=structured_entity_resolution,
                natural_style=structured_natural_response_style,
            )
            resolution_metadata = resolution_ctx.get("metadata", {})
            if resolution_ctx.get("clarification_answer"):
                answer = resolution_ctx["clarification_answer"]
                extracted_entities = original_entities
            else:
                resolved_entities = resolution_ctx.get("resolved_entities") or original_entities
                extracted_entities = list(resolved_entities)
                logger.info(f"→ Routing to EntityQueryHandler.handle_count_query(entities={resolved_entities})")
                handler = EntityQueryHandler(
                    qdrant_client,
                    collection_name,
                    structured_query_fast_mode=structured_query_fast_mode
                )
                result_data = handler.handle_count_query(
                    entity_name=resolved_entities[0] if resolved_entities else None,
                    entity_names=resolved_entities,
                    require_all_entities=require_all_entities
                )

                if structured_natural_response_style and result_data.get("total_mentions",
                                                                         0) == 0 and resolved_entities:
                    answer = _format_resolved_no_results_message(query_type, resolved_entities)
                else:
                    answer = format_count_response(result_data)
                citations = _build_structured_citations(QueryType.ENTITY_COUNT, result_data)
                answer = _prepend_resolution_note(answer, resolution_ctx.get("resolution_note"))

        elif query_type == QueryType.FILE_LOCATION:
            original_entities = _extract_requested_entities_from_params(params)
            require_all_entities = bool(params.get("require_all_entities", len(original_entities) > 1))
            resolution_ctx = _resolve_entities_for_structured_query(
                entity_names=original_entities,
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                enable_resolution=structured_entity_resolution,
                natural_style=structured_natural_response_style,
            )
            resolution_metadata = resolution_ctx.get("metadata", {})
            if resolution_ctx.get("clarification_answer"):
                answer = resolution_ctx["clarification_answer"]
                extracted_entities = original_entities
            else:
                resolved_entities = resolution_ctx.get("resolved_entities") or original_entities
                extracted_entities = list(resolved_entities)
                logger.info(f"→ Routing to EntityQueryHandler.handle_file_location_query(entities={resolved_entities})")
                handler = EntityQueryHandler(
                    qdrant_client,
                    collection_name,
                    structured_query_fast_mode=structured_query_fast_mode
                )
                result_data = handler.handle_file_location_query(
                    entity_name=resolved_entities[0] if resolved_entities else None,
                    entity_names=resolved_entities,
                    require_all_entities=require_all_entities
                )
                if structured_natural_response_style and result_data.get("total_files", 0) == 0 and resolved_entities:
                    answer = _format_resolved_no_results_message(query_type, resolved_entities)
                else:
                    answer = format_file_location_response(result_data)
                citations = _build_structured_citations(QueryType.FILE_LOCATION, result_data)
                answer = _prepend_resolution_note(answer, resolution_ctx.get("resolution_note"))

        elif query_type == QueryType.EXACT_TEXT:
            original_entities = _extract_requested_entities_from_params(params)
            require_all_entities = bool(params.get("require_all_entities", len(original_entities) > 1))
            sender = params.get("sender")
            receiver = params.get("receiver")
            include_count = bool(params.get("include_count", False))

            resolution_ctx = {
                "resolved_entities": list(original_entities),
                "resolution_note": None,
                "clarification_answer": None,
                "metadata": {}
            }
            if original_entities:
                resolution_ctx = _resolve_entities_for_structured_query(
                    entity_names=original_entities,
                    qdrant_client=qdrant_client,
                    collection_name=collection_name,
                    enable_resolution=structured_entity_resolution,
                    natural_style=structured_natural_response_style,
                )
                resolution_metadata = resolution_ctx.get("metadata", {})

            if resolution_ctx.get("clarification_answer"):
                answer = resolution_ctx["clarification_answer"]
                extracted_entities = list(original_entities)
            else:
                resolved_entities = resolution_ctx.get("resolved_entities") or list(original_entities)
                resolved_entity = resolved_entities[0] if len(resolved_entities) == 1 else None

                extracted_entities = []
                extracted_entities.extend(resolved_entities)
                if sender:
                    extracted_entities.append(sender)
                if receiver:
                    extracted_entities.append(receiver)

                logger.info(f"→ Routing to ExactTextHandler.handle_exact_text({params})")
                text_handler = ExactTextHandler(
                    qdrant_client,
                    collection_name,
                    structured_query_fast_mode=structured_query_fast_mode
                )
                allowed_handler_params = {
                    "entity_name",
                    "entity_names",
                    "require_all_entities",
                    "sender",
                    "receiver",
                    "date",
                    "keyword",
                    "max_results",
                }
                handler_params = {
                    k: v for k, v in params.items()
                    if k in allowed_handler_params and v is not None
                }
                handler_params["entity_name"] = resolved_entity
                if len(resolved_entities) > 1:
                    handler_params["entity_names"] = resolved_entities
                    handler_params["require_all_entities"] = require_all_entities
                else:
                    handler_params.pop("entity_names", None)
                    handler_params.pop("require_all_entities", None)

                text_result_data = text_handler.handle_exact_text(**handler_params)
                if (
                        structured_natural_response_style and
                        text_result_data.get("total_found", 0) == 0 and
                        resolved_entities
                ):
                    text_answer = _format_resolved_no_results_message(query_type, resolved_entities)
                else:
                    text_answer = format_exact_text_response(text_result_data)
                text_citations = _build_structured_citations(QueryType.EXACT_TEXT, text_result_data)

                if include_count and resolved_entities:
                    count_handler = EntityQueryHandler(
                        qdrant_client,
                        collection_name,
                        structured_query_fast_mode=structured_query_fast_mode
                    )
                    count_result_data = count_handler.handle_count_query(
                        entity_name=resolved_entities[0] if resolved_entities else None,
                        entity_names=resolved_entities,
                        require_all_entities=require_all_entities
                    )

                    if (
                            structured_natural_response_style and
                            count_result_data.get("total_mentions", 0) == 0
                    ):
                        answer = _format_resolved_no_results_message(QueryType.ENTITY_COUNT, resolved_entities)
                        citations = text_citations
                    else:
                        count_answer = format_count_response(count_result_data)
                        answer = f"{count_answer}\n\n{text_answer}"
                        count_citations = _build_structured_citations(QueryType.ENTITY_COUNT, count_result_data)
                        citations = _merge_structured_citations(count_citations, text_citations)
                else:
                    answer = text_answer
                    citations = text_citations

                answer = _prepend_resolution_note(answer, resolution_ctx.get("resolution_note"))

        else:
            return None

        citations = normalize_citations_sources(citations)

        # Store turn in memory
        actual_turn_number = turn_number  # Default to input if memory not available
        if memory is not None:
            turn = memory.add_turn(
                query=query,
                answer=answer,
                reformulated_query=query,
                citations=citations,
                entities=extracted_entities,  # Pass extracted entities
                confidence=classification.confidence,
                metadata={
                    "classification": classification.to_dict(),
                    "entity_resolution": resolution_metadata
                },
                query_type=classification.query_type.value,
                used_retrieval=False,
                source_documents=[]
            )

            # Use the actual turn ID from memory
            actual_turn_number = turn.turn_id

            memory.session.active_topic, memory.session.topic_entities = update_active_topic(
                turn,
                memory.session.active_topic,
                memory.session.topic_entities,
                classification.query_type.value
            )

        stage_times["query_classification"] = time.time() - stage_start
        stage_times["enhanced_handler"] = time.time() - stage_start
        stage_times["total"] = time.time() - pipeline_start

        result = ConversationalPipelineResult(
            answer=answer,
            query=query,
            citations=citations,
            metadata={
                "query_classification": classification.to_dict(),
                "used_retrieval": False,
                "handler": classification.query_type.value,
                "confidence": classification.confidence,
                "entity_resolution": resolution_metadata,
                "sources_returned": len(citations),
            },
            query_classification=classification.to_dict(),
            routing_decision={
                "handler": classification.query_type.value,
                "reason": classification.reasoning
            },
            retrieval_stats={"dense_count": 0, "sparse_count": 0, "fused_count": 0, "reranked_count": 0},
            context_stats={"total_chunks": 0, "total_tokens": 0},
            generation_stats={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            total_time=time.time() - pipeline_start,
            stage_times=stage_times,
            session_id=session_id,
            turn_number=actual_turn_number,  # Use actual turn number from memory
            was_reformulated=False,
            reformulated_query=query,
            reformulation_method="",
            detected_references=[],
            extracted_entities=extracted_entities,
            memory_stats=memory.get_statistics() if memory is not None else {}
        )

        return result

    except Exception as e:
        logger.error(f"Enhanced query handler failed: {e}", exc_info=True)
        return None


def _resolve_entity_for_structured_query(
        entity_name: Optional[str],
        qdrant_client,
        collection_name: str,
        enable_resolution: bool,
        natural_style: bool,
) -> Dict[str, Any]:
    """Resolve entity name with confidence-aware correction/clarification behavior."""
    context = {
        "resolved_entity": entity_name,
        "resolution_note": None,
        "clarification_answer": None,
        "metadata": {},
    }

    if not entity_name or not enable_resolution:
        return context

    try:
        from src.query_processing.structured_entity_resolver import StructuredEntityResolver
    except Exception as e:
        logger.debug(f"StructuredEntityResolver unavailable: {e}")
        return context

    resolver = StructuredEntityResolver(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
    )
    resolution = resolver.resolve(entity_name)

    context["metadata"] = {
        "original_entity": resolution.original_entity,
        "resolved_entity": resolution.resolved_entity,
        "confidence": resolution.confidence,
        "tier": resolution.tier,
        "was_corrected": resolution.was_corrected,
        "suggestions": resolution.suggestions,
        "reason": resolution.reason,
    }

    if resolution.tier in {"high", "medium"} and resolution.resolved_entity:
        context["resolved_entity"] = resolution.resolved_entity
        if natural_style and resolution.tier == "medium":
            context["resolution_note"] = f"Showing results for {resolution.resolved_entity}."
        return context

    if _verify_direct_entity_text_match(
            entity_name=cleaned_entity_name(entity_name),
            qdrant_client=qdrant_client,
            collection_name=collection_name,
    ):
        context["resolved_entity"] = cleaned_entity_name(entity_name)
        context["metadata"]["resolved_entity"] = cleaned_entity_name(entity_name)
        context["metadata"]["confidence"] = 1.0
        context["metadata"]["tier"] = "verified_exact_text"
        context["metadata"]["was_corrected"] = False
        context["metadata"]["reason"] = "direct_text_match_without_metadata"
        context["metadata"]["direct_text_match_fallback"] = True
        return context

    if resolution.tier in {"ambiguous", "low"} and resolution.suggestions:
        suggestion_list = ", ".join(resolution.suggestions[:3])
        context["clarification_answer"] = (
            f"I could not confidently match '{entity_name}' to a single name in the files. "
            f"Did you mean: {suggestion_list}?"
        )
        return context

    return context


def _verify_direct_entity_text_match(
        entity_name: Optional[str],
        qdrant_client,
        collection_name: str,
) -> bool:
    """Fallback correctness check when entity metadata does not capture the queried name."""
    if not entity_name:
        return False

    try:
        handler = EntityQueryHandler(
            qdrant_client,
            collection_name,
            structured_query_fast_mode=False,
        )
        return handler.has_direct_text_match(entity_name)
    except Exception as exc:
        logger.warning(
            "Direct text verification fallback failed for entity '%s': %s",
            entity_name,
            exc,
        )
        return False


def cleaned_entity_name(entity_name: Optional[str]) -> Optional[str]:
    if not isinstance(entity_name, str):
        return entity_name
    cleaned = entity_name.strip()
    return cleaned or None


def _extract_requested_entities_from_params(params: Dict[str, Any]) -> List[str]:
    """Normalize structured params into a deduplicated entity list."""
    values: List[str] = []
    entity_names = params.get("entity_names")
    if isinstance(entity_names, list):
        values.extend(entity_names)
    entity_name = params.get("entity_name")
    if isinstance(entity_name, str) and entity_name.strip():
        values.append(entity_name)

    normalized: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
    return normalized


def _resolve_entities_for_structured_query(
        entity_names: List[str],
        qdrant_client,
        collection_name: str,
        enable_resolution: bool,
        natural_style: bool,
) -> Dict[str, Any]:
    """Resolve one or multiple entities with confidence-aware behavior."""
    entities = [e for e in entity_names if isinstance(e, str) and e.strip()]
    context = {
        "resolved_entities": list(entities),
        "resolution_note": None,
        "clarification_answer": None,
        "metadata": {
            "is_multi_entity": len(entities) > 1,
            "original_entities": list(entities),
            "resolved_entities": list(entities),
            "entities": [],
        },
    }

    if not entities or not enable_resolution:
        return context

    # Preserve existing single-entity behavior exactly.
    if len(entities) == 1:
        single_ctx = _resolve_entity_for_structured_query(
            entity_name=entities[0],
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            enable_resolution=enable_resolution,
            natural_style=natural_style,
        )
        return {
            "resolved_entities": [single_ctx.get("resolved_entity") or entities[0]],
            "resolution_note": single_ctx.get("resolution_note"),
            "clarification_answer": single_ctx.get("clarification_answer"),
            "metadata": {
                "is_multi_entity": False,
                "original_entities": list(entities),
                "resolved_entities": [single_ctx.get("resolved_entity") or entities[0]],
                "entities": [single_ctx.get("metadata", {})],
            },
        }

    resolved_entities: List[str] = []
    resolution_notes: List[str] = []
    per_entity_meta: List[Dict[str, Any]] = []

    try:
        from src.query_processing.structured_entity_resolver import StructuredEntityResolver
    except Exception:
        # If resolver is unavailable, continue with raw entities.
        context["resolved_entities"] = list(entities)
        return context

    resolver = StructuredEntityResolver(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
    )

    for entity in entities:
        resolution = resolver.resolve(entity)
        entity_meta = {
            "original_entity": resolution.original_entity,
            "resolved_entity": resolution.resolved_entity,
            "confidence": resolution.confidence,
            "tier": resolution.tier,
            "was_corrected": resolution.was_corrected,
            "suggestions": resolution.suggestions,
            "reason": resolution.reason,
        }
        per_entity_meta.append(entity_meta)

        if resolution.tier in {"high", "medium"} and resolution.resolved_entity:
            resolved_entities.append(resolution.resolved_entity)
            if natural_style and resolution.tier == "medium":
                resolution_notes.append(f"Showing results for {resolution.resolved_entity}.")
            continue

        # Multi-entity queries are less brittle: auto-select best suggestion and
        # proceed, instead of blocking on a single typo.
        if resolution.suggestions:
            best = resolution.suggestions[0]
            resolved_entities.append(best)
            if natural_style:
                resolution_notes.append(f"Showing results for {best}.")
            continue

        context["clarification_answer"] = (
            f"I could not confidently match '{entity}' to a name in the files. "
            "Please provide the exact name."
        )
        context["metadata"]["entities"] = per_entity_meta
        return context

    deduped: List[str] = []
    seen = set()
    for entity in resolved_entities:
        key = entity.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)

    context["resolved_entities"] = deduped
    context["metadata"] = {
        "is_multi_entity": len(entities) > 1,
        "original_entities": list(entities),
        "resolved_entities": list(deduped),
        "entities": per_entity_meta,
    }

    if resolution_notes:
        unique_notes: List[str] = []
        seen_notes = set()
        for note in resolution_notes:
            if note in seen_notes:
                continue
            seen_notes.add(note)
            unique_notes.append(note)
        context["resolution_note"] = "\n".join(unique_notes)

    return context


def _format_entity_list_label(entity_value: Any) -> str:
    """Format one or many entities into a natural phrase."""
    if isinstance(entity_value, list):
        entities = [e for e in entity_value if isinstance(e, str) and e.strip()]
    elif isinstance(entity_value, str) and entity_value.strip():
        entities = [entity_value.strip()]
    else:
        entities = []

    if not entities:
        return "the requested entity"
    if len(entities) == 1:
        return entities[0]
    if len(entities) == 2:
        return f"{entities[0]} and {entities[1]}"
    return f"{', '.join(entities[:-1])}, and {entities[-1]}"


def _format_resolved_no_results_message(query_type: QueryType, entity_name: Any) -> str:
    """Natural no-result messaging after confident entity resolution."""
    label = _format_entity_list_label(entity_name)
    is_multi = isinstance(entity_name, list) and len(entity_name) > 1

    if query_type == QueryType.EXACT_TEXT:
        if is_multi:
            return f"I could not find exact text snippets where both {label} are mentioned in the indexed files."
        return f"I could not find exact text snippets where {label} is mentioned in the indexed files."

    if query_type == QueryType.FILE_LOCATION:
        if is_multi:
            return f"I could not find files where both {label} are mentioned in the indexed data."
        return f"I could not find files where {label} is mentioned in the indexed data."

    if is_multi:
        return f"Both {label} are not co-mentioned in the indexed files."
    return f"{label} is not mentioned in the indexed files."


def _prepend_resolution_note(answer: str, resolution_note: Optional[str]) -> str:
    """Prefix answer with medium-confidence resolution disclosure when needed."""
    if not resolution_note:
        return answer
    return f"{resolution_note}\n\n{answer}"


def _build_structured_citations(query_type: QueryType, result_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build link-free structured citations so backend can resolve file URLs."""
    citations: List[Dict[str, Any]] = []

    if query_type == QueryType.ENTITY_COUNT:
        for idx, item in enumerate(result_data.get("file_breakdown", []) or [], 1):
            source = item.get("file_name")
            if not source:
                continue
            citations.append(
                {
                    "id": idx,
                    "source": source,
                    "mentions": int(item.get("count", 0) or 0),
                    "rank": idx,
                    "query_type": query_type.value,
                }
            )
        return citations

    if query_type == QueryType.FILE_LOCATION:
        for idx, item in enumerate(result_data.get("file_details", []) or [], 1):
            source = item.get("file_name")
            if not source:
                continue
            citations.append(
                {
                    "id": idx,
                    "source": source,
                    "mentions": int(item.get("mention_count", 0) or 0),
                    "rank": idx,
                    "query_type": query_type.value,
                }
            )
        return citations

    if query_type == QueryType.EXACT_TEXT:
        snippets = result_data.get("snippets", []) or []
        by_source: Dict[str, Dict[str, Any]] = {}
        for snippet in snippets:
            source = snippet.get("source_file")
            if not source:
                continue
            entry = by_source.setdefault(
                source,
                {
                    "source": source,
                    "snippet_count": 0,
                    "pages": set(),
                    "first_snippet": snippet.get("text", ""),
                }
            )
            entry["snippet_count"] += 1
            page = snippet.get("page_number")
            if page is not None:
                entry["pages"].add(page)

        sorted_items = sorted(
            by_source.values(),
            key=lambda v: v.get("snippet_count", 0),
            reverse=True
        )
        for idx, item in enumerate(sorted_items, 1):
            citations.append(
                {
                    "id": idx,
                    "source": item["source"],
                    "mentions": int(item.get("snippet_count", 0)),
                    "pages": sorted(item.get("pages", set())),
                    "text": item.get("first_snippet", "")[:300],
                    "rank": idx,
                    "query_type": query_type.value,
                }
            )
        return citations

    return citations


def _merge_structured_citations(
        primary: List[Dict[str, Any]],
        secondary: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge citation lists by source, preserving primary order first."""
    merged: List[Dict[str, Any]] = []
    seen = set()
    for item in (primary or []) + (secondary or []):
        source = str(item.get("source", "")).strip()
        if not source:
            continue
        key = source.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)

    for idx, item in enumerate(merged, 1):
        item["id"] = idx
        item["rank"] = idx
    return merged


def _handle_structured_resolution_followup(
        query: str,
        conversation_history: list,
        memory,
        session_id: str,
        turn_number: int,
        pipeline_start: float,
        stage_times: dict,
        stage_start: float,
        ConversationalPipelineResult,
        collection_name: str,
        structured_query_fast_mode: bool,
        structured_natural_response_style: bool,
) -> Optional[Tuple[Any, str, Dict[str, Any]]]:
    """
    Handle short follow-ups (e.g., 'yes') after structured disambiguation prompts.

    Example:
    - Q1: "show files where doland trump was mentioned"
      -> "Did you mean Donald Trump, ...?"
    - Q2: "yes"
      -> run previous structured query for "Donald Trump"
    """
    last_turn = _get_last_turn_for_followup(conversation_history, memory)
    if last_turn is None:
        return None

    metadata = _turn_dict(last_turn).get("metadata", {}) or {}
    resolution_meta = metadata.get("entity_resolution", {}) or {}
    if not isinstance(resolution_meta, dict):
        resolution_meta = {}

    answer_text = _turn_answer(last_turn)
    suggestions = resolution_meta.get("suggestions") or []
    if not suggestions:
        suggestions = _extract_suggestions_from_disambiguation_answer(answer_text)
    if not suggestions:
        return None

    tier = str(resolution_meta.get("tier", "")).lower()
    has_disambiguation_prompt = bool(_extract_suggestions_from_disambiguation_answer(answer_text))
    if tier not in {"low", "ambiguous"} and not has_disambiguation_prompt:
        return None

    classification_dict = metadata.get("classification", {}) or {}
    prev_type = _parse_query_type(
        classification_dict.get("query_type") or _turn_dict(last_turn).get("query_type")
    )
    if prev_type not in {QueryType.ENTITY_COUNT, QueryType.FILE_LOCATION, QueryType.EXACT_TEXT}:
        return None

    selected = _select_suggested_entity_from_followup(query, suggestions)
    if selected is None:
        if _NEGATIVE_FOLLOWUP_RE.match((query or "").strip()):
            answer = "Okay. Please tell me the exact name you want me to search for."
            result = _build_structured_followup_result(
                query=query,
                answer=answer,
                classification_type=prev_type,
                session_id=session_id,
                turn_number=turn_number,
                pipeline_start=pipeline_start,
                stage_times=stage_times,
                stage_start=stage_start,
                memory=memory,
                ConversationalPipelineResult=ConversationalPipelineResult,
                extracted_entities=[],
                metadata_extra={
                    "entity_resolution_followup": {
                        "action": "rejected",
                        "suggestions": suggestions[:3],
                    }
                },
            )
            return result, prev_type.value, result.query_classification
        return None

    prev_params = {}
    if isinstance(classification_dict, dict):
        prev_params = dict(classification_dict.get("metadata") or {})

    prev_params["entity_name"] = selected
    followup_classification = QueryClassificationResult(
        query_type=prev_type,
        confidence=0.95,
        matched_patterns=["structured_resolution_followup"],
        reasoning=f"User confirmed structured resolution for '{selected}'",
        metadata=prev_params
    )

    result = _handle_enhanced_query(
        query=query,
        classification=followup_classification,
        params=prev_params,
        memory=memory,
        session_id=session_id,
        turn_number=turn_number,
        pipeline_start=pipeline_start,
        stage_times=stage_times,
        stage_start=stage_start,
        ConversationalPipelineResult=ConversationalPipelineResult,
        collection_name=collection_name,
        structured_query_fast_mode=structured_query_fast_mode,
        structured_entity_resolution=False,
        structured_natural_response_style=structured_natural_response_style,
    )

    if result is None:
        return None

    return result, prev_type.value, followup_classification.to_dict()


def _turn_dict(turn: Any) -> Dict[str, Any]:
    """Convert turn object/dict to a plain dict for robust metadata access."""
    if isinstance(turn, dict):
        return turn
    return {
        "query_type": getattr(turn, "query_type", None),
        "metadata": getattr(turn, "metadata", {}) or {},
    }


def _turn_answer(turn: Any) -> str:
    """Safely extract answer text from turn object or dict."""
    if isinstance(turn, dict):
        return str(turn.get("answer", "") or "")
    return str(getattr(turn, "answer", "") or "")


def _parse_query_type(value: Any) -> Optional[QueryType]:
    """Parse query type enum from enum or string value."""
    if isinstance(value, QueryType):
        return value
    if isinstance(value, str):
        for member in QueryType:
            if member.value == value:
                return member
    return None


def _normalize_followup_text(value: str) -> str:
    cleaned = (value or "").strip().lower()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = cleaned.strip(".,!?;:\"'`")
    return cleaned


def _select_suggested_entity_from_followup(query: str, suggestions: List[str]) -> Optional[str]:
    """Resolve follow-up reply to one of provided suggestions."""
    if not suggestions:
        return None

    normalized_query = _normalize_followup_text(query)
    if not normalized_query:
        return None

    if _AFFIRMATIVE_FOLLOWUP_RE.match((query or "").strip()):
        return suggestions[0]

    numeric_match = _NUMERIC_CHOICE_RE.match((query or "").strip())
    if numeric_match:
        idx = int(numeric_match.group(1)) - 1
        if 0 <= idx < len(suggestions):
            return suggestions[idx]

    suggestion_lookup = {
        _normalize_followup_text(suggestion): suggestion
        for suggestion in suggestions
    }
    return suggestion_lookup.get(normalized_query)


def _get_last_turn_for_followup(conversation_history: list, memory) -> Optional[Any]:
    """Get last conversational turn, preferring live memory when available."""
    if memory is not None and hasattr(memory, "get_last_turn"):
        try:
            last_turn = memory.get_last_turn()
            if last_turn is not None:
                return last_turn
        except Exception:
            pass

    if conversation_history:
        return conversation_history[-1]

    return None


def _extract_suggestions_from_disambiguation_answer(answer: str) -> List[str]:
    """
    Extract disambiguation suggestions from assistant answer text:
    "Did you mean: A, B, C?"
    """
    if not answer:
        return []

    match = _DID_YOU_MEAN_RE.search(answer)
    if not match:
        return []

    raw = match.group(1).strip()
    if not raw:
        return []

    # Normalize separators ("A, B or C" -> ["A", "B", "C"])
    normalized = re.sub(r"\s+\bor\b\s+", ", ", raw, flags=re.IGNORECASE)
    parts = [p.strip(" \t\n\r'\"`.,!?;:") for p in normalized.split(",")]
    suggestions = [p for p in parts if p]

    # Keep response deterministic and bounded.
    return suggestions[:5]


def _build_structured_followup_result(
        query: str,
        answer: str,
        classification_type: QueryType,
        session_id: str,
        turn_number: int,
        pipeline_start: float,
        stage_times: dict,
        stage_start: float,
        memory,
        ConversationalPipelineResult,
        extracted_entities: List[str],
        metadata_extra: Optional[Dict[str, Any]] = None,
):
    """Build a lightweight structured follow-up response and persist to memory."""
    classification = QueryClassificationResult(
        query_type=classification_type,
        confidence=0.9,
        matched_patterns=["structured_resolution_followup_meta"],
        reasoning="Structured resolution follow-up handling",
        metadata={}
    )

    actual_turn_number = turn_number
    if memory is not None:
        turn = memory.add_turn(
            query=query,
            answer=answer,
            reformulated_query=query,
            entities=extracted_entities,
            confidence=classification.confidence,
            metadata={"classification": classification.to_dict(), **(metadata_extra or {})},
            query_type=classification.query_type.value,
            used_retrieval=False,
            source_documents=[]
        )
        actual_turn_number = turn.turn_id
        memory.session.active_topic, memory.session.topic_entities = update_active_topic(
            turn,
            memory.session.active_topic,
            memory.session.topic_entities,
            classification.query_type.value
        )

    stage_times["query_classification"] = time.time() - stage_start
    stage_times["enhanced_handler"] = time.time() - stage_start
    stage_times["total"] = time.time() - pipeline_start

    result = ConversationalPipelineResult(
        answer=answer,
        query=query,
        citations=[],
        metadata={
            "query_classification": classification.to_dict(),
            "used_retrieval": False,
            "handler": classification.query_type.value,
            "confidence": classification.confidence,
            **(metadata_extra or {})
        },
        query_classification=classification.to_dict(),
        routing_decision={
            "handler": classification.query_type.value,
            "reason": classification.reasoning
        },
        retrieval_stats={"dense_count": 0, "sparse_count": 0, "fused_count": 0, "reranked_count": 0},
        context_stats={"total_chunks": 0, "total_tokens": 0},
        generation_stats={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        total_time=time.time() - pipeline_start,
        stage_times=stage_times,
        session_id=session_id,
        turn_number=actual_turn_number,
        was_reformulated=False,
        reformulated_query=query,
        reformulation_method="",
        detected_references=[],
        extracted_entities=extracted_entities,
        memory_stats=memory.get_statistics() if memory is not None else {}
    )
    return result
