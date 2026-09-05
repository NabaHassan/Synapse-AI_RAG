"""
Integrated RAG Pipeline

This module implements the complete RAG pipeline integrating:
- Query Processing (query_processing/)
- Retrieval (retrieval/)
- Context Handling (context_handling/)
- Generation (generation/)

The pipeline provides end-to-end question answering with:
- Query classification and routing
- Hybrid retrieval (dense + sparse)
- Context verification and deduplication
- Citation-based answer generation
"""

import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Query Processing
from src.query_processing.query_router import QueryRouter
from src.query_processing.query_handler import QueryHandler
from src.query_processing.query_enhancer import QueryEnhancer
from src.query_processing.query_classifier import QueryClassifier

# Retrieval
from src.retrieval.reranker import Reranker
from src.retrieval.result_fusion import ResultFusion
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

# Context Handling
from src.context_handling.context_verifier import ContextVerifier

# Generation
from src.generation.llm_generator import LLMGenerator
from src.generation.prompt_builder import RAGPromptBuilder
from src.generation.citation_extractor import CitationExtractor
from src.generation.answer_sanitizer import sanitize_generated_answer
from src.utils.source_normalization import normalize_citations_sources

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for RAG Pipeline."""
    # Collection
    collection_name: str = "knowledge_base"
    qdrant_url: str = "http://localhost:6333"

    # Query Processing
    enable_query_enhancement: bool = False  # Optional enhancement
    enable_hyde: bool = False  # Optional HyDE

    # Retrieval
    dense_top_k: int = 50
    sparse_top_k: int = 50
    dense_default_filters: Optional[Dict[str, Any]] = None
    fusion_strategy: str = "rrf"  # rrf, weighted, or haystack
    fusion_top_k: int = 20
    rerank_top_k: int = 5

    # Context Processing
    nli_threshold: float = 0.1  # Lower for more lenient filtering
    dedup_threshold: float = 0.85
    ordering_strategy: str = "lost_in_middle"

    # Generation
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens: int = 2048  # Increased for comprehensive responses
    temperature: float = 0.3
    normalize_newlines: str = "preserve"  # "preserve", "single", or "remove"

    # GPU
    use_gpu: bool = False


@dataclass
class PipelineResult:
    """Result from RAG pipeline execution."""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    # Stage results
    query_classification: Dict[str, Any]
    routing_decision: Dict[str, Any]
    retrieval_stats: Dict[str, Any]
    context_stats: Dict[str, Any]
    generation_stats: Dict[str, Any]

    # Timing
    total_time: float
    stage_times: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RAGPipeline:
    """
    Complete RAG Pipeline integrating all components.

    Implements the full flow:
    1. Query Processing: Classify and route query
    2. Retrieval: Hybrid search (dense + sparse) with fusion and reranking
    3. Context Handling: Verify relevance, deduplicate, order, prepare citations
    4. Generation: Build prompt, generate answer with citations
    """

    def __init__(
            self,
            collection_name: str = "knowledge_base",
            config: Optional[PipelineConfig] = None
    ):
        """
        Initialize RAG Pipeline.

        Args:
            collection_name: Qdrant collection name
            config: Pipeline configuration
        """
        self.collection_name = collection_name
        self.config = config or PipelineConfig()
        self.config.collection_name = collection_name

        logger.info("=" * 80)
        logger.info("Initializing RAG Pipeline")
        logger.info("=" * 80)
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Qdrant URL: {self.config.qdrant_url}")
        logger.info(f"GPU enabled: {self.config.use_gpu}")

        # Initialize all components
        self._init_query_processing()
        self._init_retrieval()
        self._init_context_handling()
        self._init_generation()

        logger.info("=" * 80)
        logger.info("RAG Pipeline initialized successfully")
        logger.info("=" * 80)
        logger.info("")

    def _init_query_processing(self):
        """Initialize query processing components."""
        logger.info("\nInitializing Query Processing...")
        logger.info("-" * 80)

        # Query Handler
        canned_responses = getattr(self.config, "canned_responses", None)
        self.query_handler = QueryHandler(canned_responses=canned_responses)
        logger.info("QueryHandler initialized")

        # Query Classifier
        self.query_classifier = QueryClassifier()
        logger.info("QueryClassifier initialized")

        # Query Router
        self.query_router = QueryRouter(backend="custom", log_decisions=True)
        logger.info("QueryRouter initialized")

        # Query Enhancer (optional)
        if self.config.enable_query_enhancement:
            self.query_enhancer = QueryEnhancer()
            logger.info("QueryEnhancer initialized")
        else:
            self.query_enhancer = None
            logger.info("QueryEnhancer skipped (disabled)")

    def _init_retrieval(self):
        """Initialize retrieval components."""
        logger.info("\nInitializing Retrieval")
        logger.info("-" * 80)

        self.dense_retriever = DenseRetriever(
            collection_name=self.collection_name,
            embedding_model="BAAI/bge-large-en-v1.5",  # Match indexing model
            qdrant_url=self.config.qdrant_url,
            top_k=self.config.dense_top_k,
            use_gpu=self.config.use_gpu,
            default_filters=self.config.dense_default_filters,
        )
        logger.info("DenseRetriever initialized")

        self.sparse_retriever = SparseRetriever(
            collection_name=self.collection_name,
            qdrant_url=self.config.qdrant_url,
            top_k=self.config.sparse_top_k,
        )
        logger.info("SparseRetriever initialized")

        # Result Fusion
        self.result_fusion = ResultFusion(
            strategy=self.config.fusion_strategy,
            top_k=self.config.fusion_top_k
        )
        logger.info(f"ResultFusion initialized (strategy={self.config.fusion_strategy})")

        # Reranker
        self.reranker = Reranker(
            top_k=self.config.rerank_top_k,
            use_gpu=self.config.use_gpu
        )
        logger.info("Reranker initialized")

    def _init_context_handling(self):
        """Initialize context handling components."""
        logger.info("\nInitializing Context Handling...")
        logger.info("-" * 80)

        self.context_verifier = ContextVerifier(
            threshold=self.config.nli_threshold,
            dedup_threshold=self.config.dedup_threshold,
            use_gpu=self.config.use_gpu
        )
        logger.info("ContextVerifier initialized")

    def _init_generation(self):
        """Initialize generation components."""
        logger.info("\nInitializing Generation...")
        logger.info("-" * 80)

        # Prompt Builder
        self.prompt_builder = RAGPromptBuilder()
        logger.info("PromptBuilder initialized")

        # LLM Generator
        from src.generation.llm_generator import GenerationConfig
        gen_config = GenerationConfig(
            model_name=self.config.llm_model,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            device="cuda" if self.config.use_gpu else "cpu",
            normalize_newlines=self.config.normalize_newlines
        )
        self.llm_generator = LLMGenerator(config=gen_config)
        logger.info("LLMGenerator initialized")

        # Citation Extractor
        self.citation_extractor = CitationExtractor()
        logger.info("CitationExtractor initialized")

    def run(
            self,
            query: str,
            user_id: Optional[str] = None,
            session_id: Optional[str] = None,
            connector: Optional[str] = None
    ) -> PipelineResult:
        """
        Run complete RAG pipeline.

        Args:
            query: User query
            user_id: Optional user identifier
            session_id: Optional session identifier

        Returns:
            PipelineResult with answer, citations, and metadata
        """
        pipeline_start = time.time()
        stage_times = {}

        logger.info("\n" + "=" * 80)
        logger.info("Starting RAG Pipeline")
        logger.info("=" * 80)
        logger.info(f"Query: {query}")
        logger.info("")

        try:
            # ===================================================================
            # STAGE 1: Query Processing
            # ===================================================================
            stage_start = time.time()
            logger.info("STAGE 1: Query Processing")
            logger.info("-" * 80)

            # 1.1: Handle query
            query_obj = self.query_handler.handle_query(
                query_text=query,
                user_id=user_id,
                session_id=session_id
            )

            if not query_obj.is_valid:
                raise ValueError(f"Invalid query: {query_obj.metadata.validation_message}")

            logger.info(f"Query validated (ID: {query_obj.metadata.query_id})")

            # 1.2: Classify query
            classification = self.query_classifier.classify(query)
            logger.info(
                f"Query classified: {classification.query_type} (confidence: {classification.confidence:.3f})"
            )

            # 1.3: Route query
            routing_decision = self.query_router.route(classification, connector=connector)
            logger.info(f"Query routed to: {routing_decision.route} ({routing_decision.reason})")

            # 1.4: Enhance query if using RAG pipeline and enhancer is enabled
            search_query = query
            enhancement_info = None
            if self.query_enhancer and routing_decision.route == "rag_pipeline":
                enhanced = self.query_enhancer.enhance(
                    query,
                    query_type=classification.query_type,
                    complexity=classification.complexity
                )

                # Use enhanced query if expansions/sub-queries were generated
                if enhanced.expanded_queries or enhanced.sub_queries:
                    # Combine original + expansions for better retrieval
                    # Use first expansion if available, otherwise use sub-queries
                    if enhanced.expanded_queries:
                        search_query = f"{query} {enhanced.expanded_queries[0]}"
                        logger.info(f"Query expanded: using first expansion for retrieval")
                    elif enhanced.sub_queries:
                        # For complex queries, use first sub-query as primary search
                        search_query = enhanced.sub_queries[0]
                        logger.info(f"Query decomposed: using first sub-query for retrieval")

                    enhancement_info = {
                        "num_expansions": len(enhanced.expanded_queries),
                        "num_sub_queries": len(enhanced.sub_queries),
                        "hyde_used": enhanced.hyde_document is not None
                    }
                else:
                    logger.info("Query enhancement produced no expansions, using original query")

            stage_times["query_processing"] = time.time() - stage_start
            logger.info(f"Query Processing: {stage_times['query_processing']:.2f}s\n")

            # Check routing decision
            if routing_decision.route == "reject":
                logger.warning("Query rejected by router")
                return self._create_rejection_result(
                    query, classification, routing_decision, stage_times, pipeline_start
                )

            # ===================================================================
            # STAGE 2: Retrieval (Parallel Execution)
            # ===================================================================
            stage_start = time.time()
            logger.info("STAGE 2: Parallel Retrieval")
            logger.info("-" * 80)

            # Run dense and sparse retrieval in parallel for faster response
            dense_docs = []
            sparse_docs = []
            parallel_success = False

            try:
                logger.info("Running dense and sparse retrieval in parallel...")

                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Submit both retrieval tasks
                    dense_future = executor.submit(
                        self.dense_retriever.retrieve,
                        search_query
                    )
                    sparse_future = executor.submit(
                        self.sparse_retriever.retrieve,
                        search_query
                    )

                    # Wait for both to complete and collect results
                    futures = {
                        dense_future: "dense",
                        sparse_future: "sparse"
                    }

                    for future in as_completed(futures):
                        retriever_type = futures[future]
                        try:
                            result = future.result(timeout=30)  # 30s timeout per retriever
                            if retriever_type == "dense":
                                dense_docs = result
                            else:
                                sparse_docs = result
                        except Exception as e:
                            logger.error(f"  {retriever_type.capitalize()} retrieval failed: {e}")
                            # Continue with empty results for failed retriever
                            if retriever_type == "dense":
                                dense_docs = []
                            else:
                                sparse_docs = []

                parallel_success = True
                logger.info("Parallel retrieval completed successfully")

            except Exception as e:
                # Fallback to sequential retrieval if parallel execution fails
                logger.warning(f"Parallel retrieval failed: {e}")
                logger.info("Falling back to sequential retrieval...")

                try:
                    dense_docs = self.dense_retriever.retrieve(search_query)
                except Exception as dense_error:
                    logger.error(f"  Dense retrieval failed: {dense_error}")
                    dense_docs = []

                try:
                    sparse_docs = self.sparse_retriever.retrieve(search_query)
                except Exception as sparse_error:
                    logger.error(f"  Sparse retrieval failed: {sparse_error}")
                    sparse_docs = []

            # 2.3: Fusion
            logger.info("Fusing results")
            fused_docs = self.result_fusion.fuse([dense_docs, sparse_docs])
            logger.info(f"   Fused to {len(fused_docs)} candidates")

            # 2.4: Reranking
            logger.info("Reranking")
            reranked_docs = self.reranker.rerank(query, fused_docs)
            logger.info(f"   Reranked to top {len(reranked_docs)} documents")

            if not reranked_docs:
                logger.warning("No documents retrieved")
                return self._create_no_results_response(
                    query, classification, routing_decision, stage_times, pipeline_start
                )

            retrieval_stats = {
                "dense_count": len(dense_docs),
                "sparse_count": len(sparse_docs),
                "fused_count": len(fused_docs),
                "reranked_count": len(reranked_docs),
                "top_score": float(reranked_docs[0].score) if reranked_docs else 0.0,
                "parallel_execution": parallel_success
            }

            stage_times["retrieval"] = time.time() - stage_start
            logger.info(f"Retrieval: {stage_times['retrieval']:.2f}s\n")

            # ===================================================================
            # STAGE 3: Context Processing
            # ===================================================================
            stage_start = time.time()
            logger.info("STAGE 3: Context Processing")
            logger.info("-" * 80)

            # 3.1: Verify relevance
            logger.info("Verifying relevance...")
            # Auto-enable keyword filter for short queries
            use_keyword_filter = len(query.split()) <= 4
            verified_docs = self.context_verifier.verify_relevance(
                query,
                reranked_docs,
                require_keyword_match=use_keyword_filter
            )
            logger.info(f"   Verified {len(verified_docs)}/{len(reranked_docs)} documents")

            if not verified_docs:
                logger.warning("No documents passed relevance verification")
                return self._create_no_results_response(
                    query, classification, routing_decision, stage_times, pipeline_start
                )

            # 3.2: Deduplicate
            logger.info("Deduplicating")
            unique_docs = self.context_verifier.deduplicate_context(verified_docs)
            logger.info(f"   Removed {len(verified_docs) - len(unique_docs)} duplicates")

            # 3.3: Order context
            logger.info("Ordering context")
            ordered_docs = self.context_verifier.order_context(
                unique_docs,
                strategy=self.config.ordering_strategy
            )
            logger.info(f"   Applied {self.config.ordering_strategy} strategy")

            # 3.4: Prepare citations
            logger.info("Preparing citations...")
            final_docs, citation_map = self.context_verifier.prepare_citations(ordered_docs)
            logger.info(f"   Prepared {len(final_docs)} documents with citations")

            context_stats = {
                "reranked": len(reranked_docs),
                "verified": len(verified_docs),
                "unique": len(unique_docs),
                "final": len(final_docs),
                "nli_threshold": self.config.nli_threshold,
                "ordering_strategy": self.config.ordering_strategy
            }

            stage_times["context_processing"] = time.time() - stage_start
            logger.info(f"Context Processing: {stage_times['context_processing']:.2f}s\n")

            # ===================================================================
            # STAGE 4: Generation
            # ===================================================================
            stage_start = time.time()
            logger.info("STAGE 4: Generation")
            logger.info("-" * 80)

            # 4.1: Build prompt
            logger.info("Building prompt")
            prompt = self.prompt_builder.build_prompt(query, final_docs)
            logger.info(f"   Prompt length: {len(prompt)} characters")

            # 4.2: Generate answer
            # NOTE: Stop sequences can truncate valid content - be careful!
            # Removed "---", "\n---", "\n\n\n" as LLM uses these in formatted responses
            logger.info("Generating answer")
            answer = self.llm_generator.generate(
                prompt,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                purpose="answer_generation",
                stop_sequences=[
                    "Question:",
                    "Context:",
                    "\n\nQuestion:",
                    "\n\nContext:",
                    "\n\nNote:",
                    "Note:",
                    "> Note",
                    "\n\nAnswer:",
                    "Instructions:",
                    "The correct response",
                    "notation applies"
                ]
            )
            logger.info(f"   Generated {len(answer)} characters")

            # 4.3: Extract citations
            logger.info("Extracting citations")

            # Convert Haystack Documents to dictionaries for citation extractor
            source_docs_dicts = []
            for doc in final_docs:
                doc_dict = {
                    'content': doc.content,
                    'source_file': doc.meta.get('source_filename', doc.meta.get('source', 'Unknown')),
                    'page': doc.meta.get('page_number', doc.meta.get('page', 0)),
                    'chunk_id': doc.meta.get('chunk_id', 'unknown'),
                    'rerank_score': doc.score if hasattr(doc, 'score') else 0.0
                }
                source_docs_dicts.append(doc_dict)

            # Extract citations (returns tuple: citations, valid_ids, invalid_ids)
            sanitized_answer = sanitize_generated_answer(answer, current_query=query)
            citations_list, valid_ids, invalid_ids = self.citation_extractor.extract_citations(
                sanitized_answer,
                source_docs_dicts
            )

            # Remove invalid citations
            clean_answer = self.citation_extractor.remove_invalid_citations(sanitized_answer, invalid_ids)
            clean_answer = sanitize_generated_answer(clean_answer, current_query=query)

            # Check if the LLM's answer indicates no relevant information
            # Common patterns when LLM recognizes context isn't relevant
            refusal_patterns = [
                "provided context does not contain",
                "context does not include",
                "cannot find",
                "no information",
                "don't have information",
                "not mentioned in the context",
                "context doesn't mention",
                "unable to answer based on",
                "cannot answer based on"
            ]

            answer_lower = clean_answer.lower()
            is_refusal = any(pattern in answer_lower for pattern in refusal_patterns)

            # If LLM refused to answer, return no-results response without citations
            if is_refusal:
                logger.warning("LLM indicated no relevant information in context, returning no-results response")
                return self._create_no_results_response(
                    query, classification, routing_decision, stage_times, pipeline_start
                )

            # Validate coverage
            coverage = self.citation_extractor.validate_citation_coverage(clean_answer)

            # Calculate simple confidence
            citation_count = len(citations_list)
            coverage_score = coverage.get('coverage', 0.0)
            avg_rerank_score = sum(doc.score for doc in final_docs) / len(final_docs) if final_docs else 0.0

            # Simple confidence formula
            confidence = (
                    coverage_score * 0.4 +
                    avg_rerank_score * 0.4 +
                    min(1.0, citation_count / 3) * 0.2  # Expect ~3 citations
            )

            # Convert Citation objects to dicts
            citations_dicts = [cit.to_dict() for cit in citations_list]

            # FALLBACK: If no citations found in response, use citation_map from prepared docs
            # This ensures we always return the source documents even if LLM didn't cite them
            # Only apply fallback if it's NOT a refusal (already checked above)
            if not citations_dicts and citation_map and "citations" in citation_map:
                logger.warning("No citations found in response, using prepared citation map as fallback")
                citations_dicts = []
                for cit_entry in citation_map["citations"]:
                    fallback_citation = {
                        "id": cit_entry.get('citation_number', 0),
                        "source": cit_entry.get('source_file', 'Unknown'),
                        "page": cit_entry.get('page', 0),
                        "chunk_id": cit_entry.get('chunk_id', 'unknown'),
                        "text": cit_entry.get('content_preview', ''),
                        "relevance": round(cit_entry.get('rerank_score', 0.0), 4)
                    }
                    citations_dicts.append(fallback_citation)
                logger.info(f"   Using {len(citations_dicts)} source documents as citations")

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

            generation_stats = {
                "prompt_length": len(prompt),
                "answer_length": len(clean_answer),
                "citations_found": len(citations_dicts),
                "citations_from_llm": len(citations_list) > 0,  # Track if citations came from LLM
                "confidence": confidence
            }

            stage_times["generation"] = time.time() - stage_start
            logger.info(f"Generation: {stage_times['generation']:.2f}s\n")

            # ===================================================================
            # Create Final Result
            # ===================================================================
            total_time = time.time() - pipeline_start

            result = PipelineResult(
                query=query,
                answer=clean_answer,
                citations=citations_dicts,
                metadata={
                    "confidence": confidence,
                    "generation_time": stage_times["generation"],
                    "model": self.config.llm_model,
                    "temperature": self.config.temperature,
                    "query_enhancement": enhancement_info  # Include enhancement info if used
                },
                query_classification=classification.to_dict(),
                routing_decision=routing_decision.to_dict(),
                retrieval_stats=retrieval_stats,
                context_stats=context_stats,
                generation_stats=generation_stats,
                total_time=total_time,
                stage_times=stage_times
            )

            logger.info("=" * 80)
            logger.info("Pipeline Complete")
            logger.info("=" * 80)
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(f"  - Query Processing: {stage_times['query_processing']:.2f}s")
            logger.info(f"  - Retrieval: {stage_times['retrieval']:.2f}s")
            logger.info(f"  - Context Processing: {stage_times['context_processing']:.2f}s")
            logger.info(f"  - Generation: {stage_times['generation']:.2f}s")
            logger.info(f"Citations: {len(result.citations)}")
            logger.info(f"Confidence: {result.metadata['confidence']:.3f}")
            logger.info("=" * 80)
            logger.info("")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    @staticmethod
    def _create_rejection_result(
            query: str,
            classification: Any,
            routing_decision: Any,
            stage_times: Dict[str, float],
            pipeline_start: float
    ) -> PipelineResult:
        """Create result for rejected queries."""
        total_time = time.time() - pipeline_start

        return PipelineResult(
            query=query,
            answer="I cannot answer this query because it was classified as out-of-scope or too generic.",
            citations=[],
            metadata={
                "confidence": 0.0,
                "rejection_reason": routing_decision.reason
            },
            query_classification=classification.to_dict(),
            routing_decision=routing_decision.to_dict(),
            retrieval_stats={},
            context_stats={},
            generation_stats={},
            total_time=total_time,
            stage_times=stage_times
        )

    @staticmethod
    def _create_no_results_response(
            query: str,
            classification: Any,
            routing_decision: Any,
            stage_times: Dict[str, float],
            pipeline_start: float
    ) -> PipelineResult:
        """Create result when no relevant documents found."""
        total_time = time.time() - pipeline_start

        return PipelineResult(
            query=query,
            answer="I couldn't find relevant information in the knowledge base to answer this question.",
            citations=[],
            metadata={
                "confidence": 0.0,
                "failure_reason": "no_relevant_documents"
            },
            query_classification=classification.to_dict(),
            routing_decision=routing_decision.to_dict(),
            retrieval_stats=stage_times.get("retrieval_stats", {}),
            context_stats={"verified": 0, "final": 0},
            generation_stats={},
            total_time=total_time,
            stage_times=stage_times
        )

    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration."""
        return {
            "collection_name": self.collection_name,
            "config": asdict(self.config)
        }

    def switch_collection(self, new_collection_name: str) -> bool:
        """Switch to a different Qdrant collection at runtime.

        This method re-initializes the dense and sparse retrievers to point to
        the new collection while keeping all other components (query
        processing, context handling, generation) unchanged.

        Returns True on success, False if initialization fails (pipeline
        continues using the previous collection in that case).
        """
        logger.info("=" * 80)
        logger.info(f"Requested collection switch: {self.collection_name} -> {new_collection_name}")
        logger.info("=" * 80)

        # Keep references to current retrievers so we can roll back on failure
        old_dense = getattr(self, "dense_retriever", None)
        old_sparse = getattr(self, "sparse_retriever", None)

        try:
            # Initialize new retrievers for the target collection WITHOUT
            # touching the current ones.
            new_dense = DenseRetriever(
                collection_name=new_collection_name,
                embedding_model="BAAI/bge-large-en-v1.5",  # Match indexing model
                qdrant_url=self.config.qdrant_url,
                top_k=self.config.dense_top_k,
                use_gpu=self.config.use_gpu,
                default_filters=self.config.dense_default_filters,
            )

            new_sparse = SparseRetriever(
                collection_name=new_collection_name,
                qdrant_url=self.config.qdrant_url,
                top_k=self.config.sparse_top_k,
            )

            # If we got here, both new retrievers initialized successfully.
            # Now we can safely switch over.
            self.collection_name = new_collection_name
            self.config.collection_name = new_collection_name
            self.dense_retriever = new_dense
            self.sparse_retriever = new_sparse

            logger.info("=" * 80)
            logger.info(f"Switched collection successfully to: {new_collection_name}")
            logger.info("=" * 80)
            logger.info("")

            # Let old retrievers be garbage-collected
            del old_dense
            del old_sparse

            return True

        except Exception as e:
            # Log error and keep existing retrievers/collection untouched
            logger.error(f"Failed to switch collection to '{new_collection_name}': {e}", exc_info=True)
            logger.info("Continuing to use previous collection: %s", self.collection_name)
            return False
