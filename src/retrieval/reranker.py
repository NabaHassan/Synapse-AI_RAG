"""
Reranker for scoring and reranking retrieved documents.

This module implements reranking using:
- Cross-encoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) via Haystack
- Scores query-document pairs for relevance
- Returns top-k documents sorted by relevance scores
- Optimized for fast inference (~50ms for 20 documents)
"""

import time
import logging
from typing import List, Optional, Dict, Any, Tuple

from haystack import Document
from haystack.utils import ComponentDevice
from haystack.components.rankers import TransformersSimilarityRanker

logger = logging.getLogger(__name__)


class Reranker:
    """
    Reranker using cross-encoder model for accurate relevance scoring.
    
    Uses a cross-encoder model that directly scores [query, document] pairs,
    providing more accurate relevance scores than simple similarity metrics.
    Typically used after hybrid retrieval fusion to select the best documents.
    """

    def __init__(
            self,
            model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k: int = 5,
            use_gpu: bool = False,
            batch_size: int = 16,
            score_threshold: Optional[float] = None
    ):
        """
        Initialize the Reranker.
        
        Args:
            model_name: Cross-encoder model name (default: ms-marco-MiniLM-L-6-v2)
            top_k: Number of documents to return after reranking (default: 5)
            use_gpu: Whether to use GPU for inference
            batch_size: Batch size for processing documents
            score_threshold: Minimum relevance score (None = no filtering)
        """
        self.model_name = model_name
        self.top_k = top_k
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.score_threshold = score_threshold

        logger.info(f" Initializing Reranker with model: {model_name}")

        # Initialize the cross-encoder ranker
        self._init_ranker()

        logger.info(
            f" Reranker initialized successfully "
            f"(top_k={top_k}, device={'GPU' if use_gpu else 'CPU'})"
        )

    def _init_ranker(self):
        """Initialize the Haystack TransformersSimilarityRanker."""
        try:
            device = ComponentDevice.from_str("cuda" if self.use_gpu else "cpu")

            self.ranker = TransformersSimilarityRanker(
                model=self.model_name,
                device=device,
                top_k=self.top_k,
                batch_size=self.batch_size
            )

            # Warm up the model
            try:
                self.ranker.warm_up()
            except AttributeError as ae:
                if "hf_device_map" in str(ae):
                    logger.warning("TransformersSimilarityRanker.warm_up failed due to missing hf_device_map. Setting device manually...")
                    self.ranker.device = device
                else:
                    raise

            logger.info(f" Cross-encoder ranker initialized on {device}")

        except Exception as e:
            logger.error(f" Failed to initialize ranker: {e}")
            raise

    def rerank(
            self,
            query: str,
            documents: List[Document],
            top_k: Optional[int] = None,
            score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Rerank documents by relevance to the query.
        
        Args:
            query: Search query
            documents: List of candidate documents to rerank
            top_k: Number of documents to return (overrides default)
            score_threshold: Minimum relevance score (overrides default)
            
        Returns:
            Reranked list of top-k documents with relevance scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        if not documents:
            logger.warning("No documents to rerank")
            return []

        k = top_k if top_k is not None else self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        try:
            logger.info(
                f" Reranking {len(documents)} documents for query: '{query[:50]}...'"
            )

            # Start timing
            start_time = time.time()

            # Run reranking
            result = self.ranker.run(
                query=query,
                documents=documents,
                top_k=k
            )

            reranked_docs = result.get("documents", [])

            # Calculate inference time
            inference_time = (time.time() - start_time) * 1000  # Convert to ms

            # Apply score threshold if specified
            if threshold is not None:
                before_filter = len(reranked_docs)
                reranked_docs = [
                    doc for doc in reranked_docs
                    if doc.score is not None and doc.score >= threshold
                ]
                if len(reranked_docs) < before_filter:
                    logger.info(
                        f" Filtered by score threshold ({threshold}): "
                        f"{before_filter} → {len(reranked_docs)} documents"
                    )

            # Log results
            if reranked_docs:
                scores = [doc.score for doc in reranked_docs if doc.score is not None]
                if scores:
                    logger.info(
                        f" Reranked to {len(reranked_docs)} documents | "
                        f"Time: {inference_time:.1f}ms | "
                        f"Score range: [{min(scores):.4f}, {max(scores):.4f}] | "
                        f"Mean: {sum(scores) / len(scores):.4f}"
                    )
                else:
                    logger.info(
                        f" Reranked to {len(reranked_docs)} documents | "
                        f"Time: {inference_time:.1f}ms"
                    )

                # Performance warning
                if inference_time > 100:
                    logger.warning(
                        f" Reranking took {inference_time:.1f}ms "
                        f"(expected ~50ms for 20 docs). Consider reducing batch size or using GPU."
                    )
            else:
                logger.warning("No documents passed reranking threshold")

            return reranked_docs

        except Exception as e:
            logger.error(f" Reranking failed: {e}")
            raise

    def rerank_batch(
            self,
            queries: List[str],
            documents_list: List[List[Document]],
            top_k: Optional[int] = None
    ) -> List[List[Document]]:
        """
        Rerank multiple query-document sets in batch.
        
        Args:
            queries: List of search queries
            documents_list: List of document lists (one per query)
            top_k: Number of documents to return per query
            
        Returns:
            List of reranked document lists
        """
        if len(queries) != len(documents_list):
            raise ValueError(
                f"Mismatch: {len(queries)} queries vs {len(documents_list)} document lists"
            )

        logger.info(f" Batch reranking {len(queries)} queries")

        results = []
        for query, docs in zip(queries, documents_list):
            reranked = self.rerank(query, docs, top_k=top_k)
            results.append(reranked)

        logger.info(f" Batch reranking complete")
        return results

    def get_scores(
            self,
            query: str,
            documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Get relevance scores for documents without reranking.
        
        Returns documents with their scores in original order.
        
        Args:
            query: Search query
            documents: List of documents to score
            
        Returns:
            List of (document, score) tuples in original order
        """
        if not documents:
            return []

        try:
            # Run ranker but return all documents with scores
            result = self.ranker.run(
                query=query,
                documents=documents,
                top_k=len(documents)  # Get scores for all
            )

            scored_docs = result.get("documents", [])

            # Create mapping of doc_id to score
            score_map = {doc.id: doc.score for doc in scored_docs if doc.score is not None}

            # Return original documents with scores
            return [(doc, score_map.get(doc.id, 0.0)) for doc in documents]

        except Exception as e:
            logger.error(f" Failed to get scores: {e}")
            return [(doc, 0.0) for doc in documents]

    def update_top_k(self, top_k: int):
        """
        Update the default top_k parameter.
        
        Args:
            top_k: New top_k value
        """
        old_k = self.top_k
        self.top_k = top_k
        logger.info(f"🔄 Updated top_k: {old_k} → {top_k}")

    def update_score_threshold(self, threshold: Optional[float]):
        """
        Update the score threshold.
        
        Args:
            threshold: New score threshold (None to disable filtering)
        """
        old_threshold = self.score_threshold
        self.score_threshold = threshold
        logger.info(f"🔄 Updated score threshold: {old_threshold} → {threshold}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get reranker statistics.
        
        Returns:
            Dictionary with configuration and stats
        """
        return {
            "model_name": self.model_name,
            "top_k": self.top_k,
            "device": "GPU" if self.use_gpu else "CPU",
            "batch_size": self.batch_size,
            "score_threshold": self.score_threshold
        }


def create_reranker(
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_k: int = 5,
        **kwargs
) -> Reranker:
    """
    Factory function to create a Reranker instance.
    
    Args:
        model_name: Cross-encoder model name
        top_k: Number of documents to return
        **kwargs: Additional arguments for Reranker
        
    Returns:
        Configured Reranker instance
    """
    return Reranker(model_name=model_name, top_k=top_k, **kwargs)
