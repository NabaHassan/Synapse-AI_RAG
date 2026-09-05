"""
Result Fusion for combining dense and sparse retrieval results.

This module implements fusion strategies:
- Reciprocal Rank Fusion (RRF) - rank-based combination
- Weighted Fusion - score-based combination
- Haystack DocumentJoiner - framework-integrated fusion
- Deduplication and score normalization
"""

import logging
from haystack import Document
from collections import defaultdict
from typing import List, Dict, Any, Optional
from haystack.components.joiners import DocumentJoiner

logger = logging.getLogger(__name__)


class ResultFusion:
    """
    Fusion engine for combining multiple retrieval results.
    
    Supports multiple fusion strategies:
    - RRF (Reciprocal Rank Fusion): Rank-based, parameter k controls decay
    - Weighted: Score-based, parameter alpha controls dense/sparse balance
    - Haystack: Uses Haystack's DocumentJoiner (supports multiple modes)
    """

    def __init__(
            self,
            strategy: str = "rrf",
            rrf_k: int = 60,
            weighted_alpha: float = 0.6,
            top_k: int = 20,
            normalize_scores: bool = True
    ):
        """
        Initialize Result Fusion.
        
        Args:
            strategy: Fusion strategy ("rrf", "weighted", "haystack")
            rrf_k: RRF constant (default 60, higher = less decay)
            weighted_alpha: Weight for first retriever in weighted fusion (0-1)
            top_k: Number of documents to return after fusion
            normalize_scores: Whether to normalize scores to 0-1 range
        """
        self.strategy = strategy.lower()
        self.rrf_k = rrf_k
        self.weighted_alpha = weighted_alpha
        self.top_k = top_k
        self.normalize_scores = normalize_scores

        # Validate strategy
        valid_strategies = ["rrf", "weighted", "haystack"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {self.strategy}. "
                f"Choose from: {valid_strategies}"
            )

        logger.info(f" Initializing ResultFusion with strategy: {self.strategy}")

        # Initialize Haystack joiner if needed
        if self.strategy == "haystack":
            self._init_haystack_joiner()

        logger.info(f" ResultFusion initialized (strategy={self.strategy}, top_k={self.top_k})")

    def _init_haystack_joiner(self):
        """Initialize Haystack DocumentJoiner."""
        try:
            self.joiner = DocumentJoiner(
                join_mode="reciprocal_rank_fusion",
                top_k=self.top_k
            )
            logger.info(" Haystack DocumentJoiner initialized with RRF")
        except Exception as e:
            logger.error(f" Failed to initialize DocumentJoiner: {e}")
            raise

    def fuse(
            self,
            retrieval_results: List[List[Document]],
            top_k: Optional[int] = None,
            query: Optional[str] = None
    ) -> List[Document]:
        """
        Fuse multiple retrieval results.
        
        Args:
            retrieval_results: List of document lists from different retrievers
                              e.g., [dense_docs, sparse_docs]
            top_k: Number of documents to return (overrides default)
            query: Original query (used for logging)
            
        Returns:
            Fused and ranked list of documents
        """
        if not retrieval_results or not any(retrieval_results):
            logger.warning("No retrieval results to fuse")
            return []

        k = top_k if top_k is not None else self.top_k

        # Log fusion details
        result_counts = [len(docs) for docs in retrieval_results]
        logger.info(
            f" Fusing {len(retrieval_results)} result sets "
            f"({', '.join(map(str, result_counts))} docs) "
            f"with {self.strategy.upper()} strategy"
        )

        # Apply fusion strategy
        if self.strategy == "rrf":
            fused_docs = self._fuse_rrf(retrieval_results, k)
        elif self.strategy == "weighted":
            fused_docs = self._fuse_weighted(retrieval_results, k)
        elif self.strategy == "haystack":
            fused_docs = self._fuse_haystack(retrieval_results, k)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Log fusion results
        if fused_docs:
            scores = [doc.score for doc in fused_docs if doc.score is not None]
            if scores:
                logger.info(
                    f" Fused to {len(fused_docs)} documents | "
                    f"Score range: [{min(scores):.3f}, {max(scores):.3f}] | "
                    f"Mean: {sum(scores) / len(scores):.3f}"
                )
            else:
                logger.info(f" Fused to {len(fused_docs)} documents")

        return fused_docs

    def _fuse_rrf(
            self,
            retrieval_results: List[List[Document]],
            top_k: int
    ) -> List[Document]:
        """
        Fuse using Reciprocal Rank Fusion (RRF).
        
        RRF Score = Σ (1 / (k + rank_i))
        where rank_i is the rank in retriever i
        
        Advantages:
        - Rank-based, doesn't depend on score scales
        - Robust to different scoring methods
        - Simple and effective
        """
        logger.debug(f"Applying RRF fusion (k={self.rrf_k})")

        # Build document rank map
        doc_ranks = defaultdict(list)  # doc_id -> [rank_in_retriever_1, rank_in_retriever_2, ...]
        doc_map = {}  # doc_id -> Document

        for retriever_idx, docs in enumerate(retrieval_results):
            for rank, doc in enumerate(docs):
                doc_id = doc.id
                doc_ranks[doc_id].append(rank + 1)  # Rank starts at 1
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Calculate RRF scores
        rrf_scores = {}
        for doc_id, ranks in doc_ranks.items():
            # RRF formula: sum of 1/(k + rank)
            rrf_score = sum(1.0 / (self.rrf_k + rank) for rank in ranks)
            rrf_scores[doc_id] = rrf_score

        # Normalize scores if requested
        if self.normalize_scores and rrf_scores:
            max_score = max(rrf_scores.values())
            min_score = min(rrf_scores.values())
            score_range = max_score - min_score

            if score_range > 0:
                rrf_scores = {
                    doc_id: (score - min_score) / score_range
                    for doc_id, score in rrf_scores.items()
                }

        # Create fused documents with RRF scores
        fused_docs = []
        for doc_id in sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]:
            doc = doc_map[doc_id]
            # Create new document with RRF score
            fused_doc = Document(
                content=doc.content,
                id=doc.id,
                meta=doc.meta,
                score=rrf_scores[doc_id]
            )
            fused_docs.append(fused_doc)

        logger.debug(f"RRF fusion: {len(doc_map)} unique docs → {len(fused_docs)} final")
        return fused_docs

    def _fuse_weighted(
            self,
            retrieval_results: List[List[Document]],
            top_k: int
    ) -> List[Document]:
        """
        Fuse using weighted score combination.
        
        Weighted Score = α * score_1 + (1-α) * score_2 + ...
        where α is the weight for each retriever
        
        Advantages:
        - Preserves score information
        - Allows tuning retriever importance
        - Good when scores are comparable
        """
        logger.debug(f"Applying weighted fusion (alpha={self.weighted_alpha})")

        # Calculate weights for each retriever
        n_retrievers = len(retrieval_results)
        if n_retrievers == 1:
            weights = [1.0]
        elif n_retrievers == 2:
            weights = [self.weighted_alpha, 1.0 - self.weighted_alpha]
        else:
            # Distribute weights with first retriever getting alpha
            remaining = 1.0 - self.weighted_alpha
            weights = [self.weighted_alpha] + [remaining / (n_retrievers - 1)] * (n_retrievers - 1)

        # Build document score map
        doc_scores = defaultdict(float)  # doc_id -> weighted_score
        doc_map = {}  # doc_id -> Document
        doc_count = defaultdict(int)  # doc_id -> count of appearances

        for retriever_idx, docs in enumerate(retrieval_results):
            weight = weights[retriever_idx]

            for doc in docs:
                doc_id = doc.id
                # Add weighted score
                if doc.score is not None:
                    doc_scores[doc_id] += weight * doc.score
                doc_count[doc_id] += 1

                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Normalize by appearance count (documents appearing in fewer retrievers get penalty)
        # Alternative: don't normalize (rewards documents in multiple retrievers)
        # doc_scores = {doc_id: score / doc_count[doc_id] for doc_id, score in doc_scores.items()}

        # Normalize scores if requested
        if self.normalize_scores and doc_scores:
            max_score = max(doc_scores.values())
            min_score = min(doc_scores.values())
            score_range = max_score - min_score

            if score_range > 0:
                doc_scores = {
                    doc_id: (score - min_score) / score_range
                    for doc_id, score in doc_scores.items()
                }

        # Create fused documents with weighted scores
        fused_docs = []
        for doc_id in sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)[:top_k]:
            doc = doc_map[doc_id]
            fused_doc = Document(
                content=doc.content,
                id=doc.id,
                meta=doc.meta,
                score=doc_scores[doc_id]
            )
            fused_docs.append(fused_doc)

        logger.debug(f"Weighted fusion: {len(doc_map)} unique docs → {len(fused_docs)} final")
        return fused_docs

    def _fuse_haystack(
            self,
            retrieval_results: List[List[Document]],
            top_k: int
    ) -> List[Document]:
        """
        Fuse using Haystack's DocumentJoiner.
        
        Uses Haystack's built-in fusion with RRF.
        """
        logger.debug("Applying Haystack DocumentJoiner fusion")

        try:
            # DocumentJoiner expects documents as separate inputs
            result = self.joiner.run(documents=retrieval_results, top_k=top_k)
            fused_docs = result.get("documents", [])

            logger.debug(
                f"Haystack fusion: {sum(len(docs) for docs in retrieval_results)} total → {len(fused_docs)} final")
            return fused_docs

        except Exception as e:
            logger.error(f" Haystack fusion failed: {e}")
            # Fallback to RRF
            logger.warning("Falling back to manual RRF fusion")
            return self._fuse_rrf(retrieval_results, top_k)

    @staticmethod
    def deduplicate(
            documents: List[Document],
            key: str = "id"
    ) -> List[Document]:
        """
        Deduplicate documents by key.
        
        Args:
            documents: List of documents
            key: Deduplication key ("id" or "content")
            
        Returns:
            Deduplicated documents (keeps first occurrence)
        """
        seen = set()
        deduped = []

        for doc in documents:
            if key == "id":
                identifier = doc.id
            elif key == "content":
                identifier = doc.content
            else:
                raise ValueError(f"Invalid deduplication key: {key}")

            if identifier not in seen:
                seen.add(identifier)
                deduped.append(doc)

        if len(deduped) < len(documents):
            logger.info(f" Deduplicated: {len(documents)} → {len(deduped)} documents")

        return deduped

    @staticmethod
    def get_fusion_stats(
            retrieval_results: List[List[Document]]
    ) -> Dict[str, Any]:
        """
        Get statistics about fusion results.
        
        Args:
            retrieval_results: List of document lists
            
        Returns:
            Dictionary with fusion statistics
        """
        stats = {
            "n_retrievers": len(retrieval_results),
            "total_docs": sum(len(docs) for docs in retrieval_results),
            "docs_per_retriever": [len(docs) for docs in retrieval_results],
        }

        # Count unique documents
        all_ids = set()
        for docs in retrieval_results:
            all_ids.update(doc.id for doc in docs)
        stats["unique_docs"] = len(all_ids)

        # Count overlap
        if len(retrieval_results) == 2:
            ids_1 = {doc.id for doc in retrieval_results[0]}
            ids_2 = {doc.id for doc in retrieval_results[1]}
            overlap = ids_1 & ids_2
            stats["overlap"] = len(overlap)
            stats["overlap_ratio"] = len(overlap) / len(all_ids) if all_ids else 0

        return stats

    def update_strategy(self, strategy: str, **kwargs):
        """
        Update fusion strategy and parameters.
        
        Args:
            strategy: New strategy ("rrf", "weighted", "haystack")
            **kwargs: Strategy-specific parameters (rrf_k, weighted_alpha, etc.)
        """
        old_strategy = self.strategy
        self.strategy = strategy.lower()

        # Update parameters
        if "rrf_k" in kwargs:
            self.rrf_k = kwargs["rrf_k"]
        if "weighted_alpha" in kwargs:
            self.weighted_alpha = kwargs["weighted_alpha"]
        if "top_k" in kwargs:
            self.top_k = kwargs["top_k"]

        # Reinitialize Haystack joiner if switching to it
        if self.strategy == "haystack" and old_strategy != "haystack":
            self._init_haystack_joiner()

        logger.info(f" Updated strategy: {old_strategy} → {self.strategy}")


def create_result_fusion(
        strategy: str = "rrf",
        top_k: int = 20,
        **kwargs
) -> ResultFusion:
    """
    Factory function to create a ResultFusion instance.
    
    Args:
        strategy: Fusion strategy ("rrf", "weighted", "haystack")
        top_k: Number of documents to return
        **kwargs: Additional arguments for ResultFusion
        
    Returns:
        Configured ResultFusion instance
    """
    return ResultFusion(strategy=strategy, top_k=top_k, **kwargs)
