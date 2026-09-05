"""
Context Verifier for relevance verification using NLI models.

This module implements the following:
- Context verification using NLI (Natural Language Inference) model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
- Entailment scoring to verify query-document relevance
- Threshold-based filtering to remove irrelevant documents
- Batch processing for efficient verification
- Semantic deduplication using Sentence-BERT embeddings and cosine similarity using sentence-transformers/all-mpnet-base-v2
- Context ordering strategies: relevance, chronological, lost-in-middle
- Citation preparation with [N] notation for LLM prompts
"""

import os
import torch
import logging
from typing import List, Optional, Dict, Any, Tuple, Literal

from haystack import Document
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline
)
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class ContextVerifier:
    """
    Verifies relevance of retrieved documents using NLI (Natural Language Inference).
    
    Uses a DeBERTa-v3 model trained on NLI tasks to check if documents are relevant
    to the query by treating it as an entailment problem:
    - Premise: Document content
    - Hypothesis: Query
    - Score: Entailment probability
    
    High entailment score indicates the document supports/answers the query.
    """

    def __init__(
            self,
            model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
            threshold: float = 0.5,
            batch_size: int = 8,
            use_gpu: bool = False,
            max_length: int = 512,
            # embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
            embedding_model: str = "BAAI/bge-large-en-v1.5",
            dedup_threshold: float = 0.85
    ):
        """
        Initialize the Context Verifier.
        
        Args:
            model_name: NLI model name (default: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)
            threshold: Minimum entailment score for relevance (0-1, default: 0.5)
            batch_size: Batch size for processing documents
            use_gpu: Whether to use GPU for inference
            max_length: Maximum sequence length for tokenization
            embedding_model: Model for semantic similarity (default: all-mpnet-base-v2)
            dedup_threshold: Similarity threshold for deduplication (default: 0.85)
        """
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_length = max_length
        self.embedding_model_name = embedding_model
        self.dedup_threshold = dedup_threshold

        logger.info(f"Initializing ContextVerifier with model: {model_name}")

        # Initialize NLI model and tokenizer
        self._init_nli_model()

        # Initialize embedding model for deduplication
        self._init_embedding_model()

        logger.info(
            f"ContextVerifier initialized successfully "
            f"(threshold={threshold}, dedup_threshold={dedup_threshold}, device={'GPU' if self.use_gpu else 'CPU'})"
        )

    def _init_nli_model(self):
        """Initialize the NLI model and tokenizer."""
        try:
            device = 0 if self.use_gpu else -1  # 0 for GPU, -1 for CPU

            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )

            # Create pipeline for easier inference
            self.nli_pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device,
                top_k=None,  # Return all class probabilities
                max_length=self.max_length,
                truncation=True
            )

            logger.info(f"NLI model initialized on {'GPU' if self.use_gpu else 'CPU'}")

        except Exception as e:
            logger.error(f"Failed to initialize NLI model: {e}")
            raise

    def _init_embedding_model(self):
        """Initialize the embedding model for semantic similarity."""
        try:
            device = 'cuda' if self.use_gpu else 'cpu'
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=device
            )
            logger.info(
                f"Embedding model initialized on {device.upper()} "
                f"for deduplication"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    @staticmethod
    def _has_keyword_match(query: str, content: str, fuzzy: bool = True) -> bool:
        """
        Check if query keywords appear in content (case-insensitive).
        
        Args:
            query: Search query
            content: Document content
            fuzzy: Allow fuzzy matching (substring matches)
            
        Returns:
            True if keywords found, False otherwise
        """
        query_lower = query.lower()
        content_lower = content.lower()

        # Direct substring match
        if query_lower in content_lower:
            return True

        if fuzzy:
            # Check if any of the query words appear in content
            query_words = query_lower.split()
            for word in query_words:
                if len(word) >= 3 and word in content_lower:  # Skip very short words
                    return True

        return False

    def verify_relevance(
            self,
            query: str,
            documents: List[Document],
            threshold: Optional[float] = None,
            require_keyword_match: bool = False
    ) -> List[Document]:
        """
        Verify relevance of documents using NLI entailment scoring.
        
        Args:
            query: Search query (treated as hypothesis in NLI)
            documents: List of candidate documents to verify
            threshold: Override default threshold for this verification
            
        Returns:
            List of relevant documents with NLI scores in metadata
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        if not documents:
            logger.warning("No documents to verify")
            return []

        threshold_val = threshold if threshold is not None else self.threshold

        logger.info(
            f"Verifying relevance of {len(documents)} documents "
            f"(threshold={threshold_val}, keyword_filter={require_keyword_match})"
        )

        try:
            # Pre-filter: Check keyword matches if required
            if require_keyword_match:
                keyword_filtered = []
                filtered_out_count = 0

                for doc in documents:
                    has_match = self._has_keyword_match(query, doc.content, fuzzy=True)
                    if has_match:
                        keyword_filtered.append(doc)
                    else:
                        filtered_out_count += 1
                        # Log first few filtered docs for debugging
                        if filtered_out_count <= 2:
                            logger.debug(
                                f"Filtered out doc (no keyword match): "
                                f"{doc.meta.get('source', 'Unknown')} - "
                            )

                if len(keyword_filtered) < len(documents):
                    logger.info(
                        f"Keyword pre-filter: {len(keyword_filtered)}/{len(documents)} "
                        f"documents contain query keywords"
                    )
                    documents = keyword_filtered

                if not documents:
                    logger.warning(
                        f"No documents contain query keywords: '{query}'"
                    )
                    logger.warning(
                        f"   Searched in {filtered_out_count} documents but none matched"
                    )
                    return []

            # Prepare inputs for NLI
            # Format: premise (document) + hypothesis (query)
            nli_inputs = []
            for doc in documents:
                # Truncate document content if too long
                content = doc.content[:self.max_length * 4]  # Rough character limit
                nli_inputs.append(f"{content} [SEP] {query}")

            # Run NLI inference in batches
            nli_scores = self._compute_entailment_scores(nli_inputs)

            # Filter documents by threshold and add NLI scores
            verified_docs = []
            for doc, score in zip(documents, nli_scores):
                if score >= threshold_val:
                    # Create new document with NLI score in metadata
                    verified_doc = Document(
                        content=doc.content,
                        id=doc.id,
                        meta={
                            **doc.meta,
                            "nli_score": score,
                            "relevance_verified": True
                        },
                        score=doc.score  # Preserve original retrieval score
                    )
                    verified_docs.append(verified_doc)

            # Log results
            if verified_docs:
                nli_scores_list = [doc.meta['nli_score'] for doc in verified_docs]
                logger.info(
                    f"Verified {len(verified_docs)}/{len(documents)} documents | "
                    f"NLI score range: [{min(nli_scores_list):.4f}, {max(nli_scores_list):.4f}] | "
                    f"Mean: {sum(nli_scores_list) / len(nli_scores_list):.4f}"
                )
            else:
                logger.warning(
                    f"No documents passed relevance threshold ({threshold_val})"
                )

            return verified_docs

        except Exception as e:
            logger.error(f"Relevance verification failed: {e}")
            raise

    def _compute_entailment_scores(self, nli_inputs: List[str]) -> List[float]:
        """
        Compute entailment scores for NLI inputs.
        
        Args:
            nli_inputs: List of formatted NLI inputs (premise [SEP] hypothesis)
            
        Returns:
            List of entailment scores (0-1)
        """
        entailment_scores = []

        # Process in batches
        for i in range(0, len(nli_inputs), self.batch_size):
            batch = nli_inputs[i:i + self.batch_size]

            # Run pipeline
            results = self.nli_pipeline(batch)

            # Extract entailment scores
            for result in results:
                # Result is a list of dicts: [{'label': 'ENTAILMENT', 'score': 0.95}, ...]
                # Find entailment label and get its score
                entailment_score = 0.0
                for label_dict in result:
                    # Different models may use different labels
                    label = label_dict['label'].upper()
                    if 'ENTAIL' in label or label == 'LABEL_2':  # LABEL_2 is often entailment
                        entailment_score = label_dict['score']
                        break

                entailment_scores.append(entailment_score)

        return entailment_scores

    def verify_batch(
            self,
            queries: List[str],
            documents_list: List[List[Document]],
            threshold: Optional[float] = None
    ) -> List[List[Document]]:
        """
        Verify relevance for multiple query-document sets.
        
        Args:
            queries: List of search queries
            documents_list: List of document lists (one per query)
            threshold: Override default threshold
            
        Returns:
            List of verified document lists
        """
        if len(queries) != len(documents_list):
            raise ValueError(
                f"Mismatch: {len(queries)} queries vs {len(documents_list)} document lists"
            )

        logger.info(f"Batch verifying {len(queries)} queries")

        results = []
        for query, docs in zip(queries, documents_list):
            verified = self.verify_relevance(query, docs, threshold=threshold)
            results.append(verified)

        logger.info(f"Batch verification complete")
        return results

    @staticmethod
    def get_verification_stats(
            documents: List[Document]
    ) -> Dict[str, Any]:
        """
        Get statistics about verified documents.
        
        Args:
            documents: List of verified documents (with nli_score in metadata)
            
        Returns:
            Dictionary with verification statistics
        """
        if not documents:
            return {
                "total_documents": 0,
                "verified_documents": 0,
                "nli_scores": []
            }

        nli_scores = [
            doc.meta.get('nli_score', 0.0)
            for doc in documents
            if doc.meta.get('relevance_verified')
        ]

        stats = {
            "total_documents": len(documents),
            "verified_documents": len([d for d in documents if d.meta.get('relevance_verified')]),
            "nli_scores": {
                "min": min(nli_scores) if nli_scores else 0.0,
                "max": max(nli_scores) if nli_scores else 0.0,
                "mean": sum(nli_scores) / len(nli_scores) if nli_scores else 0.0,
                "count": len(nli_scores)
            }
        }

        return stats

    def update_threshold(self, threshold: float):
        """
        Update the relevance threshold.
        
        Args:
            threshold: New threshold value (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        old_threshold = self.threshold
        self.threshold = threshold
        logger.info(f"Updated threshold: {old_threshold} → {threshold}")

    def deduplicate_context(
            self,
            documents: List[Document],
            threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Remove semantically duplicate documents using cosine similarity.
        
        Algorithm:
        1. Keep the first (highest relevance) document
        2. For each subsequent document, compute cosine similarity with all kept documents
        3. If max similarity < threshold, keep the document
        4. Otherwise, discard as duplicate
        
        This ensures we preserve the most relevant unique chunks.
        
        Args:
            documents: List of documents to deduplicate (should be sorted by relevance)
            threshold: Similarity threshold for deduplication (default: self.dedup_threshold)
            
        Returns:
            List of deduplicated documents
        """
        if not documents:
            logger.warning("No documents to deduplicate")
            return []

        if len(documents) == 1:
            logger.info("Only 1 document, no deduplication needed")
            return documents

        threshold_val = threshold if threshold is not None else self.dedup_threshold

        logger.info(
            f"Deduplicating {len(documents)} documents "
            f"(similarity threshold={threshold_val})"
        )

        try:
            # Generate embeddings for all documents
            contents = [doc.content for doc in documents]
            embeddings = self.embedding_model.encode(
                contents,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # Keep track of selected documents
            final_docs = [documents[0]]  # Always keep the best (first) document
            final_embeddings = [embeddings[0]]

            removed_count = 0

            # Iterate through remaining documents
            for i, (doc, embedding) in enumerate(zip(documents[1:], embeddings[1:]), start=1):
                # Compute cosine similarity with all selected documents
                similarities = [
                    util.cos_sim(embedding, selected_emb).item()
                    for selected_emb in final_embeddings
                ]

                max_similarity = max(similarities)

                # Keep document if it's sufficiently different
                if max_similarity < threshold_val:
                    final_docs.append(doc)
                    final_embeddings.append(embedding)
                else:
                    removed_count += 1
                    logger.debug(
                        f"  Removed duplicate: {doc.meta.get('chunk_id', 'N/A')} "
                        f"(similarity={max_similarity:.4f})"
                    )

            logger.info(
                f"Deduplication complete: {len(final_docs)} unique documents "
                f"({removed_count} duplicates removed)"
            )

            # Add deduplication metadata
            for doc in final_docs:
                doc.meta["deduplicated"] = True

            return final_docs

        except Exception as e:
            logger.error(f"Deduplication failed: {e}")
            raise

    def verify_and_deduplicate(
            self,
            query: str,
            documents: List[Document],
            relevance_threshold: Optional[float] = None,
            dedup_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Combined pipeline: verify relevance and remove duplicates.
        
        Args:
            query: Search query
            documents: List of candidate documents
            relevance_threshold: Override default NLI threshold
            dedup_threshold: Override default deduplication threshold
            
        Returns:
            List of verified and deduplicated documents
        """
        logger.info("Starting combined verification and deduplication")

        # Step 1: Verify relevance
        verified_docs = self.verify_relevance(
            query,
            documents,
            threshold=relevance_threshold
        )

        if not verified_docs:
            logger.warning("No documents passed verification")
            return []

        # Step 2: Deduplicate
        final_docs = self.deduplicate_context(
            verified_docs,
            threshold=dedup_threshold
        )

        logger.info(
            f"Combined processing complete: "
            f"{len(documents)} → {len(verified_docs)} → {len(final_docs)} documents"
        )

        return final_docs

    def get_deduplication_stats(
            self,
            original_docs: List[Document],
            deduplicated_docs: List[Document]
    ) -> Dict[str, Any]:
        """
        Get statistics about deduplication.
        
        Args:
            original_docs: Original document list
            deduplicated_docs: Deduplicated document list
            
        Returns:
            Dictionary with deduplication statistics
        """
        removed = len(original_docs) - len(deduplicated_docs)
        removal_rate = (removed / len(original_docs)) * 100 if original_docs else 0

        return {
            "original_count": len(original_docs),
            "final_count": len(deduplicated_docs),
            "removed_count": removed,
            "removal_rate_percent": removal_rate,
            "threshold": self.dedup_threshold
        }

    def order_context(
            self,
            documents: List[Document],
            strategy: Literal["relevance", "chronological", "lost_in_middle"] = "relevance"
    ) -> List[Document]:
        """
        Order documents according to specified strategy.
        
        Strategies:
        1. **relevance**: Keep original order (highest scores first) - DEFAULT
        2. **chronological**: Sort by timestamp/date metadata
        3. **lost_in_middle**: Mitigate "lost in the middle" effect by alternating
           high-relevance documents (best at start and end)
        
        Args:
            documents: List of documents to order
            strategy: Ordering strategy (default: "relevance")
            
        Returns:
            Ordered list of documents
        """
        if not documents:
            logger.warning("No documents to order")
            return []

        if len(documents) == 1:
            logger.info("Only 1 document, no ordering needed")
            return documents

        logger.info(f"Ordering {len(documents)} documents with strategy: {strategy}")

        if strategy == "relevance":
            # Already ordered by relevance (highest score first)
            ordered = documents
            logger.info("Documents kept in relevance order (highest score first)")

        elif strategy == "chronological":
            # Sort by timestamp/date in metadata
            ordered = self._order_chronological(documents)
            logger.info("Documents ordered chronologically")

        elif strategy == "lost_in_middle":
            # Mitigate "lost in the middle" effect
            ordered = self._order_lost_in_middle(documents)
            logger.info("Documents reordered to mitigate 'lost in the middle' effect")

        else:
            logger.warning(f"Unknown strategy '{strategy}', using 'relevance'")
            ordered = documents

        # Add ordering metadata
        for idx, doc in enumerate(ordered):
            doc.meta["display_order"] = idx + 1
            doc.meta["ordering_strategy"] = strategy

        return ordered

    @staticmethod
    def _order_chronological(documents: List[Document]) -> List[Document]:
        """
        Order documents chronologically based on metadata.
        
        Looks for timestamp/date fields in metadata:
        - timestamp
        - date
        - created_at
        - modified_at
        
        Falls back to relevance order if no timestamps found.
        """
        # Try to extract timestamps
        docs_with_time = []
        docs_without_time = []

        for doc in documents:
            timestamp = (
                    doc.meta.get('timestamp') or
                    doc.meta.get('date') or
                    doc.meta.get('created_at') or
                    doc.meta.get('modified_at')
            )

            if timestamp:
                docs_with_time.append((doc, timestamp))
            else:
                docs_without_time.append(doc)

        # Sort documents with timestamps
        if docs_with_time:
            docs_with_time.sort(key=lambda x: x[1])
            ordered = [doc for doc, _ in docs_with_time]

            # Append documents without timestamps at the end
            ordered.extend(docs_without_time)

            logger.info(
                f"  Sorted {len(docs_with_time)} docs chronologically, "
                f"{len(docs_without_time)} without timestamps appended"
            )
        else:
            logger.warning("  No timestamp metadata found, keeping relevance order")
            ordered = documents

        return ordered

    @staticmethod
    def _order_lost_in_middle(documents: List[Document]) -> List[Document]:
        """
        Mitigate "lost in the middle" effect.
        
        Research shows LLMs pay more attention to information at the
        beginning and end of context, with reduced attention in the middle.
        
        Algorithm:
        - Place most relevant document first (position 0)
        - Place second most relevant document last (end position)
        - Alternate remaining documents: odd indices → beginning, even → end
        - Result: [doc0, doc2, doc4, ..., doc5, doc3, doc1]
        
        Example with 5 documents (by score):
        Original: [A, B, C, D, E]  (A=best)
        Reordered: [A, C, E, D, B]
        - A (best) at start
        - B (2nd best) at end
        - C, E in middle-start
        - D in middle-end
        """
        n = len(documents)

        if n <= 2:
            # No reordering needed for 1-2 documents
            return documents

        # Split documents into two groups
        ordered = [None] * n

        # Most relevant at start
        ordered[0] = documents[0]

        # Second most relevant at end
        ordered[-1] = documents[1]

        # Alternate remaining documents
        # Odd-ranked docs (3rd, 5th, 7th...) → fill from start
        # Even-ranked docs (4th, 6th, 8th...) → fill from end

        left_idx = 1  # Next position from start
        right_idx = n - 2  # Next position from end

        for i in range(2, n):
            if i % 2 == 0:  # Even index (3rd, 5th, 7th... docs)
                ordered[left_idx] = documents[i]
                left_idx += 1
            else:  # Odd index (4th, 6th, 8th... docs)
                ordered[right_idx] = documents[i]
                right_idx -= 1

        logger.info(
            f"  Reordered for LLM attention: "
            f"Best at position 0, 2nd best at position {n - 1}"
        )

        return ordered

    def process_context(
            self,
            query: str,
            documents: List[Document],
            relevance_threshold: Optional[float] = None,
            dedup_threshold: Optional[float] = None,
            ordering_strategy: Literal["relevance", "chronological", "lost_in_middle"] = "lost_in_middle"
    ) -> List[Document]:
        """
        Full context processing pipeline: verify, deduplicate, and order.
        
        This is the recommended method for complete context preparation.
        
        Args:
            query: Search query
            documents: Candidate documents from reranking
            relevance_threshold: NLI threshold (default: self.threshold)
            dedup_threshold: Similarity threshold (default: self.dedup_threshold)
            ordering_strategy: How to order final documents (default: "lost_in_middle")
            
        Returns:
            Processed documents ready for LLM generation
        """
        logger.info(
            "Starting full context processing pipeline "
            f"(verify → deduplicate → order[{ordering_strategy}])"
        )

        # Step 1: Verify relevance
        verified_docs = self.verify_relevance(
            query,
            documents,
            threshold=relevance_threshold
        )

        if not verified_docs:
            logger.warning("No documents passed verification")
            return []

        # Step 2: Deduplicate
        unique_docs = self.deduplicate_context(
            verified_docs,
            threshold=dedup_threshold
        )

        if not unique_docs:
            logger.warning("No documents after deduplication")
            return []

        # Step 3: Order
        final_docs = self.order_context(
            unique_docs,
            strategy=ordering_strategy
        )

        logger.info(
            f"Context processing complete: "
            f"{len(documents)} → {len(verified_docs)} → {len(unique_docs)} → {len(final_docs)} documents"
        )

        return final_docs

    @staticmethod
    def prepare_citations(
            documents: List[Document]
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Prepare documents with citation markers and create citation map.
        
        Assigns citation IDs [1], [2], [3] etc. to documents and creates
        a citation map with source metadata for LLM prompt construction.
        
        Args:
            documents: List of documents to prepare with citations
            
        Returns:
            Tuple of:
            - List of documents with citation_id in metadata
            - Citation map dictionary with full source information
        """
        if not documents:
            logger.warning("No documents to prepare citations for")
            return [], {"citations": [], "total_citations": 0}

        logger.info(f"Preparing citations for {len(documents)} documents")

        # Prepare documents with citation IDs
        cited_documents = []
        citations = []

        for idx, doc in enumerate(documents, start=1):
            citation_id = f"[{idx}]"

            # Extract source metadata (fallback to basename of filepath if source missing)
            source_file = doc.meta.get('source') or ''
            if not source_file:
                filepath = doc.meta.get('filepath') or ''
                source_file = os.path.basename(filepath) if filepath else 'Unknown'
            page = doc.meta.get('page')
            chunk_id = doc.meta.get('chunk_id', f'chunk_{idx}')

            # Get relevance scores
            rerank_score = doc.score if doc.score is not None else 0.0
            nli_score = doc.meta.get('nli_score', 0.0)

            # Create citation entry
            citation_entry = {
                "citation_id": citation_id,
                "citation_number": idx,
                "source_file": source_file,
                "chunk_id": chunk_id,
                "rerank_score": rerank_score,
                "nli_score": nli_score,
                "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            }

            # Add page if available
            if page is not None:
                citation_entry["page"] = page

            # Add display order if available
            if 'display_order' in doc.meta:
                citation_entry["display_order"] = doc.meta['display_order']

            citations.append(citation_entry)

            # Create document with citation metadata
            cited_doc = Document(
                content=doc.content,
                id=doc.id,
                meta={
                    **doc.meta,
                    "citation_id": citation_id,
                    "citation_number": idx,
                    "citation_prepared": True
                },
                score=doc.score
            )

            cited_documents.append(cited_doc)

        # Create citation map
        citation_map = {
            "citations": citations,
            "total_citations": len(citations),
            "citation_format": "[N]",
            "instructions": "Use [N] notation to cite sources in your answer"
        }

        logger.info(
            f"Citations prepared: {len(citations)} documents "
            f"with IDs [{citations[0]['citation_id']}...{citations[-1]['citation_id']}]"
        )

        return cited_documents, citation_map

    @staticmethod
    def format_context_with_citations(
            documents: List[Document],
            include_metadata: bool = True
    ) -> str:
        """
        Format documents as context string with citations for LLM prompt.
        
        Args:
            documents: List of documents with citation_id in metadata
            include_metadata: Whether to include source metadata in output
            
        Returns:
            Formatted context string ready for LLM prompt
        """
        if not documents:
            return ""

        context_parts = []

        for doc in documents:
            citation_id = doc.meta.get('citation_id', '[?]')

            # Add citation ID and content
            context_parts.append(f"{citation_id} {doc.content}")

            # Add metadata if requested
            if include_metadata:
                source = doc.meta.get('source') or ''
                if not source:
                    fp = doc.meta.get('filepath') or ''
                    source = os.path.basename(fp) if fp else 'Unknown'
                page = doc.meta.get('page')

                if page is not None:
                    context_parts.append(f"(Source: {source}, Page: {page})")
                else:
                    context_parts.append(f"(Source: {source})")

            context_parts.append("")  # Empty line between documents

        return "\n".join(context_parts)

    @staticmethod
    def get_citation_summary(
            citation_map: Dict[str, Any]
    ) -> str:
        """
        Get human-readable summary of citations.
        
        Args:
            citation_map: Citation map from prepare_citations()
            
        Returns:
            Formatted string summarizing all citations
        """
        if not citation_map or not citation_map.get('citations'):
            return "No citations available"

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("Citation Summary")
        lines.append("=" * 60 + "\n")

        for citation in citation_map['citations']:
            cid = citation['citation_id']
            source = citation['source_file']
            page = citation.get('page', 'N/A')

            lines.append(f"{cid}  Source: {source}")
            if page != 'N/A':
                lines.append(f"     Page: {page}")
            lines.append(f"     Preview: {citation['content_preview']}")
            lines.append("")

        lines.append(f"Total Citations: {citation_map['total_citations']}")
        lines.append(f"Citation Format: {citation_map['citation_format']}")

        return "\n".join(lines)

    def get_config(self) -> Dict[str, Any]:
        """
        Get verifier configuration.
        
        Returns:
            Dictionary with configuration details
        """
        return {
            "model_name": self.model_name,
            "embedding_model": self.embedding_model_name,
            "threshold": self.threshold,
            "dedup_threshold": self.dedup_threshold,
            "batch_size": self.batch_size,
            "device": "GPU" if self.use_gpu else "CPU",
            "max_length": self.max_length
        }


def create_context_verifier(
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        threshold: float = 0.5,
        dedup_threshold: float = 0.85,
        **kwargs
) -> ContextVerifier:
    """
    Factory function to create a ContextVerifier instance.
    
    Args:
        model_name: NLI model name
        threshold: Relevance threshold
        dedup_threshold: Deduplication similarity threshold
        **kwargs: Additional arguments for ContextVerifier
        
    Returns:
        Configured ContextVerifier instance
    """
    return ContextVerifier(
        model_name=model_name,
        threshold=threshold,
        dedup_threshold=dedup_threshold,
        **kwargs
    )
