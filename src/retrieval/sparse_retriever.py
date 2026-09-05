"""
Sparse Retriever using Haystack BM25 for keyword-based search.

This module implements sparse retrieval using:
- BM25 algorithm for term-based matching
- InMemoryDocumentStore for fast keyword search
- Configurable BM25 parameters (k1, b)
- Complements dense retrieval for hybrid search
"""

import os
import logging
import math
import re
import time
from typing import List, Optional, Dict, Any
from collections import Counter, defaultdict

from haystack import Document
from qdrant_client import QdrantClient

from src.retrieval.bm25_index_cache import BM25IndexCache

logger = logging.getLogger(__name__)


# =============================================================================
# Fast Stopword Set (Hardcoded for speed - no file I/O)
# =============================================================================
STOPWORDS = {
    "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on",
    "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from", "or", "one",
    "had", "by", "word", "but", "not", "what", "all", "were", "we", "when", "your", "can",
    "said", "there", "use", "an", "each", "which", "she", "do", "how", "their", "if", "will",
    "up", "other", "about", "out", "many", "then", "them", "these", "so", "some", "her",
    "would", "make", "like", "him", "into", "time", "has", "look", "two", "more", "write",
    "go", "see", "number", "no", "way", "could", "people", "my", "than", "first", "water",
    "been", "call", "who", "oil", "its", "now", "find", "long", "down", "day", "did", "get",
    "come", "made", "may", "part"
}

# =============================================================================
# Compiled Regex Pattern (Compiled once for speed)
# =============================================================================
# Pattern breakdown:
# 1. §\s*\d+(?:\([a-z]\))?(?:\(\d+\))?  → Legal sections: § 501, § 501(c), § 501(c)(3)
# 2. \d+[a-z]?\([a-z]+\)(?:\(\d+\))?    → Complex codes: 501(c)(3), 42(a)
# 3. \b[A-Z]{2,}\b                       → Acronyms: CARVE, CAFL, IRS (preserved uppercase)
# 4. \b\w+(?:-\w+)+\b                    → Hyphenated: pre-trial, state-of-the-art
# 5. \b\w\w+\b                           → Standard words (2+ chars)
# 6. \b\d+\.?\d*\b                       → Numbers: 123, 45.6
LEGAL_TOKEN_PATTERN = re.compile(
    r'(?u)(?:'
    r'§\s*\d+(?:\([a-z]\))?(?:\(\d+\))?|'  # Legal sections
    r'\d+[a-z]?\([a-z]+\)(?:\(\d+\))?|'    # Complex codes
    r'\b[A-Z]{2,}\b|'                       # Acronyms (uppercase preserved)
    r'\b\w+(?:-\w+)+\b|'                    # Hyphenated terms
    r'\b\w\w+\b|'                           # Standard words
    r'\b\d+\.?\d*\b'                        # Numbers
    r')',
    re.UNICODE
)


class SimpleBM25:
    """
    A lightweight, fast implementation of BM25 (Okapi) for in-memory retrieval.
    Optimized for legal/technical domains with enhanced tokenization.
    
    Features:
    - Legal term preservation (§ 501(c)(3), hyphenated terms)
    - Stopword removal (speeds up scoring)
    - Acronym preservation (CARVE, IRS)
    - Optional light stemming for plurals
    """

    def __init__(
        self, 
        documents: List[Document], 
        k1: float = 1.5, 
        b: float = 0.75,
        enable_plural_stemming: bool = False
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.corpus_size = len(documents)
        self.avgdl = 0
        self.doc_lengths = []
        self.doc_freqs = []
        self.idf = {}
        self.inverted_index = defaultdict(list)
        self.enable_plural_stemming = enable_plural_stemming

        # Initialize index
        self._index_corpus()

    def _tokenize(self, text: str) -> List[str]:
        """
        Enhanced tokenizer optimized for legal/technical domains.
        
        Features:
        1. Preserves legal patterns (§ 501(c)(3), hyphenated terms)
        2. Preserves acronyms (CARVE, IRS) 
        3. Removes stopwords (speeds up BM25 scoring)
        4. Optional light plural stemming
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens (stopwords removed)
        """
        # Extract tokens using compiled regex pattern
        tokens = LEGAL_TOKEN_PATTERN.findall(text)
        
        # Process tokens
        processed = []
        for token in tokens:
            # Preserve acronyms (all uppercase, 2+ chars)
            if token.isupper() and len(token) >= 2:
                processed.append(token)  # Keep uppercase for exact match
                continue
            
            # Lowercase everything else
            token_lower = token.lower()
            
            # Skip stopwords (this speeds up BM25!)
            if token_lower in STOPWORDS:
                continue
            
            # Optional: Light plural stemming (safe for hybrid search)
            if self.enable_plural_stemming:
                token_lower = self._light_stem(token_lower)
            
            processed.append(token_lower)
        
        return processed
    
    def _light_stem(self, token: str) -> str:
        """
        Very light stemming - only handles simple plurals.
        
        This is safe for hybrid search because:
        - Dense embeddings handle semantic variations
        - BM25 needs exact matches for legal terms
        - We only normalize obvious plurals
        
        Args:
            token: Token to stem
            
        Returns:
            Stemmed token
        """
        # Only stem if word is 5+ chars (avoid breaking short words)
        if len(token) < 5:
            return token
        
        # Simple plural rules (conservative)
        if token.endswith('ies') and len(token) > 5:
            return token[:-3] + 'y'  # companies → company
        elif token.endswith('es') and len(token) > 4:
            # Only if preceded by s, x, z, ch, sh
            if token[-3] in 'sxz' or token[-4:-2] in ('ch', 'sh'):
                return token[:-2]  # boxes → box, churches → church
        elif token.endswith('s') and len(token) > 4:
            # Avoid breaking words ending in 'ss', 'us', 'is'
            if token[-2] not in 'su' and not token.endswith('is'):
                return token[:-1]  # adoptions → adoption
        
        return token

    def _index_corpus(self):
        """Build the inverted index and statistics."""
        total_length = 0
        doc_counts = Counter()

        for idx, doc in enumerate(self.documents):
            tokens = self._tokenize(doc.content)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_length += length

            # Update inverted index and document frequencies
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                self.inverted_index[token].append((idx, count))
                doc_counts[token] += 1

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0

        # Calculate IDF
        for token, freq in doc_counts.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)

    def get_scores(self, query: str) -> List[float]:
        """Calculate BM25 scores for the query against all docs."""
        scores = defaultdict(float)
        query_tokens = self._tokenize(query)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf[token]
            # Retrieve postings list for this token
            postings = self.inverted_index[token]

            for idx, freq in postings:
                doc_len = self.doc_lengths[idx]
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[idx] += idf * (numerator / denominator)

        return scores

    def retrieve(self, query: str, top_k: int = 50, scale_score: bool = True) -> List[Document]:
        """Retrieve top_k documents."""
        scores = self.get_scores(query)

        # Sort by score descending
        sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in sorted_indices:
            doc = self.documents[idx]
            # Create a copy or modify score? Haystack docs usually mutable in pipeline context
            # We'll attach the score to the document object
            final_score = self._sigmoid(score) if scale_score else score
            doc.score = final_score
            results.append(doc)

        return results

    def _sigmoid(self, x: float) -> float:
        """Scale score to 0-1 range using sigmoid."""
        return 1 / (1 + math.exp(-x))
    
    def get_state(self) -> Dict[str, Any]:
        """Extract BM25 internal state for serialization."""
        return {
            "inverted_index": dict(self.inverted_index),
            "idf": self.idf,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "corpus_size": self.corpus_size
        }
    
    @classmethod
    def from_cache(
        cls,
        documents: List[Document],
        state: Dict[str, Any],
        k1: float = 1.5,
        b: float = 0.75,
        enable_plural_stemming: bool = False
    ) -> 'SimpleBM25':
        """Restore BM25 from cached state without rebuilding index."""
        # Create instance without indexing
        instance = cls.__new__(cls)
        instance.documents = documents
        instance.k1 = k1
        instance.b = b
        instance.corpus_size = state["corpus_size"]
        instance.avgdl = state["avgdl"]
        instance.doc_lengths = state["doc_lengths"]
        instance.idf = state["idf"]
        instance.inverted_index = defaultdict(list, state["inverted_index"])
        instance.enable_plural_stemming = enable_plural_stemming
        
        return instance


class SparseRetriever:
    """
    Sparse retriever using optimized BM25 for keyword-based search.
    
    Loads documents from Qdrant and builds a fast in-memory BM25 index.
    """

    def __init__(
            self,
            collection_name: str = "knowledge_base",
            qdrant_url: str = "http://localhost:6333",
            top_k: int = 50,
            scale_score: bool = True,
            index_cache_path: Optional[str] = None,
            k1: float = 1.5,
            b: float = 0.75,
            enable_plural_stemming: bool = False,
    ):
        """
        Initialize the Sparse Retriever.
        
        Args:
            collection_name: Name of the Qdrant collection to load from
            qdrant_url: URL for Qdrant server
            top_k: Number of documents to retrieve (default 50 for broad recall)
            scale_score: Whether to scale BM25 scores to 0-1 range
            index_cache_path: Path for BM25 index cache (default: ./data/bm25_indices)
            k1: BM25 k1 parameter (default: 1.5)
            b: BM25 b parameter (default: 0.75)
            enable_plural_stemming: Enable light plural stemming (default: False)
        """
        self.collection_name = collection_name
        self.top_k = top_k
        self.scale_score = scale_score
        self.k1 = k1
        self.b = b
        self.enable_plural_stemming = enable_plural_stemming

        logger.info(f"Initializing SparseRetriever for collection: {collection_name}")
        if enable_plural_stemming:
            logger.info("  Light plural stemming: ENABLED")

        # Initialize cache manager
        cache_path = index_cache_path or "./data/bm25_indices"
        self.cache_manager = BM25IndexCache(cache_dir=cache_path)

        # Initialize Qdrant client to load documents
        self._init_qdrant_client(qdrant_url)

        # Load or build BM25 index (with caching)
        self._load_or_build_index()

        logger.info(f"SparseRetriever initialized successfully with {self.bm25.corpus_size} documents")

    def _init_qdrant_client(self, qdrant_url: str):
        """Initialize Qdrant client to load documents."""
        try:
            if not qdrant_url:
                raise ValueError("qdrant_url must be provided")
            logger.info(f" Connecting to Qdrant server: {qdrant_url}")
            self.qdrant_client = QdrantClient(
                url=qdrant_url,
                timeout=600,  # 10 minutes timeout for large operations
                prefer_grpc=False,
            )
            logger.info(" Qdrant client initialized")

        except Exception as e:
            logger.error(f" Failed to initialize Qdrant client: {e}")
            raise

    def _load_documents_from_qdrant(self) -> List[Document]:
        """Load all documents from Qdrant collection for BM25 indexing."""
        try:
            # Verify collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                raise ValueError(
                    f"Collection '{self.collection_name}' not found. "
                    f"Please run the indexing pipeline first."
                )

            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            logger.info(f" Loading {total_points} documents from Qdrant...")

            # Scroll through all documents
            documents = []
            offset = None
            batch_size = 1000

            while True:
                result, offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # Don't need vectors for BM25
                )

                if not result:
                    break

                # Convert to Haystack Documents
                for point in result:
                    source_filename = point.payload.get("source_filename", "")
                    source_filepath = point.payload.get("source_filepath", "")
                    # Derive source from filepath when missing
                    if (not source_filename) and source_filepath:
                        source_filename = os.path.basename(source_filepath)

                    doc = Document(
                        content=point.payload.get("content", ""),
                        id=str(point.id),
                        meta={
                            "source": source_filename or "Unknown",
                            "filepath": source_filepath,
                            "page": point.payload.get("page_number"),
                            "chunk_id": point.payload.get("chunk_id"),
                            "chunk_index": point.payload.get("chunk_index"),
                            "file_type": point.payload.get("file_type", ""),
                        }
                    )
                    documents.append(doc)

                if offset is None:
                    break

            logger.info(f" Loaded {len(documents)} documents for BM25 indexing")
            return documents

        except Exception as e:
            logger.error(f" Failed to load documents from Qdrant: {e}")
            raise

    def _load_or_build_index(self):
        """Load BM25 index from cache or build from scratch."""
        try:
            # Get current collection checksum
            current_checksum = self.cache_manager.get_collection_checksum(
                self.qdrant_client,
                self.collection_name
            )
            
            # Try to load from cache
            start_time = time.time()
            cached_data = self.cache_manager.load_bm25_index(
                self.collection_name,
                current_checksum
            )
            
            if cached_data:
                # Cache hit - restore from cached state
                logger.info("Loading BM25 index from cache (fast path)...")
                
                documents = cached_data["index_data"]["documents"]
                state = {
                    "inverted_index": cached_data["index_data"]["inverted_index"],
                    "idf": cached_data["index_data"]["idf"],
                    "doc_lengths": cached_data["index_data"]["doc_lengths"],
                    "avgdl": cached_data["index_data"]["avgdl"],
                    "corpus_size": cached_data["index_data"]["corpus_size"]
                }
                
                # Restore BM25 from cache
                self.bm25 = SimpleBM25.from_cache(
                    documents=documents,
                    state=state,
                    k1=self.k1,
                    b=self.b,
                    enable_plural_stemming=self.enable_plural_stemming
                )
                
                load_time = time.time() - start_time
                logger.info(
                    f"Loaded BM25 index from cache in {load_time:.2f}s | "
                    f"Documents: {len(documents):,}"
                )
                
            else:
                # Cache miss - build from scratch
                logger.info("Building BM25 index from scratch (slow path)...")
                
                # Load documents from Qdrant
                documents = self._load_documents_from_qdrant()
                
                # Build BM25 index
                self.bm25 = SimpleBM25(
                    documents=documents,
                    k1=self.k1,
                    b=self.b,
                    enable_plural_stemming=self.enable_plural_stemming
                )
                
                build_time = time.time() - start_time
                logger.info(
                    f"Built BM25 index in {build_time:.2f}s | "
                    f"Documents: {len(documents):,}"
                )
                
                # Save to cache for next time
                self._save_index_to_cache(current_checksum, documents)
                
        except Exception as e:
            logger.error(f"Failed to load/build BM25 index: {e}")
            raise
    
    def _save_index_to_cache(self, checksum: str, documents: List[Document]):
        """Save current BM25 index to cache."""
        try:
            logger.info("Saving BM25 index to cache...")
            
            # Extract BM25 state
            bm25_state = self.bm25.get_state()
            
            # Save to cache
            success = self.cache_manager.save_bm25_index(
                collection_name=self.collection_name,
                collection_checksum=checksum,
                bm25_state=bm25_state,
                documents=documents,
                bm25_params={"k1": self.k1, "b": self.b}
            )
            
            if success:
                logger.info("BM25 index cached successfully")
            else:
                logger.warning("Failed to cache BM25 index (non-fatal)")
                
        except Exception as e:
            # Non-fatal - just log the error
            logger.warning(f"Failed to save index to cache: {e}")

    def retrieve(
            self,
            query: str,
            top_k: Optional[int] = None,
            filters: Optional[Dict[str, Any]] = None,
            scale_score: Optional[bool] = None
    ) -> List[Document]:
        """
        Retrieve documents using BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides default)
            filters: Metadata filters (Not implemented in SimpleBM25 for speed, ignored)
            scale_score: Whether to scale scores to 0-1 (overrides default)
            
        Returns:
            List of Document objects with BM25 scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        k = top_k if top_k is not None else self.top_k
        use_scale = scale_score if scale_score is not None else self.scale_score

        try:
            logger.info(f" Retrieving top {k} documents for query: '{query[:50]}...'")

            # Retrieve with custom BM25
            documents = self.bm25.retrieve(
                query=query,
                top_k=k,
                scale_score=use_scale
            )

            if not documents:
                logger.warning("No documents retrieved")
                return []

            # Log retrieval statistics
            scores = [doc.score for doc in documents if doc.score is not None]
            if scores:
                logger.info(
                    f" Retrieved {len(documents)} documents | "
                    f"Score range: [{min(scores):.3f}, {max(scores):.3f}] | "
                    f"Mean: {sum(scores) / len(scores):.3f}"
                )
            else:
                logger.info(f" Retrieved {len(documents)} documents")

            return documents

        except Exception as e:
            logger.error(f" Retrieval failed: {e}")
            raise

    def retrieve_batch(
            self,
            queries: List[str],
            top_k: Optional[int] = None,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Document]]:
        """
        Retrieve documents for multiple queries in batch.
        
        Args:
            queries: List of search queries
            top_k: Number of documents per query
            filters: Metadata filters
            
        Returns:
            List of document lists (one per query)
        """
        if not queries:
            logger.warning("No queries provided")
            return []

        logger.info(f" Batch retrieval for {len(queries)} queries")

        results = []
        for i, query in enumerate(queries):
            try:
                docs = self.retrieve(query, top_k=top_k, filters=filters)
                results.append(docs)
            except Exception as e:
                logger.error(f"Failed to retrieve for query {i + 1}: {e}")
                results.append([])

        logger.info(f" Batch retrieval complete: {len(results)} result sets")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics and metadata.
        
        Returns:
            Dictionary with stats including document count, collection info, etc.
        """
        try:
            doc_count = len(self.bm25.documents)

            stats = {
                "collection_name": self.collection_name,
                "algorithm": "BM25 (Optimized)",
                "top_k": self.top_k,
                "document_count": doc_count,
                "scale_score": self.scale_score,
                "storage_mode": self.storage_mode
            }

            logger.info(f" Stats: {doc_count} documents indexed for BM25")
            return stats

        except Exception as e:
            logger.error(f" Failed to get stats: {e}")
            return {"error": str(e)}

    def update_top_k(self, new_top_k: int):
        """
        Update the default top_k value.
        
        Args:
            new_top_k: New default number of documents to retrieve
        """
        old_top_k = self.top_k
        self.top_k = new_top_k
        logger.info(f" Updated top_k: {old_top_k} → {new_top_k}")

    def test_connection(self) -> bool:
        """
        Test that the BM25 index is ready.
        
        Returns:
            True if index is ready, False otherwise
        """
        try:
            if self.bm25.corpus_size > 0:
                logger.info(f" BM25 index ready: {self.bm25.corpus_size} documents")
                return True
            else:
                logger.warning(" BM25 index is empty")
                return False
        except Exception as e:
            logger.error(f" BM25 index test failed: {e}")
            return False


def create_sparse_retriever(
        collection_name: str = None,
        storage_path: str = None,
        qdrant_url: str = None,
        top_k: int = 50,
        **kwargs
) -> SparseRetriever:
    """
    Factory function to create a SparseRetriever instance.
    
    Args:
        collection_name: Qdrant collection name
        storage_path: Local storage path
        qdrant_url: Qdrant server URL
        top_k: Number of documents to retrieve
        **kwargs: Additional arguments for SparseRetriever
        
    Returns:
        Configured SparseRetriever instance
    """
    return SparseRetriever(
        collection_name=collection_name,
        storage_path=storage_path,
        qdrant_url=qdrant_url,
        top_k=top_k,
        **kwargs
    )
