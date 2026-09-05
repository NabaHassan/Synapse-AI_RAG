"""
Dense Retriever using QdrantClient for vector-based semantic search.

This module implements dense retrieval using:
- Qdrant server for efficient similarity search
- Sentence transformers for embedding generation
- Cosine similarity for relevance scoring
- Configurable top-k retrieval for broad recall
"""

import os
import logging
from typing import List, Optional, Dict, Any

from haystack import Document
from qdrant_client import QdrantClient
from haystack.utils import ComponentDevice
from haystack.components.embedders import SentenceTransformersTextEmbedder

logger = logging.getLogger(__name__)


class DenseRetriever:
    """
    Dense retriever using vector embeddings for semantic search.
    
    Uses Qdrant as the vector store and sentence-transformers for embeddings.
    Provides high recall by returning a large number of candidates for reranking.
    """

    def __init__(
            self,
            collection_name: str = "knowledge_base",
            embedding_model: str = "BAAI/bge-large-en-v1.5",
            embedding_dim: int = 1024,
            qdrant_url: str = "http://localhost:6333",
            top_k: int = 50,
            distance_metric: str = "cosine",
            use_gpu: bool = False,
            normalize_embeddings: bool = True,
            default_filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Dense Retriever.
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Sentence transformers model name
            embedding_dim: Dimension of embedding vectors (1024 for BAAI/bge-large-en-v1.5)
            qdrant_url: URL for Qdrant server
            top_k: Number of documents to retrieve (default 50 for broad recall)
            distance_metric: Distance metric (cosine, dot, euclidean)
            use_gpu: Whether to use GPU for embeddings
            normalize_embeddings: Whether to normalize embeddings
        """
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.distance_metric = distance_metric
        self.normalize_embeddings = normalize_embeddings
        self.default_filters = dict(default_filters or {})

        logger.info(f" Initializing DenseRetriever with model: {embedding_model}")

        # Initialize embedding model
        self._init_embedder(use_gpu)

        # Initialize Qdrant client
        self._init_qdrant_client(qdrant_url)

        # Verify collection exists
        self._verify_collection()

        logger.info(f" DenseRetriever initialized successfully")

    def _init_embedder(self, use_gpu: bool):
        """Initialize the embedding model."""
        try:
            device = ComponentDevice.from_str("cuda" if use_gpu else "cpu")
            self.embedder = SentenceTransformersTextEmbedder(
                model=self.embedding_model_name,
                device=device,
                normalize_embeddings=self.normalize_embeddings
            )

            # Warm up the model
            self.embedder.warm_up()
            logger.info(f" Embedder initialized on {device}")

        except Exception as e:
            logger.error(f" Failed to initialize embedder: {e}")
            raise

    def _init_qdrant_client(self, qdrant_url: str):
        """Initialize Qdrant client."""
        try:
            if not qdrant_url:
                raise ValueError("qdrant_url must be provided")
            logger.info(f" Connecting to Qdrant server: {qdrant_url}")
            self.client = QdrantClient(
                url=qdrant_url,
                timeout=600,  # 10 minutes timeout for large operations
                prefer_grpc=False,
            )
            logger.info(" Qdrant client initialized")

        except Exception as e:
            logger.error(f" Failed to initialize Qdrant client: {e}")
            raise

    def _verify_collection(self):
        """Verify that the collection exists."""
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)

            if not collection_exists:
                logger.warning(
                    f" Collection '{self.collection_name}' does not exist. "
                    f"Please run the indexing pipeline first."
                )
                logger.info("Available collections:")
                for c in collections:
                    logger.info(f"  - {c.name}")
            else:
                # Get collection info
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f" Collection '{self.collection_name}' found")
                logger.info(f"   Points: {collection_info.points_count}")
                logger.info(f"   Vector size: {collection_info.config.params.vectors.size}")

        except Exception as e:
            logger.error(f" Failed to verify collection: {e}")
            raise

    def retrieve(
            self,
            query: str,
            top_k: Optional[int] = None,
            filters: Optional[Dict[str, Any]] = None,
            score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Retrieve documents using dense vector search.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve (overrides default)
            filters: Metadata filters for Qdrant (e.g., {"source_filename": "doc.pdf"})
            score_threshold: Minimum similarity score
            
        Returns:
            List of Document objects with similarity scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        k = top_k if top_k is not None else self.top_k

        try:
            logger.info(f" Retrieving top {k} documents for query: '{query[:50]}...'")

            # Generate query embedding
            embedding_result = self.embedder.run(text=query)
            query_embedding = embedding_result["embedding"]

            # Build Qdrant filter if provided
            query_filter = None
            effective_filters: Dict[str, Any] = dict(self.default_filters)
            if filters:
                effective_filters.update(filters)
            if effective_filters:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                field_conditions = []
                for field, value in effective_filters.items():
                    field_conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
                query_filter = Filter(must=field_conditions)

            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=score_threshold,
                query_filter=query_filter
            )

            # Convert to Haystack Documents
            documents = self._convert_to_documents(search_results)

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

    @staticmethod
    def _convert_to_documents(search_results) -> List[Document]:
        """Convert Qdrant search results to Haystack Documents."""
        documents = []
        for idx, result in enumerate(search_results):
            source_filename = result.payload.get("source_filename", "")
            source_filepath = result.payload.get("source_filepath", "")

            # Derive source from filepath when missing
            if (not source_filename) and source_filepath:
                source_filename = os.path.basename(source_filepath)

            doc = Document(
                content=result.payload.get("content", ""),
                id=str(result.id),
                score=result.score,
                meta={
                    "source": source_filename or "Unknown",
                    "filepath": source_filepath,
                    "page": result.payload.get("page_number"),
                    "chunk_id": result.payload.get("chunk_id"),
                    "chunk_index": result.payload.get("chunk_index"),
                    "file_type": result.payload.get("file_type", ""),
                }
            )
            documents.append(doc)
        return documents

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

    def retrieve_with_hyde(
            self,
            hypothetical_doc: str,
            top_k: Optional[int] = None,
            filters: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve using a hypothetical document (HyDE).
        
        Instead of retrieving with the query, retrieve using a hypothetical
        answer/document that would contain the information.
        
        Args:
            hypothetical_doc: Generated hypothetical document
            top_k: Number of documents to retrieve
            filters: Metadata filters
            
        Returns:
            List of retrieved documents
        """
        logger.info(" Retrieving with HyDE (Hypothetical Document)")
        return self.retrieve(hypothetical_doc, top_k=top_k, filters=filters)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get retriever statistics and metadata.
        
        Returns:
            Dictionary with stats including document count, collection info, etc.
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            doc_count = collection_info.points_count

            stats = {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "embedding_dim": self.embedding_dim,
                "distance_metric": self.distance_metric,
                "top_k": self.top_k,
                "document_count": doc_count,
                "normalize_embeddings": self.normalize_embeddings,
                "storage_mode": self.storage_mode
            }

            logger.info(f" Stats: {doc_count} documents in collection '{self.collection_name}'")
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
        Test the connection to Qdrant and verify collection exists.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)

            if collection_exists:
                collection_info = self.client.get_collection(self.collection_name)
                doc_count = collection_info.points_count
                logger.info(f" Connection test passed: {doc_count} documents in collection")
                return True
            else:
                logger.warning(f"  Collection '{self.collection_name}' not found")
                return False
        except Exception as e:
            logger.error(f" Connection test failed: {e}")
            return False


def create_dense_retriever(
        collection_name: str = None,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        storage_path: str = None,
        qdrant_url: str = None,
        top_k: int = 50,
        **kwargs
) -> DenseRetriever:
    """
    Factory function to create a DenseRetriever instance.
    
    Args:
        collection_name: Qdrant collection name
        embedding_model: Sentence transformers model
        storage_path: Local storage path
        qdrant_url: Qdrant server URL
        top_k: Number of documents to retrieve
        **kwargs: Additional arguments for DenseRetriever
        
    Returns:
        Configured DenseRetriever instance
    """
    return DenseRetriever(
        collection_name=collection_name,
        embedding_model=embedding_model,
        storage_path=storage_path,
        qdrant_url=qdrant_url,
        top_k=top_k,
        **kwargs
    )
