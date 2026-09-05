"""
Vector Store for RAG Pipeline
Manages vector storage and retrieval using Qdrant
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from src.indexing.embedding_generator import ChunkEmbedding

# Qdrant client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        SearchRequest,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    logging.warning("qdrant-client not available. Install with: pip install qdrant-client")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store using a Qdrant server.
    Manages embeddings storage and retrieval operations.
    """

    def __init__(
            self,
            collection_name: str = "knowledge_base",
            embedding_dim: int = 1024,  # Updated default for BAAI/bge-large-en-v1.5
            distance_metric: str = "Cosine",
            qdrant_url: str = "http://localhost:6333",
            recreate_collection: bool = False
    ):
        """
        Initialize VectorStore with Qdrant server.
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings (1024 for BAAI/bge-large-en-v1.5)
            distance_metric: Distance metric (Cosine, Euclidean, Dot)
            qdrant_url: URL for Qdrant server
            recreate_collection: Whether to recreate collection if exists
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not available. "
                "Install with: pip install qdrant-client"
            )

        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric

        if not qdrant_url:
            raise ValueError("qdrant_url must be provided")

        logger.info("Initializing Qdrant client")
        logger.info(f"  - Server URL: {qdrant_url}")
        logger.info(f"  - Collection: {collection_name}")
        self.client = QdrantClient(
            url=qdrant_url,
            timeout=600,  # 10 minutes timeout for large operations
            prefer_grpc=False,  # Use HTTP for better compatibility
        )

        # Setup collection
        self._setup_collection(recreate=recreate_collection)

        logger.info("VectorStore initialized successfully")

    def _setup_collection(self, recreate: bool = False):
        """
        Setup or create Qdrant collection
        
        Args:
            recreate: Whether to recreate collection if exists
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if collection_exists:
            if recreate:
                logger.warning(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                collection_exists = False
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                # Verify collection configuration
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"  - Points count: {collection_info.points_count}")
                logger.info(f"  - Vector size: {collection_info.config.params.vectors.size}")
                self._ensure_payload_indexes()
                return

        if not collection_exists:
            logger.info(f"Creating new collection: {self.collection_name}")

            # Map distance metric to Qdrant Distance
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            distance = distance_map.get(self.distance_metric, Distance.COSINE)

            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=distance
                )
            )

            logger.info(f" Collection created")
            logger.info(f"  - Embedding dimension: {self.embedding_dim}")
            logger.info(f"  - Distance metric: {self.distance_metric}")

            self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        """Ensure payload indexes needed by structured query handlers exist."""
        try:
            from qdrant_client.models import PayloadSchemaType

            index_fields = {
                "entity_names": PayloadSchemaType.KEYWORD,
                "is_first_chunk": PayloadSchemaType.BOOL,
                "email_sender": PayloadSchemaType.KEYWORD,
                "email_receiver": PayloadSchemaType.KEYWORD,
                "email_date": PayloadSchemaType.KEYWORD,
                "is_email": PayloadSchemaType.BOOL,
                "source_filename": PayloadSchemaType.KEYWORD,
                "file_uuid": PayloadSchemaType.KEYWORD,
                "ingest_state": PayloadSchemaType.KEYWORD,
                "ingest_job_id": PayloadSchemaType.KEYWORD,
                "ingest_version": PayloadSchemaType.KEYWORD,
            }

            created = 0
            for field_name, schema_type in index_fields.items():
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                    created += 1
                except Exception:
                    # Index may already exist (or backend may reject duplicate creation).
                    # Ignore per-field failures to keep startup resilient.
                    continue

            logger.info(
                f"  - Payload index ensure complete ({created}/{len(index_fields)} created this run)"
            )
        except Exception as e:
            logger.warning(f"  Failed to ensure payload indexes (non-fatal): {e}")

    def add_embeddings(
            self,
            chunk_embeddings: List[ChunkEmbedding],
            batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Add embeddings to the vector store
        
        Args:
            chunk_embeddings: List of ChunkEmbedding objects
            batch_size: Batch size for insertion
            
        Returns:
            Dictionary with insertion statistics
        """
        if not chunk_embeddings:
            logger.warning("No embeddings provided")
            return {"inserted": 0, "failed": 0}

        logger.info(f"Adding {len(chunk_embeddings)} embeddings to vector store")
        logger.info(f"  - Batch size: {batch_size}")

        # Insert in batches without materializing all points at once to
        # reduce peak RAM usage for large collections
        inserted = 0
        failed = 0

        for i in range(0, len(chunk_embeddings), batch_size):
            batch_embeddings = chunk_embeddings[i:i + batch_size]
            batch_points = [self._create_point(ce) for ce in batch_embeddings]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points,
                )
                inserted += len(batch_points)
                logger.debug(f"Inserted batch {i // batch_size + 1}: {len(batch_points)} points")
            except Exception as e:
                logger.error(f"Failed to insert batch: {e}")
                failed += len(batch_points)

        logger.info(f" Insertion complete")
        logger.info(f"  - Inserted: {inserted}")
        logger.info(f"  - Failed: {failed}")

        return {
            "inserted": inserted,
            "failed": failed,
            "total": len(chunk_embeddings)
        }

    def _create_point(self, chunk_embedding: ChunkEmbedding) -> PointStruct:
        """
        Create a Qdrant point from ChunkEmbedding
        
        Args:
            chunk_embedding: ChunkEmbedding object
            
        Returns:
            PointStruct for Qdrant
        """
        chunk = chunk_embedding.chunk

        # Handle both Chunk and Haystack Document objects
        # IMPORTANT: Check for 'meta' first because Haystack Documents may have chunk_id as a property
        if hasattr(chunk, 'meta'):
            # Haystack Document object
            if not hasattr(self, '_logged_branch_meta'):
                self._logged_branch_meta = True

            chunk_id = chunk.meta.get('chunk_id', 'unknown')
            content = chunk.content
            chunk_index = chunk.meta.get('chunk_index', 0)
            total_chunks = chunk.meta.get('total_chunks', 1)

            source_metadata = {
                'filename': chunk.meta.get('source_filename', chunk.meta.get('source', 'Unknown')),
                'filepath': chunk.meta.get('source_filepath', chunk.meta.get('file_path', '')),
                'file_type': chunk.meta.get('file_type', chunk.meta.get('chunker', 'unknown')),
                'page_number': chunk.meta.get('page_number', chunk.meta.get('page', 0))
            }
            start_char = chunk.meta.get('start_char')
            end_char = chunk.meta.get('end_char')
        elif hasattr(chunk, 'chunk_id'):
            # Regular Chunk object
            if not hasattr(self, '_logged_branch_chunk'):
                self._logged_branch_chunk = True

            chunk_id = chunk.chunk_id
            content = chunk.content
            chunk_index = chunk.chunk_index if hasattr(chunk, 'chunk_index') else 0
            total_chunks = chunk.total_chunks if hasattr(chunk, 'total_chunks') else 1
            source_metadata = chunk.source_metadata if hasattr(chunk, 'source_metadata') else {}
            start_char = chunk.start_char if hasattr(chunk, 'start_char') else None
            end_char = chunk.end_char if hasattr(chunk, 'end_char') else None
        else:
            raise ValueError("Unsupported chunk type - must have either chunk_id or meta attribute")

        # Create payload with metadata
        payload = {
            "chunk_id": chunk_id,
            "content": content,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "source_filename": source_metadata.get("filename", ""),
            "source_filepath": source_metadata.get("filepath", ""),
            "file_type": source_metadata.get("file_type", ""),
            "page_number": source_metadata.get("page_number"),
            "model_name": chunk_embedding.model_name,
            "embedding_dim": chunk_embedding.embedding_dim,
        }

        # Add optional metadata
        if start_char is not None:
            payload["start_char"] = start_char
        if end_char is not None:
            payload["end_char"] = end_char

        # CRITICAL: Add file_uuid for deletion support
        if hasattr(chunk, 'meta'):
            file_uuid = chunk.meta.get('file_uuid') or chunk.meta.get('file_id')
            if file_uuid:
                payload["file_uuid"] = file_uuid
                payload["file_id"] = file_uuid  # Store both for compatibility

            # CRITICAL: Copy ALL additional metadata fields from chunk.meta
            # This ensures entity extraction, email metadata, and any future fields are preserved
            excluded_keys = {
                'chunk_id', 'content', 'chunk_index', 'total_chunks',
                'source_filename', 'source_filepath', 'source', 'file_path',
                'file_type', 'chunker', 'page_number', 'page',
                'start_char', 'end_char', 'file_uuid', 'file_id'
            }
            for key, value in chunk.meta.items():
                if key not in excluded_keys and key not in payload:
                    # Only add if not already in payload and not in excluded list
                    payload[key] = value

        # Generate unique ID (use chunk_id as base)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

        return PointStruct(
            id=point_id,
            vector=chunk_embedding.embedding.tolist(),
            payload=payload
        )

    def search(
            self,
            query_vector: List[float],
            limit: int = 10,
            score_threshold: Optional[float] = None,
            filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            limit: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        logger.debug(f"Searching for {limit} similar vectors")

        # Build filter if provided
        query_filter = None
        if filter_conditions:
            query_filter = self._build_filter(filter_conditions)

        # Execute search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )

        # Format results
        results = []
        for scored_point in search_result:
            result = {
                "id": scored_point.id,
                "score": scored_point.score,
                "chunk_id": scored_point.payload.get("chunk_id"),
                "content": scored_point.payload.get("content"),
                "metadata": scored_point.payload
            }
            results.append(result)

        logger.debug(f"Found {len(results)} results")
        return results

    @staticmethod
    def _build_filter(conditions: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from conditions
        
        Args:
            conditions: Dictionary of field: value conditions
            
        Returns:
            Qdrant Filter object
        """
        field_conditions = []

        for field, value in conditions.items():
            field_conditions.append(
                FieldCondition(
                    key=field,
                    match=MatchValue(value=value)
                )
            )

        return Filter(must=field_conditions)

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)

            return {
                "collection_name": self.collection_name,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name,
                "status": collection_info.status.name,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}

    def delete_collection(self):
        """Delete the collection"""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.info("Collection deleted")

    def delete_by_file_uuid(self, file_uuid: str) -> Dict[str, Any]:
        """
        Delete all vectors belonging to a specific file UUID.
        
        Args:
            file_uuid: The UUID of the file to delete
            
        Returns:
            Dictionary with deletion statistics
        """
        try:
            logger.info(f"Deleting vectors for file UUID: {file_uuid}")

            # First, count how many points will be deleted
            points_before = self.count_points()

            # Delete points matching the file_uuid in metadata
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="file_uuid",
                            match=MatchValue(value=file_uuid)
                        )
                    ]
                )
            )

            # Count remaining points
            points_after = self.count_points()
            deleted_count = points_before - points_after

            logger.info(f"Deleted {deleted_count} vectors for file UUID: {file_uuid}")

            return {
                "success": True,
                "file_uuid": file_uuid,
                "deleted_count": deleted_count
            }

        except Exception as e:
            logger.error(f"Failed to delete file UUID {file_uuid}: {e}")
            return {
                "success": False,
                "file_uuid": file_uuid,
                "error": str(e)
            }

    def delete_by_ingest_job_id(self, ingest_job_id: str) -> Dict[str, Any]:
        """
        Delete all vectors that belong to a specific ingest job.
        """
        try:
            logger.info(f"Deleting vectors for ingest job: {ingest_job_id}")
            points_before = self.count_points()
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="ingest_job_id",
                            match=MatchValue(value=ingest_job_id),
                        )
                    ]
                ),
            )
            points_after = self.count_points()
            deleted_count = points_before - points_after
            return {
                "success": True,
                "ingest_job_id": ingest_job_id,
                "deleted_count": deleted_count,
            }
        except Exception as e:
            logger.error(f"Failed to delete ingest job {ingest_job_id}: {e}")
            return {
                "success": False,
                "ingest_job_id": ingest_job_id,
                "error": str(e),
            }

    def _scroll_point_ids(
            self,
            filter_obj: Optional[Filter],
            batch_size: int = 1000
    ) -> List[Any]:
        """
        Scroll point IDs for a filter. Uses pagination to avoid large memory spikes.
        """
        point_ids: List[Any] = []
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=batch_size,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                break

            point_ids.extend(point.id for point in points)
            if next_offset is None:
                break
            offset = next_offset

        return point_ids

    def _set_payload_for_filter(
            self,
            payload: Dict[str, Any],
            filter_obj: Filter,
            batch_size: int = 1000
    ) -> int:
        """
        Update payload for points matching a filter by scrolling IDs then setting payload in batches.
        """
        point_ids = self._scroll_point_ids(filter_obj=filter_obj, batch_size=batch_size)
        if not point_ids:
            return 0

        updated = 0
        for i in range(0, len(point_ids), batch_size):
            batch_ids = point_ids[i:i + batch_size]
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=batch_ids,
            )
            updated += len(batch_ids)
        return updated

    def commit_ingest_job(
            self,
            file_uuid: str,
            ingest_job_id: str,
            ingest_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Promote staged chunks for an ingest job to visible/ready state.
        """
        try:
            now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            commit_payload = {
                "ingest_state": "ready",
                "indexed_at": now_iso,
            }
            if ingest_version:
                commit_payload["ingest_version"] = ingest_version

            target_filter = Filter(
                must=[
                    FieldCondition(key="file_uuid", match=MatchValue(value=file_uuid)),
                    FieldCondition(key="ingest_job_id", match=MatchValue(value=ingest_job_id)),
                    FieldCondition(key="ingest_state", match=MatchValue(value="staging")),
                ]
            )
            committed = self._set_payload_for_filter(payload=commit_payload, filter_obj=target_filter)

            return {
                "success": True,
                "file_uuid": file_uuid,
                "ingest_job_id": ingest_job_id,
                "committed_count": committed,
            }
        except Exception as e:
            logger.error(f"Failed to commit ingest job {ingest_job_id}: {e}")
            return {
                "success": False,
                "file_uuid": file_uuid,
                "ingest_job_id": ingest_job_id,
                "error": str(e),
            }

    def delete_file_except_ingest_job(self, file_uuid: str, keep_ingest_job_id: str) -> Dict[str, Any]:
        """
        Delete all chunks for a file except the latest committed ingest job.
        """
        try:
            logger.info(
                "Deleting stale vectors for file_uuid=%s (keeping ingest_job_id=%s)",
                file_uuid,
                keep_ingest_job_id,
            )
            points_before = self.count_points()
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="file_uuid", match=MatchValue(value=file_uuid)),
                    ],
                    must_not=[
                        FieldCondition(key="ingest_job_id", match=MatchValue(value=keep_ingest_job_id)),
                    ],
                ),
            )
            points_after = self.count_points()
            deleted_count = points_before - points_after
            return {
                "success": True,
                "file_uuid": file_uuid,
                "keep_ingest_job_id": keep_ingest_job_id,
                "deleted_count": deleted_count,
            }
        except Exception as e:
            logger.error(f"Failed stale cleanup for file_uuid={file_uuid}: {e}")
            return {
                "success": False,
                "file_uuid": file_uuid,
                "keep_ingest_job_id": keep_ingest_job_id,
                "error": str(e),
            }

    def backfill_ingest_state_ready(self, batch_size: int = 1000, max_points: Optional[int] = None) -> Dict[str, Any]:
        """
        Best-effort one-time backfill: ensure legacy points have ingest_state=ready.
        """
        try:
            updated = 0
            offset = None
            now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            while True:
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    break

                missing_ids = [
                    point.id
                    for point in points
                    if not isinstance(point.payload, dict) or "ingest_state" not in point.payload
                ]
                if missing_ids:
                    self.client.set_payload(
                        collection_name=self.collection_name,
                        payload={"ingest_state": "ready", "indexed_at": now_iso},
                        points=missing_ids,
                    )
                    updated += len(missing_ids)

                if max_points is not None and updated >= max_points:
                    break
                if next_offset is None:
                    break
                offset = next_offset

            return {"success": True, "updated_points": updated}
        except Exception as e:
            logger.warning(f"Ingest-state backfill failed (non-fatal): {e}")
            return {"success": False, "updated_points": 0, "error": str(e)}

    def has_missing_payload_key(
            self,
            payload_key: str,
            sample_size: int = 5000,
            batch_size: int = 1000,
    ) -> bool:
        """
        Sample points to detect if a payload key is missing.

        This is a lightweight heuristic to decide whether a full backfill is needed.
        """
        try:
            checked = 0
            offset = None
            while checked < sample_size:
                limit = min(batch_size, sample_size - checked)
                points, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if not points:
                    return False
                for point in points:
                    checked += 1
                    payload = point.payload if isinstance(point.payload, dict) else {}
                    if payload_key not in payload:
                        return True
                if next_offset is None:
                    break
                offset = next_offset
            return False
        except Exception as e:
            logger.warning("Payload-key sample check failed (%s): %s", payload_key, e)
            # Conservative fallback: assume missing to avoid skipping required backfill.
            return True

    def delete_by_filename(self, filename: str) -> Dict[str, Any]:
        """
        Delete all vectors belonging to a specific filename.
        
        Args:
            filename: The filename to delete
            
        Returns:
            Dictionary with deletion statistics
        """
        try:
            logger.info(f"Deleting vectors for filename: {filename}")

            points_before = self.count_points()

            # Delete points matching the source_filename in metadata
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_filename",
                            match=MatchValue(value=filename)
                        )
                    ]
                )
            )

            points_after = self.count_points()
            deleted_count = points_before - points_after

            logger.info(f"Deleted {deleted_count} vectors for filename: {filename}")

            return {
                "success": True,
                "filename": filename,
                "deleted_count": deleted_count
            }

        except Exception as e:
            logger.error(f"Failed to delete filename {filename}: {e}")
            return {
                "success": False,
                "filename": filename,
                "error": str(e)
            }

    def get_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific point by ID

        Args:
            point_id: Point ID 
            
        Returns:
            Point data or None if not found
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )

            if points:
                point = points[0]
                return {
                    "id": point.id,
                    "payload": point.payload,
                    "vector": point.vector
                }
            return None

        except Exception as e:
            logger.error(f"Failed to get point: {e}")
            return None

    def count_points(self) -> int:
        """
        Get total number of points in collection
        
        Returns:
            Number of points
        """
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

    def scroll_points(
            self,
            limit: int = 100,
            offset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scroll through points in the collection
        
        Args:
            limit: Number of points to retrieve
            offset: Offset for pagination
            
        Returns:
            Dictionary with points and next offset
        """
        result = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            offset=offset
        )

        points, next_offset = result

        return {
            "points": [
                {
                    "id": p.id,
                    "payload": p.payload
                }
                for p in points
            ],
            "next_offset": next_offset
        }


# Convenience functions
def create_vector_store(
        collection_name: str = "knowledge_base",
        # embedding_dim: int = 768,
        embedding_dim: int = 1024,  # Updated default for BAAI/bge-large-en-v1.5
        qdrant_url: str = "http://localhost:6333",
        recreate: bool = False
) -> VectorStore:
    """
    Convenience function to create a vector store
    
    Args:
        collection_name: Name of the collection
        embedding_dim: Embedding dimension
        qdrant_url: Qdrant server URL
        recreate: Whether to recreate collection
        
    Returns:
        VectorStore instance
    """
    return VectorStore(
        collection_name=collection_name,
        embedding_dim=embedding_dim,
        qdrant_url=qdrant_url,
        recreate_collection=recreate
    )
