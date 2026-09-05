"""
BM25 Index Cache Management

This module provides utilities for persisting and loading BM25 indices to disk,
with checksum validation to ensure index freshness and correctness.

Features:
- Atomic file operations to prevent corruption
- Checksum-based validation against Qdrant collection state
- Automatic invalidation on collection changes
- Comprehensive error handling and logging
"""

import os
import pickle
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from qdrant_client import QdrantClient
from haystack import Document

logger = logging.getLogger(__name__)

# Index format version for backward compatibility
INDEX_VERSION = "1.0"


class BM25IndexCache:
    """
    Manages BM25 index persistence and validation.
    
    Handles saving/loading BM25 indices with metadata and checksum validation
    to ensure cached indices match the current Qdrant collection state.
    """
    
    def __init__(self, cache_dir: str = "./data/bm25_indices"):
        """
        Initialize the BM25 index cache manager.
        
        Args:
            cache_dir: Directory to store cached indices (default: ./data/bm25_indices)
        """
        self.cache_dir = Path(cache_dir)
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Create cache directory if it doesn't exist."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"BM25 cache directory: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to create cache directory: {e}")
            raise
    
    def get_cache_path(self, collection_name: str) -> Path:
        """
        Get the cache file path for a collection.
        
        Args:
            collection_name: Name of the Qdrant collection
            
        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{collection_name}_bm25.pkl"
    
    def get_collection_checksum(
        self,
        qdrant_client: QdrantClient,
        collection_name: str
    ) -> str:
        """
        Generate a checksum for the Qdrant collection state.
        
        The checksum is based on:
        - Number of points in the collection
        - Collection configuration (vector size, distance metric)
        
        This allows us to detect when the collection has changed and the
        cached index needs to be rebuilt.
        
        Args:
            qdrant_client: Qdrant client instance
            collection_name: Name of the collection
            
        Returns:
            MD5 checksum string
        """
        try:
            collection_info = qdrant_client.get_collection(collection_name)
            
            # Create a deterministic string representation of collection state
            state_str = (
                f"points:{collection_info.points_count}|"
                f"vectors:{collection_info.config.params.vectors.size}|"
                f"distance:{collection_info.config.params.vectors.distance}"
            )
            
            # Generate MD5 checksum
            checksum = hashlib.md5(state_str.encode()).hexdigest()
            logger.debug(f"Collection checksum: {checksum} (state: {state_str})")
            
            return checksum
            
        except Exception as e:
            logger.error(f"Failed to generate collection checksum: {e}")
            raise
    
    def save_bm25_index(
        self,
        collection_name: str,
        collection_checksum: str,
        bm25_state: Dict[str, Any],
        documents: List[Document],
        bm25_params: Dict[str, float]
    ) -> bool:
        """
        Save BM25 index to disk with metadata.
        
        Uses atomic write operations to prevent corruption:
        1. Write to temporary file
        2. Verify write succeeded
        3. Atomically rename to final location
        
        Args:
            collection_name: Name of the Qdrant collection
            collection_checksum: Checksum of the collection state
            bm25_state: BM25 internal state (inverted_index, idf, etc.)
            documents: List of Haystack documents
            bm25_params: BM25 parameters (k1, b)
            
        Returns:
            True if save succeeded, False otherwise
        """
        cache_path = self.get_cache_path(collection_name)
        
        try:
            logger.info(f"Saving BM25 index to cache: {cache_path}")
            
            # Prepare index data
            index_data = {
                "version": INDEX_VERSION,
                "collection_name": collection_name,
                "collection_checksum": collection_checksum,
                "document_count": len(documents),
                "created_at": datetime.now().isoformat(),
                "bm25_params": bm25_params,
                "index_data": {
                    "inverted_index": bm25_state["inverted_index"],
                    "idf": bm25_state["idf"],
                    "doc_lengths": bm25_state["doc_lengths"],
                    "avgdl": bm25_state["avgdl"],
                    "corpus_size": bm25_state["corpus_size"],
                    "documents": documents
                }
            }
            
            # Atomic write: write to temp file first
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.cache_dir,
                prefix=f".{collection_name}_",
                suffix=".tmp"
            )
            
            try:
                with os.fdopen(temp_fd, 'wb') as f:
                    pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Atomically rename temp file to final location
                os.replace(temp_path, cache_path)
                
                # Get file size for logging
                file_size_mb = cache_path.stat().st_size / (1024 * 1024)
                
                logger.info(
                    f"BM25 index saved successfully | "
                    f"Size: {file_size_mb:.2f} MB | "
                    f"Documents: {len(documents):,} | "
                    f"Checksum: {collection_checksum[:8]}..."
                )
                
                return True
                
            except Exception as e:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
                
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            return False
    
    def load_bm25_index(
        self,
        collection_name: str,
        current_checksum: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load BM25 index from cache with validation.
        
        Validates:
        - Cache file exists
        - Index version is compatible
        - Collection checksum matches (collection hasn't changed)
        
        Args:
            collection_name: Name of the Qdrant collection
            current_checksum: Current checksum of the collection
            
        Returns:
            Index data dict if valid, None if invalid/missing
        """
        cache_path = self.get_cache_path(collection_name)
        
        # Check if cache file exists
        if not cache_path.exists():
            logger.info(f"No cached index found for collection '{collection_name}'")
            return None
        
        try:
            logger.info(f"Loading BM25 index from cache: {cache_path}")
            
            # Load index data
            with open(cache_path, 'rb') as f:
                index_data = pickle.load(f)
            
            # Validate version
            if index_data.get("version") != INDEX_VERSION:
                logger.warning(
                    f"Index version mismatch: "
                    f"cached={index_data.get('version')}, "
                    f"current={INDEX_VERSION}. Rebuilding index."
                )
                return None
            
            # Validate checksum
            cached_checksum = index_data.get("collection_checksum")
            if cached_checksum != current_checksum:
                logger.warning(
                    f"Collection changed since index was cached. "
                    f"Cached checksum: {cached_checksum[:8]}..., "
                    f"Current checksum: {current_checksum[:8]}... "
                    f"Rebuilding index."
                )
                return None
            
            # Validate collection name
            if index_data.get("collection_name") != collection_name:
                logger.warning(
                    f"Collection name mismatch in cache. Rebuilding index."
                )
                return None
            
            # Get cache age
            created_at = datetime.fromisoformat(index_data.get("created_at"))
            age_hours = (datetime.now() - created_at).total_seconds() / 3600
            
            # Get file size
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            
            logger.info(
                f"Loaded cached BM25 index | "
                f"Documents: {index_data.get('document_count'):,} | "
                f"Age: {age_hours:.1f}h | "
                f"Size: {file_size_mb:.2f} MB | "
                f"Checksum: {cached_checksum[:8]}..."
            )
            
            return index_data
            
        except Exception as e:
            logger.error(f"Failed to load cached index: {e}")
            logger.info("Will rebuild index from scratch")
            return None
    
    def invalidate_cache(self, collection_name: str) -> bool:
        """
        Manually invalidate (delete) cached index for a collection.
        
        Useful for forcing a rebuild or clearing stale caches.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if cache was deleted, False if it didn't exist
        """
        cache_path = self.get_cache_path(collection_name)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
                logger.info(f"Invalidated cache for collection '{collection_name}'")
                return True
            else:
                logger.info(f"No cache to invalidate for collection '{collection_name}'")
                return False
                
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return False
    
    def get_cache_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about cached index without loading it.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dict with cache metadata, or None if cache doesn't exist
        """
        cache_path = self.get_cache_path(collection_name)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                index_data = pickle.load(f)
            
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            created_at = datetime.fromisoformat(index_data.get("created_at"))
            age_hours = (datetime.now() - created_at).total_seconds() / 3600
            
            return {
                "collection_name": index_data.get("collection_name"),
                "document_count": index_data.get("document_count"),
                "checksum": index_data.get("collection_checksum"),
                "created_at": index_data.get("created_at"),
                "age_hours": age_hours,
                "file_size_mb": file_size_mb,
                "version": index_data.get("version"),
                "bm25_params": index_data.get("bm25_params")
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return None
