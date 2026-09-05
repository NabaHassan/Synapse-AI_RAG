"""
Query Cache Manager for RAG Pipeline.

This module provides intelligent caching of query responses to reduce LLM inference costs.
It caches standalone queries (e.g., "What is CARVE?") but NOT follow-up queries
(e.g., "Tell me more", "What else?").

Key Features:
- Query normalization for robust matching
- Semantic similarity checking to prevent false cache hits
- Thread-safe operations for concurrent access
- JSON-based persistence with metadata
- Cache statistics and monitoring
- LRU eviction policy with configurable size limits

Usage:
    config = CacheConfig(
        cache_file="./data/query_cache.json",
        max_cache_size=1000,
        similarity_threshold=0.95
    )
    cache_manager = QueryCacheManager(config)
    
    # Check for cached response
    cached = cache_manager.get_cached_response("What is CARVE?")
    if cached:
        return cached.answer
    
    # Store new response
    cache_manager.store_response(
        query="What is CARVE?",
        answer="CARVE is...",
        citations=[...],
        entities=["CARVE"],
        confidence=0.95
    )
"""

import re
import json
import time
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple

from src.concurrency import (
    RedisBackendUnavailable,
    RedisConnection,
    RedisQueryCacheStore,
    RedisRuntimeConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CacheConfig:
    """
    Configuration for query cache manager.
    
    Attributes:
        cache_file: Path to JSON file for cache persistence
        max_cache_size: Maximum number of cached entries (LRU eviction)
        ttl_hours: Time-to-live for cache entries in hours (0 = no expiry)
        similarity_threshold: Semantic similarity threshold for matching (0.0-1.0)
        enable_semantic_matching: Whether to use semantic similarity (requires embedder)
        min_query_length: Minimum query length to cache (avoid caching trivial queries)
        max_query_length: Maximum query length to cache
    """
    cache_file: str = "./data/query_cache.json"
    max_cache_size: int = 1000
    ttl_hours: int = 168  # 7 days
    similarity_threshold: float = 0.95
    enable_semantic_matching: bool = True
    min_query_length: int = 5  # Minimum query length to cache (avoid trivial queries)
    max_query_length: int = 500
    # Redis backend settings (Phase 1). When enabled, Redis is treated as
    # source-of-truth for cache entries across workers.
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "synapse"
    redis_socket_timeout_seconds: float = 0.2
    redis_connect_timeout_seconds: float = 0.2
    # If 0, defaults to ttl_hours * 3600.
    redis_ttl_seconds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class CachedResponse:
    """
    Cached query response with metadata.
    
    Attributes:
        normalized_key: Normalized query string used as cache key
        original_query: Original query text as submitted
        answer: Generated answer text
        citations: List of citation dictionaries
        entities: Extracted entities from the response
        response_summary: Compact summary for conversation memory
        confidence: Confidence score of the response
        timestamp: When the response was first cached
        hit_count: Number of times this cache entry was accessed
        last_accessed: Timestamp of last cache access
        source_documents: Source documents used (for reuse)
        metadata: Additional metadata
    """
    normalized_key: str
    original_query: str
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    response_summary: str = ""
    confidence: float = 0.0
    timestamp: str = ""
    hit_count: int = 0
    last_accessed: str = ""
    source_documents: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedResponse":
        """Create from dictionary."""
        return cls(**data)

    def is_expired(self, ttl_hours: int) -> bool:
        """
        Check if cache entry has expired.
        
        Args:
            ttl_hours: Time-to-live in hours (0 = never expires)
            
        Returns:
            True if expired, False otherwise
        """
        if ttl_hours <= 0:
            return False
        
        try:
            cached_time = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            expiry_time = cached_time + timedelta(hours=ttl_hours)
            return datetime.now(cached_time.tzinfo) > expiry_time
        except (ValueError, AttributeError):
            # If timestamp is invalid, consider it expired
            return True


# =============================================================================
# Main QueryCacheManager Class
# =============================================================================

class QueryCacheManager:
    """
    Manages query response caching for the RAG pipeline.
    
    Provides thread-safe caching with query normalization, semantic similarity
    checking, and LRU eviction. Persists cache to JSON file.
    
    Thread-safe: All mutating operations are protected by a lock.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embedder=None  # Optional embedder for semantic similarity
    ):
        """
        Initialize query cache manager.
        
        Args:
            config: Cache configuration (uses defaults if None)
            embedder: Optional embedder for semantic similarity matching
        """
        self.config = config or CacheConfig()
        self.embedder = embedder
        self._lock = threading.RLock()

        self._redis_connection: Optional[RedisConnection] = None
        self._redis_store: Optional[RedisQueryCacheStore] = None
        self._redis_failures = 0

        if self.config.redis_enabled:
            redis_runtime = RedisRuntimeConfig(
                enabled=True,
                url=self.config.redis_url,
                key_prefix=self.config.redis_key_prefix,
                socket_timeout_seconds=self.config.redis_socket_timeout_seconds,
                connect_timeout_seconds=self.config.redis_connect_timeout_seconds,
            )
            self._redis_connection = RedisConnection(redis_runtime)
            self._redis_store = RedisQueryCacheStore(
                connection=self._redis_connection,
                default_ttl_seconds=self._effective_redis_ttl_seconds(),
                max_entries=self.config.max_cache_size,
            )
            if self._redis_connection.init_error:
                logger.warning(
                    "Query cache Redis backend initialization issue (%s). "
                    "Cache reads will fail-open as misses until backend recovers.",
                    self._redis_connection.init_error,
                )
            else:
                logger.info("QueryCacheManager initialized with Redis backend")
        
        # Cache storage: normalized_key -> CachedResponse
        self._cache: Dict[str, CachedResponse] = {}
        
        # Statistics
        self._stats = {
            "hit_count": 0,
            "miss_count": 0,
            "store_count": 0,
            "eviction_count": 0,
            "load_time": 0.0,
        }
        
        if not self.config.redis_enabled:
            # Ensure cache directory exists
            cache_path = Path(self.config.cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing cache
            self._load_cache()
        
        logger.info("QueryCacheManager initialized")
        logger.info(
            "  Cache backend: %s",
            "redis" if self.config.redis_enabled else f"json:{self.config.cache_file}",
        )
        logger.info(f"  Max size: {self.config.max_cache_size}")
        logger.info(f"  TTL: {self.config.ttl_hours} hours")
        logger.info(f"  Similarity threshold: {self.config.similarity_threshold}")
        if not self.config.redis_enabled:
            logger.info(f"  Loaded entries: {len(self._cache)}")

    # =========================================================================
    # Query Normalization
    # =========================================================================

    @staticmethod
    def normalize_query(query: str) -> str:
        """
        Normalize query for cache key generation.
        
        Applies conservative normalization to match semantically identical queries
        while avoiding false positives.
        
        Normalization steps:
        1. Convert to lowercase
        2. Remove extra whitespace
        3. Remove trailing punctuation (?, !, .)
        4. Preserve internal punctuation and structure
        
        Args:
            query: Original query text
            
        Returns:
            Normalized query string
            
        Examples:
            "What is CARVE?" -> "what is carve"
            "what is carve" -> "what is carve"
            "What is CARVE ?" -> "what is carve"
            "WHAT IS CARVE??" -> "what is carve"
        """
        if not query:
            return ""
        
        # Convert to lowercase
        normalized = query.lower()
        
        # Normalize whitespace (collapse multiple spaces)
        normalized = ' '.join(normalized.split())
        
        # Remove trailing punctuation only (preserve internal punctuation)
        normalized = normalized.rstrip('?!.,;:')
        
        # Final whitespace cleanup
        normalized = normalized.strip()
        
        return normalized

    def compute_query_hash(self, query: str) -> str:
        """
        Compute hash of normalized query for fast lookup.
        
        Args:
            query: Query text
            
        Returns:
            MD5 hash of normalized query
        """
        normalized = self.normalize_query(query)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def _effective_redis_ttl_seconds(self) -> int:
        if self.config.redis_ttl_seconds > 0:
            return int(self.config.redis_ttl_seconds)
        if self.config.ttl_hours <= 0:
            # Keep a long but bounded TTL when cache is configured as non-expiring.
            return 30 * 24 * 3600
        return int(self.config.ttl_hours * 3600)

    def _record_redis_failure(self, message: str) -> None:
        self._redis_failures += 1
        logger.warning("%s (total_failures=%s)", message, self._redis_failures)

    # =========================================================================
    # Semantic Similarity
    # =========================================================================

    def compute_semantic_similarity(self, query1: str, query2: str) -> float:
        """
        Compute semantic similarity between two queries.
        
        Uses embedder if available, otherwise falls back to exact match.
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not self.embedder or not self.config.enable_semantic_matching:
            # Fallback: exact match on normalized queries
            norm1 = self.normalize_query(query1)
            norm2 = self.normalize_query(query2)
            return 1.0 if norm1 == norm2 else 0.0
        
        try:
            # Get embeddings using Haystack's embedder interface
            # Haystack's SentenceTransformersTextEmbedder uses run() method
            if hasattr(self.embedder, 'run'):
                # Haystack embedder
                result1 = self.embedder.run(text=query1)
                result2 = self.embedder.run(text=query2)
                emb1 = result1['embedding']
                emb2 = result2['embedding']
            elif hasattr(self.embedder, 'embed_query'):
                # LangChain-style embedder
                emb1 = self.embedder.embed_query(query1)
                emb2 = self.embedder.embed_query(query2)
            else:
                raise AttributeError("Embedder does not have run() or embed_query() method")
            
            # Compute cosine similarity
            import numpy as np
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity computation failed: {e}")
            # Fallback to exact match
            norm1 = self.normalize_query(query1)
            norm2 = self.normalize_query(query2)
            return 1.0 if norm1 == norm2 else 0.0

    # =========================================================================
    # Cache Operations
    # =========================================================================

    def get_cached_response(self, query: str) -> Optional[CachedResponse]:
        """
        Retrieve cached response for a query.
        
        Performs normalization and optional semantic similarity checking.
        Updates access statistics and last_accessed timestamp on hit.
        
        Args:
            query: Query text to lookup
            
        Returns:
            CachedResponse if found and valid, None otherwise
        """
        if self.config.redis_enabled:
            return self._get_cached_response_redis(query)

        with self._lock:
            return self._get_cached_response_local(query)

    def _get_cached_response_local(self, query: str) -> Optional[CachedResponse]:
        normalized = self.normalize_query(query)

        # Check if query is too short or too long
        if len(query) < self.config.min_query_length:
            logger.debug(f"Query too short to cache: '{query}'")
            return None

        if len(query) > self.config.max_query_length:
            logger.debug(f"Query too long to cache: '{query}'")
            return None

        # Direct lookup by normalized key
        if normalized in self._cache:
            cached = self._cache[normalized]

            # Check if expired
            if cached.is_expired(self.config.ttl_hours):
                logger.info(f"Cache entry expired: '{query}'")
                del self._cache[normalized]
                self._stats["miss_count"] += 1
                return None

            # Check semantic similarity if enabled
            if self.config.enable_semantic_matching and self.embedder:
                similarity = self.compute_semantic_similarity(query, cached.original_query)
                if similarity < self.config.similarity_threshold:
                    logger.debug(
                        f"Semantic similarity too low ({similarity:.3f} < {self.config.similarity_threshold}): "
                        f"'{query}' vs '{cached.original_query}'"
                    )
                    self._stats["miss_count"] += 1
                    return None

            # Cache hit!
            cached.hit_count += 1
            from datetime import timezone
            cached.last_accessed = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            self._stats["hit_count"] += 1

            logger.info(f"✓ Cache HIT: '{query}' (hit_count: {cached.hit_count})")

            return cached

        # Cache miss
        self._stats["miss_count"] += 1
        logger.debug(f"✗ Cache MISS: '{query}'")
        return None

    def _get_cached_response_redis(self, query: str) -> Optional[CachedResponse]:
        with self._lock:
            normalized = self.normalize_query(query)

            if len(query) < self.config.min_query_length:
                logger.debug(f"Query too short to cache: '{query}'")
                return None

            if len(query) > self.config.max_query_length:
                logger.debug(f"Query too long to cache: '{query}'")
                return None

            if self._redis_store is None:
                self._stats["miss_count"] += 1
                return None

            try:
                entry = self._redis_store.get_entry(normalized)
            except RedisBackendUnavailable as exc:
                self._record_redis_failure(f"Redis cache read failed-open as miss: {exc}")
                self._stats["miss_count"] += 1
                return None

            if not entry:
                self._stats["miss_count"] += 1
                logger.debug(f"✗ Cache MISS (redis): '{query}'")
                return None

            try:
                cached = CachedResponse.from_dict(entry)
            except Exception:
                # Corrupt entry: fail-open as miss.
                self._stats["miss_count"] += 1
                logger.warning("Invalid Redis cache payload for key '%s'", normalized)
                return None

            if cached.is_expired(self.config.ttl_hours):
                self._stats["miss_count"] += 1
                try:
                    self._redis_store.delete_entry(normalized)
                except RedisBackendUnavailable:
                    pass
                return None

            if self.config.enable_semantic_matching and self.embedder:
                similarity = self.compute_semantic_similarity(query, cached.original_query)
                if similarity < self.config.similarity_threshold:
                    self._stats["miss_count"] += 1
                    return None

            from datetime import timezone
            cached.hit_count += 1
            cached.last_accessed = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            self._stats["hit_count"] += 1

            try:
                self._redis_store.set_entry(
                    normalized,
                    cached.to_dict(),
                    ttl_seconds=self._effective_redis_ttl_seconds(),
                )
                self._redis_store.touch(normalized)
            except RedisBackendUnavailable as exc:
                self._record_redis_failure(f"Redis cache touch failed: {exc}")

            logger.info(f"✓ Cache HIT (redis): '{query}' (hit_count: {cached.hit_count})")
            return cached

    def store_response(
        self,
        query: str,
        answer: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        entities: Optional[List[str]] = None,
        response_summary: Optional[str] = None,
        confidence: float = 0.0,
        source_documents: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a query response in the cache.
        
        Args:
            query: Original query text
            answer: Generated answer
            citations: List of citation dictionaries
            entities: Extracted entities
            response_summary: Compact summary for memory
            confidence: Confidence score
            source_documents: Source documents used
            metadata: Additional metadata
            
        Returns:
            True if stored successfully, False otherwise
        """
        if self.config.redis_enabled:
            return self._store_response_redis(
                query=query,
                answer=answer,
                citations=citations,
                entities=entities,
                response_summary=response_summary,
                confidence=confidence,
                source_documents=source_documents,
                metadata=metadata,
            )

        with self._lock:
            return self._store_response_local(
                query=query,
                answer=answer,
                citations=citations,
                entities=entities,
                response_summary=response_summary,
                confidence=confidence,
                source_documents=source_documents,
                metadata=metadata,
            )

    def _store_response_local(
        self,
        *,
        query: str,
        answer: str,
        citations: Optional[List[Dict[str, Any]]],
        entities: Optional[List[str]],
        response_summary: Optional[str],
        confidence: float,
        source_documents: Optional[List[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]],
    ) -> bool:
        # Check query length
        if len(query) < self.config.min_query_length:
            logger.debug(f"Query too short to cache: '{query}'")
            return False

        if len(query) > self.config.max_query_length:
            logger.debug(f"Query too long to cache: '{query}'")
            return False

        # Normalize query
        normalized = self.normalize_query(query)

        # Check if already cached
        if normalized in self._cache:
            logger.debug(f"Query already cached: '{query}'")
            return False

        # Create cached response
        from datetime import timezone
        timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        cached = CachedResponse(
            normalized_key=normalized,
            original_query=query,
            answer=answer,
            citations=citations or [],
            entities=entities or [],
            response_summary=response_summary or "",
            confidence=confidence,
            timestamp=timestamp,
            hit_count=0,
            last_accessed=timestamp,
            source_documents=source_documents or [],
            metadata=metadata or {},
        )

        # Check if cache is full (LRU eviction)
        if len(self._cache) >= self.config.max_cache_size:
            self._evict_lru()

        # Store in cache
        self._cache[normalized] = cached
        self._stats["store_count"] += 1

        logger.info(f"✓ Cached response for: '{query}'")

        # Persist to disk
        self._save_cache()

        return True

    def _store_response_redis(
        self,
        *,
        query: str,
        answer: str,
        citations: Optional[List[Dict[str, Any]]],
        entities: Optional[List[str]],
        response_summary: Optional[str],
        confidence: float,
        source_documents: Optional[List[Dict[str, Any]]],
        metadata: Optional[Dict[str, Any]],
    ) -> bool:
        with self._lock:
            if len(query) < self.config.min_query_length:
                return False
            if len(query) > self.config.max_query_length:
                return False
            if self._redis_store is None:
                return False

            normalized = self.normalize_query(query)
            from datetime import timezone
            timestamp = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            cached = CachedResponse(
                normalized_key=normalized,
                original_query=query,
                answer=answer,
                citations=citations or [],
                entities=entities or [],
                response_summary=response_summary or "",
                confidence=confidence,
                timestamp=timestamp,
                hit_count=0,
                last_accessed=timestamp,
                source_documents=source_documents or [],
                metadata=metadata or {},
            )

            try:
                existing = self._redis_store.get_entry(normalized)
            except RedisBackendUnavailable as exc:
                self._record_redis_failure(f"Redis cache read-before-write failed: {exc}")
                return False

            if existing:
                return False

            try:
                self._redis_store.set_entry(
                    normalized,
                    cached.to_dict(),
                    ttl_seconds=self._effective_redis_ttl_seconds(),
                )
                self._stats["store_count"] += 1
                return True
            except RedisBackendUnavailable as exc:
                self._record_redis_failure(f"Redis cache write failed: {exc}")
                return False

    def _evict_lru(self) -> None:
        """
        Evict least recently used cache entry.
        
        Uses last_accessed timestamp to determine LRU entry.
        """
        if not self._cache:
            return
        
        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].last_accessed
        )
        
        evicted = self._cache.pop(lru_key)
        self._stats["eviction_count"] += 1
        
        logger.info(f"Evicted LRU cache entry: '{evicted.original_query}'")

    def invalidate_entry(self, query: str) -> bool:
        """
        Invalidate (remove) a specific cache entry.
        
        Args:
            query: Query to invalidate
            
        Returns:
            True if entry was found and removed, False otherwise
        """
        if self.config.redis_enabled:
            with self._lock:
                normalized = self.normalize_query(query)
                if self._redis_store is None:
                    return False
                try:
                    self._redis_store.delete_entry(normalized)
                    return True
                except RedisBackendUnavailable as exc:
                    self._record_redis_failure(f"Redis cache invalidate failed: {exc}")
                    return False

        with self._lock:
            normalized = self.normalize_query(query)

            if normalized in self._cache:
                del self._cache[normalized]
                logger.info(f"Invalidated cache entry: '{query}'")
                self._save_cache()
                return True

            return False

    def clear_cache(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of entries cleared
        """
        if self.config.redis_enabled:
            with self._lock:
                count = 0
                if self._redis_store is not None:
                    try:
                        count = self._redis_store.count()
                        self._redis_store.clear()
                    except RedisBackendUnavailable as exc:
                        self._record_redis_failure(f"Redis cache clear failed: {exc}")
                self._stats["hit_count"] = 0
                self._stats["miss_count"] = 0
                self._stats["store_count"] = 0
                self._stats["eviction_count"] = 0
                return count

        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats["hit_count"] = 0
            self._stats["miss_count"] = 0
            self._stats["store_count"] = 0
            self._stats["eviction_count"] = 0

            logger.info(f"Cleared cache ({count} entries)")
            self._save_cache()

            return count

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_cache(self) -> None:
        """Load cache from JSON file."""
        if self.config.redis_enabled:
            return

        cache_path = Path(self.config.cache_file)
        
        if not cache_path.exists():
            logger.info("No existing cache file found, starting with empty cache")
            return
        
        try:
            start_time = time.time()
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate format
            if not isinstance(data, dict) or "cache" not in data:
                logger.warning("Invalid cache file format, starting with empty cache")
                return
            
            # Load cache entries
            loaded_count = 0
            expired_count = 0
            
            for key, entry_data in data["cache"].items():
                try:
                    cached = CachedResponse.from_dict(entry_data)
                    
                    # Check if expired
                    if cached.is_expired(self.config.ttl_hours):
                        expired_count += 1
                        continue
                    
                    self._cache[key] = cached
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load cache entry: {e}")
            
            # Load statistics if available
            if "statistics" in data:
                self._stats.update(data["statistics"])
            
            load_time = time.time() - start_time
            self._stats["load_time"] = load_time
            
            logger.info(f"Loaded {loaded_count} cache entries in {load_time:.3f}s")
            if expired_count > 0:
                logger.info(f"Skipped {expired_count} expired entries")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cache file: {e}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")

    def _save_cache(self) -> None:
        """Save cache to JSON file."""
        if self.config.redis_enabled:
            return

        try:
            cache_path = Path(self.config.cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data
            from datetime import timezone
            data = {
                "version": "1.0",
                "saved_at": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "config": self.config.to_dict(),
                "cache": {
                    key: cached.to_dict()
                    for key, cached in self._cache.items()
                },
                "statistics": self._stats
            }
            
            # Write to file
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved cache to {cache_path}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._stats["hit_count"] + self._stats["miss_count"]
            hit_rate = (
                self._stats["hit_count"] / total_requests
                if total_requests > 0 else 0.0
            )

            cache_size = len(self._cache)
            if self.config.redis_enabled and self._redis_store is not None:
                try:
                    if hasattr(self._redis_store, "count"):
                        cache_size = self._redis_store.count()
                    else:
                        cache_size = 0
                except Exception:
                    cache_size = 0

            return {
                "cache_size": cache_size,
                "max_cache_size": self.config.max_cache_size,
                "hit_count": self._stats["hit_count"],
                "miss_count": self._stats["miss_count"],
                "hit_rate": round(hit_rate, 4),
                "store_count": self._stats["store_count"],
                "eviction_count": self._stats["eviction_count"],
                "total_requests": total_requests,
                "load_time": self._stats["load_time"],
                "redis_enabled": self.config.redis_enabled,
                "redis_failures": self._redis_failures,
                "config": self.config.to_dict()
            }

    def __repr__(self) -> str:
        """String representation."""
        size = len(self._cache)
        if self.config.redis_enabled and self._redis_store is not None:
            try:
                if hasattr(self._redis_store, "count"):
                    size = self._redis_store.count()
                else:
                    size = 0
            except Exception:
                size = 0
        return (
            f"QueryCacheManager(size={size}, "
            f"hits={self._stats['hit_count']}, "
            f"misses={self._stats['miss_count']})"
        )
