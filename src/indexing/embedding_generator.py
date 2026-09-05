"""
Embedding Generator for RAG Pipeline
Generates vector embeddings for document chunks using sentence-transformers.
"""

import os
import torch
import pickle
import logging
import hashlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional

# Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import our chunk structure
from src.indexing.semantic_chunker import Chunk

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkEmbedding:
    """Structure for a chunk with its embedding"""

    def __init__(
            self,
            chunk: Chunk,
            embedding: np.ndarray,
            model_name: str,
            embedding_dim: int
    ):
        self.chunk = chunk
        self.embedding = embedding
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without embedding array for serialization)"""
        return {
            "chunk_id": self.chunk.chunk_id,
            "chunk_content": self.chunk.content,
            "chunk_metadata": self.chunk.to_dict(),
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "embedding_norm": float(np.linalg.norm(self.embedding))
        }

    def __repr__(self) -> str:
        return f"ChunkEmbedding(chunk_id={self.chunk.chunk_id}, dim={self.embedding_dim}, model={self.model_name})"


class EmbeddingGenerator:
    """
    Generate embeddings for document chunks using sentence-transformers.
    Supports batch processing, caching, and normalization.
    """

    # Default model
    DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
    def __init__(
            self,
            model_name: str = DEFAULT_MODEL,
            batch_size: int = 32,
            normalize_embeddings: bool = True,
            show_progress: bool = True,
            device: Optional[str] = None,
            cache_dir: Optional[str] = None,
            num_workers: Optional[int] = None  # For CPU parallel processing
    ):
        """
        Initialize EmbeddingGenerator
        
        Args:
            model_name: Model name
            batch_size: Batch size for embedding generation
            normalize_embeddings: Whether to normalize embeddings
            show_progress: Whether to show progress bar
            device: Device to run on ('cpu', 'cuda', 'mps', or None for auto)
            cache_dir: Directory to cache embeddings
            num_workers: Number of workers for parallel processing (None = auto)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress = show_progress
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.num_workers = num_workers or min(4, os.cpu_count() or 1)

        # Determine device if not provided
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info("=" * 60)
        logger.info("Initializing EmbeddingGenerator")
        logger.info("=" * 60)
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - Batch size: {batch_size}")

        self._init_sentence_transformers()

        # Setup cache if enabled
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  - Cache directory: {self.cache_dir}")

        # Statistics
        self.stats = {
            "total_chunks_processed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batches_processed": 0
        }

    def _init_sentence_transformers(self):
        """Initialize SentenceTransformers backend."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not available. "
                "Install with: pip install sentence-transformers"
            )

        logger.info("Loading SentenceTransformer model...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize pool for multiprocessing if on CPU
        self.pool = None
        if self.device == "cpu" and self.num_workers > 1:
            try:
                logger.info(f"Starting multi-process pool with {self.num_workers} workers...")
                self.pool = self.model.start_multi_process_pool(target_devices=['cpu'] * self.num_workers)
                logger.info("Multi-process pool started")
            except Exception as e:
                logger.warning(f"Failed to start multi-process pool: {e}")
                self.pool = None

        logger.info(f"SentenceTransformer loaded (dim={self.embedding_dim})")

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'pool') and self.pool is not None:
            try:
                logger.info("Stopping multi-process pool...")
                self.model.stop_multi_process_pool(self.pool)
            except:
                pass

    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings (n_texts x embedding_dim)
        """
        if not texts:
            return np.array([])

        if self.pool is not None:
            # Use multi-processing on CPU
            embeddings = self.model.encode_multi_process(
                texts,
                self.pool,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings
            )
        else:
            # Standard encoding (GPU/MPS/Single CPU)
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        return embeddings

    def embed_chunk(self, chunk: Chunk, use_cache: bool = True) -> ChunkEmbedding:
        """Generate embedding for a single chunk."""
        # Check cache
        if use_cache and self.cache_dir:
            cached_embedding = self._load_from_cache(chunk.chunk_id)
            if cached_embedding is not None:
                self.stats["cache_hits"] += 1
                # Preserve current chunk metadata (e.g. file_uuid/ingest_job_id)
                # while reusing cached vector values.
                return ChunkEmbedding(
                    chunk=chunk,
                    embedding=cached_embedding.embedding,
                    model_name=self.model_name,
                    embedding_dim=self.embedding_dim,
                )
            self.stats["cache_misses"] += 1

        # Generate embedding
        # Wrap single text in list for batch processing method reuse
        embedding = self.generate_embeddings_batch([chunk.content])[0]

        # Create ChunkEmbedding
        chunk_embedding = ChunkEmbedding(
            chunk=chunk,
            embedding=embedding,
            model_name=self.model_name,
            embedding_dim=self.embedding_dim
        )

        # Save to cache
        if use_cache and self.cache_dir:
            self._save_to_cache(chunk_embedding)

        self.stats["total_chunks_processed"] += 1
        return chunk_embedding

    def embed_chunks(
            self,
            chunks: List[Chunk],
            use_cache: bool = True
    ) -> List[ChunkEmbedding]:
        """
        Generate embeddings for multiple chunks with batch processing
        
        Args:
            chunks: List of chunks to embed
            use_cache: Whether to use cache
            
        Returns:
            List of ChunkEmbedding objects
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []

        logger.info(f"Generating embeddings for {len(chunks)} chunks")

        chunk_embeddings = []
        chunks_to_process = []
        chunk_indices = []

        # Check cache first
        if use_cache and self.cache_dir:
            for idx, chunk in enumerate(chunks):
                # Resolve chunk_id
                if hasattr(chunk, 'chunk_id'):
                    chunk_id = chunk.chunk_id
                elif hasattr(chunk, 'meta') and 'chunk_id' in chunk.meta:
                    chunk_id = chunk.meta['chunk_id']
                else:
                    chunk_id = f"chunk_{idx}"
                    if hasattr(chunk, 'chunk_id'): chunk.chunk_id = chunk_id

                cached = self._load_from_cache(chunk_id)
                if cached is not None:
                    # Keep the current chunk object so request-scoped metadata
                    # (such as ingest_job_id/ingest_state) is not stale.
                    chunk_embeddings.append((
                        idx,
                        ChunkEmbedding(
                            chunk=chunk,
                            embedding=cached.embedding,
                            model_name=self.model_name,
                            embedding_dim=self.embedding_dim,
                        ),
                    ))
                    self.stats["cache_hits"] += 1
                else:
                    chunks_to_process.append(chunk)
                    chunk_indices.append(idx)
                    self.stats["cache_misses"] += 1

            if self.stats["cache_hits"] > 0:
                logger.info(f"  - Cache hits: {self.stats['cache_hits']}")
                logger.info(f"  - Chunks to process: {len(chunks_to_process)}")
        else:
            chunks_to_process = chunks
            chunk_indices = list(range(len(chunks)))

        # Process remaining chunks
        if chunks_to_process:
            texts = [chunk.content for chunk in chunks_to_process]

            # Create progress bar
            if self.show_progress:
                pbar = tqdm(
                    total=len(texts),
                    desc="Embedding (sentence-transformers)",
                    unit="chunk"
                )

            # Define effective batch size
            # If using multiprocessing pool, we can feed larger batches
            eff_batch_size = self.batch_size
            if self.pool:
                eff_batch_size = self.batch_size * 4

            # Process in batches
            for i in range(0, len(texts), eff_batch_size):
                batch_texts = texts[i:i + eff_batch_size]
                batch_chunks = chunks_to_process[i:i + eff_batch_size]
                batch_indices = chunk_indices[i:i + eff_batch_size]

                # Generate embeddings for batch
                embeddings = self.generate_embeddings_batch(batch_texts)

                # Create ChunkEmbedding objects
                for idx, chunk, embedding in zip(batch_indices, batch_chunks, embeddings):
                    # Ensure chunk has ID for caching
                    if not hasattr(chunk, 'chunk_id') and hasattr(chunk, 'meta'):
                        chunk.chunk_id = chunk.meta.get('chunk_id')

                    chunk_embedding = ChunkEmbedding(
                        chunk=chunk,
                        embedding=embedding,
                        model_name=self.model_name,
                        embedding_dim=self.embedding_dim
                    )
                    chunk_embeddings.append((idx, chunk_embedding))

                    if use_cache and self.cache_dir:
                        self._save_to_cache(chunk_embedding)

                self.stats["batches_processed"] += 1
                self.stats["total_chunks_processed"] += len(batch_chunks)

                if self.show_progress:
                    pbar.update(len(batch_texts))

            if self.show_progress:
                pbar.close()

        # Sort by original index
        chunk_embeddings.sort(key=lambda x: x[0])
        return [ce for _, ce in chunk_embeddings]

    def _generate_cache_key(self, chunk_id: str) -> str:
        """Generate cache key for a chunk."""
        key = f"{self.model_name}_{chunk_id}"
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return hash_key

    def _get_cache_path(self, chunk_id: str) -> Path:
        """Get cache file path for a chunk."""
        cache_key = self._generate_cache_key(chunk_id)
        return self.cache_dir / f"{cache_key}.pkl"

    def _save_to_cache(self, chunk_embedding: ChunkEmbedding):
        """Save embedding to cache."""
        if not self.cache_dir:
            return
        try:
            # Handle potential missing chunk_id
            c_id = getattr(chunk_embedding.chunk, 'chunk_id', None)
            if not c_id and hasattr(chunk_embedding.chunk, 'meta'):
                c_id = chunk_embedding.chunk.meta.get('chunk_id')

            if not c_id: return

            cache_path = self._get_cache_path(c_id)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'chunk': chunk_embedding.chunk,
                    'embedding': chunk_embedding.embedding,
                    'model_name': chunk_embedding.model_name,
                    'embedding_dim': chunk_embedding.embedding_dim
                }, f)
        except Exception:
            pass  # Ignore caching errors to keep pipeline running

    def _load_from_cache(self, chunk_id: str) -> Optional[ChunkEmbedding]:
        """Load embedding from cache."""
        if not self.cache_dir: return None
        try:
            cache_path = self._get_cache_path(chunk_id)
            if not cache_path.exists(): return None

            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            if data['model_name'] != self.model_name: return None

            return ChunkEmbedding(
                chunk=data['chunk'],
                embedding=data['embedding'],
                model_name=data['model_name'],
                embedding_dim=data['embedding_dim']
            )
        except Exception:
            return None

    def clear_cache(self):
        """Clear all cached embeddings."""
        if self.cache_dir:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
