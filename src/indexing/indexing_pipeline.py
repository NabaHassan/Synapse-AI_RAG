"""
Complete Indexing Pipeline for RAG System
Processes documents from directory: Load → Chunk → Embed → Store
"""

import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

from src.indexing.vector_store import VectorStore
from src.indexing.embedding_generator import EmbeddingGenerator
from src.indexing.recursive_chunker import RecursiveCharacterTextSplitter
from src.indexing.document_loader import load_documents_from_directory, Document
from src.indexing.entity_extractor import EntityExtractor
from src.indexing.email_metadata_extractor import EmailMetadataExtractor
from src.indexing.anchor_term_extractor import AnchorTermExtractor
from src.config.profile_updater import ProfileUpdater

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexingPipeline:
    def __init__(
            self,
            storage_path: str = None,
            collection_name: str = "knowledge_base",
            qdrant_url: str = None,
            chunker_type: str = "recursive",
            embedding_model: str = "BAAI/bge-large-en-v1.5",
            cache_embeddings: bool = True,
            embedding_cache_dir: str = "./data/embeddings_cache",
            # Chunker params
            chunk_size: int = 300,
            chunk_overlap: int = 50,
            min_chunk_size: int = 100,
            # Batch sizes
            embedding_batch_size: int = 32,
            insertion_batch_size: int = 100,
            # Options
            show_progress: bool = True,
            recreate_collection: bool = False,
            enable_entity_extraction: bool = True,
            ner_backend: str = "spacy",
            ner_model: str = "en_core_web_lg",
            # Profile auto-update
            profile_path: Optional[str] = None,
            enable_auto_scope: bool = True
    ):
        """
        Initialize indexing pipeline

        Args:
            storage_path: Path for vector database storage
            collection_name: Name for vector store collection
            qdrant_url: URL for Qdrant server
            chunker_type: "recursive"
            embedding_model: Sentence transformer model name
            cache_embeddings: Whether to cache embeddings
            embedding_cache_dir: Directory for embedding cache
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            split_by: Haystack split unit (sentence/word/passage)
            split_length: Haystack number of units per chunk
            split_overlap: Haystack overlap in units
            breakpoint_threshold_amount: LangChain threshold (lower = more chunks)
            embedding_batch_size: Batch size for embedding generation
            insertion_batch_size: Batch size for vector insertion
            show_progress: Show progress bars
            recreate_collection: Recreate collection if exists
            enable_entity_extraction: Enable structured metadata extraction
            ner_backend: NER backend - "spacy" or "transformer"
            ner_model: NER model name
        """
        self.collection_name = collection_name

        # Handle "recursive" string or enum
        if isinstance(chunker_type, str) and chunker_type == "recursive":
            self.chunker_type_str = "recursive"

        self.show_progress = show_progress
        self.enable_entity_extraction = enable_entity_extraction

        logger.info("=" * 80)
        logger.info("Initializing Indexing Pipeline")
        logger.info("=" * 80)

        # Initialize chunker
        logger.info(f"1. Initializing {self.chunker_type_str} chunker")
        if isinstance(chunker_type, str) and chunker_type == "recursive":
            # Use RecursiveCharacterTextSplitter
            self.chunker = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size
            )
            self.is_recursive_chunker = True
        logger.info("   Recursive Chunker initialized")

        # Initialize embedding generator
        logger.info("2. Initializing embedding generator...")
        # run indexing embeddings on CPU so we don't contend with
        # serving-time GPU models (vLLM, query enhancer, online retriever).
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            show_progress=show_progress,
            cache_dir=embedding_cache_dir if cache_embeddings else None,
            device="cpu",
        )
        logger.info(f"   Embedding generator initialized (dim={self.embedding_generator.embedding_dim})")

        # Initialize vector store
        logger.info("3. Initializing vector store...")
        if storage_path is not None:
            self.vector_store = VectorStore(
                collection_name=collection_name,
                embedding_dim=self.embedding_generator.embedding_dim,
                storage_path=storage_path,
                recreate_collection=recreate_collection
            )
        elif qdrant_url is not None:
            self.vector_store = VectorStore(
                collection_name=collection_name,
                embedding_dim=self.embedding_generator.embedding_dim,
                qdrant_url=qdrant_url,
                recreate_collection=recreate_collection
            )
        logger.info("    Vector store initialized")

        self.cache_embeddings = cache_embeddings
        self.insertion_batch_size = insertion_batch_size

        # Optional metadata extractors used by structured query handlers.
        if self.enable_entity_extraction:
            try:
                self.entity_extractor = EntityExtractor(
                    backend=ner_backend,
                    model_name=ner_model
                )
                self.email_extractor = EmailMetadataExtractor()
                logger.info(
                    f"4. Entity extraction enabled (backend={ner_backend}, model={ner_model})"
                )
            except Exception as e:
                logger.warning(f"4. Failed to initialize entity extraction: {e}")
                logger.warning("   Proceeding without structured metadata extraction")
                self.entity_extractor = None
                self.email_extractor = None
                self.enable_entity_extraction = False
        else:
            self.entity_extractor = None
            self.email_extractor = None

        # Auto-scope generation
        self.profile_path = profile_path
        self.enable_auto_scope = enable_auto_scope
        if self.enable_auto_scope:
            self.scope_extractor = AnchorTermExtractor()
        else:
            self.scope_extractor = None

        logger.info("=" * 80)
        logger.info(" Pipeline initialization complete")
        logger.info("=" * 80)
        logger.info("")

    """
    Complete indexing pipeline for RAG system.
    Orchestrates: Document Loading → Chunking → Embedding → Vector Storage
    """

    def index_directory(
            self,
            input_dir: str,
            recursive: bool = True,
            file_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index all documents from a directory

        Args:
            input_dir: Directory containing documents
            recursive: Search subdirectories
            file_pattern: Optional file pattern (e.g., '*.pdf')

        Returns:
            Dictionary with indexing statistics
        """
        logger.info("=" * 80)
        logger.info("Starting Directory Indexing")
        logger.info("=" * 80)
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Chunker: {self.chunker_type_str}")
        logger.info("")

        # Step 1: Load documents
        logger.info("STEP 1: Loading documents")
        logger.info("-" * 80)
        documents = load_documents_from_directory(
            input_dir,
            recursive=recursive,
            file_pattern=file_pattern
        )

        if not documents:
            logger.warning("No documents found!")
            return {
                "success": False,
                "error": "No documents found",
                "documents_loaded": 0
            }

        logger.info(f" Loaded {len(documents)} documents")
        total_chars = sum(len(d.content) for d in documents)
        logger.info(f"   Total size: {total_chars:,} characters")
        logger.info("")

        # Step 2: Chunk documents
        logger.info("STEP 2: Chunking documents")
        logger.info("-" * 80)

        if self.is_recursive_chunker:
            # Use RecursiveCharacterTextSplitter (returns Haystack Documents)
            chunks = self.chunker.split_documents(documents)
        else:
            # Use SemanticChunker (returns Chunk objects)
            chunks = self.chunker.chunk_documents(documents)

        if not chunks:
            logger.warning("No chunks created!")
            return {
                "success": False,
                "error": "No chunks created",
                "documents_loaded": len(documents),
                "chunks_created": 0
            }

        # Get statistics
        if self.is_recursive_chunker:
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            stats = {
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0
            }
        else:
            stats = self.chunker.get_chunk_statistics(chunks)

        logger.info(f" Created {len(chunks)} chunks")
        logger.info(f"   Average size: {stats['avg_chunk_size']:.1f} characters")
        logger.info(f"   Range: {stats['min_chunk_size']}-{stats['max_chunk_size']} characters")
        logger.info("")

        # Attach metadata used by structured handlers (entity/file/exact-text).
        self._attach_structured_metadata(chunks)

        # Step 3: Generate embeddings
        logger.info("STEP 3: Generating embeddings")
        logger.info("-" * 80)
        chunk_embeddings = self.embedding_generator.embed_chunks(
            chunks,
            use_cache=self.cache_embeddings
        )

        embed_stats = self.embedding_generator.get_statistics()
        logger.info(f" Generated {len(chunk_embeddings)} embeddings")
        if self.cache_embeddings:
            logger.info(f"   Cache hits: {embed_stats['cache_hits']}")
            logger.info(f"   Cache misses: {embed_stats['cache_misses']}")
        logger.info("")

        # Step 4: Store in vector database
        logger.info("STEP 4: Storing vectors")
        logger.info("-" * 80)
        insertion_stats = self.vector_store.add_embeddings(
            chunk_embeddings,
            batch_size=self.insertion_batch_size
        )

        logger.info(f" Vectors stored")
        logger.info(f"   Inserted: {insertion_stats['inserted']}")
        logger.info(f"   Failed: {insertion_stats['failed']}")
        logger.info("")

        logger.info("=" * 80)
        logger.info("")

        # Step 5: Auto-populate KB scope (LLM)
        kb_scope = ""
        if self.enable_auto_scope and self.scope_extractor and documents:
            logger.info("STEP 5: Generating KB Scope Description (LLM)")
            logger.info("-" * 80)
            try:
                # Use first 10 docs as sample
                sample_text = "\n\n".join([d.content[:2000] for d in documents[:10]])
                kb_scope = self.scope_extractor.extract_scope_description(sample_text)
                logger.info(f"   Generated scope: {kb_scope[:100]}...")
                
                if self.profile_path:
                    logger.info(f"   Updating profile: {self.profile_path}")
                    ProfileUpdater.update_kb_scope_description(self.profile_path, kb_scope)
                else:
                    logger.info("   No profile_path provided, skipping profile update.")
            except Exception as e:
                logger.error(f"   Auto-scope generation failed: {e}")
            logger.info("")

        return {
            "success": True,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "embeddings_generated": len(chunk_embeddings),
            "vectors_inserted": insertion_stats['inserted'],
            "vectors_failed": insertion_stats['failed'],
            "collection_name": self.collection_name,
            "chunker_type": self.chunker_type_str,
            "embedding_stats": embed_stats,
            "chunk_stats": stats,
            "kb_scope": kb_scope
        }

    def index_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Index a list of documents

        Args:
            documents: List of Document objects

        Returns:
            Dictionary with indexing statistics
        """
        logger.info(f"Indexing {len(documents)} documents")

        # Chunk
        chunks = self.chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Embed
        chunk_embeddings = self.embedding_generator.embed_chunks(
            chunks,
            use_cache=self.cache_embeddings
        )
        logger.info(f"Generated {len(chunk_embeddings)} embeddings")

        # Store
        insertion_stats = self.vector_store.add_embeddings(
            chunk_embeddings,
            batch_size=self.insertion_batch_size
        )
        logger.info(f"Stored {insertion_stats['inserted']} vectors")

        return {
            "success": True,
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "embeddings_generated": len(chunk_embeddings),
            "vectors_inserted": insertion_stats['inserted']
        }

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration"""
        return {
            "collection_name": self.collection_name,
            "chunker_type": self.chunker_type_str,
            "embedding_model": self.embedding_generator.model_name,
            "embedding_dim": self.embedding_generator.embedding_dim,
            "cache_enabled": self.cache_embeddings,
            "collection_info": self.vector_store.get_collection_info()
        }

    def _attach_structured_metadata(self, chunks: List[Document]) -> None:
        """
        Attach per-chunk metadata needed by structured query handlers.

        Adds:
        - entity_names / entities
        - is_first_chunk
        - document_entity_counts (first chunk only)
        - email_sender/receiver/date/subject/is_email
        """
        if not chunks:
            return

        chunks_by_file: Dict[str, List[Document]] = defaultdict(list)
        for chunk in chunks:
            if not hasattr(chunk, "meta"):
                continue
            source_file = chunk.meta.get("source_filename", "Unknown")
            chunks_by_file[source_file].append(chunk)

        for source_file, file_chunks in chunks_by_file.items():
            ordered_chunks = sorted(
                file_chunks,
                key=lambda c: c.meta.get("chunk_index", 0) if hasattr(c, "meta") else 0
            )

            doc_entity_data = None
            email_metadata = {}
            if self.enable_entity_extraction and self.entity_extractor:
                full_text = "\n\n".join(chunk.content for chunk in ordered_chunks if getattr(chunk, "content", None))
                try:
                    doc_entity_data = self.entity_extractor.extract_from_text(full_text)
                except Exception as e:
                    logger.warning(f"Entity extraction failed for {source_file}: {e}")
                    doc_entity_data = None

                if self.email_extractor:
                    try:
                        email_metadata = self.email_extractor.extract(full_text)
                    except Exception as e:
                        logger.warning(f"Email metadata extraction failed for {source_file}: {e}")
                        email_metadata = {}

            for idx, chunk in enumerate(ordered_chunks):
                chunk.meta["is_first_chunk"] = (idx == 0)
                if idx == 0 and doc_entity_data:
                    chunk.meta["document_entity_counts"] = doc_entity_data.get("entity_counts", {})
                else:
                    chunk.meta["document_entity_counts"] = {}

                chunk.meta["email_sender"] = ""
                chunk.meta["email_receiver"] = ""
                chunk.meta["email_date"] = ""
                chunk.meta["email_subject"] = ""
                chunk.meta["is_email"] = False

                if self.enable_entity_extraction and self.entity_extractor:
                    try:
                        chunk_entities = self.entity_extractor.extract_from_chunk(chunk.content)
                        chunk.meta["entities"] = chunk_entities
                        chunk.meta["entity_names"] = [entity["name"] for entity in chunk_entities]
                    except Exception as e:
                        logger.warning(
                            f"Chunk entity extraction failed for {source_file} "
                            f"(chunk {chunk.meta.get('chunk_index', 0)}): {e}"
                        )
                        chunk.meta["entities"] = []
                        chunk.meta["entity_names"] = []

                    if email_metadata:
                        chunk.meta["email_sender"] = email_metadata.get("email_sender", "")
                        chunk.meta["email_receiver"] = email_metadata.get("email_receiver", "")
                        chunk.meta["email_date"] = email_metadata.get("email_date", "")
                        chunk.meta["email_subject"] = email_metadata.get("email_subject", "")
                        chunk.meta["is_email"] = email_metadata.get("is_email", False)
                else:
                    chunk.meta["entities"] = []
                    chunk.meta["entity_names"] = []


# Convenience function
def create_indexing_pipeline(
        collection_name: str = "knowledge_base",
        chunker_type: str = "recursive",
        **kwargs
) -> IndexingPipeline:
    """
    Convenience function to create an indexing pipeline

    Args:
        collection_name: Name for vector store collection
        chunker_type: "recursive"
        **kwargs: Additional arguments for IndexingPipeline

    Returns:
        IndexingPipeline instance
    """

    return IndexingPipeline(
        collection_name=collection_name,
        chunker_type=chunker_type,
        **kwargs
    )
