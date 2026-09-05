"""
Incremental Indexing Pipeline

Extends the base IndexingPipeline with:
- File tracking using UUIDs
- Only index new/modified files
- Delete removed files from Qdrant
- Sync directory state with vector database
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.indexing.file_tracker import FileTracker
from src.indexing.vector_store import VectorStore
from src.indexing.processed_loader import ProcessedJsonLoader
from src.indexing.embedding_generator import EmbeddingGenerator
from src.indexing.document_loader import load_document, DocumentLoader
from src.indexing.recursive_chunker import RecursiveCharacterTextSplitter
from src.indexing.entity_extractor import EntityExtractor
from src.indexing.email_metadata_extractor import EmailMetadataExtractor
from src.indexing.anchor_term_extractor import AnchorTermExtractor
from src.config.profile_updater import ProfileUpdater

logger = logging.getLogger(__name__)


class IncrementalIndexingPipeline:
    """
    Incremental indexing pipeline with UUID-based file tracking.
    
    Features:
    - Tracks indexed files with UUIDs
    - Only indexes new or modified files
    - Automatically removes deleted files from Qdrant
    - Adds file_uuid to all chunk metadata
    """

    def __init__(
            self,
            collection_name: str = "knowledge_base",
            qdrant_url: str = "http://localhost:6333",
            storage_path: str = None,
            file_tracker_db: str = "./data/file_tracker.json",
            chunk_size: int = 800,
            chunk_overlap: int = 200,
            min_chunk_size: int = 100,
            embedding_model: str = "BAAI/bge-large-en-v1.5",
            cache_embeddings: bool = True,
            embedding_cache_dir: str = "./data/embeddings_cache",
            embedding_batch_size: int = 32,
            insertion_batch_size: int = 100,
            show_progress: bool = True,
            recreate_collection: bool = False,
            use_processed_loader: bool = False,
            embedding_backend: str = "sentence-transformers",
            num_workers: int = None,
            enable_entity_extraction: bool = True,
            ner_backend: str = "spacy",
            ner_model: str = "en_core_web_lg",
            # Profile auto-update
            profile_path: Optional[str] = None,
            enable_auto_scope: bool = True
    ):
        """
        Initialize incremental indexing pipeline.
        
        Args:
            collection_name: Qdrant collection name
            qdrant_url: Qdrant server URL (preferred)
            storage_path: Local storage path (alternative to qdrant_url)
            file_tracker_db: Path to file tracking database
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            embedding_model: Sentence transformer model
            cache_embeddings: Enable embedding cache
            embedding_cache_dir: Cache directory path
            embedding_batch_size: Batch size for embeddings
            insertion_batch_size: Batch size for vector insertion
            show_progress: Show progress bars
            recreate_collection: Recreate collection if exists
            use_processed_loader: Whether to use the ProcessedJsonLoader for pre-cleaned data
            embedding_backend: 'sentence-transformers' or 'fastembed'
            num_workers: Number of workers for parallel processing
            enable_entity_extraction: Enable NER-based entity extraction during indexing
            ner_backend: NER backend - 'spacy' or 'transformer'
            ner_model: NER model name (e.g., 'en_core_web_lg' or 'dslim/bert-base-NER')
        """
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.storage_path = storage_path
        self.show_progress = show_progress
        self.cache_embeddings = cache_embeddings
        self.insertion_batch_size = insertion_batch_size
        self.use_processed_loader = use_processed_loader
        self.enable_entity_extraction = enable_entity_extraction

        logger.info("=" * 80)
        logger.info("Initializing Incremental Indexing Pipeline")
        logger.info("=" * 80)
        logger.info(f"Collection: {collection_name}")
        if qdrant_url:
            logger.info(f"Qdrant URL: {qdrant_url}")
        if storage_path:
            logger.info(f"Storage path: {storage_path}")

        # Initialize file tracker
        logger.info("\n1. Initializing File Tracker...")
        logger.info("-" * 80)
        self.file_tracker = FileTracker(db_path=file_tracker_db)
        logger.info(f"   File tracker ready (DB: {file_tracker_db})")

        # Initialize chunker
        logger.info("\n2. Initializing Chunker...")
        logger.info("-" * 80)
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )
        logger.info("   Recursive chunker initialized")

        # Initialize embedding generator
        logger.info("\n3. Initializing Embedding Generator...")
        logger.info("-" * 80)
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            show_progress=show_progress,
            cache_dir=embedding_cache_dir if cache_embeddings else None,
            backend=embedding_backend,
            num_workers=num_workers
            # device argument removed to allow auto-detection (MPS/CUDA)
        )
        logger.info(f"   Embedding generator initialized (dim={self.embedding_generator.embedding_dim})")

        # Initialize vector store
        logger.info("\n4. Initializing Vector Store...")
        logger.info("-" * 80)
        if qdrant_url:
            self.vector_store = VectorStore(
                collection_name=collection_name,
                embedding_dim=self.embedding_generator.embedding_dim,
                qdrant_url=qdrant_url,
                recreate_collection=recreate_collection
            )
        elif storage_path:
            self.vector_store = VectorStore(
                collection_name=collection_name,
                embedding_dim=self.embedding_generator.embedding_dim,
                storage_path=storage_path,
                recreate_collection=recreate_collection
            )
        else:
            raise ValueError("Must provide either qdrant_url or storage_path")

        # Initialize loader
        if self.use_processed_loader:
            self.loader = ProcessedJsonLoader()
            logger.info("   Using ProcessedJsonLoader for pre-cleaned data")
        else:
            self.loader = None  # Will be instantiated per file or use default DocumentLoader
            logger.info("   Using standard DocumentLoader")

        logger.info("   Vector store initialized")

        # Initialize entity extraction (optional)
        logger.info("\n5. Initializing Entity Extraction...")
        logger.info("-" * 80)
        if self.enable_entity_extraction:
            try:
                self.entity_extractor = EntityExtractor(
                    backend=ner_backend,
                    model_name=ner_model
                )
                self.email_extractor = EmailMetadataExtractor()
                logger.info(f"   Entity extraction enabled (backend={ner_backend}, model={ner_model})")
                logger.info("   Email metadata extraction enabled")
            except Exception as e:
                logger.warning(f"   Failed to initialize entity extraction: {e}")
                logger.warning("   Proceeding without entity extraction")
                self.entity_extractor = None
                self.email_extractor = None
                self.enable_entity_extraction = False
        else:
            self.entity_extractor = None
            self.email_extractor = None
            logger.info("   Entity extraction disabled")

        # Auto-scope generation
        self.profile_path = profile_path
        self.enable_auto_scope = enable_auto_scope
        if self.enable_auto_scope:
            self.scope_extractor = AnchorTermExtractor()
            logger.info("   Auto-scope generation enabled (LLM-based)")
        else:
            self.scope_extractor = None
            logger.info("   Auto-scope generation disabled")

        logger.info("\n" + "=" * 80)
        logger.info("Pipeline Initialization Complete")
        logger.info("=" * 80 + "\n")

    def index_directory(
            self,
            input_dir: str,
            recursive: bool = True,
            file_pattern: Optional[str] = None,
            sync_deletions: bool = True
    ) -> Dict[str, Any]:
        """
        Incrementally index a directory.
        
        Only new or modified files are processed. Optionally removes
        deleted files from Qdrant.
        
        Args:
            input_dir: Directory containing documents
            recursive: Search subdirectories
            file_pattern: Optional file pattern (e.g., '*.pdf')
            sync_deletions: Remove deleted files from Qdrant
            
        Returns:
            Dictionary with indexing statistics
        """
        logger.info("=" * 80)
        logger.info("Starting Incremental Directory Indexing")
        logger.info("=" * 80)
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Sync deletions: {sync_deletions}")
        logger.info("")

        input_path = Path(input_dir)
        if not input_path.exists():
            return {"success": False, "error": "Directory not found"}

        # Step 1: Discover files
        logger.info("STEP 1: Discovering Files")
        logger.info("-" * 80)

        # Get all supported files
        if recursive:
            if file_pattern:
                files = list(input_path.rglob(file_pattern))
            else:
                files = [f for f in input_path.rglob('*') if f.is_file()]
        else:
            if file_pattern:
                files = list(input_path.glob(file_pattern))
            else:
                files = [f for f in input_path.glob('*') if f.is_file()]

        # Filter supported files based on loader
        if self.use_processed_loader:
            supported_files = [f for f in files if f.suffix.lower() == '.json']
            logger.info(f"   Found {len(supported_files)} JSON files (processed mode)")
        else:
            supported_files = [
                f for f in files
                if f.suffix.lower() in DocumentLoader.SUPPORTED_FORMATS
            ]
            logger.info(f"   Found {len(supported_files)} supported files")

        # Step 2: Determine files to index
        logger.info("\nSTEP 2: Analyzing Files")
        logger.info("-" * 80)

        files_status = self.file_tracker.get_files_to_index(
            supported_files,
            self.collection_name
        )

        new_files = files_status['new']
        modified_files = files_status['modified']
        unchanged_files = files_status['unchanged']

        logger.info(f"   New files: {len(new_files)}")
        logger.info(f"   Modified files: {len(modified_files)}")
        logger.info(f"   Unchanged files: {len(unchanged_files)}")

        files_to_process = new_files + modified_files

        # Step 3: Handle deletions
        deleted_count = 0
        if sync_deletions:
            logger.info("\nSTEP 3: Handling Deletions")
            logger.info("-" * 80)

            deleted_files = self.file_tracker.get_deleted_files(input_path)

            if deleted_files:
                logger.info(f"   Found {len(deleted_files)} deleted files")

                for deleted_file in deleted_files:
                    file_uuid = deleted_file['file_uuid']
                    file_name = deleted_file['file_name']

                    # Delete from Qdrant
                    result = self.vector_store.delete_by_file_uuid(file_uuid)
                    if result['success']:
                        logger.info(f"   Deleted {result['deleted_count']} vectors for: {file_name}")
                        # Remove from tracking
                        self.file_tracker.remove_file(Path(deleted_file['file_path']))
                        deleted_count += 1
                    else:
                        logger.error(f"   Failed to delete: {file_name}")
            else:
                logger.info("   No deleted files found")

        # Step 4: Index new/modified files
        if not files_to_process:
            logger.info("\nNo files to index. All files are up to date!")
            logger.info("=" * 80 + "\n")
            return {
                "success": True,
                "new_files": 0,
                "modified_files": 0,
                "unchanged_files": len(unchanged_files),
                "deleted_files": deleted_count,
                "chunks_created": 0,
                "vectors_inserted": 0,
                "collection_name": self.collection_name
            }

        logger.info(f"\nSTEP 4: Indexing {len(files_to_process)} Files")
        logger.info("-" * 80)

        total_chunks = 0
        total_vectors = 0
        processed_files = 0

        for file_path in files_to_process:
            try:
                # Generate or get file UUID
                file_uuid = self.file_tracker.generate_file_uuid(file_path)

                # If file was modified, delete old vectors first
                if file_path in modified_files:
                    logger.info(f"\n   Modified file detected: {file_path.name}")
                    logger.info(f"   Deleting old vectors...")
                    self.vector_store.delete_by_file_uuid(file_uuid)

                logger.info(f"\n   Processing: {file_path.name}")
                logger.info(f"   UUID: {file_uuid}")

                # Load document using appropriate loader
                if self.use_processed_loader:
                    documents = self.loader.load(str(file_path))
                else:
                    documents = load_document(str(file_path))

                if not documents:
                    logger.warning(f"   No content loaded from {file_path.name}")
                    continue

                # Chunk documents
                chunks = self.chunker.split_documents(documents)
                logger.info(f"   Created {len(chunks)} chunks")

                # Extract document-level entity & email metadata (before chunk loop)
                doc_entity_data = None
                email_metadata = {}
                if self.enable_entity_extraction and self.entity_extractor:
                    # Get full document text for document-level analysis
                    full_text = "\n\n".join(
                        doc.content if hasattr(doc, 'content') else str(doc)
                        for doc in documents
                    )
                    try:
                        doc_entity_data = self.entity_extractor.extract_from_text(full_text)
                        logger.info(
                            f"   Extracted {len(doc_entity_data.get('entities', []))} unique entities from document")
                    except Exception as e:
                        logger.warning(f"   Entity extraction failed for document: {e}")
                        doc_entity_data = None

                    if self.email_extractor:
                        try:
                            email_metadata = self.email_extractor.extract(full_text)
                            if email_metadata.get('is_email'):
                                logger.info(f"   Email detected - From: {email_metadata.get('email_sender', '')[:40]}")
                        except Exception as e:
                            logger.warning(f"   Email metadata extraction failed: {e}")
                            email_metadata = {}

                # Add file_uuid and entity metadata to chunk metadata
                # This must be done AFTER chunking because chunks are new Haystack Document objects
                for idx, chunk in enumerate(chunks):
                    if hasattr(chunk, 'meta'):
                        chunk.meta['file_uuid'] = file_uuid
                        chunk.meta['file_id'] = file_uuid  # Store both for compatibility

                        # Entity extraction per chunk
                        if self.enable_entity_extraction and self.entity_extractor:
                            try:
                                chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                                chunk_entities = self.entity_extractor.extract_from_chunk(chunk_text)
                                chunk.meta['entities'] = chunk_entities
                                chunk.meta['entity_names'] = [e['name'] for e in chunk_entities]
                            except Exception as e:
                                logger.warning(f"   Chunk entity extraction failed: {e}")
                                chunk.meta['entities'] = []
                                chunk.meta['entity_names'] = []

                            # Email metadata (same for all chunks of same document)
                            if email_metadata:
                                chunk.meta['email_sender'] = email_metadata.get('email_sender', '')
                                chunk.meta['email_receiver'] = email_metadata.get('email_receiver', '')
                                chunk.meta['email_date'] = email_metadata.get('email_date', '')
                                chunk.meta['email_subject'] = email_metadata.get('email_subject', '')
                                chunk.meta['is_email'] = email_metadata.get('is_email', False)

                            # First chunk gets document-level aggregates
                            chunk.meta['is_first_chunk'] = (idx == 0)
                            if idx == 0 and doc_entity_data:
                                chunk.meta['document_entity_counts'] = doc_entity_data.get('entity_counts', {})
                            else:
                                chunk.meta['document_entity_counts'] = {}
                    else:
                        logger.error(f"   Chunk missing 'meta' attribute - file_uuid not set!")
                        raise ValueError(f"Chunk object missing 'meta' attribute")

                # Generate embeddings
                chunk_embeddings = self.embedding_generator.embed_chunks(
                    chunks,
                    use_cache=self.cache_embeddings
                )
                logger.info(f"   Generated {len(chunk_embeddings)} embeddings")

                # Insert into Qdrant
                insertion_stats = self.vector_store.add_embeddings(
                    chunk_embeddings,
                    batch_size=self.insertion_batch_size
                )
                logger.info(f"   Inserted {insertion_stats['inserted']} vectors")

                # Track the file
                self.file_tracker.add_file(
                    file_path,
                    self.collection_name,
                    file_uuid=file_uuid
                )

                total_chunks += len(chunks)
                total_vectors += insertion_stats['inserted']
                processed_files += 1

            except Exception as e:
                logger.error(f"   Failed to process {file_path.name}: {e}", exc_info=True)
                continue

        logger.info("=" * 80 + "\n")

        # Step 5: Auto-populate KB scope (LLM)
        kb_scope = ""
        if self.enable_auto_scope and self.scope_extractor and (new_files or modified_files):
            logger.info("STEP 5: Generating KB Scope Description (LLM)")
            logger.info("-" * 80)
            try:
                # We need all documents or a sample from the collection
                # For now, let's just use the documents from files_to_process as a representative sample
                
                # Here we reconstruct a list of Documents from files_to_process
                sample_docs = []
                for fp in (new_files + modified_files)[:20]: # Limit to 20 files for sample
                    try:
                        if self.use_processed_loader:
                            sample_docs.extend(self.loader.load(str(fp)))
                        else:
                            sample_docs.extend(load_document(str(fp)))
                    except:
                        continue
                
                if sample_docs:
                    # Combine content into a sample string
                    sample_text = "\n\n".join([d.content[:2000] for d in sample_docs[:10]])
                    kb_scope = self.scope_extractor.extract_scope_description(sample_text)
                    logger.info(f"   Generated scope: {kb_scope[:100]}...")
                    
                    if self.profile_path:
                        logger.info(f"   Updating profile: {self.profile_path}")
                        ProfileUpdater.update_kb_scope_description(self.profile_path, kb_scope)
                else:
                    logger.info("   No new/modified documents to sample for scope generation.")
            except Exception as e:
                logger.error(f"   Auto-scope generation failed: {e}")
            logger.info("")

        return {
            "success": True,
            "new_files": len(new_files),
            "modified_files": len(modified_files),
            "unchanged_files": len(unchanged_files),
            "deleted_files": deleted_count,
            "chunks_created": total_chunks,
            "vectors_inserted": total_vectors,
            "collection_name": self.collection_name,
            "kb_scope": kb_scope
        }

    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a specific file's vectors from Qdrant and untrack it.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            Deletion statistics
        """
        path = Path(file_path)

        # Get file UUID
        file_uuid = self.file_tracker.get_file_uuid(path)
        if not file_uuid:
            logger.warning(f"File not tracked: {path.name}")
            return {"success": False, "error": "File not tracked"}

        # Delete from Qdrant
        result = self.vector_store.delete_by_file_uuid(file_uuid)

        if result['success']:
            # Remove from tracking
            self.file_tracker.remove_file(path)
            logger.info(f"Successfully deleted file: {path.name}")

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        tracker_stats = self.file_tracker.get_stats()
        collection_info = self.vector_store.get_collection_info()

        return {
            "collection_name": self.collection_name,
            "tracked_files": tracker_stats['total_files'],
            "total_vectors": collection_info.get('points_count', 0),
            "collections": tracker_stats['collections'],
            "embedding_dim": self.embedding_generator.embedding_dim,
            "embedding_model": self.embedding_generator.model_name
        }
