"""
Document Manager for URL-based Indexing

Handles:
- Downloading files from SAS URLs
- Indexing without local storage
- Deletion by file_id
- Integration with file tracker
"""

import logging
import tempfile
import uuid
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from src.indexing.file_tracker import FileTracker
from src.indexing.vector_store import VectorStore
from src.indexing.embedding_generator import EmbeddingGenerator
from src.indexing.recursive_chunker import RecursiveCharacterTextSplitter
from src.indexing.entity_extractor import EntityExtractor
from src.indexing.email_metadata_extractor import EmailMetadataExtractor
from src.indexing.online_document_processor import (
    OnlineDocumentProcessor,
    OnlineDocumentProcessingError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Manages document indexing from URLs without local storage.
    
    Features:
    - Download from SAS URLs
    - Process in temporary directory
    - Clean up immediately after indexing
    - Track files by provided file_id
    """

    def __init__(
            self,
            collection_name: str,
            qdrant_url: str,
            file_tracker_db: str = "./data/file_tracker.json",
            chunk_size: int = 800,
            chunk_overlap: int = 200,
            min_chunk_size: int = 100,
            embedding_model: str = "BAAI/bge-large-en-v1.5",
            cache_embeddings: bool = True,
            embedding_cache_dir: str = "./data/embeddings_cache",
            embedding_batch_size: int = 32,
            insertion_batch_size: int = 100,
            embedding_generator: Optional[EmbeddingGenerator] = None,
            enable_entity_extraction: bool = True,
            ner_backend: str = "spacy",
            ner_model: str = "en_core_web_lg",
            indexing_mode: str = "exclusive",
            backfill_ingest_state_on_startup: bool = True,
            online_min_text_length: int = 200,
            online_min_quality_score: float = 0.4,
            online_remove_urls: bool = False,
            online_remove_emails: bool = False,
            online_use_unstructured_fallback: bool = True,
    ):
        """
        Initialize DocumentManager.
        
        Args:
            collection_name: Qdrant collection name
            qdrant_url: Qdrant server URL
            file_tracker_db: Path to file tracking database
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            min_chunk_size: Minimum chunk size
            embedding_model: Sentence transformer model
            cache_embeddings: Enable embedding cache
            embedding_cache_dir: Cache directory
            embedding_batch_size: Batch size for embeddings
            insertion_batch_size: Batch size for vector insertion
            enable_entity_extraction: Enable NER/entity metadata for structured queries
            ner_backend: NER backend - 'spacy' or 'transformer'
            ner_model: NER model name
            online_min_text_length: Minimum text length after cleaning
            online_min_quality_score: Minimum quality score after cleaning
            online_remove_urls: Remove URLs during cleaning
            online_remove_emails: Mask emails during cleaning
            online_use_unstructured_fallback: Enable Unstructured fallback in extractor
        """
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.cache_embeddings = cache_embeddings
        self.insertion_batch_size = insertion_batch_size
        self.enable_entity_extraction = enable_entity_extraction
        self.indexing_mode = (indexing_mode or "exclusive").strip().lower()
        if self.indexing_mode not in {"exclusive", "online"}:
            logger.warning("Unsupported indexing_mode='%s', defaulting to 'exclusive'", self.indexing_mode)
            self.indexing_mode = "exclusive"

        logger.info(f"Initializing DocumentManager for collection: {collection_name}")
        logger.info("DocumentManager indexing mode: %s", self.indexing_mode)

        # Initialize file tracker
        self.file_tracker = FileTracker(db_path=file_tracker_db)

        # Initialize chunker
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size
        )

        # Initialize embedding generator (or reuse shared generator)
        if embedding_generator is not None:
            self.embedding_generator = embedding_generator
        else:
            self.embedding_generator = EmbeddingGenerator(
                model_name=embedding_model,
                batch_size=embedding_batch_size,
                show_progress=False,  # Disable progress bars for API
                cache_dir=embedding_cache_dir if cache_embeddings else None,
                device="cpu"
            )

        # Initialize vector store
        self.vector_store = VectorStore(
            collection_name=collection_name,
            embedding_dim=self.embedding_generator.embedding_dim,
            qdrant_url=qdrant_url,
            recreate_collection=False
        )

        # Run online extraction + cleaning in-memory for API ingestion.
        self.document_processor = OnlineDocumentProcessor(
            min_text_length=online_min_text_length,
            min_quality_score=online_min_quality_score,
            remove_urls=online_remove_urls,
            remove_emails=online_remove_emails,
            use_unstructured_fallback=online_use_unstructured_fallback,
        )
        logger.info(
            "Online ingestion supported formats: %s",
            ", ".join(sorted(self.document_processor.supported_extensions)),
        )

        if backfill_ingest_state_on_startup:
            backfill = self.vector_store.backfill_ingest_state_ready()
            if backfill.get("success"):
                logger.info(
                    "Ingest-state backfill completed (updated_points=%s)",
                    backfill.get("updated_points", 0),
                )
            else:
                logger.warning("Ingest-state backfill failed: %s", backfill.get("error"))

        # Optional metadata extractors (entity counts, email headers).
        if self.enable_entity_extraction:
            try:
                self.entity_extractor = EntityExtractor(
                    backend=ner_backend,
                    model_name=ner_model
                )
                self.email_extractor = EmailMetadataExtractor()
                logger.info(
                    f"Entity extraction enabled for URL indexing "
                    f"(backend={ner_backend}, model={ner_model})"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize entity extraction: {e}")
                logger.warning("Continuing URL indexing without entity metadata")
                self.entity_extractor = None
                self.email_extractor = None
                self.enable_entity_extraction = False
        else:
            self.entity_extractor = None
            self.email_extractor = None

        logger.info("DocumentManager initialized successfully")

    @staticmethod
    def _normalize_local_path(local_path: str) -> Path:
        parsed = urlparse(local_path)
        if parsed.scheme == "file":
            return Path(unquote(parsed.path)).expanduser()
        return Path(local_path).expanduser()

    @staticmethod
    def _safe_temp_file_name(file_name: str) -> str:
        safe_name = Path(file_name).name
        return safe_name or f"document_{uuid.uuid4().hex}"

    @staticmethod
    def generate_source_file_id(source: str, file_name: str = "") -> str:
        """Generate a stable file id for local/batch ingestion when none is provided."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{file_name}"))

    def _find_tracked_file_by_uuid(self, file_uuid: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
        for tracked in self.file_tracker.get_all_tracked_files(self.collection_name):
            if tracked.get("file_uuid") == file_uuid and tracked.get("file_path"):
                return Path(tracked["file_path"]), tracked
        return None

    def _remove_tracker_entries_for_uuid(
            self,
            file_uuid: str,
            keep_path: Optional[Path] = None
    ) -> None:
        keep_abs = str(keep_path.absolute()) if keep_path is not None else None
        for tracked in list(self.file_tracker.get_all_tracked_files(self.collection_name)):
            tracked_path = tracked.get("file_path")
            if tracked.get("file_uuid") == file_uuid and tracked_path != keep_abs:
                self.file_tracker.remove_file(Path(tracked_path))

    def _local_tracker_path(self, file_path: Path) -> Path:
        try:
            return file_path.resolve()
        except Exception:
            return file_path.absolute()

    def discover_supported_local_files(
            self,
            directory_path: str,
            recursive: bool = True
    ) -> Dict[str, Any]:
        directory = self._normalize_local_path(directory_path)
        if not directory.exists():
            return {
                "success": False,
                "error_code": "directory_not_found",
                "error": f"Directory not found: {directory_path}",
                "directory_path": directory_path,
            }
        if not directory.is_dir():
            return {
                "success": False,
                "error_code": "not_a_directory",
                "error": f"Path is not a directory: {directory_path}",
                "directory_path": directory_path,
            }

        paths = directory.rglob("*") if recursive else directory.glob("*")
        supported: List[Path] = []
        skipped: List[Dict[str, str]] = []
        for path in paths:
            if not path.is_file():
                continue
            if self.document_processor.is_supported_filename(path.name):
                supported.append(path)
            else:
                skipped.append({
                    "local_path": str(path),
                    "file_name": path.name,
                    "reason": "unsupported_file_type",
                })

        supported.sort(key=lambda item: str(item))
        skipped.sort(key=lambda item: item["local_path"])
        return {
            "success": True,
            "directory_path": str(directory),
            "recursive": recursive,
            "files": supported,
            "skipped": skipped,
            "supported_count": len(supported),
            "skipped_count": len(skipped),
        }

    def download_file(
            self,
            sas_url: str,
            file_name: str,
            temp_dir: Path
    ) -> Optional[Path]:
        """
        Download file from SAS URL to temporary directory.
        
        Args:
            sas_url: Azure SAS URL
            file_name: Original filename
            temp_dir: Temporary directory path
            
        Returns:
            Path to downloaded file or None on failure
        """
        try:
            logger.info(f"Downloading file: {file_name}")

            # Download file with streaming
            response = requests.get(sas_url, stream=True, timeout=300)
            response.raise_for_status()

            # Save to temp directory
            temp_file_path = temp_dir / self._safe_temp_file_name(file_name)

            with open(temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size = temp_file_path.stat().st_size
            logger.info(f"Downloaded {file_name} ({file_size:,} bytes)")

            return temp_file_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {file_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error saving {file_name}: {e}")
            return None

    def _index_file_path(
            self,
            file_id: str,
            file_name: str,
            source_file_path: Path,
            tracker_path: Path,
            source_type: str,
            source_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Index a resolved local file path. The file may be an existing local file or
        a temporary file downloaded from a remote source.
        """
        logger.info(
            "Starting %s indexing for: %s (ID: %s)",
            source_type,
            file_name,
            file_id,
        )
        ingest_job_id = f"ingest_{uuid.uuid4().hex[:24]}"
        existing_uuid: Optional[str] = None

        if not self.document_processor.is_supported_filename(file_name):
            supported_formats = sorted(self.document_processor.supported_extensions)
            logger.warning(
                "Unsupported file received for online indexing: %s (supported=%s)",
                file_name,
                ", ".join(supported_formats),
            )
            return {
                "success": False,
                "error_code": "unsupported_file_type",
                "error": (
                    f"Unsupported file format for '{file_name}'. "
                    f"Supported formats: {', '.join(supported_formats)}"
                ),
                "file_id": file_id,
                "file_name": file_name,
                "supported_formats": supported_formats,
            }

        vectors_staged = False
        try:
            if not source_file_path.exists():
                return {
                    "success": False,
                    "error_code": "local_file_not_found",
                    "error": f"File not found: {source_file_path}",
                    "file_id": file_id,
                    "file_name": file_name,
                    "source_type": source_type,
                    "source_uri": source_uri,
                }
            if not source_file_path.is_file():
                return {
                    "success": False,
                    "error_code": "not_a_file",
                    "error": f"Path is not a file: {source_file_path}",
                    "file_id": file_id,
                    "file_name": file_name,
                    "source_type": source_type,
                    "source_uri": source_uri,
                }

            tracked_by_uuid = self._find_tracked_file_by_uuid(file_id)
            if tracked_by_uuid:
                existing_uuid = file_id
            else:
                existing_uuid = self.file_tracker.get_file_uuid(tracker_path)

            if existing_uuid:
                if self.indexing_mode == "exclusive":
                    logger.info(f"File {file_name} already indexed, replacing (exclusive mode)...")
                    self.vector_store.delete_by_file_uuid(existing_uuid)
                    logger.info(f"Deleted old vectors for {file_name}")
                elif self.indexing_mode == "online":
                    logger.info(
                        "File %s already indexed, using online replace mode (old vectors retained until commit)",
                        file_name,
                    )

            # Step 2: Extract + clean document using the online processor.
            logger.info("Extracting and cleaning document: %s", file_name)
            processed = self.document_processor.process_file(
                file_path=source_file_path,
                original_file_name=file_name,
            )
            documents = processed["documents"]
            processing_info = processed["processing_info"]

            if not documents:
                return {
                    "success": False,
                    "error_code": "no_content_extracted",
                    "error": "No content extracted from file",
                    "file_id": file_id,
                    "file_name": file_name,
                    "source_type": source_type,
                    "source_uri": source_uri,
                }

            logger.info(
                "Document processed: method=%s, language=%s, quality=%.3f, size=%s->%s chars",
                processing_info.get("extraction_method", "unknown"),
                processing_info.get("language", "unknown"),
                processing_info.get("quality_score", 0.0),
                processing_info.get("original_length", 0),
                processing_info.get("cleaned_length", 0),
            )

            # Extract document-level metadata once per file for structured queries.
            doc_entity_data = None
            email_metadata = {}
            if self.enable_entity_extraction and self.entity_extractor:
                full_text = "\n\n".join(
                    doc.content if hasattr(doc, 'content') else str(doc)
                    for doc in documents
                )
                try:
                    doc_entity_data = self.entity_extractor.extract_from_text(full_text)
                    logger.info(
                        f"Extracted {len(doc_entity_data.get('entities', []))} unique entities "
                        f"from {file_name}"
                    )
                except Exception as e:
                    logger.warning(f"Document-level entity extraction failed for {file_name}: {e}")
                    doc_entity_data = None

                if self.email_extractor:
                    try:
                        email_metadata = self.email_extractor.extract(full_text)
                    except Exception as e:
                        logger.warning(f"Email metadata extraction failed for {file_name}: {e}")
                        email_metadata = {}

            # Step 3: Chunk documents
            logger.info(f"Chunking document: {file_name}")
            chunks = self.chunker.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")

            # Add file_id + structured metadata to chunk payloads.
            for idx, chunk in enumerate(chunks):
                if hasattr(chunk, 'meta'):
                    chunk.meta['file_uuid'] = file_id
                    chunk.meta['file_id'] = file_id
                    chunk.meta['ingest_job_id'] = ingest_job_id
                    chunk.meta['ingest_version'] = ingest_job_id
                    chunk.meta['ingest_state'] = 'staging'
                    chunk.meta['is_first_chunk'] = (idx == 0)
                    chunk.meta['extraction_method'] = processing_info.get("extraction_method", "unknown")
                    chunk.meta['language'] = processing_info.get("language", "unknown")
                    chunk.meta['quality_score'] = processing_info.get("quality_score", 0.0)
                    if idx == 0 and doc_entity_data:
                        chunk.meta['document_entity_counts'] = doc_entity_data.get('entity_counts', {})
                    else:
                        chunk.meta['document_entity_counts'] = {}
                    chunk.meta['email_sender'] = ''
                    chunk.meta['email_receiver'] = ''
                    chunk.meta['email_date'] = ''
                    chunk.meta['email_subject'] = ''
                    chunk.meta['is_email'] = False

                    if self.enable_entity_extraction and self.entity_extractor:
                        try:
                            chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                            chunk_entities = self.entity_extractor.extract_from_chunk(chunk_text)
                            chunk.meta['entities'] = chunk_entities
                            chunk.meta['entity_names'] = [e['name'] for e in chunk_entities]
                        except Exception as e:
                            logger.warning(
                                f"Chunk-level entity extraction failed for {file_name} "
                                f"(chunk {idx}): {e}"
                            )
                            chunk.meta['entities'] = []
                            chunk.meta['entity_names'] = []

                        if email_metadata:
                            chunk.meta['email_sender'] = email_metadata.get('email_sender', '')
                            chunk.meta['email_receiver'] = email_metadata.get('email_receiver', '')
                            chunk.meta['email_date'] = email_metadata.get('email_date', '')
                            chunk.meta['email_subject'] = email_metadata.get('email_subject', '')
                            chunk.meta['is_email'] = email_metadata.get('is_email', False)
                    else:
                        chunk.meta['entities'] = []
                        chunk.meta['entity_names'] = []

            # Step 4: Generate embeddings
            logger.info(f"Generating embeddings for {file_name}")
            chunk_embeddings = self.embedding_generator.embed_chunks(
                chunks,
                use_cache=self.cache_embeddings
            )
            logger.info(f"Generated {len(chunk_embeddings)} embeddings")

            # Step 5: Insert into Qdrant
            logger.info(f"Inserting vectors into Qdrant")
            insertion_stats = self.vector_store.add_embeddings(
                chunk_embeddings,
                batch_size=self.insertion_batch_size
            )
            logger.info(f"Inserted {insertion_stats['inserted']} vectors")
            vectors_staged = insertion_stats.get("inserted", 0) > 0

            expected_count = len(chunk_embeddings)
            if insertion_stats.get("inserted", 0) != expected_count:
                raise RuntimeError(
                    f"Incomplete insert: expected={expected_count}, inserted={insertion_stats.get('inserted', 0)}"
                )

            commit_stats = self.vector_store.commit_ingest_job(
                file_uuid=file_id,
                ingest_job_id=ingest_job_id,
                ingest_version=ingest_job_id,
            )
            if not commit_stats.get("success"):
                raise RuntimeError(f"Failed to commit ingest job: {commit_stats.get('error')}")
            if commit_stats.get("committed_count", 0) != expected_count:
                raise RuntimeError(
                    f"Commit mismatch: expected={expected_count}, committed={commit_stats.get('committed_count', 0)}"
                )

            stale_cleanup = {"success": True, "deleted_count": 0}
            if self.indexing_mode == "online" and existing_uuid:
                stale_cleanup = self.vector_store.delete_file_except_ingest_job(
                    file_uuid=file_id,
                    keep_ingest_job_id=ingest_job_id,
                )
                if not stale_cleanup.get("success"):
                    logger.warning(
                        "Online stale cleanup failed for %s: %s",
                        file_name,
                        stale_cleanup.get("error"),
                    )

            # Step 6: Track the file (use file_id as UUID)
            self._remove_tracker_entries_for_uuid(file_id, keep_path=tracker_path)
            self.file_tracker.add_file(
                tracker_path,
                self.collection_name,
                file_uuid=file_id
            )

            logger.info(f"Successfully indexed {file_name}")

            return {
                "success": True,
                "file_id": file_id,
                "file_name": file_name,
                "ingest_job_id": ingest_job_id,
                "ingest_state": "ready",
                "chunks_created": len(chunks),
                "vectors_inserted": insertion_stats['inserted'],
                "committed_points": commit_stats.get("committed_count", 0),
                "stale_points_deleted": stale_cleanup.get("deleted_count", 0),
                "processing": processing_info,
                "collection_name": self.collection_name,
                "source_type": source_type,
                "source_uri": source_uri,
            }

        except UnsupportedFileTypeError as e:
            logger.warning("Unsupported file format during indexing for %s: %s", file_name, e)
            return {
                "success": False,
                "error_code": "unsupported_file_type",
                "error": str(e),
                "file_id": file_id,
                "file_name": file_name,
                "supported_formats": sorted(self.document_processor.supported_extensions),
                "ingest_job_id": ingest_job_id,
                "source_type": source_type,
                "source_uri": source_uri,
            }
        except OnlineDocumentProcessingError as e:
            logger.warning("Document preprocessing failed for %s: %s", file_name, e)
            return {
                "success": False,
                "error_code": e.code,
                "error": str(e),
                "file_id": file_id,
                "file_name": file_name,
                "ingest_job_id": ingest_job_id,
                "source_type": source_type,
                "source_uri": source_uri,
            }
        except Exception as e:
            if vectors_staged:
                rollback = self.vector_store.delete_by_ingest_job_id(ingest_job_id)
                if rollback.get("success") and rollback.get("deleted_count", 0) > 0:
                    logger.info(
                        "Rolled back %s staged vectors for ingest_job_id=%s",
                        rollback.get("deleted_count", 0),
                        ingest_job_id,
                    )
            logger.error(f"Failed to index {file_name}: {e}", exc_info=True)
            return {
                "success": False,
                "error_code": "indexing_failed",
                "error": str(e),
                "file_id": file_id,
                "file_name": file_name,
                "ingest_job_id": ingest_job_id,
                "source_type": source_type,
                "source_uri": source_uri,
            }

    def index_file_from_url(
            self,
            file_id: str,
            file_name: str,
            sas_url: str
    ) -> Dict[str, Any]:
        """
        Index a file from URL without permanent local storage.

        Args:
            file_id: Unique file identifier from backend
            file_name: Original filename
            sas_url: Azure SAS URL for download

        Returns:
            Dictionary with indexing results
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_file_path = self.download_file(sas_url, file_name, temp_dir_path)

            if not temp_file_path:
                return {
                    "success": False,
                    "error_code": "download_failed",
                    "error": "Failed to download file",
                    "file_id": file_id,
                    "file_name": file_name,
                    "source_type": "sas_url",
                    "source_uri": sas_url,
                }

            virtual_path = Path(f"url://{file_id}/{file_name}")
            return self._index_file_path(
                file_id=file_id,
                file_name=file_name,
                source_file_path=temp_file_path,
                tracker_path=virtual_path,
                source_type="sas_url",
                source_uri=sas_url,
            )

    def index_file_from_local_path(
            self,
            file_id: str,
            local_path: str,
            file_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Index a file already available on the server filesystem."""
        file_path = self._normalize_local_path(local_path)
        display_name = file_name or file_path.name

        return self._index_file_path(
            file_id=file_id,
            file_name=display_name,
            source_file_path=file_path,
            tracker_path=self._local_tracker_path(file_path),
            source_type="local_path",
            source_uri=str(file_path),
        )

    def delete_file(
            self,
            file_id: str,
            file_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a file's vectors from Qdrant by file_id.
        
        Args:
            file_id: Unique file identifier from backend
            file_name: Optional filename for logging
            
        Returns:
            Dictionary with deletion results
        """
        logger.info(f"Deleting file: {file_name or file_id} (ID: {file_id})")

        try:
            # Delete from Qdrant using file_id (which is stored as file_uuid)
            result = self.vector_store.delete_by_file_uuid(file_id)

            if result['success']:
                # Remove from tracking
                # Construct virtual path used during indexing
                virtual_path = f"url://{file_id}/{file_name or 'unknown'}"
                removed_uuid = self.file_tracker.remove_file(Path(virtual_path))
                if removed_uuid is None:
                    # Fallback: file_name may differ from ingestion-time value.
                    for tracked in self.file_tracker.get_all_tracked_files(self.collection_name):
                        if tracked.get("file_uuid") == file_id:
                            self.file_tracker.remove_file(Path(tracked["file_path"]))
                            break

                logger.info(f"Successfully deleted {result['deleted_count']} vectors for {file_name or file_id}")

                return {
                    "success": True,
                    "file_id": file_id,
                    "file_name": file_name,
                    "deleted_count": result['deleted_count']
                }
            else:
                return {
                    "success": False,
                    "file_id": file_id,
                    "file_name": file_name,
                    "error": result.get('error', 'Unknown error')
                }

        except Exception as e:
            logger.error(f"Failed to delete {file_name or file_id}: {e}", exc_info=True)
            return {
                "success": False,
                "file_id": file_id,
                "file_name": file_name,
                "error": str(e)
            }

    def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tracked file.
        
        Args:
            file_id: File identifier
            
        Returns:
            File info or None if not found
        """
        # Search through tracked files
        all_files = self.file_tracker.get_all_tracked_files(self.collection_name)

        for file_info in all_files:
            if file_info.get('file_uuid') == file_id:
                return file_info

        return None

    def list_indexed_files(self) -> Dict[str, Any]:
        """
        List all indexed files in the collection.
        
        Returns:
            Dictionary with file list and statistics
        """
        files = self.file_tracker.get_all_tracked_files(self.collection_name)
        collection_info = self.vector_store.get_collection_info()

        return {
            "collection_name": self.collection_name,
            "total_files": len(files),
            "total_vectors": collection_info.get('points_count', 0),
            "files": files
        }
