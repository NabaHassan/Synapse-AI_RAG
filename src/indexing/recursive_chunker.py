"""
Recursive Character Text Splitter - Reliable chunking with context preservation.

This chunker uses simple, predictable splitting that:
- Respects natural text boundaries (paragraphs, sentences, words)
- Preserves context with significant overlap
- Creates consistent chunk sizes
- Doesn't over-split like semantic chunkers
"""

import logging
import hashlib
from haystack import Document
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for recursive chunking."""
    chunk_size: int = 800  # Characters per chunk (enough context for embeddings)
    chunk_overlap: int = 200  # Overlap between chunks (ensures no info loss)
    min_chunk_size: int = 100  # Minimum chunk size (filters out tiny fragments)
    length_function: callable = len
    separators: List[str] = None  # Will be set in __post_init__
    keep_separator: bool = True  # Keep separators for context

    def __post_init__(self):
        if self.separators is None:
            # Order matters: try bigger separators first
            self.separators = [
                "\n\n",  # Paragraphs
                "\n",  # Lines
                ". ",  # Sentences
                ", ",  # Clauses
                " ",  # Words
                ""  # Characters (fallback)
            ]


class RecursiveCharacterTextSplitter:
    """
    Split text recursively using multiple separators.
    
    This is the RELIABLE chunker that:
    1. Creates chunks of consistent size (~800 chars)
    2. Uses 200 char overlap to prevent context loss
    3. Respects natural boundaries (paragraphs > sentences > words)
    4. Is PREDICTABLE - no semantic analysis that can fail
    """

    def __init__(
            self,
            chunk_size: int = 800,
            chunk_overlap: int = 200,
            min_chunk_size: int = 100,
            length_function: callable = len,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True
    ):
        """
        Initialize the recursive character text splitter.
        
        Args:
            chunk_size: Target size for each chunk (characters)
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (filters out tiny fragments)
            length_function: Function to measure text length
            separators: List of separators to try (None = use defaults)
            keep_separator: Whether to keep separators in output
        """
        self.config = ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            length_function=length_function,
            separators=separators,
            keep_separator=keep_separator
        )

        logger.info("   Initializing RecursiveCharacterTextSplitter")
        logger.info(f"   Chunk size: {self.config.chunk_size} chars")
        logger.info(f"   Chunk overlap: {self.config.chunk_overlap} chars")
        logger.info(f"   Min chunk size: {self.config.min_chunk_size} chars (filters tiny chunks)")
        logger.info(f"   Separators: {self.config.separators}")
        logger.info("   RecursiveCharacterTextSplitter initialized")

    def split_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Remove excessive whitespace but preserve structure
        text = text.strip()

        # If text is small enough, return as-is
        if self.config.length_function(text) <= self.config.chunk_size:
            return [text]

        # Try each separator recursively
        return self._split_text_recursive(text, self.config.separators)

    def _split_text_recursive(
            self,
            text: str,
            separators: List[str]
    ) -> List[str]:
        """
        Recursively split text using different separators.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of text chunks
        """
        # Base case: text is small enough
        if self.config.length_function(text) <= self.config.chunk_size:
            return [text]

        # Try each separator
        for i, separator in enumerate(separators):
            if separator == "":
                # Last resort: split by characters
                return self._split_by_characters(text)

            if separator in text:
                # Split by current separator
                splits = text.split(separator)

                # Keep separator if configured
                if self.config.keep_separator and separator != "":
                    splits = self._merge_separator(splits, separator)

                # Merge small splits and handle overlaps
                return self._merge_splits(splits, separators[i + 1:])

        # Fallback: split by characters
        return self._split_by_characters(text)

    @staticmethod
    def _merge_separator(
            splits: List[str],
            separator: str
    ) -> List[str]:
        """
        Re-add separator to splits for context preservation.
        
        Args:
            splits: List of split strings
            separator: Separator that was used
            
        Returns:
            Splits with separator restored
        """
        if not splits:
            return splits

        result = []
        for i, split in enumerate(splits):
            if i < len(splits) - 1:
                # Add separator back (except for last item)
                result.append(split + separator)
            else:
                result.append(split)

        return result

    def _merge_splits(
            self,
            splits: List[str],
            remaining_separators: List[str]
    ) -> List[str]:
        """
        Merge splits intelligently with overlap.
        
        Args:
            splits: List of text splits
            remaining_separators: Separators for recursive splitting
            
        Returns:
            Final list of chunks with proper overlap
        """
        final_chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = self.config.length_function(split)

            # If single split is too large, recursively split it
            if split_len > self.config.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    final_chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Recursively split the large piece
                if remaining_separators:
                    sub_chunks = self._split_text_recursive(split, remaining_separators)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.extend(self._split_by_characters(split))

                continue

            # Check if adding this split exceeds chunk size
            if current_length + split_len > self.config.chunk_size:
                if current_chunk:
                    # Save current chunk
                    final_chunks.append("".join(current_chunk))

                    # Start new chunk with overlap
                    overlap_text = "".join(current_chunk)
                    if len(overlap_text) > self.config.chunk_overlap:
                        # Keep last overlap_size characters
                        overlap_text = overlap_text[-self.config.chunk_overlap:]

                    current_chunk = [overlap_text, split]
                    current_length = self.config.length_function(overlap_text) + split_len
                else:
                    current_chunk = [split]
                    current_length = split_len
            else:
                # Add to current chunk
                current_chunk.append(split)
                current_length += split_len

        # Add remaining chunk
        if current_chunk:
            final_chunks.append("".join(current_chunk))

        return final_chunks

    def _split_by_characters(self, text: str) -> List[str]:
        """
        Split text by characters as last resort.
        
        Args:
            text: Text to split
            
        Returns:
            List of character-level chunks
        """
        chunks = []
        start = 0
        text_len = self.config.length_function(text)

        while start < text_len:
            end = min(start + self.config.chunk_size, text_len)
            chunks.append(text[start:end])
            start = end - self.config.chunk_overlap

            # Prevent infinite loop
            if start >= end:
                break

        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of Haystack Document objects
            
        Returns:
            List of chunked Document objects with metadata
        """
        logger.info(f"Splitting {len(documents)} documents into chunks")

        all_chunks = []
        total_chunks_created = 0
        total_chunks_filtered = 0

        for doc_idx, doc in enumerate(documents, 1):
            # Get document content
            content = doc.content
            if not content or not content.strip():
                logger.warning(f"   Skipping empty document {doc_idx}")
                continue

            # Get source metadata (handle both Haystack and custom Document)
            if hasattr(doc, 'meta'):
                # Haystack Document
                source_file = doc.meta.get('source', doc.meta.get('file_path', 'Unknown'))
                source_filepath = doc.meta.get('file_path', source_file)
                file_type = doc.meta.get('file_type', 'unknown')
                page = doc.meta.get('page', doc.meta.get('page_number', 0))

            elif hasattr(doc, 'metadata'):
                # Custom Document from document_loader
                metadata_dict = doc.metadata.to_dict() if hasattr(doc.metadata, 'to_dict') else {}
                source_file = metadata_dict.get('filename', metadata_dict.get('filepath', 'Unknown'))
                source_filepath = metadata_dict.get('filepath', source_file)
                file_type = metadata_dict.get('file_type', 'unknown')
                page = metadata_dict.get('page_number', 0)

            else:
                source_file = 'Unknown'
                source_filepath = ''
                file_type = 'unknown'
                page = 0
                logger.warning(f"   Document has neither 'meta' nor 'metadata' attribute!")

            logger.info(f"   Chunking document {doc_idx}/{len(documents)}: {source_file} ({len(content)} chars)")

            # Split into chunks
            text_chunks = self.split_text(content)
            chunks_before_filter = len(text_chunks)

            # Filter out tiny chunks
            filtered_chunks = []
            for chunk_text in text_chunks:
                chunk_len = len(chunk_text)
                if chunk_len >= self.config.min_chunk_size:
                    filtered_chunks.append(chunk_text)
                else:
                    total_chunks_filtered += 1
                    logger.warning(
                        f"    Filtered tiny chunk ({chunk_len} chars < {self.config.min_chunk_size} min) "
                        f"from {source_file}, page {page}"
                    )

            text_chunks = filtered_chunks
            chunks_created = len(text_chunks)
            total_chunks_created += chunks_created

            if chunks_before_filter > chunks_created:
                logger.info(
                    f"   Created {chunks_created} chunks from {source_file} "
                    f"({chunks_before_filter - chunks_created} tiny chunks filtered)"
                )
            else:
                logger.info(f"   Created {chunks_created} chunks from {source_file}")

            # Create Document objects for each chunk
            for chunk_idx, chunk_text in enumerate(text_chunks):
                # Generate unique chunk ID
                chunk_id = self._generate_chunk_id(source_file, page, chunk_idx)

                # Create new document
                # Create metadata dict compatible with both Document types
                # Create chunk document with extracted metadata
                chunk_doc = Document(
                    content=chunk_text,
                    meta={
                        'chunk_id': chunk_id,
                        'chunk_index': chunk_idx,
                        'total_chunks': chunks_created,
                        'source_filename': source_file,  # Match vector_store expectations
                        'source_filepath': source_filepath,  # Use extracted variable
                        'page_number': page,  # Match vector_store expectations
                        'file_type': file_type,  # Use extracted variable
                        'chunk_size': len(chunk_text),
                        'chunker': 'recursive'
                    }
                )

                all_chunks.append(chunk_doc)

        logger.info(f"Created {total_chunks_created} total chunks from {len(documents)} documents")
        if total_chunks_filtered > 0:
            logger.info(f"   Filtered out {total_chunks_filtered} tiny chunks (< {self.config.min_chunk_size} chars)")
            logger.info(
                f"   Filter rate: {(total_chunks_filtered / (total_chunks_created + total_chunks_filtered) * 100):.1f}%"
            )
        logger.info(f"   Average chunks per document: {total_chunks_created / len(documents):.1f}")

        # Calculate statistics
        chunk_sizes = [len(chunk.content) for chunk in all_chunks]
        if chunk_sizes:
            logger.info(f"   Chunk size range: {min(chunk_sizes)}-{max(chunk_sizes)} chars")
            logger.info(f"   Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.1f} chars")
            logger.info(f"   All chunks meet minimum size requirement ({self.config.min_chunk_size}+ chars)")

        return all_chunks

    @staticmethod
    def _generate_chunk_id(source: str, page: int, chunk_index: int) -> str:
        """
        Generate unique chunk ID.
        
        Args:
            source: Source document name
            page: Page number
            chunk_index: Index of chunk within document
            
        Returns:
            Unique chunk ID
        """
        # Create stable hash-based ID
        content = f"{source}_{page}_{chunk_index}"
        hash_obj = hashlib.md5(content.encode())
        return f"chunk_{hash_obj.hexdigest()[:8]}_{chunk_index}"

    def get_config(self) -> Dict[str, Any]:
        """Get chunker configuration."""
        return {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
            "separators": self.config.separators,
            "keep_separator": self.config.keep_separator,
            "chunker_type": "recursive"
        }


def create_recursive_chunker(
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        **kwargs
) -> RecursiveCharacterTextSplitter:
    """
    Factory function to create RecursiveCharacterTextSplitter.
    
    Args:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size (filters out tiny fragments)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured RecursiveCharacterTextSplitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        **kwargs
    )
