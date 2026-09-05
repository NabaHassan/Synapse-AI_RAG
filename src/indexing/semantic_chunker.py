"""
Semantic Chunker for RAG Pipeline
Supports both LangChain's SemanticChunker and Haystack's DocumentSplitter
"""

from typing import List, Dict, Any, Optional


class Chunk:
    """Structure for a document chunk"""

    def __init__(
            self,
            content: str,
            chunk_id: str,
            source_metadata: Dict[str, Any],
            chunk_index: int,
            total_chunks: int,
            start_char: Optional[int] = None,
            end_char: Optional[int] = None,
            overlap_with_previous: bool = False
    ):
        self.content = content
        self.chunk_id = chunk_id
        self.source_metadata = source_metadata
        self.chunk_index = chunk_index
        self.total_chunks = total_chunks
        self.start_char = start_char
        self.end_char = end_char
        self.overlap_with_previous = overlap_with_previous

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "overlap_with_previous": self.overlap_with_previous,
            "source_metadata": self.source_metadata
        }

    def __repr__(self) -> str:
        return f"Chunk(id={self.chunk_id}, index={self.chunk_index}/{self.total_chunks}, chars={len(self.content)})"
