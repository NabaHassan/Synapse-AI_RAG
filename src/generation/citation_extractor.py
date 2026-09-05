"""
Citation Extractor - Extract and validate citations from LLM responses.

This module provides functionality to:
- Extract citation markers ([N]) from generated text using regex
- Validate citation numbers against source documents
- Map citations to source metadata (filename, page, content)
- Remove hallucinated or invalid citations
- Format citations for output
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Set

from src.utils.source_normalization import normalize_source_filename

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Structured citation with metadata."""
    id: int
    source_file: str
    page: int
    chunk_id: str
    text_snippet: str
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary."""
        return {
            "id": self.id,
            "source": normalize_source_filename(self.source_file),
            "page": self.page,
            "chunk_id": self.chunk_id,
            "text": self.text_snippet,
            "relevance": round(self.relevance_score, 4)
        }


class CitationExtractor:
    """Extract and validate citations from LLM-generated responses."""

    def __init__(self):
        """Initialize the citation extractor."""
        logger.info("Initializing CitationExtractor")

        # Regex patterns for citation extraction
        self.citation_pattern = re.compile(r'\[(\d+)\]')
        self.multi_citation_pattern = re.compile(r'\[(\d+(?:,\s*\d+)*)\]')

        logger.info("CitationExtractor initialized")

    def extract_citations(
            self,
            response_text: str,
            source_documents: List[Dict[str, Any]]
    ) -> Tuple[List[Citation], Set[int], Set[int]]:
        """
        Extract citations from response text and map to source documents.
        
        Args:
            response_text: LLM-generated response with [N] citations
            source_documents: List of source documents with metadata
            
        Returns:
            Tuple of (citations, valid_ids, invalid_ids)
        """
        logger.info("Extracting citations from response")
        logger.info(f"   Response length: {len(response_text)} chars")
        logger.info(f"   Source documents: {len(source_documents)}")

        # Find all citation markers [N]
        citation_matches = self.citation_pattern.findall(response_text)
        logger.info(f"   Found {len(citation_matches)} citation markers")

        # Get unique citation IDs
        citation_ids = set(int(cid) for cid in citation_matches)
        logger.info(f"   Unique citations: {sorted(citation_ids)}")

        # Separate valid and invalid citations
        valid_ids = set()
        invalid_ids = set()
        citations = []

        for cid in sorted(citation_ids):
            idx = cid - 1  # Convert 1-indexed to 0-indexed

            if 0 <= idx < len(source_documents):
                valid_ids.add(cid)
                doc = source_documents[idx]

                # Create citation object
                citation = Citation(
                    id=cid,
                    source_file=doc.get('source_file', 'Unknown'),
                    page=doc.get('page', 0),
                    chunk_id=doc.get('chunk_id', 'unknown'),
                    text_snippet=self._truncate_text(doc.get('content', ''), max_length=200),
                    relevance_score=doc.get('rerank_score', 0.0)
                )
                citations.append(citation)
            else:
                invalid_ids.add(cid)
                logger.warning(f"Invalid citation [{cid}]: out of range")

        logger.info(f"Extracted {len(valid_ids)} valid citations")
        if invalid_ids:
            logger.warning(f"Found {len(invalid_ids)} invalid citations: {sorted(invalid_ids)}")

        return citations, valid_ids, invalid_ids

    @staticmethod
    def remove_invalid_citations(
            response_text: str,
            invalid_ids: Set[int]
    ) -> str:
        """
        Remove invalid citation markers from response text.
        
        Args:
            response_text: Original response with citations
            invalid_ids: Set of invalid citation IDs
            
        Returns:
            Cleaned response text
        """
        if not invalid_ids:
            return response_text

        logger.info(f"Removing {len(invalid_ids)} invalid citations")

        cleaned_text = response_text
        for cid in invalid_ids:
            # Remove [N] markers for invalid citations
            pattern = re.compile(rf'\[{cid}\]')
            cleaned_text = pattern.sub('', cleaned_text)

        # Clean up extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    def validate_citation_coverage(
            self,
            response_text: str,
            min_coverage: float = 0.3
    ) -> Dict[str, Any]:
        """
        Check if response has sufficient citation coverage.
        
        Args:
            response_text: Response text to validate
            min_coverage: Minimum required citation coverage (0.0-1.0)
            
        Returns:
            Validation result with coverage metrics
        """
        logger.info("Validating citation coverage")

        # Count sentences and citations
        sentences = [s.strip() for s in re.split(r'[.!?]+', response_text) if s.strip()]
        num_sentences = len(sentences)

        # Count sentences with citations
        sentences_with_citations = sum(
            1 for s in sentences if self.citation_pattern.search(s)
        )

        # Calculate coverage
        coverage = sentences_with_citations / num_sentences if num_sentences > 0 else 0.0

        # Check if coverage meets threshold
        sufficient = coverage >= min_coverage

        result = {
            "total_sentences": num_sentences,
            "cited_sentences": sentences_with_citations,
            "coverage": round(coverage, 3),
            "threshold": min_coverage,
            "sufficient": sufficient
        }

        logger.info(f"   Citation coverage: {coverage:.2%} ({sentences_with_citations}/{num_sentences})")
        logger.info(f"   Meets threshold: {sufficient}")

        return result

    def format_citations(
            self,
            citations: List[Citation],
            style: str = "numbered"
    ) -> str:
        """
        Format citations for display.
        
        Args:
            citations: List of citation objects
            style: Citation style ("numbered", "apa", "compact")
            
        Returns:
            Formatted citation string
        """
        if not citations:
            return "No citations found."

        if style == "numbered":
            formatted = []
            for cit in citations:
                formatted.append(
                    f"[{cit.id}] {cit.source_file}, p. {cit.page} "
                    f"(relevance: {cit.relevance_score:.3f})"
                )
            return "\n".join(formatted)

        elif style == "apa":
            formatted = []
            for cit in citations:
                formatted.append(
                    f"{cit.source_file}, page {cit.page}. "
                    f"Retrieved from document chunk {cit.chunk_id}"
                )
            return "\n".join(formatted)

        elif style == "compact":
            sources = set(f"{cit.source_file} (p.{cit.page})" for cit in citations)
            return "; ".join(sorted(sources))

        else:
            logger.warning(f"Unknown citation style: {style}, using 'numbered'")
            return self.format_citations(citations, style="numbered")

    def get_citation_summary(
            self,
            response_text: str,
            citations: List[Citation]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive citation summary.
        
        Args:
            response_text: Original response text
            citations: Extracted citations
            
        Returns:
            Citation summary with metrics
        """
        citation_ids = [cit.id for cit in citations]
        unique_sources = len(set(cit.source_file for cit in citations))
        avg_relevance = sum(cit.relevance_score for cit in citations) / len(citations) if citations else 0.0

        # Validate coverage
        coverage = self.validate_citation_coverage(response_text)

        summary = {
            "total_citations": len(citations),
            "unique_citation_ids": len(citation_ids),
            "citation_ids": sorted(citation_ids),
            "unique_sources": unique_sources,
            "average_relevance": round(avg_relevance, 4),
            "coverage": coverage
        }

        return summary

    @staticmethod
    def _truncate_text(text: str, max_length: int = 200) -> str:
        """Truncate text to maximum length with ellipsis."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."
