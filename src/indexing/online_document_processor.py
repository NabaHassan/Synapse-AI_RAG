"""
Online document processing for URL-based ingestion.

This module reuses the shared extraction and cleaning logic in-memory so
the online indexing flow can benefit from the same preprocessing quality
without importing script-style pipeline modules or writing intermediate
JSON artifacts to disk.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.indexing.document_loader import Document, DocumentLoader, DocumentMetadata, load_document
from src.indexing.document_extraction import DocumentExtractor
from src.indexing.text_cleaning import TextCleaner

logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension is not supported for online indexing."""

    def __init__(self, file_name: str, supported_extensions: Sequence[str]):
        self.file_name = file_name
        self.supported_extensions = tuple(sorted(supported_extensions))
        message = (
            f"Unsupported file format for '{file_name}'. "
            f"Supported formats: {', '.join(self.supported_extensions)}"
        )
        super().__init__(message)


class OnlineDocumentProcessingError(RuntimeError):
    """Raised when extraction/cleaning fails for a supported file."""

    def __init__(self, message: str, code: str = "document_processing_failed"):
        self.code = code
        super().__init__(message)


class _BasicTextCleaner:
    """
    Lightweight fallback cleaner used when the shared TextCleaner cannot be initialized.
    """

    _WS_PATTERN = re.compile(r"\s+")

    def clean_text(self, text: str, source_file: str = "") -> Tuple[str, Dict[str, Any]]:
        del source_file
        if not text:
            return "", {"empty_input": True, "reduction_ratio": 1.0}

        original_length = len(text)
        # Conservative fallback: normalize whitespace and remove duplicate lines.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        deduped_lines: List[str] = []
        prev: Optional[str] = None
        for line in lines:
            normalized = self._WS_PATTERN.sub(" ", line).strip()
            if normalized and normalized != prev:
                deduped_lines.append(normalized)
                prev = normalized

        cleaned = "\n".join(deduped_lines).strip()
        final_length = len(cleaned)
        return cleaned, {
            "original_length": original_length,
            "final_length": final_length,
            "reduction_ratio": 1 - (final_length / original_length) if original_length else 0.0,
        }

    @staticmethod
    def detect_language(text: str) -> str:
        if not text:
            return "unknown"
        ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(len(text), 1)
        return "en" if ascii_ratio > 0.9 else "unknown"

    @staticmethod
    def quality_score(text: str) -> float:
        if not text:
            return 0.0
        word_count = len(text.split())
        if word_count < 20:
            return 0.2
        if word_count < 60:
            return 0.45
        return 0.7


class OnlineDocumentProcessor:
    """
    Extract and clean a single downloaded document in-memory.
    """

    def __init__(
            self,
            min_text_length: int = 200,
            min_quality_score: float = 0.4,
            remove_urls: bool = False,
            remove_emails: bool = False,
            use_unstructured_fallback: bool = True,
    ):
        self.min_text_length = min_text_length
        self.min_quality_score = min_quality_score
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.use_unstructured_fallback = use_unstructured_fallback

        self.extractor = self._build_extractor()
        self.cleaner = self._build_cleaner()
        self.supported_extensions = self._resolve_supported_extensions()

        logger.info(
            "OnlineDocumentProcessor ready (extractor=%s, cleaner=%s, formats=%s)",
            type(self.extractor).__name__ if self.extractor else "DocumentLoader",
            type(self.cleaner).__name__,
            ", ".join(sorted(self.supported_extensions)),
        )

    def _build_extractor(self):
        try:
            return DocumentExtractor(use_unstructured_fallback=self.use_unstructured_fallback)
        except Exception as exc:
            logger.warning("Failed to initialize DocumentExtractor: %s", exc)
            return None

    def _build_cleaner(self):
        try:
            return TextCleaner(
                remove_urls=self.remove_urls,
                remove_emails=self.remove_emails,
                fix_unicode=True,
                normalize_whitespace=True,
            )
        except Exception as exc:
            logger.warning("Failed to initialize TextCleaner: %s", exc)
            return _BasicTextCleaner()

    def _resolve_supported_extensions(self) -> Set[str]:
        if self.extractor is not None and hasattr(self.extractor, "SUPPORTED_EXTENSIONS"):
            extensions = set(getattr(self.extractor, "SUPPORTED_EXTENSIONS").keys())
            return {f".{ext.lower()}" for ext in extensions}
        return set(DocumentLoader.SUPPORTED_FORMATS)

    def is_supported_filename(self, file_name: str) -> bool:
        return Path(file_name).suffix.lower() in self.supported_extensions

    def _extract_with_document_loader(self, file_path: Path) -> Tuple[str, Dict[str, Any], int, str]:
        try:
            docs = load_document(str(file_path))
        except Exception as exc:
            raise OnlineDocumentProcessingError(
                f"Failed to load '{file_path.name}' with fallback loader: {exc}",
                code="extraction_failed",
            ) from exc

        merged_text = "\n\n".join(doc.content for doc in docs if getattr(doc, "content", ""))
        if not merged_text.strip():
            raise OnlineDocumentProcessingError(
                f"No content extracted from '{file_path.name}'.",
                code="no_content_extracted",
            )

        metadata: Dict[str, Any] = {}
        if docs and hasattr(docs[0], "metadata"):
            metadata_obj = docs[0].metadata
            if hasattr(metadata_obj, "to_dict"):
                metadata = metadata_obj.to_dict()

        return merged_text, metadata, len(docs), "document_loader"

    def _extract(self, file_path: Path) -> Tuple[str, Dict[str, Any], int, str]:
        if self.extractor is None:
            return self._extract_with_document_loader(file_path)

        result = self.extractor.extract(file_path)
        if not result.get("success"):
            error = result.get("error") or "Unknown extraction error"
            raise OnlineDocumentProcessingError(
                f"Extraction failed for '{file_path.name}': {error}",
                code="extraction_failed",
            )

        full_text = result.get("full_text", "")
        if not isinstance(full_text, str) or not full_text.strip():
            raise OnlineDocumentProcessingError(
                f"No content extracted from '{file_path.name}'.",
                code="no_content_extracted",
            )

        metadata = result.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return (
            full_text,
            metadata,
            int(result.get("page_count", 0) or 0),
            str(result.get("extraction_method", "unknown")),
        )

    def process_file(
            self,
            file_path: Path,
            original_file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        file_name = original_file_name or file_path.name
        suffix = Path(file_name).suffix.lower()

        if suffix not in self.supported_extensions:
            raise UnsupportedFileTypeError(file_name=file_name, supported_extensions=self.supported_extensions)

        extracted_text, extracted_metadata, page_count, extraction_method = self._extract(file_path)
        cleaned_text, cleaning_stats = self.cleaner.clean_text(extracted_text, source_file=file_name)
        cleaned_text = cleaned_text.strip()

        if not cleaned_text:
            raise OnlineDocumentProcessingError(
                f"File '{file_name}' became empty after cleaning.",
                code="empty_after_cleaning",
            )

        language = self.cleaner.detect_language(cleaned_text) if hasattr(self.cleaner, "detect_language") else "unknown"
        quality_score = (
            float(self.cleaner.quality_score(cleaned_text))
            if hasattr(self.cleaner, "quality_score")
            else 0.0
        )

        if len(cleaned_text) < self.min_text_length:
            raise OnlineDocumentProcessingError(
                (
                    f"Cleaned content for '{file_name}' is too short "
                    f"({len(cleaned_text)} chars < {self.min_text_length})."
                ),
                code="content_too_short",
            )

        if quality_score < self.min_quality_score:
            raise OnlineDocumentProcessingError(
                (
                    f"Cleaned content quality for '{file_name}' is below threshold "
                    f"({quality_score:.2f} < {self.min_quality_score:.2f})."
                ),
                code="low_quality_content",
            )

        metadata = DocumentMetadata(
            filename=file_name,
            filepath=str(file_path.absolute()),
            file_type=suffix.lstrip(".") or "unknown",
            page_number=1,
            total_pages=page_count if page_count > 0 else None,
            file_size=file_path.stat().st_size,
        )
        metadata.extra.update({
            "original_source": file_name,
            "language": language,
            "quality_score": quality_score,
            "extraction_method": extraction_method,
            "page_count": page_count,
            "cleaning_reduction_ratio": cleaning_stats.get("reduction_ratio", 0.0),
        })

        for key, value in extracted_metadata.items():
            if isinstance(value, (str, int, float, bool)) and key not in metadata.extra:
                metadata.extra[f"meta_{key}"] = value

        return {
            "documents": [Document(content=cleaned_text, metadata=metadata)],
            "processing_info": {
                "extraction_method": extraction_method,
                "language": language,
                "quality_score": quality_score,
                "page_count": page_count,
                "original_length": len(extracted_text),
                "cleaned_length": len(cleaned_text),
                "cleaning_reduction_ratio": cleaning_stats.get("reduction_ratio", 0.0),
            },
        }
