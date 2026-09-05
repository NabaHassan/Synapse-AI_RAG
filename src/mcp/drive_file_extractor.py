"""
Shared binary-to-text extraction for Google Drive and OneDrive MCP file reads.

Uses the same indexing stack as KB ingestion (DocumentExtractor + pptx helper)
so uploaded Office files (.docx, .xlsx, .pptx, .pdf, etc.) behave consistently
across both cloud drives.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_TEXT_MIME_MARKERS = (
    "text/plain",
    "text/markdown",
    "application/json",
    "text/html",
    "application/javascript",
)


def resolve_drive_file_suffix(mime_type: str, filename: str = "") -> Optional[str]:
    """Map mime type / filename to a temp-file suffix for extraction."""
    mime_lower = (mime_type or "").lower()
    name_lower = (filename or "").lower()

    if "pdf" in mime_lower or name_lower.endswith(".pdf"):
        return ".pdf"
    if "wordprocessingml" in mime_lower or name_lower.endswith(".docx"):
        return ".docx"
    if "msword" in mime_lower or name_lower.endswith(".doc"):
        return ".doc"
    if "spreadsheetml" in mime_lower or name_lower.endswith(".xlsx"):
        return ".xlsx"
    if "ms-excel" in mime_lower or name_lower.endswith(".xls"):
        return ".xls"
    if "presentationml" in mime_lower or name_lower.endswith(".pptx"):
        return ".pptx"
    if "ms-powerpoint" in mime_lower or name_lower.endswith(".ppt"):
        return ".ppt"
    if "csv" in mime_lower or name_lower.endswith(".csv"):
        return ".csv"
    if name_lower.endswith(".txt"):
        return ".txt"
    if name_lower.endswith(".md"):
        return ".md"
    if name_lower.endswith(".json"):
        return ".json"
    if name_lower.endswith((".html", ".htm")):
        return ".html"
    return None


def _extract_from_temp_path(temp_path: str, suffix: str, *, filename: str) -> Optional[str]:
    from src.indexing.document_extraction import DocumentExtractor

    extractor = DocumentExtractor()
    path = Path(temp_path)
    result = None

    if suffix == ".pdf":
        result = extractor.extract_pdf(path)
    elif suffix == ".docx":
        result = extractor.extract_docx(path)
    elif suffix == ".doc":
        result = extractor.extract_doc(path)
    elif suffix in (".xlsx", ".xls"):
        result = extractor.extract_excel(path)
    elif suffix == ".csv":
        result = extractor.extract_csv(path)
    elif suffix == ".pptx" or suffix == ".ppt":
        from src.mcp.pptx_text_extractor import extract_pptx_text

        result = extract_pptx_text(temp_path)
    elif suffix == ".txt":
        result = extractor.extract_txt(path)
    elif suffix == ".md":
        result = extractor.extract_markdown(path)
    elif suffix == ".json":
        result = extractor.extract_json(path)
    elif suffix == ".html":
        result = extractor.extract_html(path)

    if not result:
        logger.warning("No extractor registered for suffix=%s filename=%r", suffix, filename)
        return None

    if result.get("success"):
        text = result.get("full_text") or ""
        if isinstance(text, str) and text.strip():
            logger.info(
                "drive_file_extractor: extracted suffix=%s filename=%r method=%s chars=%d",
                suffix,
                filename,
                result.get("extraction_method"),
                len(text),
            )
            return text

    error = result.get("error") or "unknown extraction error"
    logger.warning(
        "drive_file_extractor: extraction failed suffix=%s filename=%r error=%s",
        suffix,
        filename,
        error,
    )
    return None


def extract_drive_file_text(
    raw_bytes: bytes,
    *,
    mime_type: str = "",
    filename: str = "",
    source_label: str = "cloud drive",
) -> str:
    """
    Extract plain text from downloaded Drive / OneDrive file bytes.

    Args:
        raw_bytes: Raw file content from Drive or Graph API download.
        mime_type: MIME type when known.
        filename: Original filename for suffix detection.
        source_label: Used in user-facing fallback messages (e.g. "Google Drive").
    """
    if not raw_bytes:
        logger.warning("drive_file_extractor: empty bytes for filename=%r", filename)
        return f"[Unable to read empty file from {source_label}.]"

    mime_lower = (mime_type or "").lower()
    if any(marker in mime_lower for marker in _TEXT_MIME_MARKERS):
        text = raw_bytes.decode("utf-8", errors="replace")
        logger.info(
            "drive_file_extractor: decoded text file filename=%r mime=%r chars=%d",
            filename,
            mime_type,
            len(text),
        )
        return text

    suffix = resolve_drive_file_suffix(mime_type, filename)
    if not suffix:
        logger.warning(
            "drive_file_extractor: unsupported type mime=%r filename=%r",
            mime_type,
            filename,
        )
        return (
            f"[Content of type {mime_type or 'unknown'} cannot be converted to text. "
            f"Open {filename or 'the file'} in {source_label} or convert to a supported format.]"
        )

    temp_file_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=suffix) as temp_file:
            temp_file.write(raw_bytes)
            temp_file_path = temp_file.name

        extracted = _extract_from_temp_path(temp_file_path, suffix, filename=filename)
        if extracted:
            return extracted

        if suffix in (".pptx", ".ppt"):
            return (
                f"[PPTX on {source_label} has no extractable text (image-only slides or missing "
                "python-pptx on server). Try opening in Slides/PowerPoint or add text to shapes.]"
            )
        if suffix in (".xlsx", ".xls"):
            return (
                f"[Unable to extract spreadsheet text from {filename or 'file'}. "
                "Ensure openpyxl and pandas are installed on the server.]"
            )
        if suffix in (".docx", ".doc"):
            return (
                f"[Unable to extract Word document text from {filename or 'file'}. "
                "Ensure python-docx (and antiword for .doc) are available on the server.]"
            )
        return (
            f"[Unable to extract text from {mime_type or suffix} on {source_label}. "
            "Try converting to Google Docs/Sheets or a plain text format.]"
        )
    except Exception as exc:
        logger.error(
            "drive_file_extractor: unexpected failure filename=%r mime=%r: %s",
            filename,
            mime_type,
            exc,
            exc_info=True,
        )
        return f"[File extraction failed for {filename or 'file'}: {exc}]"
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError as unlink_exc:
                logger.warning(
                    "drive_file_extractor: failed to remove temp file %s: %s",
                    temp_file_path,
                    unlink_exc,
                )
