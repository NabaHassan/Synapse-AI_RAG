"""Helpers for presenting retriever source filenames to API consumers."""

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Dict, List


_OCR_TEXT_WRAPPER_EXTENSIONS = (
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".csv",
    ".tsv",
    ".rtf",
    ".html",
    ".htm",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
)


def normalize_source_filename(value: Any) -> Any:
    """Return the original source basename for OCR text-wrapper filenames."""
    if not isinstance(value, str):
        return value

    raw_value = value.strip().split("?", 1)[0].split("#", 1)[0]
    filename = PureWindowsPath(PurePosixPath(raw_value).name).name
    lowered = filename.lower()
    for original_ext in _OCR_TEXT_WRAPPER_EXTENSIONS:
        if lowered.endswith(f"{original_ext}.txt"):
            return filename[:-4]
    return filename


def normalize_citations_sources(citations: list) -> list:
    """Normalize source filename fields in citation dictionaries."""
    if not citations:
        return citations

    normalized: List[Any] = []
    for item in citations:
        if not isinstance(item, dict):
            normalized.append(item)
            continue

        updated: Dict[str, Any] = dict(item)
        for key in ("source", "source_file", "file_name"):
            if key in updated:
                updated[key] = normalize_source_filename(updated.get(key))
        normalized.append(updated)
    return normalized


def normalize_result_citations(payload: Any) -> Any:
    """Return a copy of an API result payload with normalized citations."""
    if not isinstance(payload, dict) or "citations" not in payload:
        return payload

    normalized = dict(payload)
    normalized["citations"] = normalize_citations_sources(normalized.get("citations"))
    return normalized
