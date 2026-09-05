"""Evidence-aware metadata and contextual text for newly indexed chunks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.indexing.metadata_overlay import MetadataOverlayExtractor


class EvidenceIndexingAnnotator:
    """Annotate newly indexed chunks with generic evidence metadata.

    This runs during ingestion, before embedding. It does not infer a tenant
    domain from tenant ids or display names. All annotations come from source
    metadata and document/chunk text so Ownify tenants remain fully automated.
    """

    CONTEXTUAL_VERSION = "evidence_v1_contextual"

    def __init__(
        self,
        *,
        kb_id: str = "",
        contextual_retrieval: bool = True,
        max_contextual_prefix_chars: int = 420,
    ):
        self.kb_id = kb_id or ""
        self.contextual_retrieval = bool(contextual_retrieval)
        self.max_contextual_prefix_chars = max(120, int(max_contextual_prefix_chars))
        self.extractor = MetadataOverlayExtractor()

    def annotate_chunks(
        self,
        chunks: Iterable[Any],
        *,
        document_text: str = "",
        processing_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chunks_list = list(chunks or [])
        annotated = 0
        with_contextual_text = 0
        field_counts: Dict[str, int] = {}

        for chunk in chunks_list:
            if not hasattr(chunk, "meta"):
                continue

            chunk_text = str(getattr(chunk, "content", "") or "")
            payload = self._payload_for_chunk(chunk, chunk_text, processing_info or {})
            overlay = self.extractor.extract(payload, kb_id=self.kb_id).to_payload()
            overlay["payload_version"] = "evidence_v1"
            overlay["indexing_schema_version"] = "evidence_v1"
            overlay.update(self._native_metadata(payload, chunk_text, processing_info or {}, overlay))

            for key, value in overlay.items():
                if value in (None, "", []):
                    continue
                chunk.meta[key] = value
                field_counts[key] = field_counts.get(key, 0) + 1

            if self.contextual_retrieval:
                contextual_text = self._contextual_text(chunk_text, chunk.meta, document_text)
                chunk.meta["contextual_retrieval_text"] = contextual_text
                chunk.meta["contextual_retrieval_version"] = self.CONTEXTUAL_VERSION
                with_contextual_text += 1

            annotated += 1

        return {
            "enabled": True,
            "schema_version": "evidence_v1",
            "annotated_chunks": annotated,
            "contextual_chunks": with_contextual_text,
            "field_counts": field_counts,
        }

    @staticmethod
    def _payload_for_chunk(
        chunk: Any,
        chunk_text: str,
        processing_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        meta = dict(getattr(chunk, "meta", {}) or {})
        payload = dict(meta)
        payload["content"] = chunk_text
        for key in [
            "quality_score",
            "quality_flags",
            "ocr_or_extraction_quality",
            "headings",
            "has_tables",
            "table_count",
            "table_columns",
            "sheet_names",
            "extraction_method",
        ]:
            if processing_info.get(key) is not None:
                payload[key] = processing_info.get(key)
        return payload

    def _native_metadata(
            self,
            payload: Dict[str, Any],
            chunk_text: str,
            processing_info: Dict[str, Any],
            overlay: Dict[str, Any],
    ) -> Dict[str, Any]:
        source_file = self.extractor._source_file(payload)  # pylint: disable=protected-access
        source_family = self.extractor._source_family(source_file)  # pylint: disable=protected-access
        document_id = str(payload.get("document_id") or payload.get("file_id") or payload.get("file_uuid") or overlay.get("document_id") or "")
        document_type = self._document_type(payload, source_file, chunk_text, overlay)
        heading_path = self._heading_path(payload, processing_info)
        section_title = str(payload.get("section_title") or (heading_path[-1] if heading_path else "") or "").strip()
        section_id = str(payload.get("section_id") or overlay.get("section_id") or self._section_id(section_title) or "").strip()
        quality_flags = self._quality_flags(payload, processing_info, chunk_text, overlay)
        quality = self._quality_score(payload, processing_info, overlay, quality_flags)
        table_count = int(payload.get("table_count") or processing_info.get("table_count") or 0)
        has_tables = bool(payload.get("has_tables") or processing_info.get("has_tables") or table_count)

        native = {
            "document_id": document_id,
            "source_family": source_family,
            "source_type": overlay.get("source_type") or self._source_type_from_file(source_file),
            "document_type": document_type,
            "section_id": section_id,
            "section_title": section_title,
            "heading_path": heading_path,
            "has_tables": has_tables,
            "table_count": table_count,
            "table_columns": self._string_list(payload.get("table_columns") or processing_info.get("table_columns"), limit=80),
            "sheet_names": self._string_list(payload.get("sheet_names") or processing_info.get("sheet_names"), limit=40),
            "quality_flags": quality_flags,
            "ocr_or_extraction_quality": quality,
            "is_low_quality_chunk": quality < 0.55 or bool({"low_text_quality", "low_alpha_ratio", "ocr_fragmented_lines"} & set(quality_flags)),
        }
        return native

    def _contextual_text(self, chunk_text: str, meta: Dict[str, Any], document_text: str) -> str:
        facts = self._context_facts(meta)
        doc_hint = self._document_hint(document_text)
        parts: List[str] = []
        if facts:
            parts.append("Source context: " + "; ".join(facts) + ".")
        if doc_hint:
            parts.append("Document context: " + doc_hint)
        if parts:
            prefix = " ".join(parts)
            prefix = prefix[: self.max_contextual_prefix_chars].rstrip()
            return f"{prefix}\n\n{chunk_text}"
        return chunk_text

    @staticmethod
    def _context_facts(meta: Dict[str, Any]) -> List[str]:
        facts: List[str] = []
        for label, key in [
            ("document", "source_family"),
            ("source type", "source_type"),
            ("document type", "document_type"),
            ("section", "section_id"),
            ("heading", "section_title"),
            ("document id", "document_id"),
        ]:
            value = meta.get(key)
            if value:
                facts.append(f"{label}: {value}")

        topic_path = meta.get("topic_path")
        if isinstance(topic_path, list) and topic_path:
            facts.append("topics: " + ", ".join(str(item) for item in topic_path[:6]))

        answerable_fields = meta.get("answerable_fields")
        if isinstance(answerable_fields, list) and answerable_fields:
            facts.append("answerable fields: " + ", ".join(str(item) for item in answerable_fields[:6]))

        if meta.get("has_tables"):
            facts.append(f"contains tables: {meta.get('table_count') or 1}")
        quality_flags = meta.get("quality_flags")
        if isinstance(quality_flags, list) and quality_flags:
            facts.append("quality flags: " + ", ".join(str(item) for item in quality_flags[:4]))

        entities = meta.get("primary_entities") or meta.get("entity_names")
        if isinstance(entities, list) and entities:
            facts.append("entities: " + ", ".join(str(item) for item in entities[:6]))

        return facts[:10]

    @staticmethod
    def _document_type(payload: Dict[str, Any], source_file: str, chunk_text: str, overlay: Dict[str, Any]) -> str:
        existing = str(payload.get("document_type") or overlay.get("document_type") or "").strip()
        if existing and existing != "unknown":
            return existing
        lower = f"{source_file}\n{chunk_text[:2000]}".lower()
        suffix = Path(source_file).suffix.lower()
        if payload.get("is_email") or re.search(r"^\s*from:\s+.+\n\s*to:\s+", chunk_text, re.I | re.M):
            return "email"
        if suffix in {".csv", ".xls", ".xlsx"} or payload.get("has_tables"):
            return "table"
        if "invoice" in lower:
            return "invoice"
        if "policy" in lower or "terms" in lower:
            return "policy"
        if "contract" in lower or "agreement" in lower:
            return "contract"
        if re.search(r"\b(?:family|evidence|civil|penal|probate)\s+code\b", lower):
            return "statute"
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".md", ".txt", ".doc", ".docx"}:
            return suffix.lstrip(".")
        return "unknown"

    @staticmethod
    def _source_type_from_file(source_file: str) -> str:
        suffix = Path(source_file).suffix.lower().lstrip(".")
        return suffix or "unknown"

    @classmethod
    def _heading_path(cls, payload: Dict[str, Any], processing_info: Dict[str, Any]) -> List[str]:
        existing = payload.get("heading_path")
        if isinstance(existing, list) and existing:
            return cls._string_list(existing, limit=8)
        section = str(payload.get("section_title") or "").strip()
        if section:
            return [section[:160]]
        headings = cls._string_list(processing_info.get("headings"), limit=8)
        return headings[:2]

    @staticmethod
    def _section_id(section_title: str) -> str:
        if not section_title:
            return ""
        match = re.search(r"\b(?:section|article|chapter|part)?\s*([A-Z]?\d+(?:\.\d+)*)\b", section_title, re.I)
        if match:
            return match.group(1)
        slug = re.sub(r"[^a-z0-9]+", "-", section_title.lower()).strip("-")
        return slug[:80]

    @classmethod
    def _quality_flags(
            cls,
            payload: Dict[str, Any],
            processing_info: Dict[str, Any],
            chunk_text: str,
            overlay: Dict[str, Any],
    ) -> List[str]:
        flags: List[str] = []
        for source in [overlay.get("quality_flags"), payload.get("quality_flags"), processing_info.get("quality_flags")]:
            flags.extend(cls._string_list(source, limit=20))
        if len((chunk_text or "").strip()) < 160:
            flags.append("short_chunk")
        if payload.get("has_tables") or processing_info.get("has_tables"):
            flags.append("contains_tables")
        return list(dict.fromkeys(flag for flag in flags if flag))[:20]

    @staticmethod
    def _quality_score(
            payload: Dict[str, Any],
            processing_info: Dict[str, Any],
            overlay: Dict[str, Any],
            quality_flags: List[str],
    ) -> float:
        raw = (
            payload.get("ocr_or_extraction_quality")
            or processing_info.get("ocr_or_extraction_quality")
            or overlay.get("ocr_or_extraction_quality")
            or payload.get("quality_score")
            or processing_info.get("quality_score")
            or 0.0
        )
        try:
            score = float(raw)
        except (TypeError, ValueError):
            score = 0.0
        if "replacement_characters" in quality_flags:
            score = min(score, 0.6)
        if "low_alpha_ratio" in quality_flags or "ocr_fragmented_lines" in quality_flags:
            score = min(score, 0.5)
        return max(0.0, min(1.0, score))

    @staticmethod
    def _string_list(raw: Any, *, limit: int) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            values = [raw]
        cleaned = []
        for value in values:
            item = str(value).strip()
            if item and item not in cleaned:
                cleaned.append(item[:200])
        return cleaned[:limit]

    @staticmethod
    def _document_hint(document_text: str) -> str:
        text = re.sub(r"\s+", " ", (document_text or "").strip())
        if not text:
            return ""
        # Prefer leading title/header-like text, but keep it short enough that
        # the original chunk remains the dominant embedding input.
        return text[:220].rstrip()
