"""Evidence metadata overlay extraction for existing indexed chunks.

The overlay is intentionally deterministic and conservative. It enriches
legacy payloads without reembedding or deleting points.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class OverlayMetadata:
    payload_version: str = "evidence_v1_overlay"
    document_id: str = ""
    source_family: str = ""
    source_type: str = "unknown"
    section_id: Optional[str] = None
    topic_path: List[str] = field(default_factory=list)
    jurisdiction: Optional[str] = None
    document_type: Optional[str] = None
    answerable_fields: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)
    primary_entities: List[str] = field(default_factory=list)
    ocr_or_extraction_quality: Optional[float] = None
    email_sender: Optional[str] = None
    email_recipients: List[str] = field(default_factory=list)
    email_subject: Optional[str] = None
    email_date: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, "", [])}


class MetadataOverlayExtractor:
    """Extract evidence metadata from legacy Qdrant payloads."""

    LEGAL_SECTION_RE = re.compile(
        r"\b(?:section\s*)?((?:\d{3,5})(?:\.\d+)?(?:\s*(?:-|to|through)\s*\d{3,5}(?:\.\d+)?)?)(?!\d)",
        flags=re.IGNORECASE,
    )
    EMAIL_RE = re.compile(r"[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}")
    DATE_RE = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})\b",
        flags=re.IGNORECASE,
    )

    def extract(self, payload: Dict[str, Any], *, kb_id: str = "") -> OverlayMetadata:
        payload = dict(payload or {})
        content = str(payload.get("content") or "")
        source_file = self._source_file(payload)
        source_family = self._source_family(source_file)
        lower_blob = f"{source_file}\n{content[:4000]}".lower()
        profile_hint = self._profile_hint(kb_id, lower_blob)

        overlay = OverlayMetadata(
            document_id=self._document_id(payload, source_family),
            source_family=source_family,
            source_type=self._source_type(source_file, lower_blob),
            document_type=self._document_type(source_file, lower_blob),
            quality_flags=self._quality_flags(payload, content),
            primary_entities=self._primary_entities(payload, content),
        )

        self._apply_legal_metadata(overlay, source_file, content, lower_blob, profile_hint=profile_hint)
        self._apply_email_archive_metadata(overlay, payload, content, lower_blob)
        self._apply_product_metadata(overlay, lower_blob)
        overlay.answerable_fields = self._answerable_fields(lower_blob, overlay)
        overlay.ocr_or_extraction_quality = self._quality_score(content, overlay.quality_flags)
        return overlay

    @staticmethod
    def _profile_hint(kb_id: str, lower_blob: str) -> str:
        kb_lower = (kb_id or "").lower()
        if "epstein" in kb_lower:
            return "evidence_archive"
        if "cafl" in kb_lower or "family code" in lower_blob or "evidence code" in lower_blob:
            return "legal_precision"
        return "general"

    @staticmethod
    def _source_file(payload: Dict[str, Any]) -> str:
        return str(
            payload.get("source_filename")
            or payload.get("source")
            or payload.get("file_name")
            or payload.get("filename")
            or "unknown"
        )

    @staticmethod
    def _source_family(source_file: str) -> str:
        name = Path(source_file).name
        name = re.sub(r"\.(?:txt|md|markdown|pdf|docx?|html?|csv|xlsx?|json)$", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+\(\d+\)$", "", name)
        name = re.sub(r"[_\s-]+chunk[_\s-]*\d+$", "", name, flags=re.IGNORECASE)
        return name[:220] or "unknown"

    @staticmethod
    def _document_id(payload: Dict[str, Any], source_family: str) -> str:
        existing = payload.get("document_id") or payload.get("file_uuid") or payload.get("file_id")
        if existing:
            return str(existing)
        digest = hashlib.sha1(source_family.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"doc_{digest}"

    @staticmethod
    def _source_type(source_file: str, lower_blob: str) -> str:
        filename = source_file.lower()
        if "email" in lower_blob or re.search(r"\bfrom:\s+.+\bto:\s+", lower_blob):
            return "email"
        if "code" in lower_blob or re.search(r"\bsection\s+\d", lower_blob):
            return "statute"
        if filename.endswith(".pdf") or ".pdf" in filename:
            return "pdf"
        if filename.endswith((".html", ".htm")):
            return "webpage"
        return "unknown"

    @staticmethod
    def _document_type(source_file: str, lower_blob: str) -> str:
        if "deposition" in lower_blob:
            return "deposition"
        if "email" in lower_blob or re.search(r"\bfrom:\s+.+\bto:\s+", lower_blob):
            return "email"
        if "statute" in lower_blob or "code" in lower_blob:
            return "statute"
        if "invoice" in lower_blob:
            return "invoice"
        if "flight" in lower_blob and "log" in lower_blob:
            return "flight_log"
        return "unknown"

    def _apply_legal_metadata(
        self,
        overlay: OverlayMetadata,
        source_file: str,
        content: str,
        lower_blob: str,
        *,
        profile_hint: str,
    ) -> None:
        section_match = self.LEGAL_SECTION_RE.search(source_file) or self.LEGAL_SECTION_RE.search(content[:1200])
        legal_markers = re.search(r"\b(?:family|evidence|civil|penal|probate)\s+code\b", lower_blob)
        if section_match and profile_hint == "legal_precision":
            overlay.section_id = section_match.group(1).replace(" ", "")
        elif section_match and legal_markers and profile_hint != "evidence_archive":
            overlay.section_id = section_match.group(1).replace(" ", "")
        if legal_markers:
            overlay.jurisdiction = "california"
            overlay.document_type = "statute"
            if "evidence code" in lower_blob:
                overlay.topic_path.append("evidence_code")
            if "family code" in lower_blob:
                overlay.topic_path.append("family_code")

    def _apply_email_archive_metadata(
        self,
        overlay: OverlayMetadata,
        payload: Dict[str, Any],
        content: str,
        lower_blob: str,
    ) -> None:
        overlay.email_sender = payload.get("email_sender") or self._header_value(content, "from")
        recipients = payload.get("email_recipients") or payload.get("email_receiver") or payload.get("email_receivers")
        if isinstance(recipients, str):
            overlay.email_recipients = [item.strip() for item in re.split(r"[,;]", recipients) if item.strip()]
        elif isinstance(recipients, list):
            overlay.email_recipients = [str(item).strip() for item in recipients if str(item).strip()]
        else:
            to_value = self._header_value(content, "to")
            overlay.email_recipients = [item.strip() for item in re.split(r"[,;]", to_value or "") if item.strip()]
        overlay.email_subject = payload.get("email_subject") or self._header_value(content, "subject")
        overlay.email_date = payload.get("email_date") or self._header_value(content, "date")
        if not overlay.email_date:
            date_match = self.DATE_RE.search(content[:1500])
            if date_match:
                overlay.email_date = date_match.group(0)
        if overlay.email_sender or overlay.email_recipients or overlay.email_subject:
            overlay.source_type = "email"
            overlay.document_type = "email"
            overlay.topic_path.append("email")

    @staticmethod
    def _apply_product_metadata(overlay: OverlayMetadata, lower_blob: str) -> None:
        product_markers = {
            "pricing": ["pricing", "price", "billing", "subscription", "stripe", "plan"],
            "workflow": ["workflow", "onboarding", "launch", "setup", "portal"],
            "feature": ["feature", "agent", "analytics", "whatsapp", "dashboard", "rbac"],
            "policy": ["terms", "policy", "contract", "eligibility"],
        }
        for topic, markers in product_markers.items():
            if any(marker in lower_blob for marker in markers):
                overlay.topic_path.append(topic)
        if overlay.topic_path:
            overlay.source_type = overlay.source_type if overlay.source_type != "unknown" else "product_doc"

    @classmethod
    def _header_value(cls, content: str, header: str) -> Optional[str]:
        match = re.search(rf"^\s*{re.escape(header)}\s*:\s*(.+)$", content, flags=re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip()[:500] if match else None

    @staticmethod
    def _quality_flags(payload: Dict[str, Any], content: str) -> List[str]:
        flags: List[str] = []
        existing = payload.get("quality_flags")
        if isinstance(existing, str) and existing.strip():
            flags.append(existing.strip())
        elif isinstance(existing, list):
            flags.extend(str(item).strip() for item in existing if str(item).strip())
        if len((content or "").strip()) < 80:
            flags.append("short_chunk")
        if payload.get("is_low_quality_chunk"):
            flags.append("low_quality_chunk")
        if payload.get("has_tables"):
            flags.append("contains_tables")
        if "�" in content:
            flags.append("replacement_characters")
        if payload.get("extraction_error"):
            flags.append("extraction_error")
        if not MetadataOverlayExtractor._source_file(payload) or MetadataOverlayExtractor._source_file(payload) == "unknown":
            flags.append("missing_source_file")
        return list(dict.fromkeys(flags))

    @classmethod
    def _primary_entities(cls, payload: Dict[str, Any], content: str) -> List[str]:
        raw_entities = payload.get("entity_names") or payload.get("entities") or []
        entities: List[str] = []
        if isinstance(raw_entities, str):
            entities.extend([item.strip() for item in re.split(r"[,;]", raw_entities) if item.strip()])
        elif isinstance(raw_entities, list):
            for item in raw_entities:
                if isinstance(item, dict):
                    value = item.get("name") or item.get("text")
                else:
                    value = item
                if value:
                    entities.append(str(value).strip())
        # Conservative fallback for title-cased person/org-like names.
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", content[:3000]):
            if match not in entities:
                entities.append(match)
        emails = cls.EMAIL_RE.findall(content[:3000])
        for email in emails:
            if email not in entities:
                entities.append(email)
        return entities[:25]

    @staticmethod
    def _answerable_fields(lower_blob: str, overlay: OverlayMetadata) -> List[str]:
        fields: List[str] = []
        checks = {
            "definition": ["means", "defined", "definition"],
            "requirement": ["must", "shall", "required", "requirement"],
            "price": ["price", "pricing", "fee", "cost", "subscription"],
            "date": ["date", "dated", "when"],
            "email_sender": ["from:"],
            "email_recipient": ["to:"],
            "file_lookup": [],
        }
        for field_name, markers in checks.items():
            if markers and any(marker in lower_blob for marker in markers):
                fields.append(field_name)
        if overlay.source_family:
            fields.append("file_lookup")
        if overlay.section_id:
            fields.append("section_lookup")
        return sorted(set(fields))

    @staticmethod
    def _quality_score(content: str, quality_flags: Iterable[str]) -> float:
        score = 1.0
        flags = set(quality_flags)
        if "short_chunk" in flags:
            score -= 0.25
        if "replacement_characters" in flags:
            score -= 0.2
        if "extraction_error" in flags:
            score -= 0.4
        if "missing_source_file" in flags:
            score -= 0.15
        if len((content or "").strip()) > 1000:
            score += 0.05
        return round(max(0.0, min(1.0, score)), 3)
