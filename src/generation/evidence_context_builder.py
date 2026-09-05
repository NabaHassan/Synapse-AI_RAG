"""Prompt context helpers for accepted evidence packets."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class EvidenceContextBuilder:
    """Build compact source-grouped context from evidence packets."""

    def build_context(self, evidence_packets: Iterable[Any], *, max_chars_per_packet: int = 1200) -> str:
        blocks: List[str] = []
        for index, packet in enumerate(evidence_packets or [], start=1):
            packet_dict = packet.to_dict() if hasattr(packet, "to_dict") else dict(packet or {})
            text = str(packet_dict.get("text") or "").strip()
            if len(text) > max_chars_per_packet:
                text = text[:max_chars_per_packet].rstrip() + "..."
            blocks.append(
                "\n".join(
                    [
                        f"SOURCE {index}",
                        f"source_file: {packet_dict.get('source_file', 'Unknown')}",
                        f"chunk_id: {packet_dict.get('chunk_id', 'unknown')}",
                        f"support_type: {packet_dict.get('support_type', 'unknown')}",
                        f"supported_fields: {', '.join(packet_dict.get('supports') or [])}",
                        "excerpt:",
                        text,
                    ]
                )
            )
        return "\n\n".join(blocks)

    def limitations(self, evidence_plan: Any, admission_result: Any) -> Dict[str, Any]:
        plan = evidence_plan.to_dict() if hasattr(evidence_plan, "to_dict") else dict(evidence_plan or {})
        status = getattr(admission_result, "admission_status", None)
        supported = sorted({
            item
            for packet in getattr(admission_result, "accepted_packets", []) or []
            for item in getattr(packet, "supports", []) or []
        })
        unsupported: List[str] = []
        if status in {"missing_requested_detail", "insufficient_evidence", "background_only"}:
            intent = str(plan.get("query_intent") or "requested_detail")
            unsupported.append(intent)
        return {
            "supported": supported,
            "unsupported": unsupported,
            "admission_status": status,
        }

