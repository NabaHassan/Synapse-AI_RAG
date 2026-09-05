"""Config-driven post-generation answer cleanup."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Tuple

logger = logging.getLogger(__name__)


_BOLD_LABEL_RE = re.compile(r"^(\s*(?:[-*]\s*)?\*\*[^*]+\*\*)\s*:\s+(.+)$")
_SOURCE_LIMITATION_RE = re.compile(
    r"\b(?:does\s+not|doesn't|do\s+not|don't|not|no)\s+"
    r"(?:mention|include|state|specify|provide|identify|show|contain|appear|available|require)\b"
    r"|\b(?:provided|retrieved|available)\s+(?:context|source|sources|materials?)\b.*\b"
    r"(?:does\s+not|doesn't|do\s+not|don't|not|no)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_NUMBER_TO_WORD = {value: key for key, value in _NUMBER_WORDS.items()}
_HOUR_RE = re.compile(
    r"\b(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+hours?\b",
    re.IGNORECASE,
)
_TOTAL_HOUR_RE = re.compile(
    r"\b(?:total(?:ing)?|these|the)\s+"
    r"(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+hours?\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9§.]+", " ", (text or "").lower()).strip()


def _pattern_supported(pattern: str, source_text: str) -> bool:
    normalized = _normalize(pattern)
    return bool(normalized and normalized in source_text)


def cleanup_generated_answer(
        answer: str,
        source_texts: Iterable[str],
        cleanup_config: Dict[str, Any],
        query: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Remove configured unsupported answer details after generation."""
    metadata: Dict[str, Any] = {
        "enabled": bool((cleanup_config or {}).get("enabled")),
        "removed_line_count": 0,
        "trimmed_line_count": 0,
        "stripped_trailing": False,
        "source_limited": False,
        "numeric_consistency_fixes": 0,
        "matched_patterns": [],
    }

    if not metadata["enabled"] or not answer or not answer.strip():
        return answer, metadata

    source_text = _normalize("\n".join(text for text in source_texts if text))
    cleaned = _strip_trailing_sections(answer, source_text, cleanup_config, metadata)
    cleaned = _clean_unsupported_lines(cleaned, source_text, cleanup_config, metadata)
    cleaned = _fix_inconsistent_hour_totals(cleaned, metadata)
    cleaned = _normalize_spacing(cleaned)
    source_limited = _source_limited_answer_for_missing_detail(query, source_text, cleanup_config, metadata)
    if source_limited:
        cleaned = source_limited

    if (
        metadata["removed_line_count"]
        or metadata["trimmed_line_count"]
        or metadata["stripped_trailing"]
        or metadata["numeric_consistency_fixes"]
    ):
        logger.info(
            "Answer cleanup removed=%s trimmed=%s stripped_trailing=%s numeric_fixes=%s patterns=%s",
            metadata["removed_line_count"],
            metadata["trimmed_line_count"],
            metadata["stripped_trailing"],
            metadata["numeric_consistency_fixes"],
            metadata["matched_patterns"],
        )

    return cleaned, metadata


def _source_limited_answer_for_missing_detail(
        query: str,
        source_text: str,
        cleanup_config: Dict[str, Any],
        metadata: Dict[str, Any],
) -> str:
    query_norm = _normalize(query or "")
    if not query_norm:
        return ""

    for pattern in cleanup_config.get("source_limited_query_patterns") or []:
        pattern_text = str(pattern)
        pattern_norm = _normalize(pattern_text)
        if not pattern_norm or pattern_norm not in query_norm:
            continue
        if _pattern_supported(pattern_text, source_text):
            continue
        metadata["source_limited"] = True
        metadata["matched_patterns"].append(pattern_text)
        return str(cleanup_config.get("source_limited_answer") or "The provided sources do not specify this detail.")

    return ""


def _strip_trailing_sections(
        answer: str,
        source_text: str,
        cleanup_config: Dict[str, Any],
        metadata: Dict[str, Any],
) -> str:
    cleaned = answer
    for pattern in cleanup_config.get("strip_trailing_from_patterns") or []:
        if _pattern_supported(str(pattern), source_text):
            continue
        match = re.search(re.escape(str(pattern)), cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        cleaned = cleaned[:match.start()].rstrip()
        metadata["stripped_trailing"] = True
        metadata["matched_patterns"].append(str(pattern))
        break
    return cleaned


def _clean_unsupported_lines(
        answer: str,
        source_text: str,
        cleanup_config: Dict[str, Any],
        metadata: Dict[str, Any],
) -> str:
    patterns = [str(pattern) for pattern in cleanup_config.get("remove_if_unsupported_patterns") or []]
    if not patterns:
        return answer

    preserve_label = bool(cleanup_config.get("preserve_bold_label_on_removed_detail", True))
    cleaned_lines: List[str] = []
    skip_blank = False

    for line in answer.splitlines():
        if _is_source_limitation_line(line):
            if line.strip() or not skip_blank:
                cleaned_lines.append(line)
            skip_blank = False
            continue

        matched_pattern = _first_unsupported_pattern(line, patterns, source_text)
        if not matched_pattern:
            if line.strip() or not skip_blank:
                cleaned_lines.append(line)
            skip_blank = False
            continue

        label_match = _BOLD_LABEL_RE.match(line)
        if preserve_label and label_match:
            cleaned_lines.append(label_match.group(1).rstrip())
            metadata["trimmed_line_count"] += 1
        else:
            metadata["removed_line_count"] += 1
            skip_blank = True
        metadata["matched_patterns"].append(matched_pattern)

    return "\n".join(cleaned_lines)


def _is_source_limitation_line(line: str) -> bool:
    return bool(_SOURCE_LIMITATION_RE.search(line or ""))


def _first_unsupported_pattern(line: str, patterns: List[str], source_text: str) -> str:
    line_norm = _normalize(line)
    if not line_norm:
        return ""

    for pattern in patterns:
        pattern_norm = _normalize(pattern)
        if not pattern_norm or pattern_norm not in line_norm:
            continue
        if _pattern_supported(pattern, source_text):
            continue
        return pattern

    return ""


def _normalize_spacing(text: str) -> str:
    cleaned = re.sub(r'\n{3,}', '\n\n', (text or "").strip())
    return cleaned


def _fix_inconsistent_hour_totals(answer: str, metadata: Dict[str, Any]) -> str:
    lines = (answer or "").splitlines()
    if not lines:
        return answer

    component_hours: List[int] = []
    total_line_indexes: List[int] = []
    for index, line in enumerate(lines):
        total_match = _TOTAL_HOUR_RE.search(line)
        if total_match:
            total_line_indexes.append(index)
            continue
        if not re.match(r"\s*(?:[-*]\s+|\d+[.)]\s+)", line):
            continue
        for match in _HOUR_RE.finditer(line):
            value = _parse_number(match.group("number"))
            if value is not None:
                component_hours.append(value)

    if len(component_hours) < 2 or not total_line_indexes:
        return answer

    computed_total = sum(component_hours)
    if computed_total <= 0 or computed_total > 100:
        return answer

    updated = False
    for index in total_line_indexes:
        line = lines[index]
        match = _TOTAL_HOUR_RE.search(line)
        if not match:
            continue
        stated_total = _parse_number(match.group("number"))
        if stated_total is None or stated_total == computed_total:
            continue
        replacement = _format_number_like(match.group("number"), computed_total)
        start, end = match.span("number")
        lines[index] = line[:start] + replacement + line[end:]
        updated = True

    if updated:
        metadata["numeric_consistency_fixes"] = int(metadata.get("numeric_consistency_fixes") or 0) + 1
        metadata["matched_patterns"].append("inconsistent_hour_total")
        return "\n".join(lines)
    return answer


def _parse_number(value: str) -> int:
    raw = (value or "").strip().lower()
    if raw.isdigit():
        return int(raw)
    return _NUMBER_WORDS.get(raw)


def _format_number_like(original: str, value: int) -> str:
    if (original or "").strip().isdigit():
        return str(value)
    word = _NUMBER_TO_WORD.get(value, str(value))
    if original[:1].isupper():
        return word.capitalize()
    return word
