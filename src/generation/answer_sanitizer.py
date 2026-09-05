"""
Utilities for sanitizing generated assistant answers before storage.

These helpers remove trailing model artifacts such as self-generated follow-up
questions, second-answer bleed, and meta-commentary that should never be saved
into conversation memory or fed into summarization.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


_TRAILING_META_PATTERNS = [
    r'\n+No further elaboration required.*$',
    r'\n+End of message\.?$',
    r'\n+This response provides guidance.*$',
    r'\n+unless prompted otherwise\.?$',
    r'\n+What else\s*$',
    r'\n+This response continues.*$',
    r'\n+The (above|prior) (response|discussion) (focused on|addressed|covered).*$',
    r'\n*End response\.?\n+But wait.*$',
    r'\n*after reviewing all guidelines:.*$',
    r'\n*The initial part of the response.*violating Rule.*$',
    r'\n*Also, even though.*$',
    r'\n*So yes, the connection qualifies.*$',
    r'\n*Still, because.*$',
    r'\n*Therefore, revise strictly.*$',
    r'\n*Step \d+ says:.*$',
    r'\n*Rule #\d+.*$',
    r'\n*\*Step \d+.*?\*.*$',
    r'\([Ee]nd response.*?\)',
    r'\([Tt]he (response|initial).*?\)',
    r'\([Pp]lease note.*?\)',
    r'\n\nDo you (understand|know|see|think).*?\?.*$',
    r'\n\nIt sounds like.*$',
    r'\n\nThe response contains.*?Therefore,.*$',
    r'\n\n- References to.*$',
    r'\n\n- Mentions of.*$',
    r'\n\n- Claims about.*$',
]

_QUESTION_BLOCK_RE = re.compile(
    r'^(?:q(?:uestion)?\s*[:\-]\s*)?'
    r'(?:who|what|when|where|why|how|which|whom|is|are|was|were|do|does|did|'
    r'can|could|should|would|will|has|have|had|tell|list|name|explain)\b'
    r'.{0,220}\?$',
    re.IGNORECASE,
)

_OFFER_QUESTION_RE = re.compile(
    r'^(?:would you like|do you want|want me to|shall i|should i|'
    r'let me know if you|feel free to ask)\b',
    re.IGNORECASE,
)


def sanitize_generated_answer(text: str, current_query: Optional[str] = None) -> str:
    """
    Remove trailing generation artifacts from an answer.

    Args:
        text: Raw generated assistant answer
        current_query: Optional current user query for extra follow-up detection

    Returns:
        Cleaned answer text safe for summaries and conversation memory
    """
    if not text or not text.strip():
        return text

    cleaned = text.strip()
    original_length = len(cleaned)

    cleaned = _truncate_at_injected_follow_up(cleaned, current_query=current_query)
    cleaned = _strip_trailing_meta(cleaned)
    cleaned = _normalize_spacing(cleaned)

    if len(cleaned) != original_length:
        logger.info(
            "Sanitized generated answer: %d -> %d chars",
            original_length,
            len(cleaned),
        )

    return cleaned


def _truncate_at_injected_follow_up(text: str, current_query: Optional[str] = None) -> str:
    """
    Truncate when the model starts a brand-new question after already answering.

    Example bad pattern:
      [valid answer]

      is haley robson connected to anyone else besides jephsen?

      ## Connection Analysis...
    """
    if '\n\n' not in text:
        return text

    boundaries = list(re.finditer(r'\n\s*\n', text))
    if not boundaries:
        return text

    block_starts = [0]
    block_starts.extend(match.end() for match in boundaries)
    block_ends = [match.start() for match in boundaries]
    block_ends.append(len(text))

    normalized_query = _normalize_question(current_query) if current_query else ""

    for index, (start, end) in enumerate(zip(block_starts, block_ends)):
        if index == 0:
            continue

        block = text[start:end].strip()
        if not _is_suspicious_question_block(block):
            continue

        prefix = text[:start].strip()
        if not _has_substantial_answer(prefix):
            continue

        if normalized_query and _normalize_question(block) == normalized_query:
            continue

        logger.warning("Detected injected follow-up question in generated answer: %r", block[:160])
        return text[:start].rstrip()

    return text


def _is_suspicious_question_block(block: str) -> bool:
    if not block or len(block) < 8:
        return False

    single_line = ' '.join(part.strip() for part in block.splitlines() if part.strip())
    if len(single_line) > 240:
        return False

    if single_line.startswith(('#', '*', '-')):
        return False

    if _OFFER_QUESTION_RE.match(single_line):
        return False

    return bool(_QUESTION_BLOCK_RE.match(single_line))


def _has_substantial_answer(text: str) -> bool:
    if len(text) < 140:
        return False

    sentence_endings = sum(text.count(mark) for mark in '.!?')
    return sentence_endings >= 2 or '\n\n' in text


def _strip_trailing_meta(text: str) -> str:
    cleaned = text
    for pattern in _TRAILING_META_PATTERNS:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _normalize_question(text: Optional[str]) -> str:
    if not text:
        return ""
    normalized = re.sub(r'\s+', ' ', text).strip().lower()
    normalized = normalized.strip(' "\'`.,!?;:')
    return normalized


def _normalize_spacing(text: str) -> str:
    cleaned = re.sub(r'\n{4,}', '\n\n', text)
    cleaned = re.sub(r'[ \t]+\n', '\n', cleaned)
    cleaned = re.sub(r'\n[ \t]+', '\n', cleaned)
    return cleaned.strip()
