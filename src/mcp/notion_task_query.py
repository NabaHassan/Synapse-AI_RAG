"""
Helpers for Notion task/database queries filtered by Status (and related fields).

Notion's /v1/search endpoint is full-text only — it cannot return all rows where
Status = "In Review". Use database query instead.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# User phrasing -> canonical Notion status option names (case-sensitive in API)
_STATUS_ALIASES: Dict[str, str] = {
    "in review": "In Review",
    "under review": "In Review",
    "in progress": "In Progress",
    "in-progress": "In Progress",
    "not started": "Not started",
    "done": "Done",
    "completed": "Done",
    "blocked": "Blocked",
    "cancelled": "Cancelled",
    "canceled": "Cancelled",
}

_DATABASE_HINT_PATTERNS = [
    re.compile(r"synapse\s+task\s+testing", re.I),
    re.compile(r"task\s+testing", re.I),
]


def parse_notion_task_filters(query: str) -> Optional[Dict[str, Any]]:
    """
    Detect status-filtered task list queries.

    Returns dict with status (canonical), optional database_hint, clean_query; or None.
    """
    if not query or not query.strip():
        return None

    q_lower = query.lower()
    status: Optional[str] = None

    for phrase in sorted(_STATUS_ALIASES.keys(), key=len, reverse=True):
        if phrase in q_lower:
            status = _STATUS_ALIASES[phrase]
            logger.debug("Notion task filter: phrase %r -> status %r", phrase, status)
            break

    if not status and re.search(r"\b(?:in\s+)?review\b", q_lower) and re.search(
        r"\b(?:task|tasks)\b", q_lower
    ):
        status = "In Review"
        logger.debug("Notion task filter: inferred In Review from review+tasks")

    if not status and re.search(r"\b(?:in\s+)?progress\b", q_lower) and re.search(
        r"\b(?:task|tasks)\b", q_lower
    ):
        status = "In Progress"
        logger.debug("Notion task filter: inferred In Progress from progress+tasks")

    if not status:
        return None

    database_hint: Optional[str] = None
    for pat in _DATABASE_HINT_PATTERNS:
        m = pat.search(query)
        if m:
            database_hint = m.group(0).strip()
            if "synapse" in database_hint.lower():
                database_hint = "Synapse Task Testing"
            break

    clean = query
    for phrase in _STATUS_ALIASES.keys():
        clean = re.sub(re.escape(phrase), " ", clean, flags=re.I)
    clean = re.sub(r"\b(?:list|show|all|what|are|the|my|tasks?|search|find)\b", " ", clean, flags=re.I)
    clean = re.sub(r"\s+", " ", clean).strip()

    return {
        "status": status,
        "database_hint": database_hint,
        "clean_query": clean or None,
    }


def should_use_status_database_query(query: str) -> bool:
    """True when the user is asking for tasks filtered by workflow status."""
    return parse_notion_task_filters(query) is not None


def build_status_filter(property_name: str, property_type: str, status_value: str) -> Dict[str, Any]:
    """Build a Notion API filter object for a status/select property."""
    if property_type == "status":
        return {
            "property": property_name,
            "status": {"equals": status_value},
        }
    if property_type == "select":
        return {
            "property": property_name,
            "select": {"equals": status_value},
        }
    logger.warning(
        "Unsupported Status property type %r on %r; trying status.equals",
        property_type,
        property_name,
    )
    return {
        "property": property_name,
        "status": {"equals": status_value},
    }
