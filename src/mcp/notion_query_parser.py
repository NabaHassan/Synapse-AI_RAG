"""
Parse natural-language Notion task/database queries into structured filter specs.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from src.mcp.notion_task_query import (
    _DATABASE_HINT_PATTERNS,
    _STATUS_ALIASES,
    parse_notion_task_filters,
    should_use_status_database_query,
)

logger = logging.getLogger(__name__)

_PRIORITY_ALIASES: Dict[str, str] = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "urgent": "High",
}

_CATEGORY_PATTERNS = [
    re.compile(r"\bdevelopment\b", re.I),
    re.compile(r"\bdesign\b", re.I),
    re.compile(r"\bresearch\b", re.I),
    re.compile(r"\bmarketing\b", re.I),
    re.compile(r"\boperations\b", re.I),
]

_ASSIGNEE_PATTERNS = [
    re.compile(r"(?:assigned\s+to|assignee|owned\s+by)\s+([A-Za-z][A-Za-z0-9.'\-]*(?:\s+[A-Za-z][A-Za-z0-9.'\-]*)?)", re.I),
    re.compile(r"tasks?\s+(?:for|by)\s+([A-Za-z][A-Za-z0-9.'\-]*(?:\s+[A-Za-z][A-Za-z0-9.'\-]*)?)", re.I),
]

_FOLLOW_UP_PRONOUN_PATTERNS = [
    re.compile(r"\bthis\s+task\b", re.I),
    re.compile(r"\bthat\s+task\b", re.I),
    re.compile(r"\bthe\s+task\b", re.I),
    re.compile(r"\bit\b", re.I),
    re.compile(r"\bits\b", re.I),
    re.compile(r"\bthis\s+one\b", re.I),
]

_PROPERTY_QUESTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    "priority": [re.compile(r"\bpriorit(?:y|ies)\b", re.I), re.compile(r"\bhow\s+urgent\b", re.I)],
    "assigned": [re.compile(r"\bassigned\s+to\b", re.I), re.compile(r"\bwho\s+(?:is\s+)?(?:working|owns)\b", re.I), re.compile(r"\bassignee\b", re.I)],
    "status": [re.compile(r"\bstatus\b", re.I), re.compile(r"\bwhat\s+state\b", re.I)],
    "category": [re.compile(r"\bcategor(?:y|ies)\b", re.I), re.compile(r"\bwhat\s+type\b", re.I)],
    "due": [re.compile(r"\bdue\s+date\b", re.I), re.compile(r"\bdeadline\b", re.I), re.compile(r"\bwhen\s+is\s+it\s+due\b", re.I)],
    "description": [
        re.compile(r"\bmore\s+details?\b", re.I),
        re.compile(r"\btell\s+me\s+(?:more\s+)?about\b", re.I),
        re.compile(r"\bdescription\b", re.I),
        re.compile(r"\bwhat\s+is\s+(?:it|this)\s+about\b", re.I),
    ],
}

_TEMPLATE_TITLE_PATTERNS = [
    re.compile(r"^click\s+me\s+to\b", re.I),
    re.compile(r"^click\s+the\s+blue\b", re.I),
    re.compile(r"^check\s+the\s+box\b", re.I),
    re.compile(r"^see\s+finished\s+items\b", re.I),
]

_LIST_QUERY_PATTERNS = [
    re.compile(r"\blist\s+(?:all\s+)?(?:my\s+)?(?:the\s+)?tasks?\b", re.I),
    re.compile(r"\bshow\s+(?:all\s+)?(?:my\s+)?(?:the\s+)?tasks?\b", re.I),
    re.compile(r"\bwhat\s+(?:are\s+)?(?:my\s+)?tasks?\b", re.I),
    re.compile(r"\btasks?\s+that\s+(?:have|are|with)\b", re.I),
    re.compile(r"\btasks?\s+with\s+status\b", re.I),
    re.compile(r"\b(?:have|with)\s+status\s+(?:is\s+)?\w+", re.I),
    re.compile(r"\bstatus\s+(?:is\s+|=)?(?:done|in\s+review|in\s+progress|not\s+started)\b", re.I),
    re.compile(r"/search\b", re.I),
]

_STATUS_IN_LIST_RE = re.compile(
    r"\bstatus\s+(?:is\s+|=)?(" + "|".join(re.escape(k) for k in _STATUS_ALIASES.keys()) + r")\b",
    re.I,
)


def is_notion_list_query(query: str) -> bool:
    """True when the user wants a filtered list of tasks, not a single-task follow-up."""
    if not query or not query.strip():
        return False
    q = query.strip()
    if any(pat.search(q) for pat in _LIST_QUERY_PATTERNS):
        logger.debug("Notion list query detected: %r", query[:80])
        return True
    if parse_notion_task_filters(query) is not None:
        logger.debug("Notion list query via status filter: %r", query[:80])
        return True
    return False


def is_template_notion_title(title: str) -> bool:
    if not title:
        return False
    return any(pat.search(title.strip()) for pat in _TEMPLATE_TITLE_PATTERNS)


def is_notion_follow_up_query(query: str) -> bool:
    """True when the user likely refers to a prior task/page."""
    if not query or not query.strip():
        return False
    if is_notion_list_query(query):
        return False
    q = query.strip()
    if any(pat.search(q) for pat in _FOLLOW_UP_PRONOUN_PATTERNS):
        return True
    if detect_property_question(q):
        return True
    if re.search(r"\b(?:add|give|show)\s+more\s+details?\b", q, re.I):
        return True
    return False


def detect_property_question(query: str) -> Optional[str]:
    """Return property focus key if query asks about one field."""
    if is_notion_list_query(query):
        return None
    q_stripped = (query or "").strip()
    # Single-word property queries are valid follow-ups (e.g. "status", "priority")
    if len(q_stripped.split()) <= 2:
        for key, patterns in _PROPERTY_QUESTION_PATTERNS.items():
            if any(pat.search(query) for pat in patterns):
                logger.debug("Notion property question detected: %r -> %s", query, key)
                return key
        return None
    for key, patterns in _PROPERTY_QUESTION_PATTERNS.items():
        if key == "status" and re.search(r"\btasks?\b", query, re.I):
            continue
        if any(pat.search(query) for pat in patterns):
            logger.debug("Notion property question detected: %r -> %s", query, key)
            return key
    return None


def parse_database_hint(query: str) -> Optional[str]:
    for pat in _DATABASE_HINT_PATTERNS:
        m = pat.search(query)
        if m:
            hint = m.group(0).strip()
            if "synapse" in hint.lower():
                return "Synapse Task Testing"
            return hint
    return None


def parse_notion_database_filters(query: str) -> Optional[Dict[str, Any]]:
    """
    Parse list/filter queries into a structured spec for query_database.

    Returns None when the query does not look like a filtered task list.
    """
    if not query or not query.strip():
        return None

    status_filters = parse_notion_task_filters(query)
    q_lower = query.lower()

    if not status_filters:
        status_list_match = _STATUS_IN_LIST_RE.search(q_lower)
        if status_list_match:
            phrase = status_list_match.group(1).lower()
            canonical = _STATUS_ALIASES.get(phrase)
            if canonical:
                status_filters = {
                    "status": canonical,
                    "database_hint": parse_database_hint(query),
                    "clean_query": None,
                }
                logger.debug(
                    "Notion list status phrase %r -> %r",
                    phrase,
                    canonical,
                )

    priority: Optional[str] = None
    for alias, canonical in _PRIORITY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\s+priority\b", q_lower) or re.search(
            rf"\bpriority\s+{re.escape(alias)}\b", q_lower
        ):
            priority = canonical
            break
        if re.search(rf"\b{re.escape(alias)}\b", q_lower) and re.search(r"\b(?:task|tasks|priority)\b", q_lower):
            priority = canonical
            break

    category: Optional[str] = None
    for pat in _CATEGORY_PATTERNS:
        m = pat.search(query)
        if m:
            category = m.group(0).strip().title()
            break
    if not category and re.search(r"\bdevelopment\s+categor", q_lower):
        category = "Development"

    assignee: Optional[str] = None
    for pat in _ASSIGNEE_PATTERNS:
        m = pat.search(query)
        if m:
            assignee = m.group(1).strip()
            break

    due_after: Optional[str] = None
    due_before: Optional[str] = None
    if re.search(r"\bdue\s+this\s+week\b", q_lower):
        today = date.today()
        due_after = today.isoformat()
        due_before = (today + timedelta(days=7)).isoformat()
    elif re.search(r"\boverdue\b", q_lower):
        due_before = date.today().isoformat()

    database_hint = parse_database_hint(query)
    if status_filters:
        database_hint = status_filters.get("database_hint") or database_hint
        return {
            "status": status_filters.get("status"),
            "priority": priority,
            "category": category,
            "assignee": assignee,
            "due_after": due_after,
            "due_before": due_before,
            "database_hint": database_hint,
            "clean_query": status_filters.get("clean_query"),
        }

    list_signals = re.search(r"\b(?:list|show|all|what|which)\b.*\b(?:task|tasks)\b", q_lower)
    has_filters = any([priority, category, assignee, due_after, due_before])
    if list_signals and (has_filters or database_hint):
        return {
            "status": None,
            "priority": priority,
            "category": category,
            "assignee": assignee,
            "due_after": due_after,
            "due_before": due_before,
            "database_hint": database_hint,
            "clean_query": None,
        }

    return None


def should_use_database_query(query: str) -> bool:
    return parse_notion_database_filters(query) is not None or should_use_status_database_query(query)


def find_property_by_semantic_name(
    database: Dict[str, Any],
    semantic: str,
) -> Optional[Tuple[str, str]]:
    """
    Map semantic names (status, priority, category, assignee, due) to DB column.
    Returns (property_name, property_type).
    """
    props = database.get("properties") or {}
    semantic = semantic.lower().strip()

    semantic_map = {
        "status": ("status", "state"),
        "priority": ("priority",),
        "category": ("category", "type"),
        "assignee": ("assigned to", "assignee", "owner", "person"),
        "due": ("due date", "due", "deadline"),
        "description": ("description", "notes", "details"),
    }
    candidates = semantic_map.get(semantic, (semantic,))

    for name, spec in props.items():
        name_lower = name.lower()
        ptype = spec.get("type", "")
        if any(c in name_lower for c in candidates):
            return name, ptype

    if semantic == "status":
        for name, spec in props.items():
            if spec.get("type") in ("status", "select"):
                return name, spec.get("type")

    return None


def build_compound_filter(
    database: Dict[str, Any],
    filter_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a Notion API compound filter from a parsed filter spec."""
    from src.mcp.notion_task_query import build_status_filter

    clauses: List[Dict[str, Any]] = []

    status = filter_spec.get("status")
    if status:
        match = find_property_by_semantic_name(database, "status")
        if match:
            prop_name, prop_type = match
            clauses.append(build_status_filter(prop_name, prop_type, status))

    priority = filter_spec.get("priority")
    if priority:
        match = find_property_by_semantic_name(database, "priority")
        if match:
            prop_name, prop_type = match
            if prop_type == "select":
                clauses.append({"property": prop_name, "select": {"equals": priority}})

    category = filter_spec.get("category")
    if category:
        match = find_property_by_semantic_name(database, "category")
        if match:
            prop_name, prop_type = match
            if prop_type == "select":
                clauses.append({"property": prop_name, "select": {"equals": category}})
            elif prop_type == "multi_select":
                clauses.append({"property": prop_name, "multi_select": {"contains": category}})

    due_after = filter_spec.get("due_after")
    due_before = filter_spec.get("due_before")
    if due_after or due_before:
        match = find_property_by_semantic_name(database, "due")
        if match:
            prop_name, _ = match
            if due_after and due_before:
                clauses.append({
                    "and": [
                        {"property": prop_name, "date": {"on_or_after": due_after}},
                        {"property": prop_name, "date": {"on_or_before": due_before}},
                    ]
                })
            elif due_before:
                clauses.append({"property": prop_name, "date": {"on_or_before": due_before}})
            elif due_after:
                clauses.append({"property": prop_name, "date": {"on_or_after": due_after}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"and": clauses}
