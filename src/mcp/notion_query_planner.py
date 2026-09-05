"""
Route Notion user queries to the appropriate MCP tool and parameters.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from src.mcp.notion_query_parser import (
    is_notion_follow_up_query,
    is_notion_list_query,
    parse_database_hint,
    parse_notion_database_filters,
    should_use_database_query,
)
from src.mcp.notion_task_memory import match_notion_task_from_memory, pick_active_task_from_query

logger = logging.getLogger(__name__)

_READ_PAGE_PATTERNS = [
    re.compile(r"\b(?:read|open|show)\s+(?:the\s+)?(?:page|task)\b", re.I),
    re.compile(r"\btell\s+me\s+about\b", re.I),
]


@dataclass
class NotionQueryPlan:
    tool: str
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    uses_active_task: bool = False


def plan_notion_query(
    query: str,
    session_metadata: Optional[Dict[str, Any]] = None,
    *,
    database_hint: Optional[str] = None,
    follow_up: Optional[Tuple[str, Dict[str, Any]]] = None,
    intent_tool: Optional[str] = None,
    intent_params: Optional[Dict[str, Any]] = None,
) -> NotionQueryPlan:
    """
    Choose the best Notion MCP tool for a user query.

    Priority:
      1. Database filter / list queries (status, priority, category, dates)
      2. Explicit follow-up routing (active task / property question)
      3. get_page_details for named task / read intent
      4. Intent detector fallback
      5. search_pages
    """
    session_metadata = session_metadata or {}
    q = (query or "").strip()

    db_hint = database_hint or parse_database_hint(q)
    active_db = session_metadata.get("active_notion_database") or {}
    if not db_hint and active_db.get("title"):
        db_hint = active_db.get("title")

    filter_spec = parse_notion_database_filters(q)
    if is_notion_list_query(q) or filter_spec or should_use_database_query(q):
        spec = filter_spec or {}
        if db_hint and not spec.get("database_hint"):
            spec["database_hint"] = db_hint
        logger.info("NotionQueryPlan: database query spec=%s", spec)
        return NotionQueryPlan(
            tool="query_database",
            params={"filter_spec": spec, "database_hint": spec.get("database_hint") or db_hint},
            reason="structured_database_filter",
        )

    if follow_up:
        tool, params = follow_up
        logger.info("NotionQueryPlan: follow_up -> %s params=%s", tool, params)
        return NotionQueryPlan(
            tool=tool,
            params=dict(params),
            reason="session_follow_up",
            uses_active_task=True,
        )

    matched_task = match_notion_task_from_memory(q, session_metadata)
    if matched_task and matched_task.get("page_id") and is_notion_follow_up_query(q):
        params = {"page_id": matched_task["page_id"], "title": matched_task.get("title")}
        focus = None
        from src.mcp.notion_query_parser import detect_property_question
        focus = detect_property_question(q)
        if focus:
            params["focus_property"] = focus
        logger.info("NotionQueryPlan: active task follow-up -> get_page_details")
        return NotionQueryPlan(
            tool="get_page_details",
            params=params,
            reason="active_task_follow_up",
            uses_active_task=True,
        )

    if intent_tool == "get_page_content":
        page_ref = (intent_params or {}).get("page_name") or q
        logger.info("NotionQueryPlan: intent get_page_content -> get_page_details page_ref=%r", page_ref)
        return NotionQueryPlan(
            tool="get_page_details",
            params={"page_id": page_ref, "title": page_ref},
            reason="intent_get_page_content",
        )

    if intent_tool == "search_databases":
        return NotionQueryPlan(
            tool="search_databases",
            params={"query": (intent_params or {}).get("query") or q},
            reason="intent_search_databases",
        )

    tasks = session_metadata.get("notion_task_memory") or []
    named = pick_active_task_from_query(q, tasks)
    if named and named.get("page_id"):
        logger.info("NotionQueryPlan: named task in memory -> get_page_details %r", named.get("title"))
        return NotionQueryPlan(
            tool="get_page_details",
            params={"page_id": named["page_id"], "title": named.get("title")},
            reason="named_task_in_memory",
        )

    if any(pat.search(q) for pat in _READ_PAGE_PATTERNS):
        title_guess = re.sub(
            r"\b(?:tell\s+me\s+about|read|open|show|the|page|task|this|please)\b",
            " ",
            q,
            flags=re.I,
        )
        title_guess = re.sub(r"\s+", " ", title_guess).strip()
        if len(title_guess) > 8:
            logger.info("NotionQueryPlan: read pattern -> get_page_details title_guess=%r", title_guess)
            return NotionQueryPlan(
                tool="get_page_details",
                params={"page_id": title_guess, "title": title_guess},
                reason="read_page_pattern",
            )

    enriched_query = (intent_params or {}).get("query") or q
    logger.info("NotionQueryPlan: default search_pages query=%r", enriched_query)
    return NotionQueryPlan(
        tool="search_pages",
        params={"query": enriched_query},
        reason="default_search",
    )
