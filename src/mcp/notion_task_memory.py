"""
Session memory for Notion tasks/pages — follow-up routing and persistence.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.mcp.notion_query_parser import (
    detect_property_question,
    is_notion_follow_up_query,
    is_notion_list_query,
    is_template_notion_title,
    parse_notion_database_filters,
    should_use_database_query,
)

logger = logging.getLogger(__name__)

_PAGE_ID_RE = re.compile(
    r"\bID:\s*([0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12})\b",
    re.I,
)
_TITLE_RE = re.compile(r"-\s+\*\*(.+?)\*\*\s+—")
_PROP_LINE_RE = re.compile(r"^\s+-\s+([^:]+):\s*(.+)\s*$")


def _normalize_title_text(text: str) -> str:
    """Lowercase and strip punctuation for fuzzy title matching."""
    cleaned = re.sub(r"[^\w\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _title_match_score(query: str, title: str) -> int:
    """Higher score means stronger title match."""
    q_norm = _normalize_title_text(query)
    t_norm = _normalize_title_text(title)
    if not q_norm or not t_norm:
        return 0
    if t_norm in q_norm or q_norm in t_norm:
        return 100
    stopwords = {"the", "task", "this", "that", "about", "tell", "me", "what", "is", "its", "it", "of", "and", "for"}
    tokens = [w for w in t_norm.split() if w not in stopwords and len(w) > 3]
    if not tokens:
        return 0
    hits = sum(1 for tok in tokens if tok in q_norm)
    threshold = max(2, len(tokens) // 2)
    return hits if hits >= threshold else 0


def extract_tasks_from_mcp_text(mcp_text: str) -> List[Dict[str, Any]]:
    """Parse formatted Notion MCP output into task records."""
    if not mcp_text:
        return []

    tasks: List[Dict[str, Any]] = []
    blocks = re.split(r"\n(?=-\s+\*\*)", mcp_text)
    for block in blocks:
        block = block.strip()
        if not block.startswith("- **"):
            continue

        title_m = _TITLE_RE.search(block)
        if not title_m:
            continue
        title = title_m.group(1).strip()
        if is_template_notion_title(title):
            logger.debug("Skipping template Notion title: %r", title)
            continue

        page_id_m = _PAGE_ID_RE.search(block)
        page_id = page_id_m.group(1) if page_id_m else None
        url_m = re.search(r"URL:\s*(\S+)", block)
        url = url_m.group(1) if url_m else ""

        properties: Dict[str, str] = {}
        in_props = False
        for line in block.splitlines():
            if line.strip() == "Properties:":
                in_props = True
                continue
            if in_props:
                prop_m = _PROP_LINE_RE.match(line)
                if prop_m:
                    properties[prop_m.group(1).strip()] = prop_m.group(2).strip()
                elif line.strip() and not line.startswith(" "):
                    in_props = False

        if not page_id and not title:
            continue

        tasks.append({
            "page_id": page_id,
            "title": title,
            "url": url,
            "properties": properties,
            "status": properties.get("Status", ""),
            "priority": properties.get("Priority", ""),
            "category": properties.get("Category", ""),
            "assignee": properties.get("Assigned To", ""),
            "due_date": properties.get("Due Date", properties.get("Due date", "")),
            "description": properties.get("Description", ""),
        })

    logger.info("extract_tasks_from_mcp_text: parsed %d task(s)", len(tasks))
    return tasks


def persist_notion_task_memory(
    memory: Any,
    tasks: List[Dict[str, Any]],
    *,
    active_task: Optional[Dict[str, Any]] = None,
    database_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Store Notion tasks in session metadata after a successful turn."""
    if memory is None:
        return

    clean_tasks = [t for t in tasks if t.get("page_id") or t.get("title")]
    if not clean_tasks and not active_task and not database_info:
        return

    if clean_tasks:
        memory.session.metadata["notion_task_memory"] = clean_tasks[:20]
        logger.info(
            "Persisted notion_task_memory: %d task(s), first=%r",
            len(clean_tasks[:20]),
            clean_tasks[0].get("title"),
        )

    if active_task and (active_task.get("page_id") or active_task.get("title")):
        memory.session.metadata["active_notion_task"] = active_task
        logger.info(
            "Persisted active_notion_task: page_id=%s title=%r",
            active_task.get("page_id"),
            active_task.get("title"),
        )
    elif clean_tasks:
        memory.session.metadata["active_notion_task"] = clean_tasks[0]

    if database_info:
        memory.session.metadata["active_notion_database"] = database_info
        logger.info(
            "Persisted active_notion_database: %r",
            database_info.get("title"),
        )

    memory.session.metadata["active_mcp_tool"] = "notion"


def match_notion_task_from_memory(
    query: str,
    session_metadata: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Resolve a task record from session metadata for follow-ups."""
    active = session_metadata.get("active_notion_task") or {}
    tasks = session_metadata.get("notion_task_memory") or []
    if not active and not tasks:
        return None

    best_entry: Optional[Dict[str, Any]] = None
    best_score = 0
    for entry in tasks:
        title = entry.get("title") or ""
        if not title:
            continue
        score = _title_match_score(query, title)
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_entry and best_score >= 2:
        logger.info(
            "Notion follow-up matched title %r (score=%d)",
            best_entry.get("title"),
            best_score,
        )
        return best_entry

    if is_notion_follow_up_query(query) and active.get("page_id"):
        logger.info(
            "Notion follow-up phrase detected -> active task %r",
            active.get("title"),
        )
        return active

    prop_focus = detect_property_question(query)
    if prop_focus and active.get("page_id"):
        logger.info("Notion property question %r -> active task %r", prop_focus, active.get("title"))
        return active

    return None


def apply_notion_follow_up_routing(
    query: str,
    session_metadata: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Route follow-up queries to get_page_details using session memory.

    Returns (tool_name, params) or None.
    """
    if is_notion_list_query(query) or should_use_database_query(query):
        logger.info(
            "Notion follow-up skipped for list/database query: %r",
            query[:80],
        )
        return None
    if parse_notion_database_filters(query):
        logger.info(
            "Notion follow-up skipped for parsed database filters: %r",
            query[:80],
        )
        return None

    task = match_notion_task_from_memory(query, session_metadata)
    if not task or not task.get("page_id"):
        return None

    focus = detect_property_question(query)
    params: Dict[str, Any] = {
        "page_id": task["page_id"],
        "title": task.get("title"),
    }
    if focus:
        params["focus_property"] = focus

    logger.info(
        "Notion follow-up routing -> get_page_details(page_id=%s, focus=%r)",
        task["page_id"],
        focus,
    )
    return "get_page_details", params


def pick_active_task_from_query(
    query: str,
    tasks: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick the task most likely referenced in a query from a list result."""
    if not tasks:
        return None
    best_task: Optional[Dict[str, Any]] = None
    best_score = 0
    for task in tasks:
        title = task.get("title") or ""
        score = _title_match_score(query, title)
        if score > best_score:
            best_score = score
            best_task = task
    if best_task and best_score >= 2:
        logger.info(
            "pick_active_task_from_query: matched %r (score=%d)",
            best_task.get("title"),
            best_score,
        )
        return best_task
    return None
