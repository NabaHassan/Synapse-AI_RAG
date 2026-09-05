"""
Extract human-readable values from Notion page/database property payloads.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TITLE_PROPERTY_KEYS = frozenset({"title", "name", "task name", "page name"})


def rich_text_to_plain(rich_text: List[Dict[str, Any]]) -> str:
    return "".join(rt.get("plain_text", "") for rt in (rich_text or [])).strip()


def extract_property_value(prop: Dict[str, Any]) -> str:
    """Extract a simple string representation of a single Notion property."""
    if not prop:
        return ""

    ptype = prop.get("type", "")
    if ptype in ("rich_text", "title"):
        return rich_text_to_plain(prop.get(ptype, []))
    if ptype == "select":
        sel = prop.get("select")
        return sel.get("name", "") if sel else ""
    if ptype == "multi_select":
        return ", ".join(s.get("name", "") for s in prop.get("multi_select", []) if s)
    if ptype == "status":
        stat = prop.get("status")
        return stat.get("name", "") if stat else ""
    if ptype == "date":
        date_obj = prop.get("date")
        if not date_obj:
            return ""
        start = date_obj.get("start", "")
        end = date_obj.get("end", "")
        return f"{start} to {end}" if end else start
    if ptype == "checkbox":
        return "Yes" if prop.get("checkbox") else "No"
    if ptype == "number":
        num = prop.get("number")
        return "" if num is None else str(num)
    if ptype == "url":
        return prop.get("url", "") or ""
    if ptype == "email":
        return prop.get("email", "") or ""
    if ptype == "phone_number":
        return prop.get("phone_number", "") or ""
    if ptype == "people":
        people = prop.get("people") or []
        names = []
        for person in people:
            name = (person.get("name") or "").strip()
            if name:
                names.append(name)
            elif person.get("id"):
                names.append(person["id"])
        return ", ".join(names)
    if ptype == "relation":
        relations = prop.get("relation") or []
        ids = [r.get("id", "") for r in relations if r.get("id")]
        return ", ".join(ids)
    if ptype == "rollup":
        rollup = prop.get("rollup") or {}
        rtype = rollup.get("type", "")
        if rtype == "array":
            items = rollup.get("array") or []
            parts = [extract_property_value(item) for item in items if isinstance(item, dict)]
            return ", ".join(p for p in parts if p)
        if rtype == "number":
            num = rollup.get("number")
            return "" if num is None else str(num)
        if rtype == "date":
            return extract_property_value({"type": "date", "date": rollup.get("date")})
        if rtype == "incomplete":
            return ""
    if ptype == "formula":
        formula = prop.get("formula") or {}
        ftype = formula.get("type", "")
        if ftype == "string":
            return formula.get("string", "") or ""
        if ftype == "number":
            num = formula.get("number")
            return "" if num is None else str(num)
        if ftype == "boolean":
            return "Yes" if formula.get("boolean") else "No"
        if ftype == "date":
            return extract_property_value({"type": "date", "date": formula.get("date")})
    if ptype == "files":
        files = prop.get("files") or []
        names = []
        for f in files:
            if f.get("name"):
                names.append(f["name"])
            elif f.get("external", {}).get("url"):
                names.append(f["external"]["url"])
        return ", ".join(names)
    if ptype == "created_by":
        user = prop.get("created_by") or {}
        return user.get("name", "") or user.get("id", "")
    if ptype == "last_edited_by":
        user = prop.get("last_edited_by") or {}
        return user.get("name", "") or user.get("id", "")
    if ptype == "created_time":
        return prop.get("created_time", "") or ""
    if ptype == "last_edited_time":
        return prop.get("last_edited_time", "") or ""

    logger.debug("Notion property type not mapped: %s", ptype)
    return ""


def extract_page_title(page: Dict[str, Any]) -> str:
    """Extract a human-readable title from a Notion page or database object."""
    props = page.get("properties", {})
    for key, prop in props.items():
        if prop.get("type") == "title" or prop.get("id") == "title":
            rich_text = prop.get("title", [])
            if rich_text:
                text = rich_text_to_plain(rich_text)
                if text:
                    return text

    top_title = page.get("title", [])
    if isinstance(top_title, list):
        text = rich_text_to_plain(top_title)
        if text:
            return text

    return "Untitled"


def extract_all_properties(page: Dict[str, Any]) -> Dict[str, str]:
    """Return all non-title properties as name -> value strings."""
    props = page.get("properties") or {}
    result: Dict[str, str] = {}
    for key, prop in props.items():
        if key.lower() in _TITLE_PROPERTY_KEYS or prop.get("type") == "title":
            continue
        value = extract_property_value(prop)
        if value:
            result[key] = value
    logger.debug(
        "extract_all_properties title=%r keys=%s",
        extract_page_title(page),
        list(result.keys()),
    )
    return result


def format_properties_block(properties: Dict[str, str], indent: str = "  ") -> str:
    if not properties:
        return ""
    lines = [f"{indent}Properties:"]
    for key, value in properties.items():
        lines.append(f"{indent}  - {key}: {value}")
    return "\n".join(lines)


def format_page_summary(page: Dict[str, Any]) -> str:
    """Format one page row with title, metadata, and all properties."""
    title = extract_page_title(page)
    page_id = page.get("id", "")
    url = page.get("url", "")
    edited = page.get("last_edited_time", "")
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(edited.replace("Z", "+00:00"))
        edited_display = dt.strftime("%b %d, %Y")
    except Exception:
        edited_display = edited

    properties = extract_all_properties(page)
    prop_text = format_properties_block(properties, indent="  ")

    summary = (
        f"- **{title}** — Last edited: {edited_display}\n"
        f"  URL: {url}\n"
        f"  ID: {page_id}"
    )
    if prop_text:
        summary = f"{summary}\n{prop_text}"
    return summary
