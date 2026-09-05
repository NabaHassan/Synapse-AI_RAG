"""
Connector-aware slash-command parsing for the query composer.

Maps /commands to query rewrites (handled by normal MCP routing) or direct MCP tool calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

SlashKind = Literal["help", "rewrite", "direct_mcp"]

_SLASH_PATTERN = re.compile(
    r"^/(read|search|help|unread|inbox|from|thread|free|busy|next|channel|dm|page|db|doc|sheet)"
    r"(?:\s+(.*))?$",
    re.IGNORECASE,
)

_HELP_BY_CONNECTOR: Dict[str, str] = {
    "": (
        "Slash commands change what the query box does.\n\n"
        "All: /read, /search, /help\n"
        "Gmail: /unread, /from, /thread\n"
        "Calendar: /free, /busy, /next\n"
        "Slack: /channel, /dm, /thread\n"
        "Notion: /page, /db\n"
        "Drive: /doc, /sheet\n"
        "Outlook: /inbox, /from\n"
        "OneDrive: /read, /search\n\n"
        "Also use @ (who/when/links), # (where), in the composer."
    ),
    "email": (
        "Gmail slash commands:\n"
        "• /unread — unread mail in inbox\n"
        "• /from [name] — mail from a person (e.g. /from Sarah)\n"
        "• /thread [id] — open a thread by ID\n"
        "• /search [terms] — search mail\n"
        "• /read [subject] — read-focused lookup\n\n"
        "Combine with #inbox, @today, @sender in the same query."
    ),
    "calendar": (
        "Calendar slash commands:\n"
        "• /free [@today] — find free time (default @today)\n"
        "• /busy [@today] — meetings with attendees\n"
        "• /next — next upcoming event (@soon)\n"
        "• /search [terms] — search events\n\n"
        "Use @today / @week for time and #work / #family for lens."
    ),
    "slack": (
        "Slack slash commands:\n"
        "• /channel [name] — history in a channel (#name)\n"
        "• /dm [person] — direct messages (@person)\n"
        "• /thread [topic] — search within a thread/topic\n"
        "• /read [#channel|@person] — read history\n"
        "• /search [terms] — search messages"
    ),
    "notion": (
        "Notion slash commands:\n"
        "• /page [title] — read a page by name\n"
        "• /db [keywords] — find databases\n"
        "• /read [title] — same as /page\n"
        "• /search [terms] — search pages"
    ),
    "google": (
        "Google Drive slash commands:\n"
        "• /doc [name] — find or read a Doc\n"
        "• /sheet [name] — find a Spreadsheet\n"
        "• /read [name] — read a file by name\n"
        "• /search [terms] — search Drive"
    ),
    "outlook": (
        "Outlook slash commands:\n"
        "• /unread — unread mail in inbox\n"
        "• /inbox — recent inbox messages\n"
        "• /from [name] — mail from a person\n"
        "• /thread [id] — open a message by ID\n"
        "• /read [subject] — read-focused mail lookup\n"
        "• /search [terms] — search Outlook mail\n\n"
        "Combine with #inbox, #sent, @today, @sender in the composer."
    ),
    "onedrive": (
        "OneDrive slash commands:\n"
        "• /doc [name] — read a Word/Office file\n"
        "• /read [name] — read a file by name\n"
        "• /search [terms] — search OneDrive/SharePoint"
    ),
}


@dataclass
class SlashCommandResult:
    kind: SlashKind
    command: str
    rewritten_query: Optional[str] = None
    tool: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    help_text: Optional[str] = None


def normalize_connector(connector: Optional[str]) -> str:
    if not connector:
        return ""
    c = str(connector).lower().strip()
    if c in ("email", "gmail"):
        return "email"
    if c in ("google", "google_workspace", "drive", "docs", "sheets", "presentation"):
        return "google"
    if c in ("sharepoint",):
        return "onedrive"
    return c


def parse_slash_command(
    query: str,
    connector: Optional[str] = None,
) -> Optional[SlashCommandResult]:
    """
    Parse a leading slash command. Returns None if the query is not a slash command.
    """
    if not query or not query.strip().startswith("/"):
        return None

    text = query.strip()
    match = _SLASH_PATTERN.match(text)
    if not match:
        logger.debug("Unrecognized slash command prefix in query=%r", text[:40])
        return None

    command = match.group(1).lower()
    remainder = (match.group(2) or "").strip()
    conn = normalize_connector(connector)
    logger.info(
        "Parsed slash command /%s connector=%r remainder=%r",
        command,
        conn or "(none)",
        remainder[:80] if remainder else "",
    )

    if command == "help":
        help_text = _HELP_BY_CONNECTOR.get(conn) or _HELP_BY_CONNECTOR[""]
        return SlashCommandResult(kind="help", command=command, help_text=help_text)

    if command == "search":
        rewritten = remainder or "search"
        return SlashCommandResult(kind="rewrite", command=command, rewritten_query=rewritten)

    if command == "read":
        if conn == "email":
            return _slash_read_email(remainder)
        if conn == "outlook":
            return _slash_read_outlook(remainder)
        if conn == "calendar":
            return SlashCommandResult(
                kind="rewrite",
                command=command,
                rewritten_query=(
                    f"@soon calendar events {remainder}".strip()
                    if remainder
                    else "@today what's on my calendar"
                ),
            )
        direct = _slash_to_direct_mcp("read", remainder, conn)
        if direct:
            tool, params = direct
            return SlashCommandResult(
                kind="direct_mcp",
                command=command,
                tool=tool,
                params=params,
            )
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=remainder or "read",
        )

    return _connector_slash_handler(command, remainder, conn)


def _slash_to_direct_mcp(
    command: str,
    target: str,
    connector: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Map /read-style commands to MCP tool + params."""
    if not target and command not in ("page", "db", "doc", "sheet"):
        return None

    if connector == "slack":
        channel = target.lstrip("#@").strip() if target else ""
        return "get_channel_history", {"channel_name": channel or "current"}

    if connector == "notion":
        if command in ("page", "read"):
            return "get_page_content", {"page_name": target}
        if command == "db":
            return "search_databases", {"query": target or ""}

    if connector == "google":
        if command == "sheet":
            return "search_files", {"query": f"spreadsheet {target}".strip()}
        if command in ("doc", "read"):
            return "read_drive_file", {"file_name": target}
        if command == "search":
            return "search_files", {"query": target or ""}

    if connector == "onedrive":
        if command in ("read", "doc"):
            return "read_file", {"file_name": target}
        if command == "search":
            return "search_files", {"query": target or ""}

    if connector == "slack" and command == "read" and target:
        if target.startswith("#") or target.startswith("@"):
            return "get_channel_history", {"channel_name": target.lstrip("#@").strip()}
        return "get_channel_history", {"channel_name": target}

    return None


def _slash_read_outlook(remainder: str) -> SlashCommandResult:
    if not remainder:
        rewritten = "read recent outlook inbox messages"
    elif remainder.startswith("@"):
        rewritten = f"outlook emails from {remainder.lstrip('@').strip()}"
    else:
        rewritten = f"read outlook emails about {remainder}"
    logger.info("Outlook /read rewrite: %r -> %r", remainder, rewritten)
    return SlashCommandResult(kind="rewrite", command="read", rewritten_query=rewritten)


def _slash_read_email(remainder: str) -> SlashCommandResult:
    """Gmail /read always rewrites; MCP routing handles tools (no direct_mcp)."""
    if not remainder:
        rewritten = "read recent emails #inbox"
    elif remainder.startswith("@"):
        rewritten = f"emails from {remainder.lstrip('@').strip()}"
    elif re.match(r"^[a-f0-9]{10,}$", remainder, re.I):
        rewritten = f"thread_id={remainder}"
    else:
        rewritten = f"read emails about {remainder}"
    logger.info("Gmail /read rewrite: %r -> %r", remainder, rewritten)
    return SlashCommandResult(kind="rewrite", command="read", rewritten_query=rewritten)


def _connector_slash_handler(
    command: str,
    remainder: str,
    connector: str,
) -> Optional[SlashCommandResult]:
    """Per-connector slash handlers; default to query rewrite."""

    if command == "unread":
        if connector == "outlook":
            q = "show unread outlook emails #inbox"
        elif connector and connector not in ("email", ""):
            logger.warning("/unread used with connector=%r; rewriting anyway", connector)
            q = "show unread emails #inbox"
        else:
            q = "show unread emails #inbox"
        if remainder:
            q = f"{q} {remainder}"
        return SlashCommandResult(kind="rewrite", command=command, rewritten_query=q)

    if command == "inbox":
        if connector and connector not in ("outlook", ""):
            logger.warning("/inbox used with connector=%r; rewriting anyway", connector)
        q = "recent emails in my outlook inbox"
        if remainder:
            q = f"{q} {remainder}"
        return SlashCommandResult(kind="rewrite", command=command, rewritten_query=q)

    if command == "from":
        target = remainder or ""
        if connector == "outlook":
            rewritten = f"outlook emails from {target}".strip() if target else "outlook emails from "
        else:
            rewritten = f"emails from {target}".strip() if target else "emails from "
        return SlashCommandResult(kind="rewrite", command=command, rewritten_query=rewritten)

    if command == "thread":
        if connector == "slack":
            rewritten = f"slack thread {remainder}".strip() if remainder else "slack thread"
            return SlashCommandResult(kind="rewrite", command=command, rewritten_query=rewritten)
        if connector == "outlook":
            if remainder and re.match(r"^[A-Za-z0-9_-]{20,}$", remainder):
                rewritten = f"message_id={remainder}"
            else:
                rewritten = (
                    f"outlook thread {remainder}".strip()
                    if remainder
                    else "that outlook email thread follow-up"
                )
            return SlashCommandResult(kind="rewrite", command=command, rewritten_query=rewritten)
        if remainder and re.match(r"^[a-f0-9]{10,}$", remainder, re.I):
            rewritten = f"thread_id={remainder}"
        else:
            rewritten = (
                f"thread_id={remainder} email thread"
                if remainder
                else "that email thread follow-up"
            )
        return SlashCommandResult(kind="rewrite", command=command, rewritten_query=rewritten)

    if command == "free":
        when = remainder if remainder else "@today"
        if not when.startswith("@"):
            when = f"@{when}"
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=f"{when} find free time on my calendar",
        )

    if command == "busy":
        when = remainder if remainder else "@today"
        if not when.startswith("@"):
            when = f"@{when}"
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=f"{when} @busy meetings on my calendar",
        )

    if command == "next":
        extra = f" {remainder}" if remainder else ""
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=f"@soon next upcoming calendar event{extra}".strip(),
        )

    if command == "channel":
        name = remainder.lstrip("#").strip() if remainder else ""
        prefix = f"#{name}" if name else "#"
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=f"{prefix} recent slack messages".strip(),
        )

    if command == "dm":
        name = remainder.lstrip("@").strip() if remainder else ""
        prefix = f"@{name}" if name else "@"
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=f"{prefix} recent direct messages".strip(),
        )

    if command == "page":
        direct = _slash_to_direct_mcp("page", remainder, connector or "notion")
        if direct and (connector in ("notion", "") or not connector):
            tool, params = direct
            return SlashCommandResult(
                kind="direct_mcp", command=command, tool=tool, params=params
            )
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=f"notion page {remainder}".strip() if remainder else "notion page",
        )

    if command == "db":
        direct = _slash_to_direct_mcp("db", remainder, connector or "notion")
        if direct and (connector in ("notion", "") or not connector):
            tool, params = direct
            return SlashCommandResult(
                kind="direct_mcp", command=command, tool=tool, params=params
            )
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=f"notion databases {remainder}".strip() if remainder else "notion databases",
        )

    if command == "doc":
        direct = _slash_to_direct_mcp("doc", remainder, connector or "google")
        if direct and connector in ("google", "onedrive", ""):
            tool, params = direct
            return SlashCommandResult(
                kind="direct_mcp", command=command, tool=tool, params=params
            )
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=f"google doc {remainder}".strip() if remainder else "google doc",
        )

    if command == "sheet":
        direct = _slash_to_direct_mcp("sheet", remainder, connector or "google")
        if direct and connector in ("google", ""):
            tool, params = direct
            return SlashCommandResult(
                kind="direct_mcp", command=command, tool=tool, params=params
            )
        return SlashCommandResult(
            kind="rewrite",
            command=command,
            rewritten_query=f"google sheet {remainder}".strip() if remainder else "google sheet",
        )

    logger.debug("No handler for slash command /%s", command)
    return None
