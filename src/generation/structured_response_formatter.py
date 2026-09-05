"""
Structured Response Formatter for enhanced query types.

Formats raw handler results (from EntityQueryHandler, ExactTextHandler)
into natural language answers. These responses bypass the LLM entirely
since the data is structured and doesn't need generation.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _render_entity_phrase(data: Dict[str, Any]) -> str:
    """Render one-or-many entity label for structured responses."""
    entities = data.get("entities") or []
    if isinstance(entities, list) and entities:
        if len(entities) == 1:
            return str(entities[0])
        if len(entities) == 2:
            return f"{entities[0]} and {entities[1]}"
        return ", ".join(str(e) for e in entities[:-1]) + f", and {entities[-1]}"

    entity = data.get("entity", "Unknown entity")
    return str(entity)


def _format_file_reference(file_name: str, azure_manager=None) -> str:
    """
    Format a file name for structured responses.

    Structured responses are intentionally link-free; frontend/backend owns
    link resolution based on citations/sources.
    """
    _ = azure_manager  # Intentionally unused for link-free structured answers.
    return str(file_name or "unknown")


def format_count_response(data: Dict[str, Any], azure_manager=None) -> str:
    """
    Format an entity count response into natural language.

    Example output:
        "**Jeffrey Epstein** is mentioned **127 times** across **5 files**:
        - **[epstein_emails_2015.pdf](https://...)**: 45 mentions
        - **[court_document_v2.pdf](https://...)**: 32 mentions
        ..."

    Args:
        data: Result from EntityQueryHandler.handle_count_query()
        azure_manager: Optional AzureBlobManager for generating file preview links

    Returns:
        Formatted markdown string
    """
    entity = data.get("entity", "Unknown entity")
    entities = data.get("entities") or []
    require_all_entities = bool(data.get("require_all_entities", False))
    total = data.get("total_mentions", 0)
    files_found = data.get("files_found", 0)
    breakdown = data.get("file_breakdown", [])
    error = data.get("error")

    if error:
        return f"I encountered an error while searching for mentions of {entity}: {error}"

    if total == 0:
        if require_all_entities and len(entities) > 1:
            return f"No co-mentions of {_render_entity_phrase(data)} were found in the indexed documents."
        return (
            f"No mentions of {entity} were found in the indexed documents. "
            f"This could mean the entity is not present, or the documents may need to be re-indexed "
            f"with entity extraction enabled."
        )

    # Build response
    if require_all_entities and len(entities) > 1:
        lines = [
            f"**{_render_entity_phrase(data)}** are co-mentioned **{total}** time{'s' if total != 1 else ''} "
            f"across **{files_found}** file{'s' if files_found != 1 else ''}:"
        ]
    else:
        lines = [
            f"**{entity}** is mentioned **{total}** time{'s' if total != 1 else ''} "
            f"across **{files_found}** file{'s' if files_found != 1 else ''}:"
        ]

    # Add file breakdown (limit to top 5 in answer body)
    max_visible_files = 5
    for idx, item in enumerate(breakdown[:max_visible_files], 1):
        file_name = item.get("file_name", "unknown")
        count = item.get("count", 0)
        file_display = _format_file_reference(file_name, azure_manager=azure_manager)
        lines.append(f"{idx}. {file_display} ({count} mention{'s' if count != 1 else ''})")

    if len(breakdown) > max_visible_files:
        lines.append("... and many more files.")

    return "\n".join(lines)


def format_file_location_response(data: Dict[str, Any], azure_manager=None) -> str:
    """
    Format a file location response into natural language.

    Example output:
        "**A. de Rothschild** appears in the following **3 files**:
        1. **[epstein_emails_2015.pdf](https://...)** (45 mentions)
        2. **[court_document.pdf](https://...)** (12 mentions)
        3. **[correspondence_log.docx](https://...)** (3 mentions)"

    Args:
        data: Result from EntityQueryHandler.handle_file_location_query()
        azure_manager: Optional AzureBlobManager for generating file preview links

    Returns:
        Formatted markdown string
    """
    entity = data.get("entity", "Unknown entity")
    entities = data.get("entities") or []
    require_all_entities = bool(data.get("require_all_entities", False))
    total_files = data.get("total_files", 0)
    file_details = data.get("file_details", [])
    error = data.get("error")

    if error:
        return f"I encountered an error while searching for files containing {entity}: {error}"

    if total_files == 0:
        if require_all_entities and len(entities) > 1:
            return f"No files were found where {_render_entity_phrase(data)} are mentioned together."
        return (
            f"No files were found containing {entity} in the indexed documents. "
            f"The documents may need to be re-indexed with entity extraction enabled."
        )

    if require_all_entities and len(entities) > 1:
        lines = [
            f"**{_render_entity_phrase(data)}** appear together in the following **{total_files}** file{'s' if total_files != 1 else ''}:"
        ]
    else:
        lines = [
            f"**{entity}** appears in the following **{total_files}** file{'s' if total_files != 1 else ''}:"
        ]

    max_visible_files = 5
    for i, detail in enumerate(file_details[:max_visible_files], 1):
        file_name = detail.get("file_name", "unknown")
        count = detail.get("mention_count", 0)
        count_str = f" ({count} mention{'s' if count != 1 else ''})" if count > 0 else ""
        
        file_display = _format_file_reference(file_name, azure_manager=azure_manager)
        
        lines.append(f"{i}. {file_display}{count_str}")

    if len(file_details) > max_visible_files:
        lines.append("... and many more files.")

    return "\n".join(lines)


def format_exact_text_response(data: Dict[str, Any], azure_manager=None) -> str:
    """
    Format exact text snippets as attributed quotes with source info.

    Example output:
        "Found **3 text snippets** matching your query:

        ---
        **From:** **[epstein_emails_2015.pdf](https://...)** (page 5)
        > "The meeting was arranged for March 15th. A. de Rothschild
        > confirmed attendance via email..."
        ---
        ..."

    Args:
        data: Result from ExactTextHandler.handle_exact_text()
        azure_manager: Optional AzureBlobManager for generating file preview links

    Returns:
        Formatted markdown string
    """
    total = data.get("total_found", 0)
    snippets = data.get("snippets", [])
    error = data.get("error")
    entities = data.get("entities") or []
    require_all_entities = bool(data.get("require_all_entities", False))

    if error:
        return f"I encountered an error while searching for the requested text: {error}"

    if total == 0:
        if require_all_entities and len(entities) > 1:
            return f"No text snippets were found where {_render_entity_phrase(data)} are mentioned together."
        return (
            "No matching text snippets were found in the indexed documents. "
            "Try rephrasing your query or ensure the documents have been indexed "
            "with entity extraction enabled."
        )

    max_visible_files = 5
    unique_file_snippets = []
    seen_files = set()
    for snippet in snippets:
        source_key = str(snippet.get("source_file", "unknown"))
        if source_key in seen_files:
            continue
        seen_files.add(source_key)
        unique_file_snippets.append(snippet)
        if len(unique_file_snippets) >= max_visible_files:
            break

    lines = [
        f"Found **{total}** text snippet{'s' if total != 1 else ''} matching your query:\n"
    ]

    for i, snippet in enumerate(unique_file_snippets, 1):
        source_file = snippet.get("source_file", "unknown")
        page = snippet.get("page_number")
        text = snippet.get("text", "")
        is_email = snippet.get("is_email", False)

        source_parts = [_format_file_reference(source_file, azure_manager=azure_manager)]
        
        if page is not None:
            source_parts.append(f"page {page}")

        # Add email headers if present
        lines.append(f"**Snippet {i}**")
        if is_email:
            sender = snippet.get("email_sender", "")
            receiver = snippet.get("email_receiver", "")
            date = snippet.get("email_date", "")
            subject = snippet.get("email_subject", "")

            lines.append(f"**Source:** {', '.join(source_parts)}")
            if sender:
                lines.append(f"From: {sender}")
            if receiver:
                lines.append(f"To: {receiver}")
            if date:
                lines.append(f"Date: {date}")
            if subject:
                lines.append(f"Subject: {subject}")
            lines.append("")
        else:
            lines.append(f"**Source:** {', '.join(source_parts)}")
            lines.append("")

        # Add quoted text (truncate if very long)
        if len(text) > 1000:
            quoted = text[:1000] + "..."
        else:
            quoted = text

        # Format as blockquote
        for line in quoted.split("\n"):
            lines.append(f"> {line}")
        lines.append("")
        lines.append("---")
        lines.append("")

    if len(seen_files) < len({str(s.get("source_file", "unknown")) for s in snippets}):
        lines.append("... and many more files.")

    return "\n".join(lines)
