"""
Exact Text Handler for EXACT_TEXT queries.

Generalized handler for retrieving exact text snippets, quotes, emails,
and specific passages from indexed documents. Uses Qdrant payload filtering
to find relevant chunks and returns them as attributed quotes.

Supports:
- Email retrieval (by sender/receiver)
- Entity mention quotes (by entity name)
- Keyword text search (by keyword)
"""

import logging
import re
from typing import Dict, Any, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText, MatchAny
from src.query_processing.entity_matcher import contains_entity_in_text

logger = logging.getLogger(__name__)


class ExactTextHandler:
    """
    Handles EXACT_TEXT queries — generalized handler for quotes, snippets, and emails.

    Works for:
        - Email retrieval: filter by email_sender/email_receiver
        - Entity mentions: token-aware text check on chunk content
        - Keyword search: text-match on chunk content

    Results bypass the LLM — structured data is returned directly.
    """

    def __init__(
            self,
            qdrant_client: QdrantClient,
            collection_name: str,
            structured_query_fast_mode: bool = False
    ):
        """
        Initialize the exact text handler.

        Args:
            qdrant_client: Qdrant client instance
            collection_name: Name of the Qdrant collection to query
            structured_query_fast_mode: If True, use metadata-first fast paths
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        self.structured_query_fast_mode = structured_query_fast_mode

    def handle_exact_text(
            self,
            entity_name: Optional[str] = None,
            entity_names: Optional[List[str]] = None,
            require_all_entities: bool = False,
            sender: Optional[str] = None,
            receiver: Optional[str] = None,
            date: Optional[str] = None,
            keyword: Optional[str] = None,
            max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Find exact text snippets matching the given criteria.

        Strategy:
        - If sender/receiver provided → filter by email payload fields
        - If entity_name provided → require full-token entity presence in chunk text
        - If keyword provided → text search across chunk content
        - Multiple criteria can be combined

        Args:
            entity_name: Single entity to find text snippets for
            entity_names: Optional list of entities for multi-entity co-mention queries
            require_all_entities: If True, snippet must contain all entities
            sender: Email sender to filter by
            receiver: Email receiver to filter by
            date: Email date to filter by
            keyword: Keyword to search in text
            max_results: Maximum number of snippets to return

        Returns:
            {
                "total_found": int,
                "snippets": [{
                    "text": str,
                    "source_file": str,
                    "page_number": int/None,
                    "chunk_index": int,
                    "is_email": bool,
                    "email_sender": str,
                    "email_receiver": str,
                    "email_date": str,
                    "email_subject": str,
                    "matched_entities": [str],
                }, ...]
            }
        """
        logger.info(
            f"Handling EXACT_TEXT query - entity={entity_name}, entities={entity_names}, "
            f"require_all={require_all_entities}, sender={sender}, receiver={receiver}, keyword={keyword}"
        )

        try:
            target_entities = self._normalize_entity_targets(entity_name, entity_names)
            effective_require_all = bool(require_all_entities or len(target_entities) > 1)

            # Build filter conditions based on provided params
            filter_conditions = []

            # Email-specific filters
            if sender:
                filter_conditions.append(
                    FieldCondition(
                        key="email_sender",
                        match=MatchText(text=sender)
                    )
                )
            if receiver:
                filter_conditions.append(
                    FieldCondition(
                        key="email_receiver",
                        match=MatchText(text=receiver)
                    )
                )
            if date:
                filter_conditions.append(
                    FieldCondition(
                        key="email_date",
                        match=MatchValue(value=date)
                    )
                )

            fast_entity_filter_applied = False

            # Fast mode: pre-filter exact text by indexed entity_names to avoid
            # scanning every chunk. This is most effective after full reindex.
            if target_entities and self.structured_query_fast_mode:
                entity_candidates = {
                    entity: self._build_entity_filter_candidates(entity)
                    for entity in target_entities
                }
                if all(entity_candidates.values()):
                    for entity in target_entities:
                        filter_conditions.append(
                            FieldCondition(
                                key="entity_names",
                                match=MatchAny(any=entity_candidates[entity])
                            )
                        )
                    # For multi-entity exact-text, each entity adds a must-clause.
                    fast_entity_filter_applied = True

            # If no specific filters, try keyword in content
            if not filter_conditions and keyword:
                # Fall back to scrolling all and text matching
                return self._keyword_search(keyword, max_results)

            # If only entity/entities provided (no email filters), scroll all and filter in loop
            if not filter_conditions and target_entities:
                # Will filter by entity in the results loop below
                pass  # Continue to scroll without filters
            elif not filter_conditions:
                logger.warning("No filter criteria provided for EXACT_TEXT query")
                return {"total_found": 0, "snippets": [], "error": "No search criteria provided"}

            # Execute Qdrant scroll with filters
            snippets = []
            offset = None
            seen_file_chunks = set()

            while len(snippets) < max_results:
                # Build scroll params - only add filter if we have conditions
                scroll_params = {
                    "collection_name": self.collection_name,
                    "limit": min(50, max_results - len(snippets)),
                    "offset": offset,
                    "with_payload": True,
                    "with_vectors": False
                }
                if filter_conditions:
                    scroll_params["scroll_filter"] = Filter(must=filter_conditions)
                
                results, next_offset = self.client.scroll(**scroll_params)

                if not results:
                    break

                for point in results:
                    payload = point.payload
                    content = payload.get("content", payload.get("text", ""))
                    source_file = payload.get("source_filename", "unknown")
                    chunk_idx = payload.get("chunk_index", 0)

                    # Precision-first entity check for non-fast mode or when fast-mode
                    # metadata prefilter could not be applied.
                    if target_entities:
                        if not self.structured_query_fast_mode or not fast_entity_filter_applied:
                            if effective_require_all:
                                if not all(contains_entity_in_text(entity, content) for entity in target_entities):
                                    continue
                            else:
                                if not any(contains_entity_in_text(entity, content) for entity in target_entities):
                                    continue

                    # Dedup by file+chunk
                    dedup_key = f"{source_file}_{chunk_idx}"
                    if dedup_key in seen_file_chunks:
                        continue
                    seen_file_chunks.add(dedup_key)

                    snippet = {
                        "text": content,
                        "source_file": source_file,
                        "page_number": payload.get("page_number"),
                        "chunk_index": chunk_idx,
                        "is_email": payload.get("is_email", False),
                        "email_sender": payload.get("email_sender", ""),
                        "email_receiver": payload.get("email_receiver", ""),
                        "email_date": payload.get("email_date", ""),
                        "email_subject": payload.get("email_subject", ""),
                        "matched_entities": payload.get("entity_names", []),
                    }
                    snippets.append(snippet)

                    if len(snippets) >= max_results:
                        break

                offset = next_offset
                if offset is None:
                    break

            result = {
                "total_found": len(snippets),
                "snippets": snippets,
                "entities": target_entities,
                "require_all_entities": effective_require_all
            }

            # Metadata can be sparse/inconsistent across KBs. If strict sender/receiver
            # filters return nothing, retry with relaxed entity/keyword matching so
            # follow-up exact-text queries still produce useful results.
            if not snippets and (sender or receiver):
                logger.info(
                    "No snippets found with sender/receiver filters; "
                    "retrying exact-text search with relaxed email participant filter."
                )
                relaxed_result = self.handle_exact_text(
                    entity_name=entity_name,
                    entity_names=entity_names,
                    require_all_entities=require_all_entities,
                    sender=None,
                    receiver=None,
                    date=date,
                    keyword=keyword,
                    max_results=max_results
                )
                if relaxed_result.get("total_found", 0) > 0:
                    return relaxed_result

            logger.info(f"EXACT_TEXT result: found {len(snippets)} matching snippets")
            return result

        except Exception as e:
            logger.error(f"Exact text query failed: {e}", exc_info=True)
            return {
                "total_found": 0,
                "snippets": [],
                "entities": self._normalize_entity_targets(entity_name, entity_names),
                "require_all_entities": bool(require_all_entities),
                "error": str(e)
            }

    @staticmethod
    def _normalize_entity_targets(
            entity_name: Optional[str],
            entity_names: Optional[List[str]]
    ) -> List[str]:
        """Normalize and deduplicate single/list entity inputs."""
        values: List[str] = []
        if entity_names:
            values.extend(entity_names)
        elif entity_name:
            values.append(entity_name)

        seen = set()
        normalized: List[str] = []
        for value in values:
            if not isinstance(value, str):
                continue
            cleaned = value.strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)

        return normalized

    @staticmethod
    def _build_entity_filter_candidates(entity_name: str) -> List[str]:
        """
        Build safe keyword candidates for entity_names metadata filtering.

        We avoid broad partial-token candidates to preserve precision.
        """
        raw = (entity_name or "").strip()
        if not raw:
            return []

        # Trim only edge punctuation; keep internal punctuation for initials.
        trimmed = raw.strip(".,;:!?\"'()[]{}")
        if not trimmed:
            return []

        collapsed = re.sub(r"\s+", " ", trimmed)
        punctuation_normalized = re.sub(r"[.]+", "", collapsed)
        punctuation_normalized = re.sub(r"\s+", " ", punctuation_normalized).strip()

        candidates = {
            collapsed,
            collapsed.lower(),
            collapsed.title(),
            collapsed.upper(),
        }

        if punctuation_normalized and punctuation_normalized != collapsed:
            candidates.update(
                {
                    punctuation_normalized,
                    punctuation_normalized.lower(),
                    punctuation_normalized.title(),
                    punctuation_normalized.upper(),
                }
            )

        return sorted(c for c in candidates if c)

    def _keyword_search(self, keyword: str, max_results: int) -> Dict[str, Any]:
        """
        Fall back to scrolling all chunks and text-matching the keyword.

        This is less efficient but works when no structured metadata
        filters are available.

        Args:
            keyword: Keyword to search for in chunk text
            max_results: Maximum results to return

        Returns:
            Same structure as handle_exact_text
        """
        logger.info(f"Falling back to keyword search for: '{keyword}'")

        snippets = []
        offset = None
        keyword_lower = keyword.lower()

        while len(snippets) < max_results:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=["content", "text", "source_filename", "page_number",
                              "chunk_index", "entity_names", "is_email",
                              "email_sender", "email_receiver", "email_date",
                              "email_subject"],
                with_vectors=False
            )

            if not results:
                break

            for point in results:
                payload = point.payload
                content = payload.get("content", payload.get("text", ""))

                if keyword_lower in content.lower():
                    source_file = payload.get("source_filename", "unknown")
                    snippet = {
                        "text": content,
                        "source_file": source_file,
                        "page_number": payload.get("page_number"),
                        "chunk_index": payload.get("chunk_index", 0),
                        "is_email": payload.get("is_email", False),
                        "email_sender": payload.get("email_sender", ""),
                        "email_receiver": payload.get("email_receiver", ""),
                        "email_date": payload.get("email_date", ""),
                        "email_subject": payload.get("email_subject", ""),
                        "matched_entities": payload.get("entity_names", []),
                    }
                    snippets.append(snippet)

                    if len(snippets) >= max_results:
                        break

            offset = next_offset
            if offset is None:
                break

        return {
            "total_found": len(snippets),
            "snippets": snippets
        }
