"""
Entity Query Handler for ENTITY_COUNT and FILE_LOCATION queries.

Handles structured queries by scrolling/filtering Qdrant payloads directly,
bypassing the standard retrieval pipeline. Uses the enhanced metadata
stored during indexing (entities, entity_names, document_entity_counts).
"""

import logging
import re
from typing import Dict, Any, Set, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from src.query_processing.entity_matcher import is_entity_match, count_entity_mentions_in_text

logger = logging.getLogger(__name__)


class EntityQueryHandler:
    """
    Handles entity-related queries using Qdrant payload filtering.

    Supports:
        - ENTITY_COUNT: Count total mentions across all documents
        - FILE_LOCATION: Find all files containing an entity

    These queries bypass the standard retrieval pipeline and operate
    directly on Qdrant's indexed payloads.
    """

    def __init__(
            self,
            qdrant_client: QdrantClient,
            collection_name: str,
            structured_query_fast_mode: bool = False
    ):
        """
        Initialize the entity query handler.

        Args:
            qdrant_client: Qdrant client instance
            collection_name: Name of the Qdrant collection to query
            structured_query_fast_mode: If True, prefer metadata-only fast paths
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        self.structured_query_fast_mode = structured_query_fast_mode

    def handle_count_query(
            self,
            entity_name: Optional[str] = None,
            entity_names: Optional[List[str]] = None,
            require_all_entities: bool = False
    ) -> Dict[str, Any]:
        """
        Count total mentions of entity/entities across all documents.

        Strategy:
        1. Scroll through first-chunks (is_first_chunk=True) to get document_entity_counts
        2. Fuzzy-match the entity name against stored entity names
        3. Sum counts across all matching documents

        Args:
            entity_name: Single entity to count (backward-compatible)
            entity_names: Optional list of entities for multi-entity co-mention queries
            require_all_entities: If True, return files/chunks where all entities co-occur

        Returns:
            {
                "entity": str,
                "entities": [str, ...],
                "require_all_entities": bool,
                "total_mentions": int,
                "files_found": int,
                "file_breakdown": [{"file_name": str, "count": int}, ...]
            }
        """
        target_entities = self._normalize_entity_targets(entity_name, entity_names)
        if not target_entities:
            return {
                "entity": "",
                "entities": [],
                "require_all_entities": False,
                "total_mentions": 0,
                "files_found": 0,
                "file_breakdown": [],
                "error": "No entity provided"
            }

        effective_require_all = bool(require_all_entities or len(target_entities) > 1)
        entity_label = self._build_entity_label(target_entities, effective_require_all)
        logger.info(
            f"Handling ENTITY_COUNT query for entities={target_entities} "
            f"(require_all={effective_require_all})"
        )

        try:
            if len(target_entities) == 1 and not effective_require_all:
                primary_entity = target_entities[0]
                metadata_matches = self._collect_metadata_matches(primary_entity)
                metadata_only = bool(metadata_matches)

                if not metadata_matches:
                    logger.info(
                        "No metadata matches found for '%s'; falling back to direct collection scan",
                        primary_entity,
                    )
                    verified_counts = self._scan_collection_for_entity_mentions(primary_entity)
                    file_breakdown = []
                    for file_name, count in verified_counts.items():
                        if count <= 0:
                            continue
                        file_breakdown.append(
                            {
                                "file_name": file_name,
                                "count": count,
                            }
                        )

                    file_breakdown.sort(key=lambda x: x["count"], reverse=True)
                    total_count = sum(item["count"] for item in file_breakdown)
                    result = {
                        "entity": primary_entity,
                        "entities": [primary_entity],
                        "require_all_entities": False,
                        "total_mentions": total_count,
                        "files_found": len(file_breakdown),
                        "file_breakdown": file_breakdown,
                        "metadata_fallback_used": True,
                    }
                elif self.structured_query_fast_mode:
                    result = self._build_fast_count_result(primary_entity, metadata_matches)
                    if metadata_only:
                        result["metadata_only"] = True
                else:
                    verified_counts = self._count_mentions_in_files(primary_entity, set(metadata_matches.keys()))

                    file_breakdown = []
                    for file_name, count in verified_counts.items():
                        if count <= 0:
                            continue

                        details = metadata_matches.get(file_name, {})
                        breakdown_item = {
                            "file_name": file_name,
                            "count": count,
                        }
                        matched_entities = sorted(details.get("matched_entities", set()))
                        if matched_entities:
                            breakdown_item["matched_entities"] = matched_entities
                        file_breakdown.append(breakdown_item)

                    # Sort by count descending
                    file_breakdown.sort(key=lambda x: x["count"], reverse=True)
                    total_count = sum(item["count"] for item in file_breakdown)

                    result = {
                        "entity": primary_entity,
                        "entities": [primary_entity],
                        "require_all_entities": False,
                        "total_mentions": total_count,
                        "files_found": len(file_breakdown),
                        "file_breakdown": file_breakdown
                    }
            else:
                metadata_by_entity = {
                    e: self._collect_metadata_matches(e)
                    for e in target_entities
                }
                candidate_files = self._select_candidate_files(metadata_by_entity, effective_require_all)

                if self.structured_query_fast_mode:
                    file_breakdown = []
                    total_count = 0
                    for file_name in sorted(candidate_files):
                        entity_counts = {}
                        matched_entities: Set[str] = set()
                        for entity in target_entities:
                            details = metadata_by_entity.get(entity, {}).get(file_name, {})
                            count = details.get("count", 0)
                            if count <= 0 and details.get("matched_entities"):
                                count = 1
                            entity_counts[entity] = count
                            matched_entities.update(details.get("matched_entities", set()))

                        if effective_require_all and any(v <= 0 for v in entity_counts.values()):
                            continue
                        file_count = min(entity_counts.values()) if effective_require_all else sum(
                            entity_counts.values())
                        if file_count <= 0:
                            continue

                        item = {
                            "file_name": file_name,
                            "count": file_count,
                            "entity_counts": entity_counts,
                        }
                        if matched_entities:
                            item["matched_entities"] = sorted(matched_entities)
                        file_breakdown.append(item)
                        total_count += file_count
                else:
                    verified_counts_by_entity = {
                        entity: self._count_mentions_in_files(entity, candidate_files)
                        for entity in target_entities
                    }
                    file_breakdown = []
                    total_count = 0

                    for file_name in sorted(candidate_files):
                        entity_counts = {
                            entity: verified_counts_by_entity.get(entity, {}).get(file_name, 0)
                            for entity in target_entities
                        }
                        if effective_require_all and any(v <= 0 for v in entity_counts.values()):
                            continue
                        file_count = min(entity_counts.values()) if effective_require_all else sum(
                            entity_counts.values())
                        if file_count <= 0:
                            continue

                        matched_entities: Set[str] = set()
                        for entity in target_entities:
                            details = metadata_by_entity.get(entity, {}).get(file_name, {})
                            matched_entities.update(details.get("matched_entities", set()))

                        item = {
                            "file_name": file_name,
                            "count": file_count,
                            "entity_counts": entity_counts,
                        }
                        if matched_entities:
                            item["matched_entities"] = sorted(matched_entities)

                        file_breakdown.append(item)
                        total_count += file_count

                file_breakdown.sort(key=lambda x: x["count"], reverse=True)
                result = {
                    "entity": entity_label,
                    "entities": target_entities,
                    "require_all_entities": effective_require_all,
                    "total_mentions": total_count,
                    "files_found": len(file_breakdown),
                    "file_breakdown": file_breakdown
                }

            logger.info(
                f"ENTITY_COUNT result: {result.get('total_mentions', 0)} mentions across "
                f"{result.get('files_found', 0)} files"
            )
            return result

        except Exception as e:
            logger.error(f"Entity count query failed: {e}", exc_info=True)
            return {
                "entity": entity_label,
                "entities": target_entities,
                "require_all_entities": effective_require_all,
                "total_mentions": 0,
                "files_found": 0,
                "file_breakdown": [],
                "error": str(e)
            }

    def handle_file_location_query(
            self,
            entity_name: Optional[str] = None,
            entity_names: Optional[List[str]] = None,
            require_all_entities: bool = False
    ) -> Dict[str, Any]:
        """
        Find all files containing one or multiple entities.

        Strategy:
        1. Scroll through all points where entity_names contains the entity
        2. Collect unique source filenames

        Args:
            entity_name: Single entity to search for (backward-compatible)
            entity_names: Optional list of entities for multi-entity co-mention queries
            require_all_entities: If True, only files containing all entities are returned

        Returns:
            {
                "entity": str,
                "entities": [str, ...],
                "require_all_entities": bool,
                "files": [str, ...],
                "total_files": int
            }
        """
        target_entities = self._normalize_entity_targets(entity_name, entity_names)
        if not target_entities:
            return {
                "entity": "",
                "entities": [],
                "require_all_entities": False,
                "files": [],
                "file_details": [],
                "total_files": 0,
                "error": "No entity provided"
            }

        effective_require_all = bool(require_all_entities or len(target_entities) > 1)
        entity_label = self._build_entity_label(target_entities, effective_require_all)
        logger.info(
            f"Handling FILE_LOCATION query for entities={target_entities} "
            f"(require_all={effective_require_all})"
        )

        try:
            if len(target_entities) == 1 and not effective_require_all:
                primary_entity = target_entities[0]
                metadata_matches = self._collect_metadata_matches(primary_entity)
                metadata_only = bool(metadata_matches)

                if not metadata_matches:
                    logger.info(
                        "No metadata matches found for '%s'; falling back to direct collection scan",
                        primary_entity,
                    )
                    verified_counts = self._scan_collection_for_entity_mentions(primary_entity)
                    file_details = []
                    for file_name, count in verified_counts.items():
                        if count <= 0:
                            continue
                        file_details.append(
                            {
                                "file_name": file_name,
                                "mention_count": count,
                            }
                        )
                    file_details.sort(key=lambda x: x.get("mention_count", 0), reverse=True)
                elif self.structured_query_fast_mode:
                    file_details = []
                    for file_name, details in metadata_matches.items():
                        mention_count = details.get("count", 0)
                        item = {
                            "file_name": file_name,
                            "mention_count": mention_count
                        }
                        matched_entities = sorted(details.get("matched_entities", set()))
                        if matched_entities:
                            item["matched_entities"] = matched_entities
                        file_details.append(item)

                    file_details.sort(key=lambda x: x.get("mention_count", 0), reverse=True)
                else:
                    verified_counts = self._count_mentions_in_files(primary_entity, set(metadata_matches.keys()))
                    file_details = []
                    for file_name, count in verified_counts.items():
                        if count <= 0:
                            continue

                        details = metadata_matches.get(file_name, {})
                        item = {
                            "file_name": file_name,
                            "mention_count": count
                        }
                        matched_entities = sorted(details.get("matched_entities", set()))
                        if matched_entities:
                            item["matched_entities"] = matched_entities
                        file_details.append(item)

                    # Sort by mention count descending
                    file_details.sort(key=lambda x: x.get("mention_count", 0), reverse=True)
            else:
                metadata_by_entity = {
                    e: self._collect_metadata_matches(e)
                    for e in target_entities
                }
                candidate_files = self._select_candidate_files(metadata_by_entity, effective_require_all)

                if self.structured_query_fast_mode:
                    file_details = []
                    for file_name in sorted(candidate_files):
                        entity_counts = {}
                        matched_entities: Set[str] = set()
                        for entity in target_entities:
                            details = metadata_by_entity.get(entity, {}).get(file_name, {})
                            count = details.get("count", 0)
                            if count <= 0 and details.get("matched_entities"):
                                count = 1
                            entity_counts[entity] = count
                            matched_entities.update(details.get("matched_entities", set()))

                        if effective_require_all and any(v <= 0 for v in entity_counts.values()):
                            continue
                        mention_count = min(entity_counts.values()) if effective_require_all else sum(
                            entity_counts.values())
                        if mention_count <= 0:
                            continue

                        item = {
                            "file_name": file_name,
                            "mention_count": mention_count,
                            "entity_counts": entity_counts,
                        }
                        if matched_entities:
                            item["matched_entities"] = sorted(matched_entities)
                        file_details.append(item)
                else:
                    verified_counts_by_entity = {
                        entity: self._count_mentions_in_files(entity, candidate_files)
                        for entity in target_entities
                    }
                    file_details = []
                    for file_name in sorted(candidate_files):
                        entity_counts = {
                            entity: verified_counts_by_entity.get(entity, {}).get(file_name, 0)
                            for entity in target_entities
                        }
                        if effective_require_all and any(v <= 0 for v in entity_counts.values()):
                            continue
                        mention_count = min(entity_counts.values()) if effective_require_all else sum(
                            entity_counts.values())
                        if mention_count <= 0:
                            continue

                        matched_entities: Set[str] = set()
                        for entity in target_entities:
                            details = metadata_by_entity.get(entity, {}).get(file_name, {})
                            matched_entities.update(details.get("matched_entities", set()))

                        item = {
                            "file_name": file_name,
                            "mention_count": mention_count,
                            "entity_counts": entity_counts,
                        }
                        if matched_entities:
                            item["matched_entities"] = sorted(matched_entities)
                        file_details.append(item)

                file_details.sort(key=lambda x: x.get("mention_count", 0), reverse=True)

            result = {
                "entity": entity_label,
                "entities": target_entities,
                "require_all_entities": effective_require_all,
                "files": [f["file_name"] for f in file_details],
                "file_details": file_details,
                "total_files": len(file_details)
            }
            if len(target_entities) == 1 and not effective_require_all and not metadata_only:
                result["metadata_fallback_used"] = True

            logger.info(
                f"FILE_LOCATION result: entity found in {len(file_details)} files"
            )
            return result

        except Exception as e:
            logger.error(f"File location query failed: {e}", exc_info=True)
            return {
                "entity": entity_label,
                "entities": target_entities,
                "require_all_entities": effective_require_all,
                "files": [],
                "file_details": [],
                "total_files": 0,
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
    def _build_entity_label(entity_names: List[str], require_all_entities: bool) -> str:
        """Create display label for one/many entities."""
        if not entity_names:
            return ""
        if len(entity_names) == 1:
            return entity_names[0]
        if len(entity_names) == 2:
            base = f"{entity_names[0]} and {entity_names[1]}"
        else:
            base = f"{', '.join(entity_names[:-1])}, and {entity_names[-1]}"
        if require_all_entities:
            return f"both/all of {base}"
        return base

    @staticmethod
    def _select_candidate_files(
            metadata_by_entity: Dict[str, Dict[str, Dict[str, Any]]],
            require_all_entities: bool
    ) -> Set[str]:
        """Select candidate files by union/intersection over per-entity metadata matches."""
        file_sets = [set(matches.keys()) for matches in metadata_by_entity.values() if matches]
        if not file_sets:
            return set()
        if require_all_entities:
            return set.intersection(*file_sets)
        return set.union(*file_sets)

    def _build_fast_count_result(self, entity_name: str, metadata_matches: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Build count response from first-chunk metadata only (fast mode)."""
        file_breakdown = []
        total_count = 0
        estimated_count_used = False

        for file_name, details in metadata_matches.items():
            count = details.get("count", 0)

            # Metadata fallback for older indexes: if only entity_names matched,
            # keep file visible with a minimal estimated count.
            if count <= 0 and details.get("matched_entities"):
                count = 1
                estimated_count_used = True

            if count <= 0:
                continue

            item = {"file_name": file_name, "count": count}
            matched_entities = sorted(details.get("matched_entities", set()))
            if matched_entities:
                item["matched_entities"] = matched_entities
            file_breakdown.append(item)
            total_count += count

        file_breakdown.sort(key=lambda x: x["count"], reverse=True)

        result = {
            "entity": entity_name,
            "entities": [entity_name],
            "require_all_entities": False,
            "total_mentions": total_count,
            "files_found": len(file_breakdown),
            "file_breakdown": file_breakdown
        }
        if estimated_count_used:
            result["count_estimated_from_entity_names"] = True

        return result

    def _collect_metadata_matches(self, entity_name: str) -> Dict[str, Dict[str, Any]]:
        """
        Collect first-chunk metadata matches for an entity.

        Returns:
            {
                "<file_name>": {
                    "count": int,                  # from document_entity_counts
                    "matched_entities": set[str],  # matched entity variants
                },
                ...
            }
        """
        # Fast mode optimization:
        # use payload index on entity_names to avoid scanning all first chunks.
        if self.structured_query_fast_mode:
            fast_candidates = self._build_entity_filter_candidates(entity_name)
            if fast_candidates:
                fast_matches = self._collect_metadata_matches_with_optional_prefilter(
                    entity_name=entity_name,
                    entity_filter_candidates=fast_candidates
                )
                if fast_matches:
                    return fast_matches
                logger.info(
                    "Fast metadata prefilter returned no matches for '%s'; "
                    "falling back to full first-chunk scan for robustness.",
                    entity_name,
                )

        return self._collect_metadata_matches_with_optional_prefilter(
            entity_name=entity_name,
            entity_filter_candidates=None
        )

    def _collect_metadata_matches_with_optional_prefilter(
            self,
            entity_name: str,
            entity_filter_candidates: Optional[List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Collect first-chunk metadata matches, optionally prefiltered by entity_names."""
        matches: Dict[str, Dict[str, Any]] = {}
        offset = None

        must_conditions = [
            FieldCondition(
                key="is_first_chunk",
                match=MatchValue(value=True)
            )
        ]
        if entity_filter_candidates:
            must_conditions.append(
                FieldCondition(
                    key="entity_names",
                    match=MatchAny(any=entity_filter_candidates)
                )
            )

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=must_conditions),
                limit=512,
                offset=offset,
                with_payload=["entity_names", "document_entity_counts", "source_filename"],
                with_vectors=False
            )

            if not results:
                break

            for point in results:
                payload = point.payload or {}
                source_file = payload.get("source_filename", "unknown")
                if not source_file:
                    continue

                doc_counts = payload.get("document_entity_counts") or {}
                entity_names = payload.get("entity_names") or []

                matched_entities: Set[str] = set()
                aggregated_count = 0

                if isinstance(doc_counts, dict):
                    for stored_entity, stored_count in doc_counts.items():
                        if not isinstance(stored_entity, str):
                            continue
                        if is_entity_match(entity_name, stored_entity):
                            matched_entities.add(stored_entity)
                            aggregated_count += self._safe_int(stored_count)

                if isinstance(entity_names, list):
                    for stored_entity in entity_names:
                        if not isinstance(stored_entity, str):
                            continue
                        if is_entity_match(entity_name, stored_entity):
                            matched_entities.add(stored_entity)

                if not matched_entities:
                    continue

                entry = matches.setdefault(
                    source_file,
                    {"count": 0, "matched_entities": set()}
                )
                entry["count"] += aggregated_count
                entry["matched_entities"].update(matched_entities)

            offset = next_offset
            if offset is None:
                break

        return matches

    @staticmethod
    def _build_entity_filter_candidates(entity_name: str) -> List[str]:
        """
        Build safe exact-match candidates for entity_names payload filtering.

        This narrows fast-mode scans while keeping precision constraints.
        """
        raw = (entity_name or "").strip()
        if not raw:
            return []

        trimmed = raw.strip(".,;:!?\"'()[]{}")
        if not trimmed:
            return []

        collapsed = re.sub(r"\s+", " ", trimmed).strip()
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

        # Common possessive variants in OCR'd/legal corpora.
        for base in list(candidates):
            if not base.endswith("'s"):
                candidates.add(f"{base}'s")

        return sorted(c for c in candidates if c)

    @staticmethod
    def _safe_int(value: Any) -> int:
        """Best-effort integer conversion for metadata counts."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _count_mentions_in_files(self, entity_name: str, files: set) -> Dict[str, int]:
        """
        Count verified entity mentions from actual chunk text for a set of files.

        This provides no-reindex precision by validating metadata-derived
        candidates against real content at query time.
        """
        if not files:
            return {}

        verified_counts = {file_name: 0 for file_name in files}
        file_list = list(files)
        batch_size = 50

        for i in range(0, len(file_list), batch_size):
            batch = file_list[i:i + batch_size]
            try:
                self._accumulate_batch_counts(entity_name, batch, verified_counts)
            except Exception as e:
                logger.warning(
                    f"Batch verification failed for {len(batch)} files ({e}); "
                    f"falling back to per-file verification"
                )
                for file_name in batch:
                    self._accumulate_single_file_counts(entity_name, file_name, verified_counts)

        return verified_counts

    def has_direct_text_match(self, entity_name: str) -> bool:
        """Cheap correctness fallback when metadata-based entity indexes miss a name."""
        if not entity_name:
            return False
        matches = self._scan_collection_for_entity_mentions(entity_name, stop_after_first_match=True)
        return bool(matches)

    def _scan_collection_for_entity_mentions(
            self,
            entity_name: str,
            stop_after_first_match: bool = False
    ) -> Dict[str, int]:
        """
        Scan collection chunk text directly for entity mentions.

        This is slower than metadata-based matching, so it is only used as a
        fallback when entity metadata fails to represent the queried name.
        """
        if not entity_name:
            return {}

        counts: Dict[str, int] = {}
        offset = None

        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=["source_filename", "content", "text"],
                with_vectors=False,
            )

            if not results:
                break

            for point in results:
                payload = point.payload or {}
                source_file = payload.get("source_filename", "")
                if not source_file:
                    continue

                content = payload.get("content", payload.get("text", ""))
                if not content:
                    continue

                mention_count = count_entity_mentions_in_text(entity_name, content)
                if mention_count <= 0:
                    continue

                counts[source_file] = counts.get(source_file, 0) + mention_count
                if stop_after_first_match:
                    return counts

            offset = next_offset
            if offset is None:
                break

        return counts

    def _accumulate_batch_counts(self, entity_name: str, batch: list, counts: Dict[str, int]) -> None:
        """Accumulate mention counts for a batch of files using MatchAny."""
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_filename",
                            match=MatchAny(any=batch)
                        )
                    ]
                ),
                limit=200,
                offset=offset,
                with_payload=["source_filename", "content", "text"],
                with_vectors=False
            )

            if not results:
                break

            for point in results:
                payload = point.payload
                source_file = payload.get("source_filename", "")
                if source_file not in counts:
                    continue
                content = payload.get("content", payload.get("text", ""))
                if not content:
                    continue
                counts[source_file] += count_entity_mentions_in_text(entity_name, content)

            offset = next_offset
            if offset is None:
                break

    def _accumulate_single_file_counts(self, entity_name: str, file_name: str, counts: Dict[str, int]) -> None:
        """Fallback verifier using per-file scrolling with MatchValue."""
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_filename",
                            match=MatchValue(value=file_name)
                        )
                    ]
                ),
                limit=200,
                offset=offset,
                with_payload=["content", "text"],
                with_vectors=False
            )

            if not results:
                break

            for point in results:
                payload = point.payload
                content = payload.get("content", payload.get("text", ""))
                if not content:
                    continue
                counts[file_name] += count_entity_mentions_in_text(entity_name, content)

            offset = next_offset
            if offset is None:
                break
