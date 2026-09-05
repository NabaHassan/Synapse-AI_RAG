"""Index-backed lexical search backends for sparse retrieval.

The current in-process BM25 retriever is a safe fallback, but very large
collections benefit from a persistent lexical index that can answer keyword
candidate-generation queries without walking Python postings lists.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from haystack import Document
from qdrant_client import QdrantClient

from src.retrieval.bm25_index_cache import BM25IndexCache

logger = logging.getLogger(__name__)

LEXICAL_INDEX_VERSION = "sqlite_fts_v1"
DEFAULT_LEXICAL_PAYLOAD_FIELDS = [
    "content",
    "source",
    "source_filename",
    "source_filepath",
    "file_name",
    "filename",
    "file_path",
    "page",
    "page_number",
    "chunk_id",
    "chunk_index",
    "file_type",
    "document_id",
    "file_id",
    "source_family",
    "section_id",
    "document_type",
    "normalized_source",
    "ingest_state",
    "ingest_version",
    "ingest_job_id",
    "payload_version",
    "is_low_quality_chunk",
    "quality_flags",
    "ocr_or_extraction_quality",
    "quality_score",
    "email_sender",
    "email_receiver",
    "email_recipient",
    "sent_date",
    "date",
    "entity_names",
    "document_entity_counts",
    "legal_codes",
]

_TOKEN_PATTERN = re.compile(r"(?u)[\w§][\w§.-]*")
_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "give", "how", "i", "in", "is", "it", "list", "me", "mentioned",
    "of", "on", "or", "show", "tell", "that", "the", "there", "these", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


@dataclass(frozen=True)
class LexicalIndexStatus:
    ready: bool
    reason: str
    index_path: str
    collection_checksum: Optional[str] = None
    document_count: int = 0


def _safe_index_filename(collection_name: str) -> str:
    safe = _SAFE_NAME_PATTERN.sub("_", collection_name or "collection").strip("._")
    return f"{safe or 'collection'}.sqlite3"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _metadata_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload or {}
    source_filename = (
        payload.get("source_filename")
        or payload.get("source")
        or payload.get("file_name")
        or payload.get("filename")
        or ""
    )
    source_filepath = payload.get("source_filepath") or payload.get("file_path") or ""
    if not source_filename and source_filepath:
        source_filename = os.path.basename(str(source_filepath))

    meta = {
        "source": source_filename or "Unknown",
        "filepath": source_filepath,
        "page": payload.get("page_number", payload.get("page")),
        "chunk_id": payload.get("chunk_id"),
        "chunk_index": payload.get("chunk_index"),
        "file_type": payload.get("file_type", ""),
    }
    excluded = {
        "content",
        "source",
        "source_filename",
        "source_filepath",
        "file_name",
        "filename",
        "file_path",
        "page_number",
        "page",
        "chunk_id",
        "chunk_index",
        "file_type",
    }
    for key, value in payload.items():
        if key not in excluded and key not in meta:
            meta[key] = value
    return meta


def _fts_tokens(query: str) -> List[str]:
    tokens = []
    for token in _TOKEN_PATTERN.findall(query or ""):
        cleaned = token.strip(".-").lower()
        if cleaned in _LEXICAL_STOPWORDS:
            continue
        if len(cleaned) >= 2 or cleaned.isdigit():
            tokens.append(cleaned)
    return tokens


def _fts_match_expression(query: str, *, max_terms: int = 16, operator: str = "AND") -> str:
    tokens = _fts_tokens(query)
    if not tokens:
        return ""

    unique_tokens: List[str] = []
    seen = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique_tokens.append(token)
        if len(unique_tokens) >= max_terms:
            break

    # Prefix matching helps common document-search behavior without forcing
    # exact morphology, while still keeping the query index-backed.
    escaped = [f'"{token}"*' for token in unique_tokens]
    joiner = " OR " if str(operator).upper() == "OR" else " "
    return joiner.join(escaped)


class SQLiteFTSLexicalSearchBackend:
    """SQLite FTS5 lexical sidecar built from existing Qdrant payloads."""

    def __init__(
        self,
        *,
        collection_name: str,
        qdrant_client: Optional[QdrantClient] = None,
        index_dir: str = "./data/lexical_indices",
        cache_manager: Optional[BM25IndexCache] = None,
    ) -> None:
        self.collection_name = collection_name
        self.qdrant_client = qdrant_client
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / _safe_index_filename(collection_name)
        self.cache_manager = cache_manager or BM25IndexCache(cache_dir="./data/bm25_indices")
        self.last_query_debug: Dict[str, Any] = {}
        self._last_status: Optional[LexicalIndexStatus] = None

    def signature(self) -> str:
        status = self._last_status or self.status()
        return _json_dumps(
            {
                "backend": "sqlite_fts",
                "version": LEXICAL_INDEX_VERSION,
                "collection": self.collection_name,
                "checksum": status.collection_checksum,
                "document_count": status.document_count,
            }
        )

    def status(self) -> LexicalIndexStatus:
        if not self.index_path.exists():
            self._last_status = LexicalIndexStatus(False, "missing_index", str(self.index_path))
            return self._last_status

        try:
            with self._connect(readonly=True) as con:
                meta = self._read_meta(con)
                if meta.get("index_version") != LEXICAL_INDEX_VERSION:
                    self._last_status = LexicalIndexStatus(False, "version_mismatch", str(self.index_path))
                    return self._last_status
                if meta.get("collection_name") != self.collection_name:
                    self._last_status = LexicalIndexStatus(False, "collection_mismatch", str(self.index_path))
                    return self._last_status
                count = int(meta.get("document_count") or 0)
                if count <= 0:
                    self._last_status = LexicalIndexStatus(False, "empty_index", str(self.index_path))
                    return self._last_status

                checksum = meta.get("collection_checksum")
                current_checksum = self._current_collection_checksum()
                if current_checksum and checksum and checksum != current_checksum:
                    self._last_status = LexicalIndexStatus(
                        False,
                        "stale_checksum",
                        str(self.index_path),
                        collection_checksum=checksum,
                        document_count=count,
                    )
                    return self._last_status

                self._last_status = LexicalIndexStatus(
                    True,
                    "ready",
                    str(self.index_path),
                    collection_checksum=checksum,
                    document_count=count,
                )
                return self._last_status
        except Exception as exc:
            logger.warning("Lexical index status check failed for %s: %s", self.collection_name, exc)
            self._last_status = LexicalIndexStatus(False, f"status_error:{type(exc).__name__}", str(self.index_path))
            return self._last_status

    def is_ready(self) -> bool:
        return self.status().ready

    def search(self, query: str, *, top_k: int = 50, scale_score: bool = True) -> List[Document]:
        start = time.perf_counter()
        match_expr = _fts_match_expression(query, operator="AND")
        if not match_expr:
            self.last_query_debug = {
                "lexical_backend": "sqlite_fts",
                "lexical_backend_ready": self.is_ready(),
                "lexical_backend_used": True,
                "lexical_fallback_reason": "empty_match_expression",
                "returned": 0,
                "total_ms": round((time.perf_counter() - start) * 1000, 3),
            }
            return []

        with self._connect(readonly=True) as con:
            sql_start = time.perf_counter()
            rows = self._execute_match(con, match_expr, top_k)
            match_strategy = "and"
            if not rows:
                fallback_expr = _fts_match_expression(query, operator="OR")
                if fallback_expr and fallback_expr != match_expr:
                    rows = self._execute_match(con, fallback_expr, top_k)
                    match_strategy = "or_fallback"
            sql_ms = (time.perf_counter() - sql_start) * 1000

        docs = self._rows_to_documents(rows, scale_score=scale_score)
        self.last_query_debug = {
            "lexical_backend": "sqlite_fts",
            "lexical_backend_ready": True,
            "lexical_backend_used": True,
            "lexical_fallback_reason": None,
            "tokenize_ms": 0.0,
            "score_ms": round(sql_ms, 3),
            "sort_ms": 0.0,
            "build_results_ms": 0.0,
            "total_ms": round((time.perf_counter() - start) * 1000, 3),
            "query_token_count": len(_fts_tokens(query)),
            "unique_query_token_count": len(set(_fts_tokens(query))),
            "matched_terms": None,
            "postings_scanned": None,
            "candidate_count": len(rows),
            "corpus_size": (self._last_status.document_count if self._last_status else len(rows)),
            "returned": len(docs),
            "top_k": top_k,
            "scale_score": scale_score,
            "index_path": str(self.index_path),
            "match_strategy": match_strategy,
        }
        return docs

    def _execute_match(self, con: sqlite3.Connection, match_expr: str, top_k: int) -> List[sqlite3.Row]:
        return con.execute(
            """
            SELECT
                f.point_id,
                c.content,
                c.payload_json,
                bm25(chunks_fts) AS rank
            FROM chunks_fts AS f
            JOIN chunks AS c ON c.point_id = f.point_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank ASC
            LIMIT ?
            """,
            (match_expr, int(top_k)),
        ).fetchall()

    def build_from_qdrant(
        self,
        *,
        batch_size: int = 1000,
        limit: Optional[int] = None,
        payload_fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if self.qdrant_client is None:
            raise ValueError("qdrant_client is required to build a lexical index")

        collections = self.qdrant_client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            raise ValueError(f"Collection '{self.collection_name}' not found")

        checksum = self._current_collection_checksum()
        started = time.perf_counter()
        payload_selector: Any = list(payload_fields or DEFAULT_LEXICAL_PAYLOAD_FIELDS)
        inserted = 0
        offset = None

        with self._connect(readonly=False) as con:
            self._ensure_schema(con)
            self._reset_index(con)

            while True:
                current_limit = batch_size
                if limit is not None:
                    remaining = int(limit) - inserted
                    if remaining <= 0:
                        break
                    current_limit = min(current_limit, remaining)

                points, offset = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=current_limit,
                    offset=offset,
                    with_payload=payload_selector,
                    with_vectors=False,
                )
                if not points:
                    break

                self._insert_points(con, points)
                inserted += len(points)
                con.commit()

                if offset is None:
                    break

            self._write_meta(
                con,
                {
                    "index_version": LEXICAL_INDEX_VERSION,
                    "collection_name": self.collection_name,
                    "collection_checksum": checksum or "",
                    "document_count": str(inserted),
                    "built_at": str(time.time()),
                },
            )
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            con.commit()

        elapsed = time.perf_counter() - started
        logger.info(
            "Built SQLite FTS lexical index for %s: %s docs in %.2fs at %s",
            self.collection_name,
            inserted,
            elapsed,
            self.index_path,
        )
        return {
            "collection_name": self.collection_name,
            "index_path": str(self.index_path),
            "document_count": inserted,
            "collection_checksum": checksum,
            "elapsed_seconds": round(elapsed, 3),
        }

    def build_from_documents(
        self,
        documents: Iterable[Document],
        *,
        collection_checksum: str = "test",
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        inserted = 0
        with self._connect(readonly=False) as con:
            self._ensure_schema(con)
            self._reset_index(con)
            for doc in documents:
                payload = dict(getattr(doc, "meta", {}) or {})
                payload["content"] = getattr(doc, "content", "") or ""
                point_id = str(getattr(doc, "id", None) or payload.get("chunk_id") or inserted)
                self._insert_payload(con, point_id=point_id, payload=payload)
                inserted += 1
            self._write_meta(
                con,
                {
                    "index_version": LEXICAL_INDEX_VERSION,
                    "collection_name": self.collection_name,
                    "collection_checksum": collection_checksum,
                    "document_count": str(inserted),
                    "built_at": str(time.time()),
                },
            )
            con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
            con.commit()

        return {
            "collection_name": self.collection_name,
            "index_path": str(self.index_path),
            "document_count": inserted,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def upsert_payloads(self, point_payloads: Iterable[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Incrementally upsert payloads into the sidecar index after online ingest."""
        started = time.perf_counter()
        upserted = 0
        with self._connect(readonly=False) as con:
            self._ensure_schema(con)
            for point_id, payload in point_payloads:
                self._delete_point(con, str(point_id))
                self._insert_payload(con, point_id=str(point_id), payload=dict(payload or {}))
                upserted += 1
            count = int(con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 0)
            checksum = self._current_collection_checksum() or ""
            meta = self._read_meta(con)
            meta.update({
                "index_version": LEXICAL_INDEX_VERSION,
                "collection_name": self.collection_name,
                "collection_checksum": checksum,
                "document_count": str(count),
                "updated_at": str(time.time()),
            })
            if "built_at" not in meta:
                meta["built_at"] = str(time.time())
            self._write_meta(con, meta)
            con.commit()
        self._last_status = None
        return {
            "collection_name": self.collection_name,
            "index_path": str(self.index_path),
            "upserted": upserted,
            "document_count": count,
            "collection_checksum": checksum,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def _connect(self, *, readonly: bool) -> sqlite3.Connection:
        if readonly:
            uri = f"file:{self.index_path}?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=30)
        else:
            con = sqlite3.connect(str(self.index_path), timeout=60)
        con.row_factory = sqlite3.Row
        if not readonly:
            con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA temp_store=MEMORY")
        return con

    def _ensure_schema(self, con: sqlite3.Connection) -> None:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                point_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                point_id UNINDEXED,
                content,
                source_filename,
                document_id,
                chunk_id,
                tokenize='unicode61'
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS lexical_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

    def _reset_index(self, con: sqlite3.Connection) -> None:
        con.execute("DELETE FROM chunks")
        con.execute("DELETE FROM chunks_fts")
        con.execute("DELETE FROM lexical_meta")
        con.commit()

    def _insert_points(self, con: sqlite3.Connection, points: Sequence[Any]) -> None:
        for point in points:
            self._insert_payload(con, point_id=str(point.id), payload=dict(point.payload or {}))

    def _insert_payload(self, con: sqlite3.Connection, *, point_id: str, payload: Dict[str, Any]) -> None:
        content = str(payload.get("contextual_retrieval_text") or payload.get("content") or "")
        if not content.strip():
            return
        source_filename = str(
            payload.get("source_filename")
            or payload.get("source")
            or payload.get("file_name")
            or payload.get("filename")
            or ""
        )
        document_id = str(payload.get("document_id") or payload.get("file_id") or "")
        chunk_id = str(payload.get("chunk_id") or point_id)
        con.execute(
            "INSERT OR REPLACE INTO chunks(point_id, content, payload_json) VALUES (?, ?, ?)",
            (point_id, content, _json_dumps(payload)),
        )
        con.execute(
            """
            INSERT INTO chunks_fts(point_id, content, source_filename, document_id, chunk_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (point_id, content, source_filename, document_id, chunk_id),
        )

    def _delete_point(self, con: sqlite3.Connection, point_id: str) -> None:
        con.execute("DELETE FROM chunks WHERE point_id = ?", (point_id,))
        con.execute("DELETE FROM chunks_fts WHERE point_id = ?", (point_id,))

    def _rows_to_documents(self, rows: Sequence[sqlite3.Row], *, scale_score: bool) -> List[Document]:
        if not rows:
            return []
        ranks = [float(row["rank"]) for row in rows]
        min_rank = min(ranks)
        max_rank = max(ranks)
        span = max(max_rank - min_rank, 1e-9)
        docs: List[Document] = []
        for idx, row in enumerate(rows):
            payload = json.loads(row["payload_json"] or "{}")
            score = self._score_from_rank(float(row["rank"]), min_rank=min_rank, span=span, rank_index=idx)
            if not scale_score:
                score = -float(row["rank"])
            docs.append(
                Document(
                    content=row["content"] or "",
                    id=str(row["point_id"]),
                    score=score,
                    meta=_metadata_from_payload(payload),
                )
            )
        return docs

    def _score_from_rank(self, rank: float, *, min_rank: float, span: float, rank_index: int) -> float:
        if math.isfinite(rank):
            normalized = 1.0 - ((rank - min_rank) / span)
            return max(0.0, min(1.0, normalized))
        return 1.0 / (rank_index + 1)

    def _read_meta(self, con: sqlite3.Connection) -> Dict[str, str]:
        rows = con.execute("SELECT key, value FROM lexical_meta").fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def _write_meta(self, con: sqlite3.Connection, values: Dict[str, str]) -> None:
        for key, value in values.items():
            con.execute(
                "INSERT OR REPLACE INTO lexical_meta(key, value) VALUES (?, ?)",
                (str(key), str(value)),
            )

    def _current_collection_checksum(self) -> Optional[str]:
        if self.qdrant_client is None:
            return None
        return self.cache_manager.get_collection_checksum(self.qdrant_client, self.collection_name)
