import json
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

REDIS_PREFIX = "synapse:session_docs"
SESSION_TTL_SECONDS = 7200
MAX_FILE_BYTES = 25 * 1024 * 1024


def store_session_docs(
    redis_client,
    session_id: str,
    chunks: list[dict],
    file_meta: dict,
    ttl: int = SESSION_TTL_SECONDS,
) -> None:
    """
    Serialize chunks and file_meta to JSON and write them as fields of a single Redis HASH
    under the key synapse:session_docs:{session_id}. Immediately after the hset, call expire
    on the same key with ttl. Do not use separate keys per chunk — the single HASH keeps TTL atomic.
    """
    key = f"{REDIS_PREFIX}:{session_id}"
    redis_client.hset(
        key,
        mapping={
            "chunks": json.dumps(chunks),
            "file_meta": json.dumps(file_meta),
        },
    )
    redis_client.expire(key, ttl)


def get_session_doc_info(redis_client, session_id: str) -> Optional[dict]:
    """
    Return session document metadata and chunk counts for UI/status checks.
    Returns None when no document is stored for the session.
    """
    key = f"{REDIS_PREFIX}:{session_id}"
    if not redis_client.exists(key):
        return None

    raw_meta = redis_client.hget(key, "file_meta")
    raw_chunks = redis_client.hget(key, "chunks")
    file_meta = json.loads(raw_meta) if raw_meta else {}
    chunks = json.loads(raw_chunks) if raw_chunks else []
    ttl = redis_client.ttl(key)

    return {
        "filename": file_meta.get("filename"),
        "upload_time": file_meta.get("upload_time"),
        "total_chunks": file_meta.get("total_chunks", len(chunks)),
        "chunks_stored": len(chunks),
        "expires_in_seconds": ttl if ttl >= 0 else None,
        "ttl_seconds": SESSION_TTL_SECONDS,
        "available": len(chunks) > 0,
    }


def session_has_uploaded_docs(redis_client, session_id: str) -> bool:
    """Return True when the session has at least one uploaded document in Redis."""
    if not redis_client or not session_id:
        return False
    key = f"{REDIS_PREFIX}:{session_id}"
    try:
        exists = bool(redis_client.exists(key))
        logger.debug("session_has_uploaded_docs session_id=%s exists=%s", session_id, exists)
        return exists
    except Exception as exc:
        logger.warning("session_has_uploaded_docs check failed for %s: %s", session_id, exc)
        return False


def get_session_docs(redis_client, session_id: str) -> Optional[list[dict]]:
    """
    Build the key from REDIS_PREFIX and session_id. Call hget for the "chunks" field.
    Return None if the key does not exist or the field is missing.
    Otherwise deserialize and return the list of chunk dicts.
    """
    key = f"{REDIS_PREFIX}:{session_id}"
    raw = redis_client.hget(key, "chunks")
    if raw is None:
        return None
    return json.loads(raw)


def delete_session_docs(redis_client, session_id: str) -> None:
    """
    Delete the entire HASH key synapse:session_docs:{session_id}.
    One redis_client.delete() call.
    """
    key = f"{REDIS_PREFIX}:{session_id}"
    redis_client.delete(key)


def cosine_similarity_search(
    query_embedding: list[float],
    chunks: list[dict],
    top_k: int = 3,
    session_id: str | None = None,     # add this
) -> list[dict]:
    if not chunks:
        return []

    # Pre-filter — don't score chunks that can't possibly be relevant
    if session_id:
        chunks = [c for c in chunks if c.get("metadata", {}).get("session_id") == session_id]
    
    if not chunks:
        return []

    q_vec = np.array(query_embedding, dtype=np.float32)
    q_vec /= (np.linalg.norm(q_vec) + 1e-10)

    # Vectorized scoring — no Python loop
    matrix = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= (norms + 1e-10)
    scores = matrix @ q_vec                        # single matmul, not a loop

    top_indices = np.argpartition(scores, -min(top_k, len(scores)))[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [{"score": float(scores[i]), **chunks[i]} for i in top_indices]