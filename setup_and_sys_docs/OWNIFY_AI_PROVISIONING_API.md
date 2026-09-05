# Ownify AI Provisioning API Guide

This guide covers the Ownify-specific AI APIs used by the Ownify backend.

The intended flow is:

1. Provision the tenant AI workspace.
2. Add or delete tenant documents.
3. Query the tenant KB by tenant id.
4. Delete the tenant AI workspace when the tenant is removed.

These endpoints are additive. Existing `/kb/...` APIs still work and should not be changed for deployed clients.

## Important Concepts

- One Ownify tenant maps to one KB.
- `tenant_id` is also the public `kb_id`.
- Runtime tenant config is stored in `data/kb_configs/<tenant_id>.yaml`.
- Provisioning, document ingestion, document deletion, and queries run through async job/executor paths.
- Query traffic uses the query executor and does not wait on provisioning or indexing jobs.
- Indexing runs in `online` mode, so document writes do not take the older exclusive read/write path that blocks queries.

## Async Behavior

The Ownify API may wait for a final response when that is easier for the backend, but the actual work is still offloaded to queues/executors.

| Operation | HTTP response | Work execution |
| --- | --- | --- |
| Provision tenant AI | Returns `202 accepted` immediately | Ownify provisioning queue + indexing executor |
| Update prompt/config | Returns after snapshot write | Indexing executor, no server restart |
| Add documents | Returns after indexing finishes | Ownify provisioning queue + indexing executor |
| Delete document | Returns after delete finishes | Ownify provisioning queue + indexing executor |
| Delete tenant AI | Returns after deletion finishes | Ownify provisioning queue + indexing executor |
| List/status | Returns current state | Executor-backed reads where Qdrant/file IO is involved |
| Query tenant | Returns `202 accepted` immediately | Async query queue + query executor |

## Authentication

Set this on the AI service:

```bash
export OWNIFY_PROVISIONING_API_KEY="your-internal-secret"
```

Then the Ownify backend sends either:

```bash
-H "Authorization: Bearer your-internal-secret"
```

or:

```bash
-H "X-Ownify-API-Key: your-internal-secret"
```

If `OWNIFY_PROVISIONING_API_KEY` is not set, `/ownify/...` endpoints are allowed without auth for local development, and the server logs a warning.

## 1. Provision Tenant AI

Use this when the tenant signs up, or when you need to recreate/update the tenant workspace config.

This endpoint creates or updates:

- tenant KB registration
- tenant display name and description
- tenant system prompt
- tenant AI config
- runtime config snapshot

It does not upload documents. Documents are added in the next step.

```bash
curl -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/provision \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-internal-secret" \
  -d '{
    "display_name": "Acme AI",
    "description": "AI assistant for Acme customers",
    "system_prompt": "You are Acme AI. Answer using Acme knowledge base content first.",
    "idempotency_key": "ownify-signup-acme-ai-v1",
    "replace_existing": false,
    "ai_config": {
      "generation": {
        "max_tokens": 1024,
        "temperature": 0.4,
        "top_p": 0.8
      },
      "grounding": {
        "allow_general_knowledge_fallback": false,
        "collection_anchor_terms": ["acme", "support", "faq"]
      },
      "canned_responses": {
        "greeting": "Hello! How can I help you today?",
        "acknowledgment": "You are welcome. Let me know if you have another question.",
        "meaningless": "I can help with questions related to this knowledge base."
      }
    }
  }' | jq
```

Expected response:

```json
{
  "status": "accepted",
  "job_id": "abc123",
  "tenant_id": "acme_ai",
  "kb_id": "acme_ai",
  "request_id": "req123",
  "job_status": "queued",
  "queued_at": "2026-04-15T10:00:00Z",
  "timeout_seconds": 900.0
}
```

Poll the job:

```bash
curl http://localhost:8000/ownify/tenants/acme_ai/ai/jobs/abc123 \
  -H "Authorization: Bearer your-internal-secret" | jq
```

Terminal job statuses:

- `succeeded`
- `succeeded_with_errors`
- `failed`
- `cancelled`
- `timed_out`

If you send documents to the provision endpoint, the API returns `400`. Use the document endpoint below.

## 2. Update Prompt Or AI Config

Use this when the tenant already exists and you only need to change prompt/config metadata.

No restart is required. The cached tenant pipeline is evicted automatically.

```bash
curl -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/config \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-internal-secret" \
  -d '{
    "display_name": "Acme Support AI",
    "system_prompt": "You are Acme Support AI. Be concise and use Acme documents as the source of truth.",
    "ai_config": {
      "generation": {
        "max_tokens": 768,
        "temperature": 0.3
      },
      "grounding": {
        "allow_general_knowledge_fallback": false
      }
    }
  }' | jq
```

## 3. Add Documents

Use the tenant id in the URL. Send one or more documents. Each document must provide exactly one source:

- `sas_url`: a temporary URL the AI service can download
- `local_path`: a path to a file already present on the AI service filesystem

For SAS URLs, send `file_id` and `file_name` from the Ownify backend when available. For local files, `file_id` and `file_name` can be omitted; the AI service uses the basename and generates a stable source-based ID.

The response returns only after indexing finishes. Internally, the document work still runs through the Ownify job queue and indexing executor, so normal query traffic and other API requests are not blocked while this request waits for the final result.

SAS URL documents:

```bash
curl -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/documents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-internal-secret" \
  -d '{
    "idempotency_key": "acme-doc-upload-batch-001",
    "documents": [
      {
        "file_id": "doc-uuid-456",
        "file_name": "pricing.pdf",
        "sas_url": "https://storage.blob.core.windows.net/..."
      },
      {
        "file_id": "doc-uuid-789",
        "file_name": "policies.docx",
        "sas_url": "https://storage.blob.core.windows.net/..."
      }
    ]
  }' | jq
```

Mixed SAS URL and server-local files:

```bash
curl -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/documents \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-internal-secret" \
  -d '{
    "idempotency_key": "acme-doc-upload-batch-002",
    "documents": [
      {
        "file_id": "doc-uuid-456",
        "file_name": "pricing.pdf",
        "sas_url": "https://storage.blob.core.windows.net/..."
      },
      {
        "local_path": "/data/uploads/acme/faq.md"
      }
    ]
  }' | jq
```

The Ownify document endpoint does not scan directories. To ingest a server-local directory directly, use the lower-level KB endpoint `POST /kb/{kb_id}/documents/add-batch` with `kb_id == tenant_id`.

Successful response:

```json
{
  "status": "success",
  "job_id": "4d177480cf8f4bd9a2775288c642a43a",
  "tenant_id": "acme_ai",
  "kb_id": "acme_ai",
  "request_id": "66455c776be143589b2802a3bf83f385",
  "job_status": "succeeded",
  "phase": "finalizing",
  "queued_at": "2026-04-16T09:35:37.256909Z",
  "started_at": "2026-04-16T09:35:37.257101Z",
  "completed_at": "2026-04-16T09:35:42.932400Z",
  "timeout_seconds": 900.0,
  "documents": [
    {
      "file_id": "doc-uuid-456",
      "file_name": "pricing.pdf",
      "status": "succeeded",
      "result": {
        "success": true,
        "file_id": "doc-uuid-456",
        "file_name": "pricing.pdf",
        "ingest_job_id": "ingest_abc123",
        "chunks_created": 18,
        "vectors_inserted": 18,
        "committed_points": 18,
        "stale_points_deleted": 0,
        "collection_name": "kb_acme_ai",
        "source_type": "sas_url",
        "source_uri": "https://storage.blob.core.windows.net/..."
      }
    }
  ],
  "result": {
    "documents_total": 1,
    "documents_failed": 0
  },
  "error": null
}
```

`POST /ownify/tenants/{tenant_id}/ai/documents/jobs` is also supported as a compatibility alias, but it now returns the same completed response.

If one document fails, the response returns `status: "succeeded_with_errors"` and includes per-document failure details.

## 4. List Or Delete Documents

List the tenant documents:

```bash
curl http://localhost:8000/ownify/tenants/acme_ai/ai/documents \
  -H "Authorization: Bearer your-internal-secret" | jq
```

Delete a document by its Ownify `file_id`.

The response returns only after the delete finishes. Internally, the delete still runs through the Ownify job queue and indexing executor.

```bash
curl -X DELETE "http://localhost:8000/ownify/tenants/acme_ai/ai/documents/doc-uuid-456?file_name=pricing.pdf&idempotency_key=acme-doc-delete-456-v1" \
  -H "Authorization: Bearer your-internal-secret" | jq
```

Successful response:

```json
{
  "status": "success",
  "job_id": "64fe29acb31548d0ae1f81fdd7628cc8",
  "tenant_id": "acme_ai",
  "kb_id": "acme_ai",
  "request_id": "657422aa6c6a44299c97d245bc6426bf",
  "job_status": "succeeded",
  "phase": "deleting_document",
  "queued_at": "2026-04-16T09:40:01.100000Z",
  "started_at": "2026-04-16T09:40:01.100000Z",
  "completed_at": "2026-04-16T09:40:01.540000Z",
  "timeout_seconds": 900.0,
  "file_id": "doc-uuid-456",
  "file_name": "pricing.pdf",
  "deleted_count": 18,
  "result": {
    "success": true,
    "file_id": "doc-uuid-456",
    "file_name": "pricing.pdf",
    "deleted_count": 18
  }
}
```

`file_name` is optional. It is only used for lookup/logging when available.

## 5. Check Tenant Status

```bash
curl http://localhost:8000/ownify/tenants/acme_ai/ai/status \
  -H "Authorization: Bearer your-internal-secret" | jq
```

Use this to confirm:

- tenant KB exists
- Qdrant collection status
- document count
- latest provisioning/document job
- current KB metadata

## 6. Query The Tenant

The easiest Ownify-facing flow is to create a session and submit an async query job.

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/session/new \
  -H "Authorization: Bearer your-internal-secret" | jq -r '.session_id')

JOB_ID=$(curl -s -X POST http://localhost:8000/ownify/tenants/acme_ai/ai/query/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-internal-secret" \
  -d '{"query": "What does Acme offer?", "session_id": "'$SESSION_ID'"}' | jq -r '.job_id')

curl -s http://localhost:8000/ownify/tenants/acme_ai/ai/query/jobs/$JOB_ID \
  -H "Authorization: Bearer your-internal-secret" | jq
```

You can also keep using the existing deployed KB APIs directly because `tenant_id == kb_id`:

```bash
curl -X POST http://localhost:8000/kb/acme_ai/query/jobs \
  -H "Content-Type: application/json" \
  -d '{"query": "What does Acme offer?"}' | jq
```

## 7. Delete Tenant AI

Use this when the Ownify tenant is removed and the AI workspace must be fully deleted.

The response returns only after deletion finishes. Internally, tenant delete runs through the Ownify job queue and indexing executor. Jobs for the same tenant are serialized, so a tenant delete will not race a document upload for that tenant. Other tenants and normal query traffic continue on their own executors.

Tenant delete removes:

- KB registry entry
- Qdrant collection
- runtime config snapshot
- document/file tracker
- conversation history
- query cache
- BM25 cache
- cached in-process pipeline
- Ownify provisioning/document job records for that tenant

```bash
curl -X DELETE "http://localhost:8000/ownify/tenants/acme_ai/ai?idempotency_key=acme-delete-ai-v1" \
  -H "Authorization: Bearer your-internal-secret" | jq
```

Successful response:

```json
{
  "status": "success",
  "job_id": "8e0c7a04a7a6406cb7c3eebc0e2d6a18",
  "tenant_id": "acme_ai",
  "kb_id": "acme_ai",
  "request_id": "5f96ea61c5d04e92a9413d0a0c62ad49",
  "job_status": "succeeded",
  "phase": "finalizing",
  "queued_at": "2026-04-16T10:00:01.100000Z",
  "started_at": "2026-04-16T10:00:01.101000Z",
  "completed_at": "2026-04-16T10:00:02.250000Z",
  "timeout_seconds": 900.0,
  "result": {
    "kb_id": "acme_ai",
    "tenant_id": "acme_ai",
    "existed": true,
    "delete": {
      "status": "success",
      "kb_id": "acme_ai",
      "collection_name": "kb_acme_ai"
    },
    "removed_job_records": 4
  },
  "error": null
}
```

If the tenant AI workspace was already deleted, the endpoint still succeeds with `existed: false`.

## Practical Rules

- Use one stable `tenant_id` per Ownify tenant.
- `tenant_id` must be 3-64 chars: letters, numbers, underscore, or dash, starting with a letter or number.
- Always send an `idempotency_key` for signup/provisioning and document upload batches.
- For document uploads, provide exactly one source per document: `sas_url` or `local_path`.
- Only use `local_path` for files that already exist on the AI service host.
- Use `replace_existing: true` only when intentionally recreating a tenant KB from scratch.
- Do not upload documents in the provisioning request.
- Use tenant delete only when the tenant is being removed from Ownify.
- Do not write tenant prompt/profile files under `src/config`; tenant config belongs in runtime snapshots.
- Existing `/kb/...` APIs are unchanged and remain valid.
