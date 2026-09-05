# CURL + API Commands

## Conversational RAG (Single‑KB)

### Query (sync)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is Nikola Tesla?"}'
```

### Query (async jobs)

```bash
# Submit async query job
curl -X POST http://localhost:8000/query/jobs \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is Nikola Tesla?"}'
```

### Health & Runtime Diagnostics

```bash
# Root status
curl http://localhost:8000/

# Liveness probe
curl http://localhost:8000/health/liveness | jq

# Readiness probe (deps + runtime readiness)
curl http://localhost:8000/health/readiness | jq

# Backward-compatible health (same as readiness)
curl http://localhost:8000/health | jq

# Runtime stats snapshot
curl http://localhost:8000/stats | jq
```

### Sessions

```bash
# Create session
SESSION_ID=$(curl -s -X POST http://localhost:8000/session/new | jq -r '.session_id')

# Query with session
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the case summary?", "session_id": "'$SESSION_ID'"}'

# History
curl http://localhost:8000/session/$SESSION_ID/history | jq

# Session info
curl http://localhost:8000/session/$SESSION_ID | jq

# List all sessions
curl http://localhost:8000/sessions | jq

# Clear a session
curl -X POST http://localhost:8000/session/$SESSION_ID/clear | jq

# Delete a session
curl -X DELETE http://localhost:8000/session/$SESSION_ID | jq
```

### Cache (Admin)

```bash
# Cache stats
curl http://localhost:8000/cache/stats | jq

# Clear cache
curl -X POST http://localhost:8000/cache/clear | jq

# Delete a cache entry by key
curl -X DELETE http://localhost:8000/cache/entry \
  -H "Content-Type: application/json" \
  -d '{"cache_key": "your_cache_key"}' | jq
```

### Switch Collection (Admin)

```bash
curl -X POST http://localhost:8000/admin/switch-collection \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "new_kb"}'
```

## Multi‑KB Server

### Create KB

```bash
curl -X POST http://localhost:8000/kb/create \
  -H "Content-Type: application/json" \
  -d '{"kb_id": "client_abc_legal_docs", "display_name": "ABC Legal Docs"}'
```

### List KBs

```bash
curl http://localhost:8000/kb/list
```

### Register Existing Collection

```bash
curl -X POST http://localhost:8000/kb/create \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": "client_legal_kb",
    "display_name": "Client Legal KB",
    "existing_collection": "CAFL_data"
  }'
```

### Add Document

SAS URL:

```bash
curl -X POST http://localhost:8000/kb/client_abc_legal_docs/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "doc-uuid-123",
    "file_name": "contract.pdf",
    "sas_url": "https://storage.blob.core.windows.net/..."
  }'
```

Local file on the server:

```bash
curl -X POST http://localhost:8000/kb/client_abc_legal_docs/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "local_path": "/data/uploads/contract.pdf"
  }'
```

Directory batch:

```bash
curl -X POST http://localhost:8000/kb/client_abc_legal_docs/documents/add-batch \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/data/uploads/legal_docs",
    "recursive": true
  }'
```

### Create Session

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/kb/client_abc_legal_docs/session/new | jq -r '.session_id')
```

### Query (sync)

```bash
curl -X POST http://localhost:8000/kb/client_abc_legal_docs/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the termination clauses?", "session_id": "'$SESSION_ID'"}'
```

### Query (async jobs)

```bash
# Submit async query job
JOB_ID=$(curl -s -X POST http://localhost:8000/kb/client_abc_legal_docs/query/jobs \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the NDA", "session_id": "'$SESSION_ID'", "timeout_seconds": 300}' | jq -r '.job_id')

# Poll job status
curl -s http://localhost:8000/kb/client_abc_legal_docs/query/jobs/$JOB_ID | jq

# Cancel job
curl -X POST http://localhost:8000/kb/client_abc_legal_docs/query/jobs/$JOB_ID/cancel | jq
```
