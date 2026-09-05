# Synapse AI. (Core AI Infrastructure)

### How to Get Started

> More reference documents
> are: [System and Service Commands](./setup_and_sys_docs/SYSTEMD_README.md) | [CURL and Client API Commands](./setup_and_sys_docs/curl_and_apis.md)

## Quick Start

1) Install services:

```bash
sudo ./scripts/setup_systemd_services.sh
```

2) Start Redis, Qdrant, and the API services:

```bash
sudo systemctl start synapse-redis
sudo systemctl start qdrant
sudo systemctl start multi-kb-server
```

3) Check health:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/readiness | jq
```

## Indexing

Indexing is handled online through the production Multi-KB/Ownify APIs. The old offline KB builder was removed from this repo; offline-only indexing now lives outside this service codebase.

```bash
curl -X POST http://localhost:8000/documents/add \
  -H "Content-Type: application/json" \
  -d '{"file_id": "doc-uuid-123", "file_name": "contract.pdf", "sas_url": "https://storage.blob.core.windows.net/..."}'
```

## Multi‑KB Orchestrator

The multi‑KB server exposes KB‑scoped endpoints so you can manage multiple KBs and sessions in parallel while reusing a
single LLM instance.

Start the multi‑KB server (if not already running):

```bash
sudo systemctl start multi-kb-server
./scripts/manage_services.sh logs multi
```

Example flow:

```bash
# Create a KB (kb_id optional)
curl -X POST http://localhost:8000/kb/create \
  -H "Content-Type: application/json" \
  -d '{"kb_id": "client_abc_legal_docs", "display_name": "ABC Legal Docs"}'

# Create a session
SESSION=$(curl -s -X POST http://localhost:8000/kb/client_abc_legal_docs/session/new | jq -r '.session_id')

# Query
curl -X POST http://localhost:8000/kb/client_abc_legal_docs/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What are the termination clauses?\", \"session_id\": \"$SESSION\"}"
```

### Adding A New KB End-To-End

This is the production flow for onboarding a new KB into the multi-KB system.

1. Pick the final `kb_id`.

- Use the exact `kb_id` you want clients to use later.
- The config system now resolves profiles by exact filename:
  - `src/config/profiles/<kb_id>.yaml`
  - `src/config/prompts/<kb_id>_system_prompt.j2`

2. Add KB-specific config only if this KB needs custom behavior.

- If the KB should have custom prompt/tokens/canned responses/routing rules, create:
  - `src/config/profiles/<kb_id>.yaml`
  - `src/config/prompts/<kb_id>_system_prompt.j2`
- If those files do not exist, the system automatically falls back to:
  - `src/config/profiles/default.yaml`
  - the default general prompt

3. Register the KB.

- If the vector DB already exists in Qdrant, register it against the existing collection:

```bash
curl -X POST http://localhost:8000/kb/create \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": "client_legal_kb",
    "display_name": "Client Legal KB",
    "existing_collection": "knowledge_base_name"
  }'
```

- If you want the infra to create a new empty collection for this KB, omit `existing_collection`.

4. Let the system create the KB snapshot automatically.

- On `/kb/create`, the system automatically:
  - resolves the profile using `src/config/profiles/<kb_id>.yaml` if present
  - resolves the prompt using the profile or `src/config/prompts/<kb_id>_system_prompt.j2`
  - writes the effective snapshot to `data/kb_configs/<kb_id>.yaml`
  - stores that snapshot path in the KB registry

- Normally, you do **not** need to run:
  - `scripts/backfill_kb_configs.py`
  - `scripts/set_kb_config_profile.py`

5. Create a session and use the KB.

- Recommended production path is the async job API:

```bash
# Create session
SESSION=$(curl -s -X POST http://localhost:8000/kb/client_legal_kb/session/new | jq -r '.session_id')

# Submit async query job
JOB=$(curl -s -X POST http://localhost:8000/kb/client_legal_kb/query/jobs \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What does this KB contain?\", \"session_id\": \"$SESSION\"}" | jq -r '.job_id')

# Poll job status
curl -s http://localhost:8000/kb/client_legal_kb/query/jobs/$JOB | jq
```

6. Only run the helper scripts in these cases.

- Run `python3 scripts/backfill_kb_configs.py --force` when:
  - you changed profile or prompt files for already-registered KBs
  - you changed config structure and need to regenerate stored snapshots
- Run `python3 scripts/set_kb_config_profile.py ...` only when:
  - you want to force a KB to use a different profile than its default exact-`kb_id` profile

### How KB Config Resolution Works

At runtime, the multi-KB pipeline loads the stored snapshot from `data/kb_configs/<kb_id>.yaml`.

- If `src/config/profiles/<kb_id>.yaml` exists at KB creation/backfill time, that profile is used.
- If not, the system falls back to `default`.
- If the profile does not explicitly declare a prompt, the system looks for:
  - `src/config/prompts/<kb_id>_system_prompt.j2`
- If that prompt file does not exist, the system uses the default general prompt.

This means adding a new KB is now convention-based and dynamic, not hard-coded.

### Models

- **Embedding**: BAAI/bge-large-en-v1.5
- **Context Verifier**: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Query Enhancer**: Qwen/Qwen2.5-1.5B-Instruct
- **Query Classifier**: cross-encoder/nli-deberta-v3-base
