# KB Operations Guide

This document covers the operational steps for updating, replacing, and maintaining KBs in the multi-KB system.

## Overview

Use this guide when you need to:

- change a KB prompt or profile
- regenerate KB config snapshots
- replace an existing KB with a new Qdrant collection
- register a new KB
- verify that a KB is working end to end

## Important Concepts

Each KB uses:

- a registry entry in `data/kb_registry.json`
- a generated runtime snapshot in `data/kb_configs/<kb_id>.yaml`
- optional KB-specific config files:
  - `src/config/profiles/<kb_id>.yaml`
  - `src/config/prompts/<kb_id>_system_prompt.j2`

The runtime uses the stored snapshot, so if you change prompt/profile files for an already-registered KB, you must regenerate the snapshot.

## 1. When You Change A Prompt Or Profile

Examples:

- you edited `src/config/prompts/<kb_id>_system_prompt.j2`
- you changed `max_tokens`, `temperature`, canned responses, routing flags, or grounding rules in `src/config/profiles/<kb_id>.yaml`

After making those changes, run:

```bash
python3 scripts/backfill_kb_configs.py --force
sudo systemctl restart multi-kb-server
```

If you only want to regenerate one KB:

```bash
python3 scripts/backfill_kb_configs.py --kb-id client_epstein_kb_09_03_2025 --force
sudo systemctl restart multi-kb-server
```

## 2. When You Want To Force A KB To Use A Different Profile

Normally the system uses:

- `src/config/profiles/<kb_id>.yaml` if it exists
- otherwise `default`

If you explicitly want to force a KB to use another profile template:

```bash
python3 scripts/set_kb_config_profile.py \
  --kb-id client_epstein_kb_09_03_2025 \
  --profile-template-id default

sudo systemctl restart multi-kb-server
```

Use this only when you intentionally want to override the normal exact-`kb_id` profile convention.

## 3. Register A New KB

### Register Against An Existing Qdrant Collection

```bash
curl -X POST http://localhost:8000/kb/create \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": "client_legal_kb",
    "display_name": "Client Legal KB",
    "existing_collection": "knowledge_base_name"
  }'
```

Normally, no extra script is needed after this.

The system automatically:

- creates the KB registry entry
- resolves the profile and prompt
- writes `data/kb_configs/<kb_id>.yaml`

### Create A New Empty KB Collection

```bash
curl -X POST http://localhost:8000/kb/create \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": "client_new_kb",
    "display_name": "Client New KB"
  }'
```

## 4. Replace An Existing KB With A New Collection

This is the usual flow when you rebuild or replace a KB collection outside the live tenant path and want to swap the KB in production while keeping the same `kb_id`.

### Step 1: Delete The Old KB Registration

```bash
curl -X DELETE http://localhost:8000/kb/client_epstein_kb_09_03_2025
```

This removes:

- the KB registry entry
- the stored snapshot
- KB-local cache/session/tracker files

It also attempts to delete the currently-registered Qdrant collection, so only do this when you are done with the old registered collection.

### Step 2: Register The New Collection Under The Same `kb_id`

```bash
curl -X POST http://localhost:8000/kb/create \
  -H "Content-Type: application/json" \
  -d '{
    "kb_id": "client_epstein_kb_09_03_2025",
    "display_name": "Client Epstein KB",
    "existing_collection": "epstein_v2_collection"
  }'
```

Normally, no backfill script is needed after this because `/kb/create` generates the new snapshot automatically.

### Step 3: Create New Sessions

Old sessions tied to the deleted KB should be treated as invalid. Create fresh sessions:

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/kb/client_epstein_kb_09_03_2025/session/new | jq -r '.session_id')
```

## 5. Add KB-Specific Prompt And Profile Files

If a KB needs its own behavior, create:

```text
src/config/profiles/<kb_id>.yaml
src/config/prompts/<kb_id>_system_prompt.j2
```

Example:

```text
src/config/profiles/client_epstein_kb_09_03_2025.yaml
src/config/prompts/client_epstein_kb_09_03_2025_system_prompt.j2
```

Then either:

- register the KB if it is new, or
- run backfill if the KB is already registered

## 6. Add Or Delete A Document In A KB

### Add Document

From a SAS URL:

```bash
curl -X POST http://localhost:8000/kb/client_epstein_kb_09_03_2025/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "doc-uuid-123",
    "file_name": "contract.pdf",
    "sas_url": "https://storage.blob.core.windows.net/..."
  }'
```

From a local file on the server:

```bash
curl -X POST http://localhost:8000/kb/client_epstein_kb_09_03_2025/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "doc-uuid-123",
    "file_name": "contract.pdf",
    "local_path": "/data/uploads/contract.pdf"
  }'
```

`file_id` and `file_name` can be omitted for local files; the server will use the basename and generate a stable source-based ID.

### Add A Directory Batch

```bash
curl -X POST http://localhost:8000/kb/client_epstein_kb_09_03_2025/documents/add-batch \
  -H "Content-Type: application/json" \
  -d '{
    "directory_path": "/data/uploads/legal_docs",
    "recursive": true
  }'
```

The batch endpoint indexes supported files and reports unsupported files in `skipped`. You can also send a `documents` array with mixed `sas_url` and `local_path` items.

### Delete Document

```bash
curl -X DELETE "http://localhost:8000/kb/client_epstein_kb_09_03_2025/documents/doc-uuid-123?file_name=contract.pdf"
```

## 7. Verify A KB After Changes

### Check KB registration

```bash
curl http://localhost:8000/kb/client_epstein_kb_09_03_2025 | jq
```

### List KBs

```bash
curl http://localhost:8000/kb/list | jq
```

### List documents

```bash
curl http://localhost:8000/kb/client_epstein_kb_09_03_2025/documents | jq
```

### Run a quick query

```bash
SESSION_ID=$(curl -s -X POST http://localhost:8000/kb/client_epstein_kb_09_03_2025/session/new | jq -r '.session_id')

curl -X POST http://localhost:8000/kb/client_epstein_kb_09_03_2025/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this KB about?", "session_id": "'$SESSION_ID'"}'
```

## 8. Practical Rules

### Run `backfill_kb_configs.py` when:

- you changed a prompt file
- you changed a profile YAML
- you changed config schema/behavior and want fresh snapshots

### Do not run `backfill_kb_configs.py` when:

- you just created a brand-new KB using `/kb/create`

### Run `set_kb_config_profile.py` when:

- you intentionally want a KB to use a different profile than its normal exact-`kb_id` profile

### Replace a KB with the same public identity:

- delete the old KB
- recreate it with the same `kb_id`
- point it to the new Qdrant collection

## 9. Recommended Restart Commands

After config changes:

```bash
sudo systemctl restart multi-kb-server
```

If you changed system services:

```bash
sudo ./scripts/setup_systemd_services.sh
sudo systemctl restart synapse-redis
sudo systemctl restart qdrant
sudo systemctl restart multi-kb-server
```
