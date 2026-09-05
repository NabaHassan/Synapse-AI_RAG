# Synapse AI — Multi-KB RAG

Production RAG API for multiple isolated knowledge bases: hybrid retrieval (dense + BM25), citation-grounded answers, online document indexing, and conversational sessions. Built so one shared LLM/embedding stack can serve many tenants without mixing their data.

## Demo

<video src="docs/Ownify.mp4" controls width="100%"></video>

[Watch the demo](docs/Ownify.mp4)

## Why

Single-KB RAG breaks when you need many clients, custom prompts per KB, and safe concurrent query/indexing. This service gives you KB-scoped APIs, Redis-backed session safety, Qdrant collections per KB, and async query/provisioning jobs.

## Optional MCP tools

Queries can use the knowledge base alone, or you can **connect MCP tools** so the assistant also reads live workspace data. Each connector is optional — enable the ones you need in `.env`, then OAuth-connect per user.

| Connector | What it unlocks | Enable with |
|-----------|-----------------|-------------|
| **Google Workspace** | Gmail, Drive, Calendar | `GOOGLE_MCP_ENABLED=true` + OAuth client vars |
| **Microsoft 365** | Outlook, OneDrive / SharePoint | `MICROSOFT_MCP_ENABLED=true` + OAuth client vars |
| **Slack** | Channels, DMs, message search | `SLACK_MCP_ENABLED=true` + Slack app vars |
| **Notion** | Pages, databases, tasks | `NOTION_MCP_ENABLED=true` + Notion OAuth vars |

Connect a user via the `/mcp/.../auth` routes (Google, Microsoft, Slack, Notion). After that, queries and slash commands can pull from those tools in addition to the KB. See `.env.example` for the full variable list.

## Requirements

- Python 3.10+
- [Redis](https://redis.io/) (`localhost:6379`)
- [Qdrant](https://qdrant.tech/) (`localhost:6333`)
- CUDA GPU for local vLLM (default answer model)

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in any connector secrets you need (MCP tools above are optional).

Core env vars (defaults work for local Redis/Qdrant):

```bash
QDRANT_URL=http://localhost:6333
DATA_DIR=./data
REDIS_ENABLED=true
LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

## Run

```bash
export PYTHONPATH=.
python src/api/multi_kb_server.py --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Quick API flow

```bash
# Create a KB
curl -X POST http://localhost:8000/kb/create \
  -H "Content-Type: application/json" \
  -d '{"kb_id": "demo_kb", "display_name": "Demo KB"}'

# Session + query
SESSION=$(curl -s -X POST http://localhost:8000/kb/demo_kb/session/new | jq -r '.session_id')

curl -X POST http://localhost:8000/kb/demo_kb/query \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is in this knowledge base?\", \"session_id\": \"$SESSION\"}"
```

Add documents via `POST /kb/{kb_id}/documents/add` (online indexing). Python client helpers live in `client_backend_code_multi_kb_async.py`.

## Layout

| Path | Role |
|------|------|
| `src/api/` | FastAPI multi-KB + Ownify tenant APIs |
| `src/pipeline/` | Conversational + multi-KB RAG orchestration |
| `src/retrieval/` | Dense, BM25, fusion, rerank, web |
| `src/indexing/` | Chunking, embeddings, online ingest |
| `src/mcp/` | Optional Google / Microsoft / Slack / Notion MCP clients |
| `src/config/` | Profiles, prompts, runtime policy |
| `scripts/` | Ops helpers (BM25, evals, systemd) |
| `docs/` | Demo video and assets |
| `setup_and_sys_docs/` | Longer ops / API guides |

Production systemd setup: [SYSTEMD_README](./setup_and_sys_docs/SYSTEMD_README.md) · API details: [curl_and_apis](./setup_and_sys_docs/curl_and_apis.md)