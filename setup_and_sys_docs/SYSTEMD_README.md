# Systemd Service Setup for Synapse AI

This directory contains systemd service files for the production Multi-KB stack:

- `synapse-redis.service`
- `qdrant.service`
- `multi-kb-server.service`

The legacy single-KB `conversational-rag-server` service has been removed from this repo. Production tenants use `multi-kb-server` and Ownify APIs.

## Structure

```text
systemd/
├── qdrant.service                      # Qdrant vector database service
├── synapse-redis.service               # Redis service
└── multi-kb-server.service             # Production Multi-KB RAG API service

scripts/
├── start_qdrant_local.sh               # Start Qdrant on port 6333
├── start_redis_local.sh                # Start Redis on port 6379
├── setup_systemd_services.sh           # Install production services to systemd
└── manage_services.sh                  # Manage services
```

## Install Services

```bash
sudo ./scripts/setup_systemd_services.sh
```

This installs/enables:

- `synapse-redis.service`
- `qdrant.service`
- `multi-kb-server.service`

If an older host still has `conversational-rag-server.service` installed, the setup script disables its boot auto-start.

## Start Services

Production/default:

```bash
./scripts/manage_services.sh start
./scripts/manage_services.sh status
```

The default service target is `multi`, so those commands operate on Redis, Qdrant, and `multi-kb-server`.

Explicit targets:

```bash
./scripts/manage_services.sh start multi
./scripts/manage_services.sh status multi
./scripts/manage_services.sh logs multi

./scripts/manage_services.sh status redis
./scripts/manage_services.sh status qdrant
```

## Management Script Commands

```bash
./scripts/manage_services.sh start
./scripts/manage_services.sh stop
./scripts/manage_services.sh restart
./scripts/manage_services.sh status
./scripts/manage_services.sh logs
./scripts/manage_services.sh enable
./scripts/manage_services.sh disable
```

Supported targets:

- `multi`: production Multi-KB API service plus Redis and Qdrant for stack-level actions.
- `redis`: Redis only.
- `qdrant`: Qdrant only.

## Direct `systemctl` Commands

```bash
sudo systemctl start synapse-redis
sudo systemctl start qdrant
sudo systemctl start multi-kb-server

sudo systemctl status synapse-redis
sudo systemctl status qdrant
sudo systemctl status multi-kb-server

sudo journalctl -u multi-kb-server -f
```

## Service Details

### Redis Service (`synapse-redis.service`)

Local Redis instance for distributed locks, session store, rate limiting, and query cache.

Health check:

```bash
redis-cli -h 127.0.0.1 -p 6379 ping
```

### Qdrant Service (`qdrant.service`)

Vector database for document embeddings. Runtime code connects through `QDRANT_URL`; embedded local Qdrant client storage is no longer supported.

Health check:

```bash
curl -sS http://127.0.0.1:6333/collections
```

### Multi-KB Service (`multi-kb-server.service`)

Production API service with:

- KB management
- Ownify tenant APIs
- local file and directory ingestion
- async query jobs
- Redis-backed safety controls
- shared LLM, embedding, reranker, and context-verifier resources

Health checks:

```bash
curl -sS http://127.0.0.1:8000/health/liveness
curl -sS http://127.0.0.1:8000/health/readiness
```
