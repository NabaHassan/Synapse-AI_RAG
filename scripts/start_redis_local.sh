#!/bin/bash
# Start Redis for Synapse AI.
#
# Strategy:
# 1) Use local redis-server binary if available.
# 2) Fallback to Docker-managed redis container if redis-server is unavailable.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_IMAGE="${REDIS_IMAGE:-redis:7-alpine}"
REDIS_CONTAINER_NAME="${REDIS_CONTAINER_NAME:-synapse-redis}"
REDIS_DATA_DIR="${REDIS_DATA_DIR:-$PROJECT_DIR/data/redis}"

mkdir -p "$REDIS_DATA_DIR"

if [ ! -d "$REDIS_DATA_DIR" ]; then
    echo "ERROR: Redis data directory does not exist: $REDIS_DATA_DIR"
    exit 1
fi

if [ ! -w "$REDIS_DATA_DIR" ]; then
    echo "ERROR: Redis data directory is not writable: $REDIS_DATA_DIR"
    ls -ld "$REDIS_DATA_DIR" || true
    exit 1
fi

# Keep Redis rooted in a stable writable directory rather than the repo cwd.
cd "$REDIS_DATA_DIR"

if command -v redis-server >/dev/null 2>&1; then
    echo "Starting redis-server binary on port ${REDIS_PORT}..."
    exec redis-server \
        --bind 0.0.0.0 \
        --port "${REDIS_PORT}" \
        --dir "${REDIS_DATA_DIR}" \
        --appendonly yes \
        --save 60 1000
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Neither redis-server nor docker is available on this host."
    exit 1
fi

echo "redis-server not found. Falling back to Docker container ${REDIS_CONTAINER_NAME} (${REDIS_IMAGE})..."

# Pull image if missing.
if ! docker image inspect "${REDIS_IMAGE}" >/dev/null 2>&1; then
    docker pull "${REDIS_IMAGE}"
fi

# Create container once if it doesn't exist.
if ! docker ps -a --format '{{.Names}}' | grep -Fxq "${REDIS_CONTAINER_NAME}"; then
    docker create \
        --name "${REDIS_CONTAINER_NAME}" \
        --publish "${REDIS_PORT}:6379" \
        --volume "${REDIS_DATA_DIR}:/data" \
        "${REDIS_IMAGE}" \
        redis-server --appendonly yes
fi

# If already running (e.g., manual start), follow logs.
if docker ps --format '{{.Names}}' | grep -Fxq "${REDIS_CONTAINER_NAME}"; then
    exec docker logs -f "${REDIS_CONTAINER_NAME}"
fi

# Attach to container process so systemd can supervise it.
exec docker start -a "${REDIS_CONTAINER_NAME}"
