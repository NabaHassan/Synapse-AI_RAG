#!/bin/bash
# Start Qdrant server locally with optimized settings for large-scale indexing

set -e

QDRANT_DIR="./bin/qdrant"
STORAGE_DIR="./data/qdrant_storage"

# ============================================================================
# CRITICAL: Increase file descriptor limits to prevent "Too many open files"
# ============================================================================
echo "Setting file descriptor limits..."

# Check current limits
CURRENT_SOFT=$(ulimit -Sn)
CURRENT_HARD=$(ulimit -Hn)
echo "  Current soft limit: $CURRENT_SOFT"
echo "  Current hard limit: $CURRENT_HARD"

# Set to maximum available (or 131072 if hard limit allows)
# This prevents "Too many open files" errors during heavy indexing
ulimit -n 131072 2>/dev/null || ulimit -n $(ulimit -Hn)

NEW_LIMIT=$(ulimit -n)
echo "  New limit: $NEW_LIMIT"
echo ""

if [ "$NEW_LIMIT" -lt 10000 ]; then
    echo "WARNING: File descriptor limit is low ($NEW_LIMIT)"
    echo "   For large-scale indexing, you may need to increase system limits."
    echo "   See: https://qdrant.tech/documentation/guides/common-errors/#too-many-open-files"
    echo ""
fi

# Check if qdrant binary exists
if [ ! -f "$QDRANT_DIR/qdrant" ]; then
    echo "Downloading Qdrant binary..."
    mkdir -p "$QDRANT_DIR"

    # Detect OS and architecture
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    ARCH=$(uname -m)

    if [ "$OS" = "darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            URL="https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-aarch64-apple-darwin.tar.gz"
        else
            URL="https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-apple-darwin.tar.gz"
        fi
    elif [ "$OS" = "linux" ]; then
        URL="https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz"
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi

    echo "Downloading from: $URL"
    curl -L "$URL" | tar xz -C "$QDRANT_DIR"
    chmod +x "$QDRANT_DIR/qdrant"
    echo "✓ Qdrant binary downloaded"
fi

# Create the storage directory
mkdir -p "$STORAGE_DIR"

# Create optimized config file for large-scale indexing
CONFIG_FILE="$QDRANT_DIR/config.yaml"
cat > "$CONFIG_FILE" << EOF
storage:
  storage_path: ../../$STORAGE_DIR
  # Optimize for large collections
  performance:
    max_search_threads: 0  # Auto-detect CPU cores
  # Reduce memory pressure during indexing
  optimizers:
    deleted_threshold: 0.2
    vacuum_min_vector_number: 1000
    default_segment_number: 2
    max_segment_size_kb: 200000  # 200MB segments
    memmap_threshold_kb: 50000   # Use memory mapping for large segments
    indexing_threshold_kb: 20000 # Start indexing at 20MB
    flush_interval_sec: 30       # Flush every 30 seconds
    max_optimization_threads: 2  # Limit concurrent optimizations

service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
  # Increase timeouts for large operations
  max_request_size_mb: 128
  max_workers: 0  # Auto-detect

# Logging configuration
log_level: INFO
EOF

echo "Starting Qdrant server with optimized settings..."
echo "  Storage: $STORAGE_DIR"
echo "  URL: http://localhost:6333"
echo "  File descriptor limit: $NEW_LIMIT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start Qdrant with config file
cd "$QDRANT_DIR"
./qdrant --config-path ./config.yaml
