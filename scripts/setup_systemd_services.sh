#!/bin/bash

# Script to setup systemd services for Qdrant, Redis, and the Multi-KB RAG server
# Run this script with sudo on your Azure VM

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: sudo $0"
            echo ""
            echo "Installs and enables the production Multi-KB stack:"
            echo "  synapse-redis, qdrant, multi-kb-server"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo "Usage: sudo $0"
            exit 1
            ;;
    esac
done

if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run this script with sudo${NC}"
    exit 1
fi

echo "Setting up systemd services for Synapse AI..."

ACTUAL_USER=${SUDO_USER:-$USER}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo -e "${YELLOW}Project directory: $PROJECT_DIR${NC}"
echo -e "${YELLOW}Running as user: $ACTUAL_USER${NC}"

chmod +x "$PROJECT_DIR/scripts/start_redis_local.sh" || true

REDIS_DATA_DIR="/var/lib/synapse-redis"
mkdir -p "$REDIS_DATA_DIR"
chown -R "$ACTUAL_USER:$ACTUAL_USER" "$REDIS_DATA_DIR"
chmod 755 "$REDIS_DATA_DIR"

echo "Creating service files with correct paths..."

sed "s|PROJECT_USER|$ACTUAL_USER|g; s|PROJECT_DIR|$PROJECT_DIR|g" "$PROJECT_DIR/systemd/qdrant.service" > /tmp/qdrant.service.tmp
sed "s|PROJECT_USER|$ACTUAL_USER|g; s|PROJECT_DIR|$PROJECT_DIR|g" "$PROJECT_DIR/systemd/synapse-redis.service" > /tmp/synapse-redis.service.tmp
sed "s|PROJECT_USER|$ACTUAL_USER|g; s|PROJECT_DIR|$PROJECT_DIR|g" "$PROJECT_DIR/systemd/multi-kb-server.service" > /tmp/multi-kb-server.service.tmp

echo "Installing service files to /etc/systemd/system/..."
cp /tmp/qdrant.service.tmp /etc/systemd/system/qdrant.service
cp /tmp/synapse-redis.service.tmp /etc/systemd/system/synapse-redis.service
cp /tmp/multi-kb-server.service.tmp /etc/systemd/system/multi-kb-server.service

rm /tmp/qdrant.service.tmp
rm /tmp/synapse-redis.service.tmp
rm /tmp/multi-kb-server.service.tmp

chmod 644 /etc/systemd/system/qdrant.service
chmod 644 /etc/systemd/system/synapse-redis.service
chmod 644 /etc/systemd/system/multi-kb-server.service

if systemctl list-unit-files conversational-rag-server.service >/dev/null 2>&1; then
    echo -e "${YELLOW}Disabling removed conversational-rag-server service.${NC}"
    systemctl disable conversational-rag-server.service >/dev/null 2>&1 || true
fi

echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "Enabling services to start on boot..."
systemctl enable synapse-redis.service
systemctl enable qdrant.service
systemctl enable multi-kb-server.service

echo -e "${GREEN}✓ Services installed and enabled successfully!${NC}"
echo ""
echo "To start the services, run:"
echo "  sudo systemctl start synapse-redis"
echo "  sudo systemctl start qdrant"
echo "  sudo systemctl start multi-kb-server"
echo ""
echo "Or use the management script:"
echo "  ./scripts/manage_services.sh start"
echo ""
echo "To check status:"
echo "  ./scripts/manage_services.sh status"
echo ""
echo "To view logs:"
echo "  ./scripts/manage_services.sh logs"
echo ""
