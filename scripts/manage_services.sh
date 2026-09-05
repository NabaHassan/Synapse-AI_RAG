#!/bin/bash

# Management script for Synapse AI production services
# Usage: ./manage_services.sh [start|stop|restart|status|logs|enable|disable] [multi|redis|qdrant]

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

COMMAND=${1:-status}
SERVICE=${2:-multi}

resolve_service() {
    if [ "$1" = "multi" ]; then
        echo "multi-kb-server"
    elif [ "$1" = "redis" ]; then
        echo "synapse-redis"
    elif [ "$1" = "qdrant" ]; then
        echo "qdrant"
    elif [ "$1" = "tunnel" ]; then
        echo "synapse-tunnel"
    else
        echo "unknown"
    fi
}

TARGET_SERVICE=$(resolve_service "$SERVICE")

if [ "$TARGET_SERVICE" = "unknown" ]; then
    echo -e "${RED}Invalid service target: $SERVICE${NC}"
    echo "Usage: $0 [start|stop|restart|status|logs|enable|disable] [multi|redis|qdrant]"
    exit 1
fi

manage_dependency_service() {
    local action=$1
    local service_name=$2
    sudo systemctl "$action" "$service_name"
    case "$action" in
        start) echo -e "${GREEN}✓ ${service_name} started${NC}" ;;
        stop) echo -e "${GREEN}✓ ${service_name} stopped${NC}" ;;
        restart) echo -e "${GREEN}✓ ${service_name} restarted${NC}" ;;
        enable) echo -e "${GREEN}✓ ${service_name} enabled${NC}" ;;
        disable) echo -e "${GREEN}✓ ${service_name} disabled${NC}" ;;
        *) echo -e "${GREEN}✓ ${service_name}: ${action}${NC}" ;;
    esac
}

case "$COMMAND" in
    start)
        echo -e "${YELLOW}Starting services...${NC}"
        if [ "$TARGET_SERVICE" = "synapse-redis" ] || [ "$TARGET_SERVICE" = "qdrant" ] || [ "$TARGET_SERVICE" = "synapse-tunnel" ]; then
            manage_dependency_service start "$TARGET_SERVICE"
            exit 0
        fi

        sudo systemctl start synapse-redis
        echo -e "${GREEN}✓ Redis started${NC}"
        sleep 2
        sudo systemctl start qdrant
        echo -e "${GREEN}✓ Qdrant started${NC}"
        sleep 3
        sudo systemctl start multi-kb-server
        echo -e "${GREEN}✓ Multi-KB Server started${NC}"
        sleep 2
        sudo systemctl start synapse-tunnel
        echo -e "${GREEN}✓ Localtunnel started${NC}"
        ;;

    stop)
        echo -e "${YELLOW}Stopping services...${NC}"
        if [ "$TARGET_SERVICE" = "synapse-redis" ] || [ "$TARGET_SERVICE" = "qdrant" ] || [ "$TARGET_SERVICE" = "synapse-tunnel" ]; then
            manage_dependency_service stop "$TARGET_SERVICE"
            exit 0
        fi

        sudo systemctl stop synapse-tunnel
        echo -e "${GREEN}✓ Localtunnel stopped${NC}"
        sudo systemctl stop multi-kb-server
        echo -e "${GREEN}✓ Multi-KB Server stopped${NC}"
        sudo systemctl stop qdrant
        echo -e "${GREEN}✓ Qdrant stopped${NC}"
        sudo systemctl stop synapse-redis
        echo -e "${GREEN}✓ Redis stopped${NC}"
        ;;

    restart)
        echo -e "${YELLOW}Restarting services...${NC}"
        if [ "$TARGET_SERVICE" = "synapse-redis" ] || [ "$TARGET_SERVICE" = "qdrant" ] || [ "$TARGET_SERVICE" = "synapse-tunnel" ]; then
            manage_dependency_service restart "$TARGET_SERVICE"
            exit 0
        fi

        sudo systemctl restart synapse-redis
        echo -e "${GREEN}✓ Redis restarted${NC}"
        sleep 2
        sudo systemctl restart qdrant
        echo -e "${GREEN}✓ Qdrant restarted${NC}"
        sleep 3
        sudo systemctl restart multi-kb-server
        echo -e "${GREEN}✓ Multi-KB Server restarted${NC}"
        sleep 2
        sudo systemctl restart synapse-tunnel
        echo -e "${GREEN}✓ Localtunnel restarted${NC}"
        ;;

    status)
        if [ "$TARGET_SERVICE" = "synapse-redis" ] || [ "$TARGET_SERVICE" = "qdrant" ] || [ "$TARGET_SERVICE" = "synapse-tunnel" ]; then
            echo -e "${YELLOW}=== ${TARGET_SERVICE} Status ===${NC}"
            sudo systemctl status "$TARGET_SERVICE" --no-pager || true
            exit 0
        fi

        echo -e "${YELLOW}=== Redis Status ===${NC}"
        sudo systemctl status synapse-redis --no-pager || true
        echo ""
        echo -e "${YELLOW}=== Qdrant Status ===${NC}"
        sudo systemctl status qdrant --no-pager || true
        echo ""
        echo -e "${YELLOW}=== Multi-KB Server Status ===${NC}"
        sudo systemctl status multi-kb-server --no-pager || true
        echo ""
        echo -e "${YELLOW}=== Localtunnel Status ===${NC}"
        sudo systemctl status synapse-tunnel --no-pager || true
        ;;

    logs)
        SERVICE=${2:-multi}
        if [ "$SERVICE" = "redis" ]; then
            echo -e "${YELLOW}Showing Redis logs (Ctrl+C to exit)...${NC}"
            sudo journalctl -u synapse-redis -f
        elif [ "$SERVICE" = "qdrant" ]; then
            echo -e "${YELLOW}Showing Qdrant logs (Ctrl+C to exit)...${NC}"
            sudo journalctl -u qdrant -f
        elif [ "$SERVICE" = "multi" ]; then
            echo -e "${YELLOW}Showing Multi-KB Server logs (Ctrl+C to exit)...${NC}"
            sudo journalctl -u multi-kb-server -f
        elif [ "$SERVICE" = "tunnel" ]; then
            echo -e "${YELLOW}Showing Localtunnel logs (Ctrl+C to exit)...${NC}"
            sudo journalctl -u synapse-tunnel -f
        else
            echo -e "${YELLOW}Showing production stack logs (Ctrl+C to exit)...${NC}"
            sudo journalctl -u synapse-redis -u qdrant -u multi-kb-server -u synapse-tunnel -f
        fi
        ;;

    enable)
        echo -e "${YELLOW}Enabling services to start on boot...${NC}"
        if [ "$TARGET_SERVICE" = "synapse-redis" ] || [ "$TARGET_SERVICE" = "qdrant" ] || [ "$TARGET_SERVICE" = "synapse-tunnel" ]; then
            manage_dependency_service enable "$TARGET_SERVICE"
        else
            sudo systemctl enable synapse-redis
            sudo systemctl enable qdrant
            sudo systemctl enable multi-kb-server
            sudo systemctl enable synapse-tunnel
            echo -e "${GREEN}✓ Production stack enabled${NC}"
        fi
        ;;

    disable)
        echo -e "${YELLOW}Disabling services from starting on boot...${NC}"
        if [ "$TARGET_SERVICE" = "synapse-redis" ] || [ "$TARGET_SERVICE" = "qdrant" ] || [ "$TARGET_SERVICE" = "synapse-tunnel" ]; then
            manage_dependency_service disable "$TARGET_SERVICE"
        else
            sudo systemctl disable multi-kb-server
            sudo systemctl disable synapse-tunnel
            echo -e "${GREEN}✓ Multi-KB Server and Localtunnel disabled${NC}"
        fi
        ;;

    *)
        echo -e "${RED}Invalid command: $COMMAND${NC}"
        echo "Usage: $0 [start|stop|restart|status|logs|enable|disable] [multi|redis|qdrant]"
        echo ""
        echo "Commands:"
        echo "  start    - Start Redis, Qdrant, and multi-kb-server"
        echo "  stop     - Stop multi-kb-server, Qdrant, and Redis"
        echo "  restart  - Restart Redis, Qdrant, and multi-kb-server"
        echo "  status   - Show status of Redis, Qdrant, and multi-kb-server"
        echo "  logs     - Show live logs (optional: 'redis', 'qdrant', or 'multi')"
        echo "  enable   - Enable services to start on boot"
        echo "  disable  - Disable multi-kb-server from starting on boot"
        exit 1
        ;;
esac
