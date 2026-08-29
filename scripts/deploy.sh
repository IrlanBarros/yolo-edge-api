#!/bin/bash
set -euo pipefail
DEPLOY_PATH="${DEPLOY_PATH:-~/yolo-edge-api}"
cd "$DEPLOY_PATH"
docker compose pull
docker compose up -d
sleep 10
if curl -sf http://localhost:8000/health > /dev/null; then
    echo "Deploy bem-sucedido."
else
    echo "Falha no health check. Revertendo..."
    docker compose down
    exit 1
fi
