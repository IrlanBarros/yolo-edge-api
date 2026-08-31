#!/bin/bash
set -euo pipefail
DEPLOY_PATH="/home/miguel/yolo-edge-api"
cd "$DEPLOY_PATH"

# Força o build local na Pi usando os arquivos atualizados
docker compose up --build -d

# Aguarda a API aquecer o motor (respeitando o start_period)
sleep 15

# Testa o health check usando Python puro (já que a imagem não tem curl)
if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null; then
    echo "Deploy bem-sucedido."
else
    echo "Falha no health check. Revertendo..."
    docker compose down
    exit 1
fi
