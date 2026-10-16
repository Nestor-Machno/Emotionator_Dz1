#!/usr/bin/env bash
set -e

IMAGE_NAME="emotionator-app:latest"
CONTAINER_NAME="emotionator-server"
PORT=${PORT:-8000}

echo "[1/3] Проверка наличия собранного артефакта"
if [ ! -f "artifacts/emotionator-artifact.tar.gz" ]; then
    echo "Артефакт не найден. Запуск build.sh..."
    chmod +x build.sh
    ./build.sh
fi

echo "[2/3] Упаковка артефакта в Docker-образ"
docker build -t "$IMAGE_NAME" .

echo "[3/3] Деплой контейнера на сервер"
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Останавливаем и удаляем старую версию контейнера..."
    docker stop "$CONTAINER_NAME" || true
    docker rm "$CONTAINER_NAME" || true
fi

docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "${PORT}:8000" \
    "$IMAGE_NAME"

echo "[УСПЕШНО] Приложение развернуто."
echo "Приложение: http://localhost:${PORT}"
echo "Мониторинг: http://localhost:${PORT}/monitoring"
echo "Health check: http://localhost:${PORT}/health"

if [ -n "$RENDER_DEPLOY_HOOK_URL" ]; then
    echo "Отправка сигнала деплоя на облачный сервер Render..."
    curl -s -X POST "$RENDER_DEPLOY_HOOK_URL"
    echo "Облачный деплой запущен."
fi
