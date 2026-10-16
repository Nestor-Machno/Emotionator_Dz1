#!/usr/bin/env bash
set -e

echo "[1/3] Подготовка директории для артефактов"
mkdir -p artifacts

VERSION=${GITHUB_SHA:-"local-$(date +%Y%m%d-%H%M%S)"}
ARTIFACT_NAME="emotionator-artifact.tar.gz"
VERSIONED_ARTIFACT="emotionator-${VERSION}.tar.gz"

echo "[2/3] Проверка наличия ключевых компонентов и моделей"
for item in bacend.py frontend.html requirements.txt models icons static; do
    if [ ! -e "$item" ]; then
        echo "[ОШИБКА] Не найден обязательный файл или папка: $item"
        exit 1
    fi
done

echo "[3/3] Упаковка собранного артефакта"
tar -czf "artifacts/${ARTIFACT_NAME}" \
    bacend.py \
    frontend.html \
    requirements.txt \
    models/ \
    icons/ \
    static/

cp "artifacts/${ARTIFACT_NAME}" "artifacts/${VERSIONED_ARTIFACT}"

echo "[УСПЕШНО] Сборка завершена."
echo "Основной артефакт: artifacts/${ARTIFACT_NAME}"
echo "Версионный артефакт: artifacts/${VERSIONED_ARTIFACT}"
