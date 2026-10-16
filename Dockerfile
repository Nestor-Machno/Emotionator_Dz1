FROM python:3.11-slim

# Устанавливаем системные библиотеки для работы OpenCV и curl для Healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Копируем исходный код и скрипт сборки
COPY . /build/

# Если артефакт еще не собран снаружи — запускаем скрипт сборки артефакта
RUN chmod +x build.sh && \
    if [ ! -f "artifacts/emotionator-artifact.tar.gz" ]; then ./build.sh; fi

# --- Этап упаковки артефакта в рабочее окружение контейнера ---
WORKDIR /app

# Берем сохраненный артефакт и распаковываем его в рабочую директорию сервера
RUN cp /build/artifacts/emotionator-artifact.tar.gz /app/ && \
    tar -xzf emotionator-artifact.tar.gz && \
    rm -rf /build

# Устанавливаем зависимости из распакованного артефакта
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт приложения
ENV PORT=8000
EXPOSE 8000

# Встроенная проверка работоспособности контейнера (Docker Healthcheck)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Запуск сервера
CMD ["python", "bacend.py"]
