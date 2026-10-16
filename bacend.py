import os
import sys
import time
import asyncio
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import psutil
import uvicorn
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotionSystem")

app = FastAPI(title="Emotionator Server", version="1.0.0")

# ДАННЫЕ ДЛЯ МОНИТОРИНГА 
START_TIME = time.time()
visit_timestamps = deque()       # Хранит время посещений главной страницы
frame_timestamps = deque()       # Хранит время обработанных кадров
total_visits = 0
total_frames_processed = 0

def clean_old_metrics():
    """Удаляет записи старше 1 часа (3600 секунд) для метрики 'в час'"""
    now = time.time()
    while visit_timestamps and now - visit_timestamps[0] > 3600:
        visit_timestamps.popleft()
    while frame_timestamps and now - frame_timestamps[0] > 3600:
        frame_timestamps.popleft()

# Глобальная переменная для REST API
last_analysis_result = {"success": False, "error": "No data yet", "faces": []}

# Конфигурация путей к моделям
YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
FER_PATH = os.path.join(BASE_DIR, "models", "emotion-ferplus-8.onnx")

models_loaded = False
detector = None
ort_session = None
INPUT_NAME = None

if os.path.exists(YUNET_PATH) and os.path.exists(FER_PATH):
    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (640, 480), score_threshold=0.6, nms_threshold=0.3)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    ort_session = ort.InferenceSession(FER_PATH, providers=['CPUExecutionProvider'], sess_options=opts)
    INPUT_NAME = ort_session.get_inputs()[0].name
    models_loaded = True
    logger.info("Модели ONNX успешно загружены!")
else:
    logger.error(f"Модели .onnx не найдены по путям: {YUNET_PATH}, {FER_PATH}")

executor = ThreadPoolExecutor(max_workers=2)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

emotions_list = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']
ALPHA_PROBS = 0.85
ALPHA_BOX = 0.85

biases = np.array([
    -0.8,  # neutral
    1.0,   # happy
    1.8,   # surprise
    -0.2,  # sad
    1.5,   # angry
    1.8,   # disgust
    3.5,   # fear
    2.96   # contempt
])

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denominator = float(boxAArea + boxBArea - interArea)
    return interArea / denominator if denominator > 0 else 0

def process_frame_sync(frame_bytes, client_state):
    global total_frames_processed
    start_time = time.time()
    if len(frame_bytes) < 1024:
        return {"success": False, "error": "Invalid frame payload"}

    try:
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"success": False, "error": "Frame decode failed"}

        # Обновляем размер входа детектора, если кадр отличается от 640x480
        h_img, w_img = frame.shape[:2]
        detector.setInputSize((w_img, h_img))

        _, faces = detector.detect(frame)
        results = []
        current_frame_faces = {}

        if faces is not None:
            for face in faces:
                raw_box = face[:4].astype(np.float32)
                x, y, w, h = raw_box.astype(np.int32)

                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                if x2 <= x1 + 10 or y2 <= y1 + 10:
                    continue

                matched_id = None
                best_iou = 0.3
                for face_id, state in client_state.items():
                    iou = get_iou(raw_box, state['box'])
                    if iou > best_iou:
                        best_iou = iou
                        matched_id = face_id

                if matched_id is None:
                    matched_id = os.urandom(4).hex()
                    smoothed_box = raw_box
                    smoothed_probs = None
                else:
                    smoothed_box = ALPHA_BOX * raw_box + (1 - ALPHA_BOX) * client_state[matched_id]['box']
                    smoothed_probs = client_state[matched_id]['probs']

                face_crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                face_equalized = clahe.apply(face_crop)
                face_resized = cv2.resize(face_equalized, (64, 64), interpolation=cv2.INTER_LINEAR)
                input_tensor = np.reshape(face_resized.astype(np.float32), (1, 1, 64, 64))

                raw_outputs = ort_session.run(None, {INPUT_NAME: input_tensor})[0][0]
                raw_probs = np.exp(raw_outputs - np.max(raw_outputs))
                raw_probs /= raw_probs.sum()

                dynamic_biases = biases.copy()
                dynamic_biases[7] = -100.0

                if raw_probs[6] > 0.03:
                    dynamic_biases[6] += 2.0
                if np.max(raw_probs[1:]) < 0.20:
                    dynamic_biases[0] = 0.0

                adjusted_logits = raw_outputs + dynamic_biases
                probs = np.exp(adjusted_logits - np.max(adjusted_logits))
                probs /= probs.sum()

                if smoothed_probs is not None:
                    probs = ALPHA_PROBS * probs + (1 - ALPHA_PROBS) * smoothed_probs

                current_frame_faces[matched_id] = {'box': smoothed_box, 'probs': probs}
                max_idx = np.argmax(probs)

                results.append({
                    "id": matched_id,
                    "emotion": emotions_list[max_idx],
                    "confidence": int(probs[max_idx] * 100),
                    "box": [int(smoothed_box[0]), int(smoothed_box[1]), int(smoothed_box[2]), int(smoothed_box[3])],
                    "all_emotions": {em: int(p * 100) for em, p in zip(emotions_list, probs) if em != 'contempt'}
                })

        client_state.clear()
        client_state.update(current_frame_faces)
        process_time_ms = int((time.time() - start_time) * 1000)

        # Записываем метрику активности
        frame_timestamps.append(time.time())
        total_frames_processed += 1

        return {"success": True, "faces": results, "process_time_ms": process_time_ms}

    except Exception as e:
        return {"success": False, "error": str(e)}



icons_dir = os.path.join(BASE_DIR, "icons")
static_dir = os.path.join(BASE_DIR, "static")

if os.path.exists(icons_dir):
    app.mount("/icons", StaticFiles(directory=icons_dir), name="icons")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_frontend():
    global total_visits
    visit_timestamps.append(time.time())
    total_visits += 1
    html_path = os.path.join(BASE_DIR, "frontend.html")
    return FileResponse(html_path)

@app.post("/api/v1/shutdown")
async def shutdown():
    return {"success": True, "message": "Running in cloud mode"}

@app.get("/api/v1/emotional")
async def get_emotional_data():
    return last_analysis_result


#МАРШРУТЫ МОНИТОРИНГА 

@app.get("/health")
async def health_check():
    """Метрика 1: Health Check (проверка работоспособности)"""
    status = "UP" if models_loaded else "DEGRADED"
    http_code = 200 if models_loaded else 503
    return JSONResponse(
        status_code=http_code,
        content={
            "status": status,
            "models_loaded": models_loaded,
            "uptime_seconds": int(time.time() - START_TIME)
        }
    )

@app.get("/api/v1/metrics")
async def get_metrics_json():
    clean_old_metrics()
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    process = psutil.Process(os.getpid())
    proc_mem_mb = round(process.memory_info().rss / (1024 * 1024), 2)

    return {
        "health_check": {
            "status": "HEALTHY" if models_loaded else "UNHEALTHY",
            "models_loaded": models_loaded,
            "uptime_seconds": int(time.time() - START_TIME)
        },
        "resources": {
            "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
            "app_ram_used_mb": proc_mem_mb,
            "system_ram_used_mb": round(vm.used / (1024 * 1024), 2),
            "system_ram_available_mb": round(vm.available / (1024 * 1024), 2),
            "system_ram_percent": vm.percent,
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
            "disk_percent": disk.percent
        },
        "usage_metrics": {
            "visits_last_hour": len(visit_timestamps),
            "total_visits": total_visits,
            "frames_analyzed_last_hour": len(frame_timestamps),
            "total_frames_analyzed": total_frames_processed
        }
    }

@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Мониторинг — Эмоцинатор</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }
            h1 { margin-top: 0; color: #38bdf8; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
            .card { background: #1e293b; border-radius: 14px; padding: 24px; border: 1px solid #334155; box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
            .card h2 { margin-top: 0; font-size: 18px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
            .badge { display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 16px; background: #10b981; color: #fff; }
            .stat { display: flex; justify-content: space-between; margin: 14px 0; font-size: 16px; border-bottom: 1px solid #334155; padding-bottom: 8px; }
            .val { font-weight: bold; color: #38bdf8; }
            a { color: #38bdf8; text-decoration: none; }
        </style>
    </head>
    <body>
        <h1>📊 Панель мониторинга: Эмоцинатор</h1>
        <p>Автообновление каждые 2 секунды | <a href="/">← Вернуться в приложение</a> | <a href="/api/v1/metrics" target="_blank">JSON API</a> | <a href="/health" target="_blank">Health Check</a></p>
        <div class="grid">
            <div class="card">
                <h2>1. Health Check (Работоспособность)</h2>
                <div class="stat"><span>Статус сервиса:</span> <span id="hc-status" class="badge">ЗАГРУЗКА...</span></div>
                <div class="stat"><span>ONNX нейросети:</span> <span class="val" id="hc-models">-</span></div>
                <div class="stat"><span>Время непрерывной работы (Uptime):</span> <span class="val" id="hc-uptime">-</span></div>
            </div>
            <div class="card">
                <h2>2. Потребляемые и доступные ресурсы</h2>
                <div class="stat"><span>Загрузка CPU:</span> <span class="val" id="res-cpu">-</span></div>
                <div class="stat"><span>RAM контейнера (приложения):</span> <span class="val" id="res-app-ram">-</span></div>
                <div class="stat"><span>RAM доступно (свободно):</span> <span class="val" id="res-avail-ram">-</span></div>
                <div class="stat"><span>Диск свободно:</span> <span class="val" id="res-disk">-</span></div>
            </div>
            <div class="card">
                <h2>3. Активность (Посещения в час)</h2>
                <div class="stat"><span>Посещений за последний час:</span> <span class="val" id="use-visits-hr">-</span></div>
                <div class="stat"><span>Всего посещений:</span> <span class="val" id="use-visits-tot">-</span></div>
                <div class="stat"><span>Распознано кадров за час:</span> <span class="val" id="use-frames-hr">-</span></div>
                <div class="stat"><span>Всего распознано кадров:</span> <span class="val" id="use-frames-tot">-</span></div>
            </div>
        </div>
        <script>
            async function updateMetrics() {
                try {
                    const res = await fetch('/api/v1/metrics');
                    const data = await res.json();
                    document.getElementById('hc-status').innerText = data.health_check.status;
                    document.getElementById('hc-models').innerText = data.health_check.models_loaded ? 'Активны (YuNet + FERPlus)' : 'Ошибка';
                    document.getElementById('hc-uptime').innerText = data.health_check.uptime_seconds + ' сек.';
                    document.getElementById('res-cpu').innerText = data.resources.cpu_usage_percent + ' %';
                    document.getElementById('res-app-ram').innerText = data.resources.app_ram_used_mb + ' МБ';
                    document.getElementById('res-avail-ram').innerText = data.resources.system_ram_available_mb + ' МБ (' + (100 - data.resources.system_ram_percent).toFixed(1) + '% своб.)';
                    document.getElementById('res-disk').innerText = data.resources.disk_free_gb + ' ГБ';
                    document.getElementById('use-visits-hr').innerText = data.usage_metrics.visits_last_hour;
                    document.getElementById('use-visits-tot').innerText = data.usage_metrics.total_visits;
                    document.getElementById('use-frames-hr').innerText = data.usage_metrics.frames_analyzed_last_hour;
                    document.getElementById('use-frames-tot').innerText = data.usage_metrics.total_frames_analyzed;
                } catch (e) {
                    document.getElementById('hc-status').innerText = 'OFFLINE';
                    document.getElementById('hc-status').style.background = '#ef4444';
                }
            }
            updateMetrics();
            setInterval(updateMetrics, 2000);
        </script>
    </body>
    </html>
    """


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global last_analysis_result
    await websocket.accept()
    frame_queue = asyncio.Queue(maxsize=1)
    client_state = {}

    async def receive_frames():
        try:
            while True:
                data = await websocket.receive_bytes()
                if frame_queue.full():
                    frame_queue.get_nowait()
                await frame_queue.put(data)
        except Exception:
            pass

    async def process_and_send():
        global last_analysis_result
        try:
            while True:
                frame_bytes = await frame_queue.get()
                result = await asyncio.get_running_loop().run_in_executor(
                    executor, process_frame_sync, frame_bytes, client_state
                )
                last_analysis_result = result
                await websocket.send_json(result)
        except Exception:
            pass

    producer = asyncio.create_task(receive_frames())
    consumer = asyncio.create_task(process_and_send())
    try:
        await asyncio.gather(producer, consumer)
    finally:
        producer.cancel()
        consumer.cancel()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
