import base64
import io
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from PIL import Image
import numpy as np

# Importações dos seus módulos locais
from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    MetricsResponse,
    Detection
)
from app.model import load_model

app = FastAPI(
    title="YOLO Inference API",
    version="1.0.0",
    description="API REST para inferência com YOLOv8 no Raspberry Pi 5"
)

# Métricas globais da API
_metrics = {
    "total": 0,
    "success": 0,
    "total_ms": 0.0
}

def log_event(event: str, level: str = "INFO", **kwargs):
    """Emite um evento estruturado em JSON para stdout (conforme Aula 3)."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level":     level,
        "event":     event,
        **kwargs,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)

def _decode_image(b64_str: str) -> np.ndarray:
    """Decodifica uma string Base64 em um array NumPy RGB."""
    try:
        image_data = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao decodificar imagem base64: {str(e)}")

@app.get("/health")
def health_check():
    model_name = os.getenv("MODEL_NAME", "yolov8n.pt")
    model_loaded = False
    try:
        load_model(model_name)
        model_loaded = True
    except Exception:
        pass
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_name": model_name
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    _metrics["total"] += 1
    log_event("predict_start", model=request.model_name, confidence=request.confidence)

    try:
        if not request.image_base64:
            raise HTTPException(status_code=422, detail="Forneça image_base64.")

        img_rgb = _decode_image(request.image_base64)
        model = load_model(request.model_name)

        t0 = time.perf_counter()
        results = model(img_rgb, conf=request.confidence, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms

        # Processa as detecções para o formato do schema
        detections = []
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append(Detection(label=label, confidence=round(conf, 4), bbox=[round(x, 2) for x in xyxy]))

        img_height, img_width = img_rgb.shape[:2]

        log_event("predict_complete",
                  model=request.model_name,
                  detections=len(detections),
                  inference_ms=round(elapsed_ms, 2),
                  image_size=f"{img_width}x{img_height}")

        return PredictResponse(
            detections=detections,
            inference_ms=round(elapsed_ms, 2),
            model_used=request.model_name,
            image_width=img_width,
            image_height=img_height
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        log_event("predict_error", level="ERROR", reason=str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_event("predict_error", level="ERROR", reason=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    t_total = time.perf_counter()
    results = []
    for img_b64 in request.images_base64:
        img = _decode_image(img_b64)
        # Execução simplificada para batch reutilizando o fluxo padrão
        model = load_model(request.model_name)
        t0 = time.perf_counter()
        res = model(img, conf=request.confidence, verbose=False)[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        detections = []
        if res.boxes is not None:
            for box in res.boxes:
                cls_id = int(box.cls[0])
                detections.append(Detection(
                    label=res.names[cls_id],
                    confidence=round(float(box.conf[0]), 4),
                    bbox=[round(x, 2) for x in box.xyxy[0].tolist()]
                ))
        h, w = img.shape[:2]
        results.append(PredictResponse(
            detections=detections,
            inference_ms=round(elapsed_ms, 2),
            model_used=request.model_name,
            image_width=w,
            image_height=h
        ))

    total_ms = (time.perf_counter() - t_total) * 1000
    return BatchPredictResponse(results=results, total_inference_ms=round(total_ms, 2))

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    avg = (_metrics["total_ms"] / _metrics["success"] if _metrics["success"] > 0 else 0.0)
    return MetricsResponse(
        total_requests=_metrics["total"],
        successful_requests=_metrics["success"],
        avg_inference_ms=round(avg, 2),
    )
