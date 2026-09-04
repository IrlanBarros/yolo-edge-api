import base64
import io
import json
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image

from app.model import load_model

# Importações dos seus módulos locais
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Detection,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)
from preprocessing.preprocessor import CONFIG_DEFAULT, Preprocessor

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

_preprocessor = Preprocessor(CONFIG_DEFAULT)

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
        raise HTTPException(status_code=400, detail=f"Erro ao decodificar imagem base64: {e!s}")


def _run_inference(image_rgb: np.ndarray, model_name: str, confidence: float) -> PredictResponse:
    """Executa o mesmo pipeline explicito para requisicoes simples e em lote."""
    model = load_model(model_name)
    preprocessed = _preprocessor.process(image_rgb[:, :, ::-1])

    t0 = time.perf_counter()
    model_results = model(preprocessed.frame, conf=confidence, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for result in model_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            box_processed = box.xyxy[0].detach().cpu().numpy().reshape(1, 4)
            box_original = _preprocessor.adjust_boxes(box_processed, preprocessed)[0]
            class_id = int(box.cls[0].item())
            detections.append(Detection(
                label=model.names[class_id],
                confidence=round(float(box.conf[0].item()), 4),
                bbox=[round(float(coordinate), 2) for coordinate in box_original],
            ))

    height, width = image_rgb.shape[:2]
    return PredictResponse(
        detections=detections,
        inference_ms=round(elapsed_ms, 2),
        model_used=model_name,
        image_width=width,
        image_height=height,
    )

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
        response = _run_inference(img_rgb, request.model_name, request.confidence)
        elapsed_ms = response.inference_ms

        _metrics["success"] += 1
        _metrics["total_ms"] += elapsed_ms

        img_height, img_width = img_rgb.shape[:2]

        log_event("predict_complete",
                  model=request.model_name,
                  detections=len(response.detections),
                  inference_ms=round(elapsed_ms, 2),
                  image_size=f"{img_width}x{img_height}")

        return response

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
        results.append(_run_inference(img, request.model_name, request.confidence))

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
