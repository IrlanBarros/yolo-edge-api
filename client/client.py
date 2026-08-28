import os
from pathlib import Path
import httpx
from PIL import Image
import io
import base64

API_URL = os.getenv("API_URL", "http://yolo-api:8000")
IMAGES_DIR = Path("/client/images")

def wait_for_api():
    print(f"Aguardando a API em {API_URL}...")
    while True:
        try:
            response = httpx.get(f"{API_URL}/health", timeout=5.0)
            if response.status_code == 200:
                print("API está pronta!")
                break
        except Exception:
            pass
        import time
        time.sleep(2)

def run_single_inference(image_path: Path):
    print(f"\n[Inferência Individual] Enviando {image_path.name}...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    payload = {"image_base64": encoded, "model_name": "yolov8n.pt"}
    
    response = httpx.post(f"{API_URL}/predict", json=payload, timeout=30.0)
    if response.status_code == 200:
        data = response.json()
        print(f"Sucesso! {len(data['detections'])} objetos detectados em {data['inference_ms']} ms.")
    else:
        print(f"Erro na requisição: {response.status_code} - {response.text}")

def run_batch_inference(image_paths: list[Path]):
    print(f"\n[Inferência em Lote] Enviando {len(image_paths)} imagens...")
    images_payload = []
    
    for path in image_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            images_payload.append({"image_base64": b64, "model_name": "yolov8n.pt"})
            
    payload = {"images": images_payload}
    response = httpx.post(f"{API_URL}/predict/batch", json=payload, timeout=60.0)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Lote processado com sucesso em {data['total_inference_ms']} ms total.")
        for i, r in enumerate(data["results"]):
            print(f"  Imagem {i+1}: {len(r['detections'])} detecções em {r['inference_ms']} ms")
    else:
        print(f"Erro no lote: {response.status_code} - {response.text}")

if __name__ == "__main__":
    wait_for_api()

    images = sorted(IMAGES_DIR.glob("*.jpg")) + \
             sorted(IMAGES_DIR.glob("*.png"))

    if not images:
        print("[AVISO] Nenhuma imagem encontrada em /client/images/")
    else:
        run_single_inference(images[0])
        if len(images) > 1:
            run_batch_inference(images)

    metrics = httpx.get(f"{API_URL}/metrics").json()
    print(f"\n─── Métricas da API ───")
    print(f"  Total de requisições : {metrics['total_requests']}")
    print(f"  Latência média        : {metrics['avg_inference_ms']} ms")
