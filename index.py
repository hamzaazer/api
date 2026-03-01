import os
import hashlib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from ecg_signal import image_to_1d_signal  # CYTHON

CLASS_NAMES = ['f', 'm', 'n', 'q', 's', 'v']
MODELS_DIR = "models"

app = FastAPI(title="ECG Multi-Model Comparison API")

models = {}              # {filename: model}
_models_fingerprint = "" # last known folder state


def folder_fingerprint(models_dir: str) -> str:
    """
    Create a fingerprint of all .keras files based on name + size + mtime.
    If anything changes, fingerprint changes.
    """
    if not os.path.exists(models_dir):
        return ""

    items = []
    for f in sorted(os.listdir(models_dir)):
        if f.endswith(".keras"):
            p = os.path.join(models_dir, f)
            st = os.stat(p)
            items.append(f"{f}|{st.st_size}|{int(st.st_mtime)}")

    raw = "||".join(items).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def load_all_models():
    global models
    models.clear()

    if not os.path.exists(MODELS_DIR):
        raise RuntimeError(f"Folder '{MODELS_DIR}' not found")

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")]
    if not model_files:
        raise RuntimeError("No .keras models found")

    for file in model_files:
        path = os.path.join(MODELS_DIR, file)
        models[file] = tf.keras.models.load_model(path, compile=False)


def ensure_models_up_to_date():
    """
    Reload models automatically if models/ folder changed.
    """
    global _models_fingerprint
    fp = folder_fingerprint(MODELS_DIR)

    # First time or changed
    if fp != _models_fingerprint:
        load_all_models()
        _models_fingerprint = fp


@app.on_event("startup")
def startup():
    ensure_models_up_to_date()


@app.get("/health")
def health():
    ensure_models_up_to_date()
    return {"status": "ok", "models_loaded": len(models)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        ensure_models_up_to_date()
    except Exception as ex:
        raise HTTPException(status_code=503, detail=f"Model load error: {ex}")

    if not models:
        raise HTTPException(status_code=503, detail="No models loaded")

    suffix = os.path.splitext(file.filename)[-1].lower() or ".png"
    temp_path = f"__temp_ecg{suffix}"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        with open(temp_path, "wb") as f:
            f.write(content)

        signal = image_to_1d_signal(temp_path)

        results = []
        for name, model in models.items():
            preds = model(signal)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            probs = preds[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))
            label = CLASS_NAMES[pred_idx]

            results.append({
                "name": name,
                "label": label,
                "confidence": confidence,
                "probs": {c: float(p) for c, p in zip(CLASS_NAMES, probs)}
            })

        best = max(results, key=lambda x: x["confidence"])

        return JSONResponse({
            "best_model": best["name"],
            "best_label": best["label"],
            "best_confidence": best["confidence"],
            "results": results
        })

    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
