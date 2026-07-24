"""
Vehicle Number Plate Recognition (ANPR) Backend
-------------------------------------------------
FastAPI service that detects and reads vehicle number plates from
uploaded images, using fast-alpr (YOLOv9 detector + CRNN-style OCR,
both running on ONNX Runtime -- no PyTorch, so it stays light on RAM).

Endpoints:
  GET  /health                -> health check (no API key required)
  POST /detect-plate          -> upload ONE image, get plate text back
  POST /detect-plate-batch    -> upload MULTIPLE images in one request

Auth:
  All /detect-plate* endpoints require a header:  x-api-key: <your key>
  Set the expected key via the API_KEY environment variable on Railway.
  If API_KEY is not set, auth is disabled (useful for local dev only --
  always set it in production).
"""

import io
import os
import logging

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List

from fast_alpr import ALPR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anpr")

app = FastAPI(
    title="Vehicle Number Plate Recognition API",
    description="Detects and reads vehicle number plates from uploaded images.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API key auth --------------------------------------------------------
API_KEY = os.environ.get("API_KEY")  # set this in Railway -> Variables

# Cap how many files a single batch request can contain, so one request
# can't tie up the server for a very long time or exhaust memory.
MAX_BATCH_SIZE = 10


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY is None:
        # No key configured on the server -> auth disabled (dev mode).
        return
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


# --- Lazy-loaded model -----------------------------------------------
# ALPR bundles its own detector + OCR models. Loading it downloads
# small ONNX model files (~10MB total) on first use, so we build it
# once, lazily, rather than at import time.
_alpr = None


def get_alpr() -> ALPR:
    global _alpr
    if _alpr is None:
        logger.info("Loading ALPR models (first request only)...")
        _alpr = ALPR()
    return _alpr


def read_image(file_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return np.array(image)[:, :, ::-1]  # RGB -> BGR for OpenCV-style array
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")


def detect_plate_in_image(file_bytes: bytes) -> dict:
    img = read_image(file_bytes)
    alpr = get_alpr()
    results = alpr.predict(img)

    detections = []
    for r in results:
        detections.append({
            "plate_text": r.ocr.text if r.ocr else None,
            "detection_confidence": round(float(r.detection.confidence), 3),
            "ocr_confidence": (
                round(float(np.mean(r.ocr.confidence)), 3)
                if r.ocr and r.ocr.confidence else None
            ),
            "bounding_box": {
                "x1": r.detection.bounding_box.x1,
                "y1": r.detection.bounding_box.y1,
                "x2": r.detection.bounding_box.x2,
                "y2": r.detection.bounding_box.y2,
            },
        })

    best = max(detections, key=lambda d: d["detection_confidence"]) if detections else None

    return {
        "plates_found": len(detections),
        "best_match": best,
        "all_detections": detections,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Vehicle ANPR backend is running. POST an image to /detect-plate"}


@app.post("/detect-plate", dependencies=[Depends(require_api_key)])
async def detect_plate(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    file_bytes = await file.read()
    result = detect_plate_in_image(file_bytes)
    return JSONResponse({"filename": file.filename, **result})


@app.post("/detect-plate-batch", dependencies=[Depends(require_api_key)])
async def detect_plate_batch(files: List[UploadFile] = File(...)):
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max {MAX_BATCH_SIZE} per request.",
        )

    output = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            output.append({
                "filename": file.filename,
                "error": "Not an image file, skipped.",
            })
            continue

        try:
            file_bytes = await file.read()
            result = detect_plate_in_image(file_bytes)
            output.append({"filename": file.filename, **result})
        except HTTPException as exc:
            output.append({"filename": file.filename, "error": exc.detail})
        except Exception as exc:
            logger.exception("Unexpected error processing %s", file.filename)
            output.append({"filename": file.filename, "error": str(exc)})

    return JSONResponse({"count": len(output), "results": output})
