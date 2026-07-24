"""
Vehicle Number Plate Recognition (ANPR) Backend
-------------------------------------------------
FastAPI service that accepts an image, locates the number plate region
using OpenCV, and reads the text on it using Tesseract OCR.

Tesseract is used instead of EasyOCR because EasyOCR pulls in PyTorch,
which needs far more RAM than low-tier hosting plans (like Railway's
free/trial tier) provide, causing the process to be OOM-killed.
Tesseract has no deep-learning-framework dependency and runs
comfortably in ~512MB of RAM.

Endpoints:
  GET  /health                -> simple health check (used by Railway)
  POST /detect-plate          -> upload an image, get back detected plate text
"""

import io
import re
import logging

import cv2
import numpy as np
import pytesseract
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anpr")

app = FastAPI(
    title="Vehicle Number Plate Recognition API",
    description="Detects and reads vehicle number plates from uploaded images.",
    version="1.0.0",
)

# Allow calls from any frontend (tighten this to your domain in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_plate_cascade = None

# Basic pattern for Indian plates, e.g. "KA01AB1234". Adjust/remove
# this if you need to support other countries' plate formats.
PLATE_PATTERN = re.compile(r"[A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,3}\s?[0-9]{3,4}")


def get_plate_cascade():
    global _plate_cascade
    if _plate_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
        _plate_cascade = cv2.CascadeClassifier(cascade_path)
    return _plate_cascade


def read_image(file_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")


def locate_plate_regions(img: np.ndarray):
    """Return a list of cropped candidate plate regions (numpy arrays)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    cascade = get_plate_cascade()
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))

    crops = []
    for (x, y, w, h) in plates:
        pad_x, pad_y = int(w * 0.1), int(h * 0.2)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)
        crops.append(img[y1:y2, x1:x2])

    return crops


def preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    """Grayscale + threshold + upscale small crops to help Tesseract read plates."""
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    if gray.shape[1] < 300:
        scale = 300 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def run_ocr(image_region: np.ndarray):
    processed = preprocess_for_ocr(image_region)

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    data = pytesseract.image_to_data(
        processed, config=config, output_type=pytesseract.Output.DICT
    )

    results = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        conf = float(conf)
        if text and conf > 0:
            results.append({"text": text, "confidence": round(conf / 100, 3)})
    return results


def best_plate_match(texts):
    """Pick the OCR result that best matches a plate-like pattern."""
    candidates = []
    for item in texts:
        cleaned = item["text"].upper().replace(" ", "")
        if PLATE_PATTERN.search(cleaned):
            candidates.append({**item, "normalized": cleaned})
    if candidates:
        return max(candidates, key=lambda c: c["confidence"])
    return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Vehicle ANPR backend is running. POST an image to /detect-plate"}


@app.post("/detect-plate")
async def detect_plate(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    file_bytes = await file.read()
    img = read_image(file_bytes)

    plate_crops = locate_plate_regions(img)

    # If the cascade found no candidate region, fall back to running OCR
    # on the whole image (works fine for close-up plate photos).
    regions_to_scan = plate_crops if plate_crops else [img]

    all_texts = []
    for region in regions_to_scan:
        if region.size == 0:
            continue
        all_texts.extend(run_ocr(region))

    match = best_plate_match(all_texts)

    return JSONResponse({
        "plate_regions_found": len(plate_crops),
        "best_match": match,
        "all_detected_text": all_texts,
    })
