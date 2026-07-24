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
    """Return a list of cropped candidate plate regions (numpy arrays).

    Tries the Haar cascade first (fast, but trained on Russian-style
    plates so it often misses others), then falls back to a classical
    contour-based method that looks for a bright, rectangular,
    high-contrast region -- which is what a plate looks like against a
    car body/bumper. This catches most real-world plate photos.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)

    cascade = get_plate_cascade()
    plates = cascade.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))

    crops = []
    for (x, y, w, h) in plates:
        pad_x, pad_y = int(w * 0.1), int(h * 0.2)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)
        crops.append(img[y1:y2, x1:x2])

    if crops:
        return crops

    return _contour_based_plate_regions(img, gray)


def _contour_based_plate_regions(img: np.ndarray, gray: np.ndarray):
    blurred = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(blurred, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    img_h, img_w = gray.shape[:2]
    candidates = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect_ratio = w / h
        area_fraction = (w * h) / (img_w * img_h)

        # A plate is a wide rectangle, not too small, not the whole photo.
        if 2.0 <= aspect_ratio <= 6.0 and 0.005 <= area_fraction <= 0.35:
            pad_x, pad_y = int(w * 0.08), int(h * 0.15)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)
            candidates.append((area_fraction, img[y1:y2, x1:x2]))

    # Largest plausible candidates first; cap at 5 to keep OCR calls sane.
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [crop for _area, crop in candidates[:5]]


def preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    """Grayscale + upscale + threshold to help Tesseract read plates.

    Tuned empirically: a moderate upscale (target width ~250px) with a
    straight Otsu threshold reads plate characters more accurately than
    a larger upscale or an extra blur step, which tend to soften
    character edges and cause digit/letter confusion (e.g. 6 vs G).
    """
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    target_width = 250
    if gray.shape[1] != target_width:
        scale = target_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def run_ocr(image_region: np.ndarray, psm: int = 7):
    processed = preprocess_for_ocr(image_region)

    config = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    data = pytesseract.image_to_data(
        processed, config=config, output_type=pytesseract.Output.DICT
    )

    results = []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        conf = float(conf)
        if text and conf >= 0:
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

    all_texts = []
    if plate_crops:
        # Cascade found candidate plate regions: each crop is close to a
        # single line of text, so psm 7 (single text line) works best.
        for region in plate_crops:
            if region.size == 0:
                continue
            all_texts.extend(run_ocr(region, psm=7))
    else:
        # No candidate region found (common for full car/scene photos).
        # psm 11 (sparse text) looks for scattered text blocks anywhere
        # in the image instead of assuming one line, which works much
        # better here. We also try psm 6 as a second pass since plates
        # sometimes read better as a single uniform block.
        all_texts.extend(run_ocr(img, psm=11))
        all_texts.extend(run_ocr(img, psm=6))

    match = best_plate_match(all_texts)

    return JSONResponse({
        "plate_regions_found": len(plate_crops),
        "best_match": match,
        "all_detected_text": all_texts,
    })
