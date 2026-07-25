"""
Vehicle Number Plate Recognition (ANPR) Backend
-------------------------------------------------
FastAPI service that accepts an image, locates the number plate region
using OpenCV, and reads the text on it using Tesseract OCR.

This is the free, no-billing, no-API-key version: everything runs
locally on the server using open-source tools (OpenCV + Tesseract),
with no external paid API involved. Cloud Vision API was tried and
reads plates more accurately, but requires a Google Cloud billing
account -- this version avoids that entirely at the cost of lower
OCR accuracy on small/angled/cluttered plates.

Plate localization uses three passes, merged together:
  1. Haar cascade (fast, but trained on Russian-style plates so it
     frequently misses Indian ones)
  2. Colour-mask detection tuned for Indian plates -- yellow background
     (commercial vehicles) or white background (private vehicles) with
     dark text, since that's a much stronger signal than generic edges
  3. Generic contour-based detection (bright, rectangular, high-contrast
     region) as a catch-all fallback

Detected regions are OCR'd individually and only text that matches the
Indian plate format is considered a candidate -- this stops large,
high-contrast text elsewhere in the photo (ad banners, stickers, phone
numbers on auto-rickshaws etc.) from being picked as the plate.

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
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_plate_cascade = None

# Standard Indian plate: SS DD LL DDDD (state code, district, series, number)
# e.g. "MH12NW8556", "KA01AB1234". BH-series ("22BH1234AB") handled separately.
PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
BH_SERIES_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB", "AN", "CH", "DN", "DD", "DL", "JK", "LA", "LD", "PY",
}


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


# ---------------------------------------------------------------------------
# Plate region localization
# ---------------------------------------------------------------------------

def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _dedupe_boxes(boxes, iou_thresh: float = 0.4):
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def _haar_boxes(eq_gray: np.ndarray):
    cascade = get_plate_cascade()
    plates = cascade.detectMultiScale(eq_gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))
    boxes = []
    for (x, y, w, h) in plates:
        pad_x, pad_y = int(w * 0.1), int(h * 0.2)
        boxes.append((x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y))
    return boxes


def _color_mask_boxes(img: np.ndarray):
    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 40, 255))

    boxes = []
    for mask in (yellow_mask, white_mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect_ratio = w / h
            area_fraction = (w * h) / (img_w * img_h)

            if 2.0 <= aspect_ratio <= 6.0 and 0.003 <= area_fraction <= 0.30:
                pad_x, pad_y = int(w * 0.08), int(h * 0.2)
                boxes.append((x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y))

    return boxes


def _contour_boxes(gray: np.ndarray):
    blurred = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(blurred, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    img_h, img_w = gray.shape[:2]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect_ratio = w / h
        area_fraction = (w * h) / (img_w * img_h)
        if 2.0 <= aspect_ratio <= 6.0 and 0.005 <= area_fraction <= 0.35:
            pad_x, pad_y = int(w * 0.08), int(h * 0.15)
            boxes.append((x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y))
    return boxes


def locate_plate_regions(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    img_h, img_w = gray.shape[:2]

    color_boxes = _color_mask_boxes(img)
    haar_boxes = _haar_boxes(eq)
    contour_boxes = _contour_boxes(gray) if not color_boxes else []

    all_boxes = color_boxes + haar_boxes + contour_boxes
    all_boxes = [
        (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))
        for (x1, y1, x2, y2) in all_boxes
        if x2 > x1 and y2 > y1
    ]
    all_boxes = _dedupe_boxes(all_boxes)[:6]

    crops = [img[y1:y2, x1:x2] for (x1, y1, x2, y2) in all_boxes]
    return crops, all_boxes


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def preprocess_for_ocr(region: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    target_width = 250
    if gray.shape[1] > 0 and gray.shape[1] != target_width:
        scale = target_width / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _thresh, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def run_ocr(image_region: np.ndarray, psm: int = 7):
    if image_region is None or image_region.size == 0:
        return []
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


# ---------------------------------------------------------------------------
# Text normalization + validation + scoring
# ---------------------------------------------------------------------------

def normalize_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def is_valid_plate_format(cleaned: str) -> bool:
    if PLATE_PATTERN.match(cleaned):
        return cleaned[:2] in VALID_STATE_CODES
    if BH_SERIES_PATTERN.match(cleaned):
        return True
    return False


def looks_like_phone_number(cleaned: str) -> bool:
    digits = re.sub(r"\D", "", cleaned)
    return len(digits) >= 10 and digits == cleaned


def looks_like_pure_word(cleaned: str) -> bool:
    return cleaned.isalpha()


def is_obviously_not_a_plate(cleaned: str) -> bool:
    if not cleaned:
        return True
    if looks_like_phone_number(cleaned):
        return True
    if looks_like_pure_word(cleaned):
        return True
    if len(cleaned) < 6 or len(cleaned) > 11:
        return True
    return False


def best_plate_match(texts, region_scores=None):
    region_scores = region_scores or {}
    candidates = []

    for item in texts:
        cleaned = normalize_text(item["text"])
        if is_obviously_not_a_plate(cleaned):
            continue
        if not is_valid_plate_format(cleaned):
            continue

        region_bonus = region_scores.get(item.get("region_index"), 0.5)
        score = (item["confidence"] * 0.6) + (region_bonus * 0.4)

        candidates.append({
            "text": item["text"],
            "confidence": item["confidence"],
            "normalized": cleaned,
            "score": round(score, 3),
        })

    if not candidates:
        return None
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[0]


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

    plate_crops, plate_boxes = locate_plate_regions(img)

    all_texts = []
    region_scores = {}

    if plate_crops:
        img_h = img.shape[0]
        for idx, (region, box) in enumerate(zip(plate_crops, plate_boxes)):
            _x1, y1, _x2, y2 = box
            vertical_center = (y1 + y2) / 2
            region_scores[idx] = 0.7 if vertical_center > img_h * 0.4 else 0.4

            for res in run_ocr(region, psm=7):
                res["region_index"] = idx
                all_texts.append(res)
            for res in run_ocr(region, psm=8):
                res["region_index"] = idx
                all_texts.append(res)
    else:
        for res in run_ocr(img, psm=11):
            res["region_index"] = None
            all_texts.append(res)
        for res in run_ocr(img, psm=6):
            res["region_index"] = None
            all_texts.append(res)

    match = best_plate_match(all_texts, region_scores)

    return JSONResponse({
        "plate_regions_found": len(plate_crops),
        "best_match": match,
        "all_detected_text": all_texts,
    })
