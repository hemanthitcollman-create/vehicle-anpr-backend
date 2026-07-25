"""
Vehicle Number Plate Recognition (ANPR) Backend
-------------------------------------------------
FastAPI service that accepts an image, locates the number plate region
using OpenCV, and reads the text on it using PaddleOCR.

Endpoints:
  GET  /health                -> simple health check (used by Railway)
  POST /detect-plate          -> upload an image, get back detected plate text
"""

import io
import re
import logging

import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Set logging to DEBUG so we can see exactly what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("anpr")

app = FastAPI(
    title="Vehicle Number Plate Recognition API",
    description="Detects and reads vehicle number plates from uploaded images.",
    version="2.5.0",
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
PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$")
BH_SERIES_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB", "AN", "CH", "DN", "DD", "DL", "JK", "LA", "LD", "PY",
}

PLATE_NOISE_WORDS = {
    "IND", "INDIA", "BHARAT", "BS", "BS4", "BS6", "CNG", "LPG",
    "RE", "BH", "HR", "OD", "UP", "TN", "KA", "MH", "DL", "GJ", "RJ",
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
# Plate region localization (OpenCV-based)
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


def _color_mask_boxes(img: np.ndarray):
    """Detect yellow/white plate regions using HSV color masking."""
    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, (15, 50, 50), (40, 255, 255))
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
            if 1.5 <= aspect_ratio <= 8.0 and 0.001 <= area_fraction <= 0.30:
                vertical_center = y + h / 2
                if vertical_center > img_h * 0.5:
                    pad_x, pad_y = int(w * 0.1), int(h * 0.25)
                    boxes.append((
                        max(0, x - pad_x), max(0, y - pad_y),
                        min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)
                    ))
    return boxes


def _contour_boxes(gray: np.ndarray):
    """Detect rectangular regions using edge detection."""
    blurred = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(blurred, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    img_h, img_w = gray.shape[:2]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect_ratio = w / h
        area_fraction = (w * h) / (img_w * img_h)
        if 1.5 <= aspect_ratio <= 8.0 and 0.001 <= area_fraction <= 0.35:
            pad_x, pad_y = int(w * 0.1), int(h * 0.15)
            boxes.append((
                max(0, x - pad_x), max(0, y - pad_y),
                min(img_w, x + w + pad_x), min(img_h, y + h + pad_y)
            ))
    return boxes


def _bottom_region_boxes(img: np.ndarray):
    """Always crop the bottom portion of the image."""
    img_h, img_w = img.shape[:2]
    boxes = []
    boxes.append((0, int(img_h * 0.70), img_w, img_h))
    boxes.append((int(img_w * 0.2), int(img_h * 0.65), img_w, img_h))
    return boxes


def locate_plate_regions(img: np.ndarray):
    """Merge all OpenCV-based plate region candidates."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]

    color_boxes = _color_mask_boxes(img)
    contour_boxes = _contour_boxes(gray)
    bottom_boxes = _bottom_region_boxes(img)

    all_boxes = color_boxes + contour_boxes + bottom_boxes
    all_boxes = [
        (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))
        for (x1, y1, x2, y2) in all_boxes
        if x2 > x1 and y2 > y1
    ]
    all_boxes = _dedupe_boxes(all_boxes)[:10]

    crops = [img[y1:y2, x1:x2] for (x1, y1, x2, y2) in all_boxes]
    return crops, all_boxes


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def upscale_for_ocr(region: np.ndarray, target_width: int = 800) -> np.ndarray:
    """CLAHE contrast enhancement + upscale for better OCR."""
    h, w = region.shape[:2]
    if w == 0 or h == 0:
        return region

    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    region = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if w < target_width:
        scale = target_width / w
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return region


# ---------------------------------------------------------------------------
# OCR (PaddleOCR v3.x predict API)
# ---------------------------------------------------------------------------

_ocr_engine = None
_ocr_loaded = False


def get_ocr_engine() -> PaddleOCR:
    """Lazily construct the PaddleOCR pipeline."""
    global _ocr_engine, _ocr_loaded
    if _ocr_engine is None:
        logger.info("Loading PaddleOCR models (first request only)...")
        _ocr_engine = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
        _ocr_loaded = True
        logger.info("PaddleOCR models loaded successfully!")
    return _ocr_engine


def run_ocr(image_region: np.ndarray, label: str = "unknown"):
    """
    Run PaddleOCR predict() on an image region.
    Returns list of {"text": str, "confidence": float}.
    """
    if image_region is None or image_region.size == 0:
        logger.warning(f"  [{label}] Empty image region, skipping OCR")
        return []

    region = upscale_for_ocr(image_region)
    engine = get_ocr_engine()

    try:
        result = engine.predict(region)
    except Exception as e:
        logger.error(f"  [{label}] PaddleOCR predict() FAILED with exception: {e}")
        return []

    results = []
    for page in result:
        texts = page.get("rec_texts", []) if hasattr(page, "get") else getattr(page, "rec_texts", [])
        scores = page.get("rec_scores", []) if hasattr(page, "get") else getattr(page, "rec_scores", [])
        for text, score in zip(texts, scores):
            text = (text or "").strip()
            if text:
                results.append({"text": text, "confidence": round(float(score), 3)})

    if not results:
        logger.warning(f"  [{label}] PaddleOCR returned NO text")
    else:
        logger.info(f"  [{label}] Found {len(results)} text(s): {[(r['text'], r['confidence']) for r in results]}")

    return results


# ---------------------------------------------------------------------------
# Text normalization + validation
# ---------------------------------------------------------------------------

def normalize_text(raw: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", raw.upper())


def is_valid_plate_format(cleaned: str) -> bool:
    if PLATE_PATTERN.match(cleaned):
        return cleaned[:2] in VALID_STATE_CODES
    if BH_SERIES_PATTERN.match(cleaned):
        return True
    return False


def is_obviously_not_a_plate(cleaned: str) -> bool:
    if not cleaned:
        return True
    digits = re.sub(r"\D", "", cleaned)
    if len(digits) >= 10 and digits == cleaned:
        return True
    if cleaned.isalpha():
        return True
    if len(cleaned) < 6 or len(cleaned) > 11:
        return True
    return False


def _looks_like_plate_partial(text: str) -> bool:
    """Check if text could be a partial plate number."""
    cleaned = normalize_text(text)
    if len(cleaned) < 3 or len(cleaned) > 7:
        return False
    if not any(c.isdigit() for c in cleaned):
        return False
    if cleaned.isdigit() and len(cleaned) >= 7:
        return False
    if cleaned in PLATE_NOISE_WORDS:
        return False
    return True


# ---------------------------------------------------------------------------
# Plate matching
# ---------------------------------------------------------------------------

def best_plate_match(texts, region_scores=None):
    """Pick the OCR result that best matches a valid Indian plate format."""
    region_scores = region_scores or {}
    candidates = []

    for item in texts:
        cleaned = normalize_text(item["text"])
        if is_obviously_not_a_plate(cleaned):
            logger.debug(f"  REJECTED '{cleaned}' - obviously not a plate")
            continue
        if not is_valid_plate_format(cleaned):
            logger.debug(f"  REJECTED '{cleaned}' - invalid plate format")
            continue

        region_bonus = region_scores.get(item.get("region_index"), 0.5)
        score = (item["confidence"] * 0.6) + (region_bonus * 0.4)
        logger.info(f"  CANDIDATE '{cleaned}' score={score:.3f}")
        candidates.append({
            "text": item["text"],
            "confidence": item["confidence"],
            "normalized": cleaned,
            "score": round(score, 3),
        })

    if not candidates:
        logger.info("  No valid plate candidates found")
        return None

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]
    logger.info(f"  BEST MATCH: {best['normalized']} (score={best['score']})")
    return {
        "text": best["normalized"],
        "confidence": best["confidence"],
        "score": best["score"],
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

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
    img_h, img_w = img.shape[:2]

    logger.info(f"=== DETECT-PLATE START === Image: {img_w}x{img_h}, {len(file_bytes)} bytes")

    all_texts = []
    region_scores = {}
    next_idx = 0

    # ============================================================
    # STRATEGY 1: Full-image OCR (no preprocessing)
    # ============================================================
    logger.info("--- Strategy 1: Full-image OCR ---")
    full_image_results = run_ocr(img, "full_image")
    for res in full_image_results:
        res["region_index"] = next_idx
        all_texts.append(res)
    region_scores[next_idx] = 0.5
    next_idx += 1

    match = best_plate_match(all_texts, region_scores)

    # ============================================================
    # STRATEGY 2: Targeted re-OCR on bottom region
    # ============================================================
    if match is None:
        logger.info("--- Strategy 2: Targeted re-OCR on bottom region ---")
        has_partial = any(_looks_like_plate_partial(t["text"]) for t in all_texts)
        if has_partial:
            logger.info("  Found partial plate-like text, running targeted re-OCR")
            # Crop the bottom 65% of the image and run OCR
            crop_y1 = int(img_h * 0.60)
            crop = img[crop_y1:, :]

            scale = 1200 / max(crop.shape[1], 1)
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            targeted_results = run_ocr(crop, "bottom_region_targeted")
            for res in targeted_results:
                cleaned = normalize_text(res["text"])
                if cleaned in PLATE_NOISE_WORDS:
                    continue
                res["region_index"] = next_idx
                all_texts.append(res)
            region_scores[next_idx] = 0.95
            next_idx += 1

            match = best_plate_match(all_texts, region_scores)
        else:
            logger.info("  No partial plate-like text found, skipping targeted re-OCR")

    # ============================================================
    # STRATEGY 3: OpenCV-based localization
    # ============================================================
    if match is None:
        logger.info("--- Strategy 3: OpenCV-based localization ---")
        plate_crops, plate_boxes = locate_plate_regions(img)
        logger.info(f"  Found {len(plate_crops)} plate region candidates")

        for idx, (region, box) in enumerate(zip(plate_crops, plate_boxes)):
            _x1, y1, _x2, y2 = box
            vertical_center = (y1 + y2) / 2
            if vertical_center > img_h * 0.65:
                region_scores[next_idx] = 0.9
            elif vertical_center > img_h * 0.45:
                region_scores[next_idx] = 0.7
            else:
                region_scores[next_idx] = 0.3

            region_results = run_ocr(region, f"opencv_crop_{next_idx}")
            for res in region_results:
                res["region_index"] = next_idx
                all_texts.append(res)
            next_idx += 1

        match = best_plate_match(all_texts, region_scores)

    logger.info(f"=== RESULT: best_match={match} ===")

    return JSONResponse({
        "plate_regions_found": len(plate_boxes) if match is None else 0,
        "best_match": match,
        "all_detected_text": [
            {"text": t["text"], "confidence": t["confidence"], "region_index": t.get("region_index")}
            for t in all_texts
        ],
    })
