"""
Vehicle Number Plate Recognition (ANPR) Backend
-------------------------------------------------
FastAPI service that accepts an image and reads any vehicle number
plate on it using Google Cloud Vision API's TEXT_DETECTION.

Why Cloud Vision instead of Tesseract/PaddleOCR:
  - Tesseract could not reliably read real Indian plates in testing
    (returned empty/garbage even on a perfectly hand-cropped plate).
  - PaddleOCR reads better but needs more RAM than this project's
    Railway plan comfortably supports, and could not be verified
    end-to-end in the development environment.
  - Cloud Vision is a hosted API call -- no local model to load, no
    RAM/build concerns, and it's colour-agnostic: it finds text
    regardless of whether the plate background is yellow or white,
    unlike the old colour-mask localization step, which broke on a
    shadowed white plate and a yellow-on-yellow plate in testing.

Because Cloud Vision does its own text detection across the whole
image, this version does NOT do local plate cropping first -- it
sends the full image and lets Vision find all text, then the same
strict Indian-plate-format validator (unchanged from before) picks
out the plate from everything else Vision finds (ad banners, phone
numbers, shop signs, etc. all still get correctly rejected).

Setup required:
  Set the environment variable GOOGLE_VISION_API_KEY (on Railway:
  Service -> Variables) to an API key with Cloud Vision API enabled.

Endpoints:
  GET  /health                -> simple health check (used by Railway)
  POST /detect-plate          -> upload an image, get back detected plate text
"""

import base64
import logging
import os
import re

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anpr")

app = FastAPI(
    title="Vehicle Number Plate Recognition API",
    description="Detects and reads vehicle number plates from uploaded images using Google Cloud Vision.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")
VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

# Standard Indian plate: SS DD LL DDDD (state code, district, series, number)
# e.g. "MH12NW8556", "KA01AB1234". BH-series ("22BH1234AB") handled separately.
PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
BH_SERIES_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

VALID_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB", "AN", "CH", "DN", "DD", "DL", "JK", "LA", "LD", "PY",
}


# ---------------------------------------------------------------------------
# Cloud Vision call
# ---------------------------------------------------------------------------

def call_vision_ocr(file_bytes: bytes) -> list:
    """
    Send the image to Cloud Vision's TEXT_DETECTION and return a flat
    list of {"text": str, "confidence": float}. TEXT_DETECTION (as
    opposed to DOCUMENT_TEXT_DETECTION) returns one entry per detected
    text block/word plus one entry for the full concatenated text as
    element [0] -- we skip element [0] and use the individual pieces so
    each candidate can be validated against the plate regex on its own.

    Cloud Vision doesn't return a numeric "confidence" per text block
    for TEXT_DETECTION the way Tesseract did, so every candidate here
    gets the same fixed confidence weight; format validation via regex
    is the real filter (as it already was for Tesseract results too).
    """
    if not VISION_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_VISION_API_KEY is not set. Add it in Railway's Variables tab.",
        )

    encoded_image = base64.b64encode(file_bytes).decode("utf-8")
    payload = {
        "requests": [
            {
                "image": {"content": encoded_image},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }

    try:
        response = httpx.post(
            VISION_ENDPOINT,
            params={"key": VISION_API_KEY},
            json=payload,
            timeout=30.0,
        )
    except httpx.RequestError as exc:
        logger.exception("Cloud Vision request failed")
        raise HTTPException(status_code=502, detail=f"Could not reach Cloud Vision API: {exc}")

    if response.status_code != 200:
        logger.error("Cloud Vision returned %s: %s", response.status_code, response.text)
        raise HTTPException(
            status_code=502,
            detail=f"Cloud Vision API error ({response.status_code}): {response.text[:300]}",
        )

    data = response.json()
    api_response = data.get("responses", [{}])[0]

    if "error" in api_response:
        logger.error("Cloud Vision API error: %s", api_response["error"])
        raise HTTPException(status_code=502, detail=f"Cloud Vision API error: {api_response['error']}")

    annotations = api_response.get("textAnnotations", [])
    if not annotations:
        return []

    # annotations[0] is the full block of concatenated text -- skip it,
    # we want the individual words/fragments in annotations[1:]
    results = []
    for ann in annotations[1:]:
        text = (ann.get("description") or "").strip()
        if text:
            results.append({"text": text, "confidence": 0.9})  # Vision doesn't give per-word confidence here
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


def _combine_adjacent(texts: list) -> list:
    """
    Vision often splits a plate into separate word fragments, e.g.
    "MH12K" and "R1145" as two annotations instead of one. Try
    combining every pair of consecutive raw text entries (in the
    order Vision returned them) as an extra candidate, in case the
    plate got split across two words -- this mirrors how a plate
    reads top-to-bottom or left-to-right in the original image.
    """
    combined = []
    for i in range(len(texts) - 1):
        merged = normalize_text(texts[i]["text"] + texts[i + 1]["text"])
        combined.append({"text": merged, "confidence": 0.8})
    return combined


def best_plate_match(texts: list):
    all_candidates_raw = texts + _combine_adjacent(texts)
    candidates = []

    for item in all_candidates_raw:
        cleaned = normalize_text(item["text"])
        if is_obviously_not_a_plate(cleaned):
            continue
        if not is_valid_plate_format(cleaned):
            continue

        candidates.append({
            "text": item["text"],
            "confidence": item["confidence"],
            "normalized": cleaned,
            "score": item["confidence"],
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

    all_texts = call_vision_ocr(file_bytes)
    match = best_plate_match(all_texts)

    return JSONResponse({
        "best_match": match,
        "all_detected_text": all_texts,
    })
