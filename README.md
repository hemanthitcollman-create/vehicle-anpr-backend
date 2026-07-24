# Vehicle Number Plate Recognition (ANPR) Backend

FastAPI backend that detects and reads vehicle number plates from
uploaded images using OpenCV (plate localization) + EasyOCR (text
reading).

## Project structure

```
vehicle-anpr-backend/
├── main.py            # FastAPI app
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build used by Railway
├── railway.json        # Railway build/deploy config
├── .gitignore
└── README.md
```

# Vehicle Number Plate Recognition (ANPR) Backend

FastAPI backend that detects and reads vehicle number plates from
uploaded images using **fast-alpr** (a YOLOv9 plate detector + OCR
model, both running on ONNX Runtime — no PyTorch, so it stays light
on RAM and works well on low-memory hosting tiers).

## Project structure

```
vehicle-anpr-backend/
├── main.py            # FastAPI app
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build used by Railway
├── railway.json        # Railway build/deploy config
├── .gitignore
└── README.md
```

## API

| Method | Path                  | Auth required | Description                              |
|--------|-----------------------|:--------------:|-------------------------------------------|
| GET    | `/health`             | No             | Health check (used by Railway)            |
| GET    | `/`                   | No             | Basic info message                        |
| POST   | `/detect-plate`       | Yes            | Upload ONE image, get detected plate text |
| POST   | `/detect-plate-batch` | Yes            | Upload MULTIPLE images (max 10) at once   |

### Authentication

Protected endpoints require a header:
```
x-api-key: <your key>
```
The expected key is read from the `API_KEY` environment variable.
**If `API_KEY` is not set, auth is disabled** — fine for local testing,
but always set it before making the backend public (see step 3 below).

**Example single-image request:**
```bash
curl -X POST "https://<your-app>.up.railway.app/detect-plate" \
  -H "x-api-key: YOUR_SECRET_KEY" \
  -F "file=@car.jpg"
```

**Example response:**
```json
{
  "filename": "car.jpg",
  "plates_found": 1,
  "best_match": {
    "plate_text": "TN39BU6084",
    "detection_confidence": 0.882,
    "ocr_confidence": 0.999,
    "bounding_box": {"x1": 206, "y1": 730, "x2": 389, "y2": 785}
  },
  "all_detections": [ /* one entry per plate found in the image */ ]
}
```

**Example batch request (multiple images in one call):**
```bash
curl -X POST "https://<your-app>.up.railway.app/detect-plate-batch" \
  -H "x-api-key: YOUR_SECRET_KEY" \
  -F "files=@car1.jpg" \
  -F "files=@car2.jpg" \
  -F "files=@car3.jpg"
```
Response is `{"count": 3, "results": [...]}`, one result object per
uploaded file, in the same shape as the single-image response (each
also includes an `"error"` field instead if that particular file
failed or wasn't an image).

> Batch requests are capped at **10 files** per call (`MAX_BATCH_SIZE`
> in `main.py`) to keep any single request from tying up the server or
> using too much memory. Raise or lower this if needed.

---

## 1. Run locally (optional, before deploying)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
API_KEY=devkey123 uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI. In
Swagger, click the padlock icon (or use "Try it out") to enter your
API key before calling protected endpoints.

---

## 2. Push the project to GitHub

```bash
cd vehicle-anpr-backend
git add .
git commit -m "Add API key auth, batch endpoint, upgraded plate detector"
git push
```
(If this is a brand-new repo instead of an update, see the git init /
remote add steps from your first setup.)

---

## 3. Set your API key on Railway (do this before going live)

1. Open your service on Railway → **Variables** tab
2. Add a new variable: `API_KEY` = *(pick a long random string, e.g.
   generate one with `openssl rand -hex 32`)*
3. Save — Railway will redeploy automatically
4. Keep this key secret; anyone with it can call your endpoints. Share
   it only with whatever frontend/service is meant to call this API.

Without this step, your endpoints are open to anyone who finds your
URL.

---

## 4. Deploy / redeploy on Railway

Railway auto-deploys on every push to your connected branch — just
`git push` and watch the **Deployments** tab. First deploy after this
update will reinstall dependencies (fast-alpr instead of the old
OCR stack), so it may take a few minutes; after that, redeploys are
fast.

### Resource notes
- fast-alpr downloads its ONNX model files (~10MB total) the first
  time `/detect-plate` is called after a deploy/restart, so the
  **first request will take a few extra seconds**. After that it's
  fast (well under 1s per image on CPU).
- Peak memory usage is roughly 150–200MB for the model + inference,
  much lighter than a PyTorch-based OCR stack — should run
  comfortably even on Railway's free/trial tier.

---

## Notes / possible next steps
- Tighten CORS (`allow_origins=["*"]`) to your actual frontend's
  domain once you have one.
- Add per-key rate limiting if this becomes public-facing.
- The bundled detector/OCR models are general-purpose; if you need
  higher accuracy for a specific plate format or camera setup, you
  can fine-tune your own YOLO model and point `ALPR(detector_model=...)`
  at it — see the [fast-alpr docs](https://github.com/ankandrew/fast-alpr).

