# Vehicle Number Plate Recognition (ANPR) Backend

FastAPI backend that detects and reads vehicle number plates from
uploaded images using OpenCV (plate localization: colour-masking +
Haar cascade + contour fallback) and Tesseract OCR (text reading).

> Note: Tesseract is used instead of EasyOCR because EasyOCR pulls in
> PyTorch, which needs more RAM than low-tier hosting plans provide.
> Detected text is validated against the Indian plate format
> (`SS DD LL DDDD`, with real state-code checking) before being
> returned as `best_match`, so ad banners, stickers, and phone numbers
> in the photo don't get mistaken for the plate.

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

| Method | Path            | Description                              |
|--------|-----------------|-------------------------------------------|
| GET    | `/health`       | Health check (used by Railway)            |
| GET    | `/`             | Basic info message                        |
| POST   | `/detect-plate` | Upload an image, get detected plate text  |

**Example request:**
```bash
curl -X POST "https://<your-app>.up.railway.app/detect-plate" \
  -F "file=@car.jpg"
```

**Example response:**
```json
{
  "plate_regions_found": 1,
  "best_match": {
    "text": "KA01AB1234",
    "confidence": 0.87,
    "normalized": "KA01AB1234",
    "score": 0.79
  },
  "all_detected_text": [
    {"text": "KA01AB1234", "confidence": 0.87, "region_index": 0}
  ]
}
```

`best_match` is `null` if nothing in the image matched a valid Indian
plate format — this is intentional. The endpoint will not guess; it
only returns text that passed format + state-code validation.

> The regex in `main.py` (`PLATE_PATTERN`) is set up for Indian plate
> formats (e.g. `KA01AB1234`). Edit it if you need a different
> country's format.

### Known limitation
Localization uses classical computer vision (colour masking, Haar
cascade, contour detection) — no trained model. This works well on
front/rear-on, reasonably lit shots, but can miss plates that are
small, angled, or in heavy shadow, since there's little brightness
contrast between the plate and its surroundings in that case. If you
need reliable results on harder real-world photos (angled shots,
low light, distant vehicles), swap `locate_plate_regions()` for a
YOLOv8-based plate detector — it's a bigger change but will meaningfully
outperform rule-based CV on the "will it find the plate at all" step.

---

## 1. Run locally (optional, before deploying)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---

## 2. Push the project to GitHub

```bash
cd vehicle-anpr-backend
git init
git add .
git commit -m "Initial commit: ANPR backend"

# Create a new repo on GitHub first (via github.com or gh CLI), then:
git branch -M main
git remote add origin https://github.com/<your-username>/vehicle-anpr-backend.git
git push -u origin main
```

Using GitHub CLI instead:
```bash
gh repo create vehicle-anpr-backend --public --source=. --remote=origin --push
```

---

## 3. Deploy on Railway

**Option A — Railway dashboard (recommended for first deploy)**
1. Go to [railway.app](https://railway.app) and log in.
2. Click **New Project → Deploy from GitHub repo**.
3. Select your `vehicle-anpr-backend` repository.
4. Railway detects the `Dockerfile` automatically and builds the image.
5. Once deployed, go to **Settings → Networking → Generate Domain**
   to get a public URL like `https://vehicle-anpr-backend.up.railway.app`.
6. Test it: `GET https://<your-domain>/health` should return `{"status": "ok"}`.

**Option B — Railway CLI**
```bash
npm install -g @railway/cli
railway login
railway init
railway up
railway domain   # generates a public URL
```

### Environment variables
No required env vars for this project — Railway automatically injects
`PORT`, which `main.py`/Dockerfile already read. Add any of your own
(e.g. an API key to protect the endpoint) via **Settings → Variables**
on Railway.

### Resource notes
- EasyOCR downloads its recognition model (~65 MB) the first time
  `/detect-plate` is called, so the **first request after a deploy or
  restart will be slow** (10–30s). Later requests are fast.
- EasyOCR pulls in PyTorch, so the build is a few hundred MB — pick at
  least Railway's default plan; the free trial tier usually has enough
  RAM (≥512 MB recommended, 1 GB+ preferred).

---

## 4. Redeploying after changes

Railway auto-deploys on every push to your connected branch:
```bash
git add .
git commit -m "Update detection logic"
git push
```

---

## Notes / next steps you may want
- Add an API key check (e.g. a header `x-api-key`) before exposing this publicly.
- Swap the Haar cascade for a YOLOv8-based plate detector if you need
  higher accuracy on angled/dirty/far-away plates.
- Add a `/detect-plate` rate limit if this will be public-facing.
