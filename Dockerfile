FROM python:3.11-slim

# System deps: libgl1/libglib needed by OpenCV (used internally by fast-alpr)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects the PORT env var at runtime; default to 8000 locally
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
