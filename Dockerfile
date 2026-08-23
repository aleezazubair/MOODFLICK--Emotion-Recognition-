# Container image for any container host (Render, Fly.io, Cloud Run).
# Vercel does not use this -- it builds from vercel.json and api/index.py.
FROM python:3.11-slim

# libglib2 is still needed by opencv-python-headless; libgomp1 by onnxruntime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Copied first so the dependency layer is cached across code-only changes.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

ENV PORT=7860
EXPOSE 7860

# Shell form so $PORT expands: hosts assign it at runtime.
# One worker, since each loads its own copy of the runtime; threads absorb the
# concurrent /predict calls the page makes while scanning.
CMD gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 4 \
    --timeout 120 --access-logfile - app:app
