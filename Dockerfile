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

# One thread per pool: a free instance gets a fraction of a core, so the
# default thread pools only add contention.
ENV PORT=7860 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1
EXPOSE 7860

# Shell form so $PORT expands: hosts assign it at runtime.
# One worker, since each would load its own copy of the runtime. Two threads
# absorb the concurrent /predict calls the page makes while scanning; more than
# that just contends for a fraction of a core.
CMD gunicorn --bind "0.0.0.0:${PORT}" --workers 1 --threads 2 \
    --timeout 120 --graceful-timeout 30 \
    --access-logfile - --error-logfile - --log-level info app:app
