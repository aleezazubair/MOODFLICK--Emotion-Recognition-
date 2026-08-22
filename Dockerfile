# Container image for Hugging Face Spaces (SDK: docker).
FROM python:3.11-slim

# libglib2 is still needed by opencv-python-headless; libgomp1 by TensorFlow.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000.
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

ENV TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860
EXPOSE 7860

# One worker: each would load its own copy of TensorFlow. Threads absorb the
# concurrent /predict calls the page makes while scanning.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", \
     "--timeout", "120", "--access-logfile", "-", "app:app"]
