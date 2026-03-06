FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/hf-cache
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/hf-cache
ENV LTX_CACHE_ROOT=/runpod-volume/ltx
ENV LTX_WORK_ROOT=/tmp/ltx
ENV LTX_REPO_DIR=/opt/LTX-2
ENV LTX_REPO_COMMIT=9e8a28e17ac4dd9e49695223d50753a1ebda36fe
ENV PYTHONPATH=/opt/LTX-2/packages/ltx-core/src:/opt/LTX-2/packages/ltx-pipelines/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone repo, install deps, clean caches — all in one layer to minimize disk use
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /opt/LTX-2 \
    && cd /opt/LTX-2 \
    && git fetch --depth 1 origin ${LTX_REPO_COMMIT} \
    && git checkout ${LTX_REPO_COMMIT} \
    && uv sync --frozen --no-dev \
    && uv pip install --python .venv/bin/python --no-cache-dir runpod huggingface_hub requests \
    && rm -rf /root/.cache/uv /root/.cache/pip \
    && rm -rf .git

COPY handler.py /opt/ltx-worker/handler.py
COPY http_server.py /opt/ltx-worker/http_server.py

WORKDIR /opt/ltx-worker

# Default: serverless mode. Set LTX_MODE=http for pod mode.
ENV LTX_MODE=serverless
CMD ["/bin/sh", "-c", "if [ \"$LTX_MODE\" = 'http' ]; then exec /opt/LTX-2/.venv/bin/python http_server.py; else exec /opt/LTX-2/.venv/bin/python handler.py; fi"]
