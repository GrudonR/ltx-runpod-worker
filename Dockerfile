FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
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

RUN git clone https://github.com/Lightricks/LTX-2.git /opt/LTX-2 \
    && cd /opt/LTX-2 \
    && git checkout ${LTX_REPO_COMMIT}

WORKDIR /opt/LTX-2

RUN uv sync --frozen --no-dev \
    && uv pip install --python .venv/bin/python --no-cache-dir runpod huggingface_hub requests

COPY handler.py /opt/ltx-worker/handler.py

WORKDIR /opt/ltx-worker

CMD ["/opt/LTX-2/.venv/bin/python", "handler.py"]
