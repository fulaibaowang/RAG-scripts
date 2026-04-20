FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin:/usr/local/cuda/bin:${PATH}

WORKDIR /app

# -----
# System packages
# -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    gcc \
    g++ \
    make \
    git \
    curl \
    wget \
    ca-certificates \
    tini \
    procps \
    htop \
    less \
    vim-tiny \
    unzip \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    openjdk-21-jre-headless \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

COPY requirements-docker-pytorch.txt requirements-docker.txt ./

# -----
# PyTorch (CUDA 12.8 wheels) then pinned Python deps (see requirements-docker*.txt)
# -----
RUN python -m pip install -r requirements-docker-pytorch.txt

RUN python -m pip install -r requirements-docker.txt \
 && python - <<'PY'
import nltk
nltk.download("punkt_tab")
nltk.download("punkt")
print("nltk data ready")
PY

RUN python - <<'PY'
from FlagEmbedding import FlagLLMReranker
print("FlagEmbedding import ok:", FlagLLMReranker is not None)
PY

# -----
# Build-time import smoke test
# -----
RUN python - <<'PY'
import sys
import torch
import transformers
print('Python:', sys.version)
print('Torch:', torch.__version__)
print('Torch CUDA build:', torch.version.cuda)
print('Transformers:', transformers.__version__)
PY

ENTRYPOINT ["tini", "--"]
CMD ["bash", "-lc", "python - <<'PY'\nimport torch\nprint('torch', torch.__version__)\nprint('cuda available', torch.cuda.is_available())\nprint('cuda build', torch.version.cuda)\nprint('device count', torch.cuda.device_count())\nif torch.cuda.is_available():\n    print('device 0', torch.cuda.get_device_name(0))\n    print('capability', torch.cuda.get_device_capability(0))\nPY"]
