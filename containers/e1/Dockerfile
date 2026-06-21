# Profluent-E1 model image for the protlms container contract (single-sequence).
#
# Build (150m, default / CI):
#   docker build --build-arg E1_CHECKPOINT=E1-150m -t protlms-e1:150m containers/e1
# Build (300m / 600m):
#   docker build --build-arg E1_CHECKPOINT=E1-300m -t protlms-e1:300m containers/e1
#   docker build --build-arg E1_CHECKPOINT=E1-600m -t protlms-e1:600m containers/e1
#
# Weights are baked in at build time, so runtime needs no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.
#
# The E1 package requires Python >=3.12 (the official pytorch images ship 3.11),
# so this base is python:3.12-slim and torch is pulled transitively by `pip
# install E1`. flash-attn is intentionally not installed; E1 falls back to
# flex_attention.

ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ARG E1_CHECKPOINT=E1-150m
ARG E1_REF=bfd2620a602248499f3d2583d85a7ecddf0b6e02
ENV E1_CHECKPOINT=${E1_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

# git is needed for the VCS install; build-essential covers any sdist deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "E1 @ git+https://github.com/Profluent-AI/E1.git@${E1_REF}"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

# Bake the checkpoint's weights into the image (populates the HF cache layer).
RUN python /app/entrypoint.py _prefetch

# Enforce offline weights at runtime for reproducibility.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
