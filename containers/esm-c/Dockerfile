# ESM-C model image for the protlms container contract.
#
# ESM-C via HuggingFace transformers, using the MIT-licensed biohub ESM-C weights.
# The biohub `esm` package registers the ESMCForMaskedLM architecture with
# transformers' auto classes (native transformers has no `esmc` support).
#
# Build (300M, default / CI):
#   docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t protlms-esm-c:300m containers/esm-c
# Build (600M / 6B):
#   docker build --build-arg ESMC_CHECKPOINT=esmc_600m -t protlms-esm-c:600m containers/esm-c
#   docker build --build-arg ESMC_CHECKPOINT=esmc_6b   -t protlms-esm-c:6b   containers/esm-c
#
# Weights are baked in at build time, so runtime needs no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.

ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

# Pinned biohub esm package commit (no PyPI release yet). Resolve via:
#   git ls-remote https://github.com/Biohub/esm.git main
ARG ESM_REF=cf002f1d2fb9220a0918fc8c5ea4e3b56d28fbaf
ARG ESMC_CHECKPOINT=esmc_300m
ENV ESMC_CHECKPOINT=${ESMC_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

# git + build tools for the git-based esm install; esm pulls torch + transformers (~4.57.x).
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "esm @ git+https://github.com/Biohub/esm.git@${ESM_REF}"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

# Bake the checkpoint's weights into the image (populates the HF cache layer).
RUN python /app/entrypoint.py _prefetch

# Enforce offline weights at runtime for reproducibility.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
