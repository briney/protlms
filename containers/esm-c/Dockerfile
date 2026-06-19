# ESM-C model image for the plms container contract.
#
# Build (300M, default / CI):
#   docker build --build-arg ESMC_CHECKPOINT=esmc_300m -t plms-esm-c:300m containers/esm-c
# Build (600M):
#   docker build --build-arg ESMC_CHECKPOINT=esmc_600m -t plms-esm-c:600m containers/esm-c
#
# Weights are baked in at build time, so runtime needs no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.
#
# The esm package requires Python 3.12 (>=3.12,<3.13), so this base differs from
# the ESM2 image's pytorch base. flash-attn is intentionally not installed, so the
# SDK uses standard attention.

ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ARG ESMC_CHECKPOINT=esmc_300m
ENV ESMC_CHECKPOINT=${ESMC_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "esm==3.2.3"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

# Bake the checkpoint's weights into the image (populates the HF cache layer).
RUN python /app/entrypoint.py _prefetch

# Enforce offline weights at runtime for reproducibility.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
