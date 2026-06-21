# ESM2 model image for the protlms container contract.
#
# Build (tiny demo model):
#   docker build --build-arg ESM2_CHECKPOINT=esm2_t6_8M -t protlms-esm2:t6_8M containers/esm2
# Build (standard workhorse):
#   docker build --build-arg ESM2_CHECKPOINT=esm2_t33_650M -t protlms-esm2:t33_650M containers/esm2
#
# Weights are baked in at build time, so runtime needs no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG ESM2_CHECKPOINT=esm2_t6_8M
ENV ESM2_CHECKPOINT=${ESM2_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "transformers==4.46.3"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

# Bake the checkpoint's weights into the image (populates the HF cache layer).
RUN python /app/entrypoint.py _prefetch

# Enforce offline weights at runtime for reproducibility.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
