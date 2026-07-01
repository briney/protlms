# Shared ESM masked-LM image for the protlms container contract.
#
# Serves ESM-1b and all ESM-2 sizes (same EsmForMaskedLM architecture) via
# HuggingFace transformers; the checkpoint is chosen by build args.
#
# Build (ESM-2 tiny demo):
#   docker build --build-arg ESM_HF_ID=facebook/esm2_t6_8M_UR50D \
#     --build-arg ESM_MODEL_NAME=esm2_t6_8M --build-arg ESM_MODEL_FAMILY=esm2 \
#     -t protlms-esm:t6_8M containers/esm
# Build (ESM-1b):
#   docker build --build-arg ESM_HF_ID=facebook/esm1b_t33_650M_UR50S \
#     --build-arg ESM_MODEL_NAME=esm1b_t33_650M --build-arg ESM_MODEL_FAMILY=esm1b \
#     -t protlms-esm:esm1b_650M containers/esm
#
# Weights are baked in at build time, so runtime needs no network access.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG ESM_HF_ID=facebook/esm2_t6_8M_UR50D
ARG ESM_MODEL_NAME=esm2_t6_8M
ARG ESM_MODEL_FAMILY=esm2
ENV ESM_HF_ID=${ESM_HF_ID} \
    ESM_MODEL_NAME=${ESM_MODEL_NAME} \
    ESM_MODEL_FAMILY=${ESM_MODEL_FAMILY} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir "transformers==4.46.3"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

RUN python /app/entrypoint.py _prefetch

ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
