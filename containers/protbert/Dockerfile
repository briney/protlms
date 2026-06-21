# ProtBERT model image for the protlms container contract.
#
# Build (UniRef100 / CI default):
#   docker build --build-arg PROTBERT_CHECKPOINT=prot_bert -t protlms-protbert:uniref100 containers/protbert
# Build (BFD):
#   docker build --build-arg PROTBERT_CHECKPOINT=prot_bert_bfd -t protlms-protbert:bfd containers/protbert
#
# Weights are baked in at build time, so runtime needs no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG PROTBERT_CHECKPOINT=prot_bert
ENV PROTBERT_CHECKPOINT=${PROTBERT_CHECKPOINT} \
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
