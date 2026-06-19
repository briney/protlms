# ProGen2 model image for the plms container contract.
#
# Build (small model, ~600 MB weights):
#   docker build --build-arg PROGEN2_CHECKPOINT=progen2-small -t plms-progen2:small containers/progen2
# Build (base model):
#   docker build --build-arg PROGEN2_CHECKPOINT=progen2-base -t plms-progen2:base containers/progen2
#
# Weights are baked in at build time; runtime requires no network access.
# The image runs on CPU by default and uses the GPU when launched with --gpus.
#
# NOTE: trust_remote_code=True is required because the hugohrban/progen2-*
# port ships a custom modeling_progen.py alongside the weights.
ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG PROGEN2_CHECKPOINT=progen2-small
ENV PROGEN2_CHECKPOINT=${PROGEN2_CHECKPOINT} \
    HF_HOME=/opt/hf-cache \
    PYTHONUNBUFFERED=1

# transformers for AutoModelForCausalLM + AutoConfig; tokenizers for the
# ProGen2-specific Tokenizer.from_pretrained() call (not AutoTokenizer).
RUN pip install --no-cache-dir "transformers==4.46.3" "tokenizers==0.20.3"

WORKDIR /app
COPY entrypoint.py /app/entrypoint.py

# Bake the checkpoint's weights into the image (populates the HF cache layer).
RUN python /app/entrypoint.py _prefetch

# Enforce offline weights at runtime for reproducibility.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "/app/entrypoint.py"]
