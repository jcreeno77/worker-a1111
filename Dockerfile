# ---------------------------------------------------------------------------- #
#                         Stage 1: Download the models                         #
# ---------------------------------------------------------------------------- #
FROM alpine/git:2.43.0 as download

# NOTE: CivitAI usually requires an API token, so you need to add it in the header
#       of the wget command if you're using a model from CivitAI.
RUN apk add --no-cache wget curl && \
    wget -q -O /model.safetensors "https://huggingface.co/Lykon/dreamshaper-xl-lightning/resolve/main/DreamShaperXL_Lightning.safetensors" && \
    wget -q -O /TLRS_Style.safetensors "https://get-lora.s3.us-east-1.amazonaws.com/TLRS_Style.safetensors"

# ---------------------------------------------------------------------------- #
#                        Stage 2: Build the final image                        #
# ---------------------------------------------------------------------------- #
FROM python:3.10.14-slim as build_final_image

ARG A1111_RELEASE=v1.9.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    ROOT=/stable-diffusion-webui \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget libgoogle-perftools-dev libtcmalloc-minimal4 procps libgl1 libglib2.0-0 && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y
    
COPY --from=download /TLRS_Style.safetensors /TLRS_Style.safetensors
COPY --from=download /model.safetensors /model.safetensors

RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git && \
    cd stable-diffusion-webui && \
    git reset --hard ${A1111_RELEASE} && \
    pip install xformers && \
    pip install -r requirements_versions.txt && \
    git clone https://github.com/dimitribarbot/sd-webui-birefnet.git extensions/sd-webui-birefnet && \
    mkdir -p models/Lora && \
    mkdir -p models/Stable-diffusion && \
    cd .. && \
    cp TLRS_Style.safetensors stable-diffusion-webui/models/Lora/ && \
    cp model.safetensors stable-diffusion-webui/models/Stable-diffusion/ && \
    cd stable-diffusion-webui && \
    python -c "from launch import prepare_environment; prepare_environment()" --skip-torch-cuda-test

# install dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

COPY test_input.json .

ADD src .

RUN chmod +x /start.sh
CMD /start.sh
