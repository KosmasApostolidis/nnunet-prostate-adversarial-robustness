# Runtime image for the LSP6 segmentation service.  One image per WG/Zones model
# combination, selected at build time (see scripts/build_images.sh):
#   dimzaridis/faith_lsp6:wg-v1-zones-v2, :wg-v2-zones-v1, :wg-v2-zones-v2
# (v1/v1 is the original dimzaridis/faith_lsp6:0.1 and is not rebuilt).
# CUDA 12.8 + cuDNN runtime on Ubuntu 22.04 (system Python 3.10), torch cu128 wheels.
# Build with BuildKit (default since Docker 23): the .dockerignore re-includes under
# nnUnet_paths/** rely on its walker; the legacy builder may prune the directory.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python-is-python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Code, Utils shims and __main__.py resolve paths relative to /, as before.
WORKDIR /

# Dependencies first so code or checkpoint changes do not re-install them.
COPY requirements.txt .
RUN python -m pip install \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

# Everything .dockerignore lets through: src/, Utils/, __main__.py, pyproject.toml
# and the four nnU-Net model folders (v1 fold_0 + v2 fold_all, WG and Zones).
COPY . .

# Editable install so the console scripts (mri-segmentor, ...) resolve; deps are present.
RUN python -m pip install --no-deps -e .

# Define mountable directories
VOLUME ["/Pats", "/Outputs", "/dicom_outputs"]

# Model versions baked into this image (see mri_prostate_seg.models.nnunet_call);
# the LSP6 app also passes the same values as environment, which takes precedence.
ARG WG_MODEL=v1
ARG ZONES_MODEL=v1
ENV WG_MODEL=${WG_MODEL} \
    ZONES_MODEL=${ZONES_MODEL}

# Run model.py when the container launches
CMD ["python", "./__main__.py"]
