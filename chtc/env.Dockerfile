# Environment-only image for OpenPI training on CHTC.
# Does NOT bake source code — code is transferred by HTCondor at runtime.
# This mirrors the Infinity/pixi pattern: image = deps only.
#
# Build:
#   docker build -t openpi_env -f chtc/env.Dockerfile .
#
# Push to DockerHub:
#   docker tag openpi_env <dockerhub_user>/openpi_env:latest
#   docker push <dockerhub_user>/openpi_env:latest

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential clang curl \
    ca-certificates \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

# Copy from cache instead of linking (container filesystem)
ENV UV_LINK_MODE=copy
# Venv at a fixed path so run_train.sh can reference /.venv/bin/python
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

# Install Python and all project dependencies from the lockfile.
# --no-install-project: skip installing openpi itself (code not in image).
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev && \
    chmod -R a+rwX $UV_PROJECT_ENVIRONMENT

# Bundle PaliGemma tokenizer (GCS bucket is not publicly accessible).
COPY chtc/assets/paligemma_tokenizer.model /opt/openpi-cache/big_vision/paligemma_tokenizer.model
RUN chmod -R a+rwX /opt/openpi-cache

ENV PATH="/.venv/bin:$PATH"

CMD ["/bin/bash"]
