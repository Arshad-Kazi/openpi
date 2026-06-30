#!/usr/bin/env bash
set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
# $1: config name (required)   e.g. pi05_collab_lora
# $2: experiment name          e.g. block_pick_123  (defaults to timestamp)
# $3+: extra args forwarded to train.py  e.g. --checkpoint-base-dir=... --resume
CONFIG_NAME="${1:?Usage: execute.sh <config_name> [exp_name] [extra train.py args...]}"
EXP_NAME="${2:-run_$(date +%Y%m%d_%H%M%S)}"
PYTHON=/.venv/bin/python

echo "OpenPI CHTC job: config=${CONFIG_NAME}  exp=${EXP_NAME}"
nvidia-smi || true

# ── Install project packages ───────────────────────────────────────────────────
# HTCondor transfers the openpi source directory from the access point (git pull).
# All heavy deps are already baked into the image — only install the project itself.
CODE_DIR="${_CONDOR_SCRATCH_DIR:-.}/openpi"
echo "Installing openpi from: $CODE_DIR"
/.venv/bin/pip install --no-deps -q \
    -e "$CODE_DIR" \
    -e "$CODE_DIR/packages/openpi-client"

# ── Cache directories ─────────────────────────────────────────────────────────
export HF_HOME="${_CONDOR_SCRATCH_DIR:-.}/.cache/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export OPENPI_DATA_HOME="${_CONDOR_SCRATCH_DIR:-.}/.cache/openpi"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$OPENPI_DATA_HOME"

# ── Tokenizer ─────────────────────────────────────────────────────────────────
# PaliGemma tokenizer is baked into the image at /opt/openpi-cache.
# Copy it to scratch so train.py can locate it without hitting GCS.
if [ -d /opt/openpi-cache ]; then
    cp -rn /opt/openpi-cache/* "$OPENPI_DATA_HOME/" 2>/dev/null || true
fi
# Fail fast — a missing tokenizer causes an opaque GCS 401 error later.
TOKENIZER_PATH="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
if [ ! -f "$TOKENIZER_PATH" ]; then
    echo "ERROR: Missing tokenizer: $TOKENIZER_PATH" >&2
    exit 2
fi

# ── Dataset ───────────────────────────────────────────────────────────────────
# HTCondor transfers the dataset tarball(s) from staging into scratch.
# Extract each *_dataset.tar.gz so lerobot can find them under HF_LEROBOT_HOME.
export HF_LEROBOT_HOME="${_CONDOR_SCRATCH_DIR:-.}/lerobot_data"
mkdir -p "$HF_LEROBOT_HOME"
for tarball in *_dataset.tar.gz; do
    [ -f "$tarball" ] || continue
    echo "Extracting dataset: $tarball"
    tar -xzf "$tarball" -C "$HF_LEROBOT_HOME"
    rm -f "$tarball"
done
echo "Datasets available under $HF_LEROBOT_HOME:"
ls "$HF_LEROBOT_HOME/local/" 2>/dev/null || echo "(none found)"

# ── JAX / SSL ─────────────────────────────────────────────────────────────────
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
# Use certifi's CA bundle — the container's system certs may be stale or missing.
export SSL_CERT_FILE=$($PYTHON -c "import certifi; print(certifi.where())")

# ── W&B ───────────────────────────────────────────────────────────────────────
# Passed from the submit node via `getenv = WANDB_API_KEY` in the .sub file.
# Fallback: read from wandb_api_key.txt if it was transferred alongside the job.
if [ -z "${WANDB_API_KEY:-}" ] && [ -f wandb_api_key.txt ]; then
    export WANDB_API_KEY=$(cat wandb_api_key.txt)
fi

# ── Normalization statistics ───────────────────────────────────────────────────
echo "Computing normalization statistics for: $CONFIG_NAME"
$PYTHON "$CODE_DIR/scripts/compute_norm_stats.py" --config-name "$CONFIG_NAME"

# ── Training ──────────────────────────────────────────────────────────────────
echo "Starting training: config=$CONFIG_NAME  exp=$EXP_NAME"
$PYTHON "$CODE_DIR/scripts/train.py" "$CONFIG_NAME" \
    --exp-name="$EXP_NAME" \
    "${@:3}"
