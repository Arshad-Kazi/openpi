#!/bin/bash
set -euo pipefail

# Usage: ./chtc/stage_data.sh <netid> <dataset_name_under_local> [dataset_dir]
#
# <dataset_name_under_local> is the name the dataset takes on CHTC: execute.sh
# unpacks the tarball into $HF_LEROBOT_HOME and lerobot resolves it as
# local/<dataset_name_under_local>.
#
# [dataset_dir] optionally points at a LeRobot dataset living outside the local
# HF cache (e.g. one written with merge_robot_human_to_lerobot.py --save_dir).
# The stored paths are rewritten so the tarball still unpacks to
# local/<dataset_name_under_local>, which is what chtc/execute.sh expects.
NETID="${1:?Usage: stage_data.sh <netid> <dataset_name_under_local> [dataset_dir]}"
DATASET_NAME="${2:?Usage: stage_data.sh <netid> <dataset_name_under_local> [dataset_dir]}"
DATASET_DIR_OVERRIDE="${3:-}"

LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
REPO_ID="local/${DATASET_NAME}"

TRANSFER_HOST="transfer.chtc.wisc.edu"
GROUP_STAGING_DIR="/staging/aakazi/"
STAGED_TARBALL_NAME="${DATASET_NAME}_dataset.tar.gz"
TARBALL="${TMPDIR:-/tmp}/${STAGED_TARBALL_NAME}"

if [ -n "$DATASET_DIR_OVERRIDE" ]; then
  DATASET_DIR="${DATASET_DIR_OVERRIDE%/}"
else
  DATASET_DIR="${LEROBOT_HOME}/${REPO_ID}"
fi

if [ ! -d "$DATASET_DIR" ]; then
  echo "ERROR: dataset directory not found: $DATASET_DIR"
  exit 1
fi
if [ ! -f "${DATASET_DIR}/meta/info.json" ]; then
  echo "ERROR: not a LeRobot dataset (no meta/info.json): $DATASET_DIR"
  exit 1
fi

echo "Creating tarball from: $DATASET_DIR"
# The archive must hold local/<name>/... at its root, because execute.sh
# extracts it straight into $HF_LEROBOT_HOME. When the source lives elsewhere
# on disk, rewrite the stored prefix so the layout on CHTC is identical either
# way. (Note: a symlink into the cache would NOT work here -- tar stores
# symlinks as symlinks unless -h is given.)
tar -czf "$TARBALL" \
  -C "$(dirname "$DATASET_DIR")" \
  --transform "s|^$(basename "$DATASET_DIR")|${REPO_ID}|" \
  "$(basename "$DATASET_DIR")"

echo "Tarball size: $(du -sh "$TARBALL" | cut -f1)"
echo "Archive root: $(tar -tzf "$TARBALL" | head -n 1)"

echo "Uploading to ${TRANSFER_HOST}:${GROUP_STAGING_DIR}/${STAGED_TARBALL_NAME}"
scp "$TARBALL" "${NETID}@${TRANSFER_HOST}:${GROUP_STAGING_DIR}/${STAGED_TARBALL_NAME}"
rm -f "$TARBALL"

echo "Done."
