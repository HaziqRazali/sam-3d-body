#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SAM3D_PYTHON:-/home/haziq/anaconda3/envs/sam_3d_body/bin/python}"
WORKER="$SCRIPT_DIR/my_scripts/data_processing/run_on_mmpose_dataset.py"

if [[ ! -x "$PYTHON" ]]; then
  echo "[ERROR] SAM3D Python not found or not executable: $PYTHON" >&2
  echo "        Set SAM3D_PYTHON to the sam_3d_body environment's Python." >&2
  exit 2
fi

cd "$SCRIPT_DIR"
exec "$PYTHON" "$WORKER" "$@"
