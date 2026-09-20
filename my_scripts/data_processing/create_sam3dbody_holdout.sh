#!/usr/bin/env bash
set -euo pipefail

: <<'USAGE'
Create a non-destructive, SAM-3D-Body-only grouped train/val view.

Default: 70:30 holdout preview/materialization

    # Preview counts; source and output trees are unchanged.
    DRY_RUN=1 bash create_sam3dbody_holdout.sh

    # Create /data/haziq/sam3dbody/splits/sam3dbody_70_30.
    bash create_sam3dbody_holdout.sh

    # Create another independent view; the 70:30 view is untouched.
    VAL_RATIO=0.40 bash create_sam3dbody_holdout.sh

    # Choose an explicit output directory.
    VAL_RATIO=0.50 OUTPUT_ROOT=/data/haziq/sam3dbody/splits/sam3dbody_50_50 \
        bash create_sam3dbody_holdout.sh

Environment overrides:
    VAL_RATIO              validation row ratio (default: 0.40)
    SEED                   deterministic group-order seed (default: 20260825)
    OUTPUT_ROOT            derived view path (default: sam3dbody_<train>_<val>)
    DRY_RUN                set to 1 to scan and print counts only
    SOURCE_ROOT            official SAM-3D-Body converted source root
    SPLIT_ROOT             parent directory for derived views
    PYTHON_BIN             Python interpreter with pyarrow installed
    DATASETS               comma-separated SAM-3D dataset names
    MMPOSE_MODEL           MMPose output directory name
USAGE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAL_RATIO="${VAL_RATIO:-0.30}"
SEED="${SEED:-20260825}"
DRY_RUN="${DRY_RUN:-0}"
SOURCE_ROOT="${SOURCE_ROOT:-/data/haziq/sam3dbody}"
SPLIT_ROOT="${SPLIT_ROOT:-$SOURCE_ROOT/splits}"
PYTHON_BIN="${PYTHON_BIN:-/home/haziq/anaconda3/envs/pytorch_env_cu128/bin/python}"
DATASETS="${DATASETS:-coco,mpii,aic,sa1b,3dpw,harmony4d,egohumans}"
MMPOSE_MODEL="${MMPOSE_MODEL:-rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122}"

if ! awk -v ratio="$VAL_RATIO" 'BEGIN { exit !(ratio > 0 && ratio < 1) }'; then
    echo "ERROR: VAL_RATIO must be strictly between 0 and 1: $VAL_RATIO" >&2
    exit 1
fi
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "ERROR: DRY_RUN must be 0 or 1: $DRY_RUN" >&2
    exit 1
fi

VAL_PERCENT="$(awk -v ratio="$VAL_RATIO" 'BEGIN { printf "%.0f", ratio * 100 }')"
TRAIN_PERCENT=$((100 - VAL_PERCENT))
OUTPUT_ROOT="${OUTPUT_ROOT:-$SPLIT_ROOT/sam3dbody_${TRAIN_PERCENT}_${VAL_PERCENT}}"
BUILDER="$SCRIPT_DIR/create_grouped_sam3dbody_split.py"

for required_path in "$SOURCE_ROOT" "$BUILDER"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required path does not exist: $required_path" >&2
        exit 1
    fi
done
if [[ "$PYTHON_BIN" == */* ]]; then
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "ERROR: Python interpreter is not executable: $PYTHON_BIN" >&2
        exit 1
    fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter was not found: $PYTHON_BIN" >&2
    exit 1
fi

args=(
    "$BUILDER"
    --source-root "$SOURCE_ROOT"
    --output-root "$OUTPUT_ROOT"
    --datasets "$DATASETS"
    --val-ratio "$VAL_RATIO"
    --seed "$SEED"
    --mmpose-model "$MMPOSE_MODEL"
)
if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
fi

echo "Source root:      $SOURCE_ROOT"
echo "Output root:      $OUTPUT_ROOT"
echo "Validation ratio: $VAL_RATIO"
echo "Seed:              $SEED"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "Mode:              dry run"
else
    echo "Mode:              materialize relative symlinks"
fi

exec "$PYTHON_BIN" -u "${args[@]}"
