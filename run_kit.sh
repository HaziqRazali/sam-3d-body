#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

KIT_ROOT="/media/haziq/Haziq/mocap/data/kit"
CHECKPOINT="./checkpoints/sam-3d-body-dinov3/model.ckpt"
MHR="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"

TEST_MODE="${TEST_MODE:-1}"

# Args: --shard K --num_shards N
SHARD=0
NUM_SHARDS=1
if [[ "${1:-}" == "--shard" && "${3:-}" == "--num_shards" ]]; then
  SHARD="$2"
  NUM_SHARDS="$4"
fi

echo "[START] $(date) GPU=${CUDA_VISIBLE_DEVICES:-unset} TEST_MODE=$TEST_MODE SHARD=$SHARD/$NUM_SHARDS"

# Build ONE deterministic, sorted list of videos across train+val and mp4+avi
mapfile -t VIDS < <(
  find "$KIT_ROOT" -type f \( \
      -path "*/train/*/videos/cam1/*.mp4" -o -path "*/train/*/videos/cam1/*.avi" -o \
      -path "*/val/*/videos/cam1/*.mp4"  -o -path "*/val/*/videos/cam1/*.avi"  \
    \) | sort
)

echo "[INFO] Total videos found: ${#VIDS[@]}"

if (( ${#VIDS[@]} == 0 )); then
  echo "[WARN] No videos matched. Check paths under $KIT_ROOT/*/(train|val)/*/videos/cam1/*.(mp4|avi)"
  exit 0
fi

for idx in "${!VIDS[@]}"; do
  vid="${VIDS[$idx]}"

  # shard filter
  if (( idx % NUM_SHARDS != SHARD )); then
    continue
  fi

  # split + sequence
  split="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$vid")")")")")"   # train|val
  seq="$(basename "$(dirname "$(dirname "$(dirname "$vid")")")")"                  # files_motions_xxxx

  base="$(basename "$vid")"
  base="${base%.*}"  # strip extension

  out_dir="$KIT_ROOT/$split/$seq/mhr/cam1"
  out_npz="$out_dir/${base}_mhr_outputs.npz"

  echo "================================================"
  echo "IDX   : $idx"
  echo "GPU   : ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "SPLIT : $split"
  echo "SEQ   : $seq"
  echo "VIDEO : $vid"
  echo "OUTPUT: $out_dir"
  echo "NPZ   : $out_npz"
  echo "================================================"

  if [[ -f "$out_npz" ]]; then
    echo "[SKIP] NPZ already exists"
    echo
    continue
  fi

  if [[ "$TEST_MODE" -eq 1 ]]; then
    echo "[TEST_MODE] Would run:"
    echo python demo.py \
      --video_path "$vid" \
      --output_folder "$out_dir" \
      --checkpoint_path "$CHECKPOINT" \
      --mhr_path "$MHR" \
      --save_npz
    echo
  else
    mkdir -p "$out_dir"
    python demo.py \
      --video_path "$vid" \
      --output_folder "$out_dir" \
      --checkpoint_path "$CHECKPOINT" \
      --mhr_path "$MHR" \
      --save_npz
  fi
done
