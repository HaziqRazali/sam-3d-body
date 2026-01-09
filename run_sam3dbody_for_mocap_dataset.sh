#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Qianli 
# CUDA_VISIBLE_DEVICES=0 FORCE=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /home/haziq/datasets/mocap/data/fit3d/ --shard 0 --num_shards 4 --ignore-cams "50591643,58860488" 2>&1 | tee fit3d_shard0_part0.txt
# CUDA_VISIBLE_DEVICES=1 FORCE=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /home/haziq/datasets/mocap/data/fit3d/ --shard 1 --num_shards 4 --ignore-cams "50591643,58860488" 2>&1 | tee fit3d_shard1_part0.txt
# CUDA_VISIBLE_DEVICES=2 FORCE=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /home/haziq/datasets/mocap/data/fit3d/ --shard 2 --num_shards 4 --ignore-cams "50591643,58860488" 2>&1 | tee fit3d_shard2_part0.txt
# CUDA_VISIBLE_DEVICES=3 FORCE=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /home/haziq/datasets/mocap/data/fit3d/ --shard 3 --num_shards 4 --ignore-cams "50591643,58860488" 2>&1 | tee fit3d_shard3_part0.txt

# Ali
# CUDA_VISIBLE_DEVICES=0 FORCE=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/sc3d/ --shard 0 --num_shards 2 --ignore-cams "50591643,58860488" 2>&1 | tee sc3d_shard0_part0.txt
# CUDA_VISIBLE_DEVICES=1 FORCE=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/sc3d/ --shard 1 --num_shards 2 --ignore-cams "50591643,58860488" 2>&1 | tee sc3d_shard1_part0.txt

# Cheston
# CUDA_VISIBLE_DEVICES=0 FORCE=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/humaneva/ --shard 0 --num_shards 3 --ignore-cams "BW1,BW2,C2" 2>&1 | tee humaneva_shard0_part0.txt
# CUDA_VISIBLE_DEVICES=1 FORCE=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/humaneva/ --shard 1 --num_shards 3 --ignore-cams "BW1,BW2,C2" 2>&1 | tee humaneva_shard1_part0.txt
# CUDA_VISIBLE_DEVICES=2 FORCE=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/humaneva/ --shard 2 --num_shards 3 --ignore-cams "BW1,BW2,C2" 2>&1 | tee humaneva_shard2_part0.txt

# CUDA_VISIBLE_DEVICES=0 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/self --shard 0 --num_shards 2 2>&1 | tee self_shard0_part0.txt
# CUDA_VISIBLE_DEVICES=1 TEST_MODE=0 ./run_sam3dbody_for_mocap_dataset.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/self --shard 1 --num_shards 2 2>&1 | tee self_shard1_part0.txt

# Default (can be overridden via --DATA_ROOT or env var DATA_ROOT)
DATA_ROOT="${DATA_ROOT:-/media/haziq/Haziq/mocap/data/kit}"
CHECKPOINT="./checkpoints/sam-3d-body-dinov3/model.ckpt"
MHR="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"

TEST_MODE="${TEST_MODE:-1}"

# Optional: --DATA_ROOT /path/to/kit
if [[ "${1:-}" == "--DATA_ROOT" ]]; then
  DATA_ROOT="$2"
  shift 2
fi

# Args: --shard K --num_shards N
SHARD=0
NUM_SHARDS=1
if [[ "${1:-}" == "--shard" && "${3:-}" == "--num_shards" ]]; then
  SHARD="$2"
  NUM_SHARDS="$4"
  shift 4
fi

# Args: --ignore-cams "cam1,cam2,cam3"
IGNORE_CAMS=""
if [[ "${1:-}" == "--ignore-cams" ]]; then
  IGNORE_CAMS="$2"
  shift 2
fi

echo "[START] $(date) GPU=${CUDA_VISIBLE_DEVICES:-unset} TEST_MODE=$TEST_MODE DATA_ROOT=$DATA_ROOT SHARD=$SHARD/$NUM_SHARDS"

# Build ONE deterministic, sorted list of videos across train+val and mp4+avi
# Expected layout example:
#   $DATA_ROOT/train/<seq>/videos/<cam_name>/<video>.{mp4,avi}
#   $DATA_ROOT/val/<seq>/videos/<cam_name>/<video>.{mp4,avi}
mapfile -t VIDS < <(
  find "$DATA_ROOT" -type f \( \
      -path "*/train/*/videos/*/*.mp4" -o -path "*/train/*/videos/*/*.avi" -o \
      -path "*/val/*/videos/*/*.mp4"  -o -path "*/val/*/videos/*/*.avi"  \
    \) | sort
)

echo "[INFO] Total videos found: ${#VIDS[@]}"

if (( ${#VIDS[@]} == 0 )); then
  echo "[WARN] No videos matched. Check paths under $DATA_ROOT/(train|val)/*/videos/*/*.(mp4|avi)"
  exit 0
fi

for idx in "${!VIDS[@]}"; do
  vid="${VIDS[$idx]}"

  # split + sequence + camera
  # For: .../$split/$seq/videos/$cam/$video
  split="$(basename "$(dirname "$(dirname "$(dirname "$(dirname "$vid")")")")")"  # train|val
  seq="$(basename "$(dirname "$(dirname "$(dirname "$vid")")")")"                 # e.g., haziq
  cam="$(basename "$(dirname "$vid")")"                                          # e.g., laptop_webcam

  # shard filter
  if (( idx % NUM_SHARDS != SHARD )); then
    continue
  fi

  # camera ignore filter
  if [[ -n "$IGNORE_CAMS" ]]; then
    for ignore_cam in $(echo "$IGNORE_CAMS" | tr ',' ' '); do
      if [[ "$cam" == "$ignore_cam" ]]; then
        echo "[SKIP] Camera ignored: $cam"
        echo
        continue 2  # continue outer for loop
      fi
    done
  fi

  base="$(basename "$vid")"
  base="${base%.*}"  # strip extension

  out_dir="$DATA_ROOT/$split/$seq/mhr/$cam"
  out_npz="$out_dir/${base}_mhr_outputs.npz"

  echo "================================================"
  echo "IDX   : $idx"
  echo "GPU   : ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "SPLIT : $split"
  echo "SEQ   : $seq"
  echo "CAM   : $cam"
  echo "VIDEO : $vid"
  echo "OUTPUT: $out_dir"
  echo "NPZ   : $out_npz"
  echo "================================================"

  FORCE="${FORCE:-0}"
  if [[ -f "$out_npz" && "$FORCE" -ne 1 ]]; then
    echo "[SKIP] NPZ already exists (use FORCE=1 to overwrite)"
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
