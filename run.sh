#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

#CUDA_VISIBLE_DEVICES=0 FORCE=0 TEST_MODE=1 ./run.sh --DATA_ROOT /data/haziq/mocap/data/brett

#CUDA_VISIBLE_DEVICES=0 FORCE=0 TEST_MODE=1 ./run.sh --DATA_ROOT /data/haziq/mocap/data/laptop_webcam --shard 0 --num_shards 3
#CUDA_VISIBLE_DEVICES=0 FORCE=0 TEST_MODE=1 ./run.sh --DATA_ROOT /data/haziq/mocap/data/laptop_webcam --shard 1 --num_shards 3
#CUDA_VISIBLE_DEVICES=0 FORCE=0 TEST_MODE=1 ./run.sh --DATA_ROOT /data/haziq/mocap/data/laptop_webcam --shard 2 --num_shards 3

#CUDA_VISIBLE_DEVICES=0 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /home/haziq/datasets/mocap/data/basler_2026_03_20/ --shard 0 --num_shards 3 2>&1 | tee basler_2026_03_20_shard0_part0.txt
#CUDA_VISIBLE_DEVICES=1 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /home/haziq/datasets/mocap/data/basler_2026_03_20/ --shard 1 --num_shards 3 2>&1 | tee basler_2026_03_20_shard1_part0.txt
#CUDA_VISIBLE_DEVICES=2 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /home/haziq/datasets/mocap/data/basler_2026_03_20/ --shard 2 --num_shards 3 2>&1 | tee basler_2026_03_20_shard2_part0.txt

#CUDA_VISIBLE_DEVICES=0 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /home/haziq/datasets/mocap/data/basler_2026_03_21/ --shard 0 --num_shards 3 2>&1 | tee basler_2026_03_21_shard0_part0.txt
#CUDA_VISIBLE_DEVICES=1 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /home/haziq/datasets/mocap/data/basler_2026_03_21/ --shard 1 --num_shards 3 2>&1 | tee basler_2026_03_21_shard1_part0.txt
#CUDA_VISIBLE_DEVICES=2 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /home/haziq/datasets/mocap/data/basler_2026_03_21/ --shard 2 --num_shards 3 2>&1 | tee basler_2026_03_21_shard2_part0.txt

# Qianli 
# CUDA_VISIBLE_DEVICES=0 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /data/haziq/mocap/data/fit3d/ --shard 0 --num_shards 4 --ignore-cams "50591643,58860488" 2>&1 | tee fit3d_shard0_part0.txt
# CUDA_VISIBLE_DEVICES=1 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /data/haziq/mocap/data/fit3d/ --shard 1 --num_shards 4 --ignore-cams "50591643,58860488" 2>&1 | tee fit3d_shard1_part0.txt
# CUDA_VISIBLE_DEVICES=2 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /data/haziq/mocap/data/fit3d/ --shard 2 --num_shards 4 --ignore-cams "50591643,58860488" 2>&1 | tee fit3d_shard2_part0.txt
# CUDA_VISIBLE_DEVICES=3 FORCE=0 TEST_MODE=0 ./run.sh --DATA_ROOT /data/haziq/mocap/data/fit3d/ --shard 3 --num_shards 4 --ignore-cams "50591643,58860488" 2>&1 | tee fit3d_shard3_part0.txt

# Ali
# CUDA_VISIBLE_DEVICES=0 FORCE=1 TEST_MODE=0 ./run.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/sc3d/ --shard 0 --num_shards 2 --ignore-cams "50591643,58860488" 2>&1 | tee sc3d_shard0_part0.txt
# CUDA_VISIBLE_DEVICES=1 FORCE=1 TEST_MODE=0 ./run.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/sc3d/ --shard 1 --num_shards 2 --ignore-cams "50591643,58860488" 2>&1 | tee sc3d_shard1_part0.txt

# Cheston
# CUDA_VISIBLE_DEVICES=0 FORCE=1 TEST_MODE=0 ./run.sh --DATA_ROOT /home/haziq/datasets/mocap/data/humaneva/ --shard 0 --num_shards 3 --ignore-cams "BW1,BW2,C2" 2>&1 | tee humaneva_shard0_part0.txt
# CUDA_VISIBLE_DEVICES=1 FORCE=1 TEST_MODE=0 ./run.sh --DATA_ROOT /home/haziq/datasets/mocap/data/humaneva/ --shard 1 --num_shards 3 --ignore-cams "BW1,BW2,C2" 2>&1 | tee humaneva_shard1_part0.txt
# CUDA_VISIBLE_DEVICES=2 FORCE=1 TEST_MODE=0 ./run.sh --DATA_ROOT /home/haziq/datasets/mocap/data/humaneva/ --shard 2 --num_shards 3 --ignore-cams "BW1,BW2,C2" 2>&1 | tee humaneva_shard2_part0.txt

# CUDA_VISIBLE_DEVICES=0 TEST_MODE=0 ./run.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/self --shard 0 --num_shards 2 2>&1 | tee self_shard0_part0.txt
# CUDA_VISIBLE_DEVICES=1 TEST_MODE=0 ./run.sh --DATA_ROOT /media/haziq/Haziq/mocap/data/self --shard 1 --num_shards 2 2>&1 | tee self_shard1_part0.txt

# Default (can be overridden via --DATA_ROOT or env var DATA_ROOT)
DATA_ROOT="${DATA_ROOT:-/media/haziq/Haziq/mocap/data/kit}"
CHECKPOINT="./checkpoints/sam-3d-body-dinov3/model.ckpt"
MHR="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
MHR_TO_SMPL_PY="/home/haziq/MHR/tools/mhr_smpl_conversion/mhr_to_smpl.py"
VISUALIZE_PY="/home/haziq/sam-3d-body/my_scripts/visualize_smplx.py"
VIDEO_CODEC="${VIDEO_CODEC:-mp4v}"

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

  out_dir="$DATA_ROOT/$split/$seq/sam3d/$cam"
  out_npz="$out_dir/${base}_mhr_outputs.npz"
  out_rendered="$out_dir/${base}_rendered.mp4"
  out_json="$out_dir/${base}_smplx.json"
  out_vis="$out_dir/${base}_smplx_vis.mp4"

  echo "================================================"
  echo "IDX   : $idx"
  echo "GPU   : ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "SPLIT : $split"
  echo "SEQ   : $seq"
  echo "CAM   : $cam"
  echo "VIDEO : $vid"
  echo "OUTPUT: $out_dir"
  echo "NPZ      : $out_npz"
  echo "RENDERED : $out_rendered"
  echo "JSON     : $out_json"
  echo "VIS      : $out_vis"
  echo "================================================"

  FORCE="${FORCE:-0}"

  # ── Step 1: SAM-3D-Body inference ──────────────────────────────────────────
  if [[ -f "$out_npz" && -f "$out_rendered" && "$FORCE" -ne 1 ]]; then
    echo "[SKIP] NPZ + rendered video already exist (use FORCE=1 to overwrite)"
  elif [[ "$TEST_MODE" -eq 1 ]]; then
    echo "[TEST_MODE] Would run:"
    echo "  python demo.py \\"
    echo "    --video_path \"$vid\" \\"
    echo "    --output_folder \"$out_dir\" \\"
    echo "    --checkpoint_path \"$CHECKPOINT\" \\"
    echo "    --mhr_path \"$MHR\" \\"
    echo "    --save_npz \\"
    echo "    --video_codec \"$VIDEO_CODEC\""
  else
    if [[ -f "$out_npz" && ! -f "$out_rendered" ]]; then
      echo "[INFO] NPZ exists but rendered video missing – re-running Step 1 to regenerate"
    fi
    mkdir -p "$out_dir"
    python demo.py \
      --video_path "$vid" \
      --output_folder "$out_dir" \
      --checkpoint_path "$CHECKPOINT" \
      --mhr_path "$MHR" \
      --save_npz \
      --video_codec "$VIDEO_CODEC"
  fi

  # ── Step 2: MHR → SMPL-X conversion ───────────────────────────────────────
  if [[ -f "$out_json" && "$FORCE" -ne 1 ]]; then
    echo "[SKIP] SMPL-X JSON already exists (use FORCE=1 to overwrite)"
  elif [[ "$TEST_MODE" -eq 1 ]]; then
    echo "[TEST_MODE] Would run:"
    echo "  conda run -n mhr_new python $MHR_TO_SMPL_PY \\"
    echo "    --mhr_path \"$out_npz\" \\"
    echo "    --out_json \"$out_json\""
  elif [[ ! -f "$out_npz" ]]; then
    echo "[WARN] NPZ not found – skipping MHR→SMPL-X conversion"
  else
    conda run -n mhr_new python "$MHR_TO_SMPL_PY" \
      --mhr_path "$out_npz" \
      --out_json "$out_json"
  fi

  # ── Step 3: Visualize SMPL-X overlay ──────────────────────────────────────
  if [[ -f "$out_vis" && "$FORCE" -ne 1 ]]; then
    echo "[SKIP] Visualization video already exists (use FORCE=1 to overwrite)"
  elif [[ "$TEST_MODE" -eq 1 ]]; then
    echo "[TEST_MODE] Would run:"
    echo "  conda run -n sam_3d_body python $VISUALIZE_PY \\"
    echo "    --video_path \"$vid\" \\"
    echo "    --smplx_json \"$out_json\" \\"
    echo "    --out_video  \"$out_vis\""
  elif [[ ! -f "$out_json" ]]; then
    echo "[WARN] SMPL-X JSON not found – skipping visualization"
  else
    conda run -n sam_3d_body python "$VISUALIZE_PY" \
      --video_path "$vid" \
      --smplx_json "$out_json" \
      --out_video  "$out_vis"
  fi
done
