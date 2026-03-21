
#################### the commands below run sam3dbody, converts it from mhr to smpl, before visualization
# Generic function: run_sam3d <cameras_space_separated> <dataset> <split> <subject> <action> <ext>
# Example:  run_sam3d "50591643 58860488" fit3d train s03 squat mp4
run_sam3d() {
  local cameras=($1)
  local dataset=$2
  local split=$3
  local subject=$4
  local action=$5
  local ext=${6:-mp4}

  local data_root=/home/haziq/datasets/mocap/data/$dataset/$split/$subject

  cd /home/haziq/sam-3d-body
  for camera in "${cameras[@]}"; do
    echo "========== Processing camera: $camera =========="

    python demo.py \
      --video_path $data_root/videos/$camera/$action.$ext \
      --output_folder $data_root/sam3d/$camera \
      --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
      --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
      --save_npz

    conda run -n mhr_new --cwd /home/haziq/MHR/tools/mhr_smpl_conversion python mhr_to_smpl.py \
      --mhr_path $data_root/sam3d/$camera/${action}_mhr_outputs.npz \
      --out_json $data_root/sam3d/$camera/${action}_smplx.json

    conda run -n mhr_new python /home/haziq/sam-3d-body/my_scripts/visualize_smplx.py \
      --video_path $data_root/videos/$camera/$action.$ext \
      --smplx_json $data_root/sam3d/$camera/${action}_smplx.json \
      --mhr_npz    $data_root/sam3d/$camera/${action}_mhr_outputs.npz \
      --out_video  $data_root/sam3d/$camera/${action}_smplx_vis.mp4 \
      --max_frames 100
  done
}

# fit3d — s03, squat, cameras: 50591643 58860488 60457274 65906101
# run_sam3d "50591643 58860488 60457274 65906101" fit3d train s03 squat mp4

# humaneva — S1, Jog_1, camera: C2
# source /home/haziq/sam-3d-body/run_telept.sh && run_sam3d "BW1 BW2 BW3 BW4 C1 C2 C3" humaneva train S1 Jog_1 avi

####################

# 00:21.744 left shoulder flexion
# 00:17.978 left shoulder extension
# 00:31.970 left elbow flexion
# 01:28.586 left shoulder abduction
# 01:38.357 left shoulder internal rotation
# 01:41.239 left shoulder external rotation
# python demo.py \
#   --video_path $HOME/datasets/telept/data/ipad/rgb_1764569430654.mp4 \
#   --video_timestamps \
#   --timestamps "00:21.744,00:17.978,00:31.970,01:38.357,01:28.586,01:38.357,01:41.239" \
#   --output_folder $HOME/datasets/telept/data/ipad/rgb_1764569430654 \
#   --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
#   --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
#   --detector_name sam3 \
#   --save_npz

# python demo.py \
#   --video_path $HOME/datasets/telept/data/ipad/rgb_1764569695903.mp4 \
#   --video_timestamps \
#   --timestamps "00:11.254,00:31.270,00:49.261,00:52.457" \
#   --output_folder $HOME/datasets/telept/data/ipad/rgb_1764569695903 \
#   --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
#   --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
#   --detector_name sam3 \
#   --save_npz

# python demo.py \
#   --video_path $HOME/datasets/telept/data/ipad/rgb_1764569971278.mp4 \
#   --video_timestamps \
#   --timestamps "00:20.347,00:22.695" \
#   --output_folder $HOME/datasets/telept/data/ipad/rgb_1764569971278 \
#   --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
#   --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
#   --detector_name sam3 \
#   --save_npz