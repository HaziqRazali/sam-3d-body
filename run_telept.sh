


# 00:21.744 left shoulder flexion
# 00:17.978 left shoulder extension
# 00:31.970 left elbow flexion
# 01:28.586 left shoulder abduction
# 01:38.357 left shoulder internal rotation
# 01:41.239 left shoulder external rotation
python demo.py \
  --video_path $HOME/datasets/telept/data/ipad/rgb_1764569430654.mp4 \
  --video_timestamps \
  --timestamps "00:21.744,00:17.978,00:31.970,01:38.357,01:28.586,01:38.357,01:41.239" \
  --output_folder $HOME/datasets/telept/data/ipad/rgb_1764569430654 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz

python demo.py \
  --video_path $HOME/datasets/telept/data/ipad/rgb_1764569695903.mp4 \
  --video_timestamps \
  --timestamps "00:11.254,00:31.270,00:49.261,00:52.457" \
  --output_folder $HOME/datasets/telept/data/ipad/rgb_1764569695903 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz

python demo.py \
  --video_path $HOME/datasets/telept/data/ipad/rgb_1764569971278.mp4 \
  --video_timestamps \
  --timestamps "00:20.347,00:22.695" \
  --output_folder $HOME/datasets/telept/data/ipad/rgb_1764569971278 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz