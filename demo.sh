
# run on a single image
python demo.py \
    --image_folder /home/haziq/sam-3d-body/example_data/images/ \
    --output_folder /home/haziq/sam-3d-body/example_data/results/ \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --detector_name sam3

# run on a video with timestamps
python demo.py \
  --video_path /home/haziq/datasets/telept/data/ipad/rgb_1764569430654.mp4 \
  --video_timestamps \
  --timestamps "01:37.409,01:38.357" \
  --output_folder /home/haziq/datasets/telept/data/ipad/rgb_1764569430654 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3

# run on every frame of the video
python demo.py \
  --video_path /home/haziq/datasets/telept/data/ipad/rgb_1764569430654.mp4 \
  --output_folder /home/haziq/datasets/telept/data/ipad/rgb_1764569430654 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3