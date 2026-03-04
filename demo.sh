
# 1) run kit.sh which runs 3) on every video in kit
# 2) run mhr_to_smpl.sh which converts from mhr to smpl

# 1) run on a single image
python demo.py \
    --image_folder /home/haziq/sam-3d-body/example_data/images/ \
    --output_folder /home/haziq/sam-3d-body/example_data/results/ \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
    --detector_name sam3 \
    --save_npz \
    --center_person_only

# 3) run on every frame of the video
python demo.py \
  --video_path /home/haziq/datasets/telept/data/ipad/rgb_1764569430654_trim_0_149_crop.mp4 \
  --output_folder /home/haziq/datasets/telept/data/ipad/rgb_1764569430654_trim_0_149_crop \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz \
  --center_person_only

python demo.py \
  --video_path /home/haziq/datasets/motion-x++/data/video_ground_truth/animation/Ways_to_Jump_In_+_Swim_+_Get_Out_of_a_Pool_Backwards_Fall_clip1.mp4 \
  --output_folder /home/haziq/datasets/motion-x++/data/sam3d/animation/Ways_to_Jump_In_+_Swim_+_Get_Out_of_a_Pool_Backwards_Fall_clip1 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz \
  --center_person_only

# # # # #
# IROS  #
# # # # #

# # # # #
# IROS  #
# # # # #

python demo.py \
  --video_path /home/haziq/datasets/motion-x++/data/video/animation/Ways_to_Open_a_Christmas_Gift_Wrong_Gift_clip1.mp4 \
  --output_folder /home/haziq/datasets/motion-x++/data/sam3d/animation/Ways_to_Open_a_Christmas_Gift_Wrong_Gift_clip1 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz \
  --center_person_only \
  --render_only

# Ways_to_Go_to_Sleep_Watching_TV_clip1
python demo.py \
  --video_path /home/haziq/datasets/motion-x++/data/video/animation/Ways_to_Go_to_Sleep_Watching_TV_clip1.mp4 \
  --output_folder /home/haziq/datasets/motion-x++/data/sam3d/animation/Ways_to_Go_to_Sleep_Watching_TV_clip1 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz \
  --center_person_only \
  --render_only

python my_scripts/pose_editor.py \
  --npz_path /home/haziq/datasets/motion-x++/data/sam3d/animation/Ways_to_Go_to_Sleep_Watching_TV_clip1/Ways_to_Go_to_Sleep_Watching_TV_clip1_mhr_outputs.npz \
  --video_path /home/haziq/datasets/motion-x++/data/video/animation/Ways_to_Go_to_Sleep_Watching_TV_clip1.mp4 \
  --frame 0 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

# # # # # #
# TelePT  #
# # # # # #

# run on a video with timestamps
python demo.py \
  --video_path /home/haziq/datasets/telept/data/ipad/rgb_1764569430654.mp4 \
  --video_timestamps \
  --timestamps "01:38.357,00:59.825" \
  --output_folder /home/haziq/datasets/telept/data/ipad/rgb_1764569430654 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz

python demo.py \
  --video_path /home/haziq/datasets/telept/data/ipad/rgb_1764569695903.mp4 \
  --video_timestamps \
  --timestamps "00:11.254,00:31.270,00:49.261,00:52.457" \
  --output_folder /home/haziq/datasets/telept/data/ipad/rgb_1764569695903 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz

python demo.py \
  --video_path /home/haziq/datasets/telept/data/ipad/rgb_1764569971278.mp4 \
  --video_timestamps \
  --timestamps "00:20.347,00:22.695" \
  --output_folder /home/haziq/datasets/telept/data/ipad/rgb_1764569971278 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --detector_name sam3 \
  --save_npz