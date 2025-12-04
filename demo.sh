python demo.py \
    --image_folder "/home/haziq/datasets/ipad/251201/rgb_1764569430654/" \
    --output_folder "/home/haziq/datasets/ipad/251201/sam_rgb_1764569430654/" \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
    
CUDA_VISIBLE_DEVICES="1" python demo.py \
    --image_folder "/home/haziq/datasets/ipad/251201/rgb_1764569695903/" \
    --output_folder "/home/haziq/datasets/ipad/251201/sam_rgb_1764569695903/" \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
    
CUDA_VISIBLE_DEVICES="2" python demo.py \
    --image_folder "/home/haziq/datasets/ipad/251201/rgb_1764569971278/" \
    --output_folder "/home/haziq/datasets/ipad/251201/sam_rgb_1764569971278/" \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
    
python demo.py \
    --image_folder "/home/haziq/datasets/ipad/251201/rgb_1764569430654/" \
    --output_folder "/home/haziq/datasets/ipad/251201/sam_rgb_1764569430654/" \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
    
python demo.py \
    --image_folder "/home/haziq/datasets/ipad/251201/rgb_1764569695903/" \
    --output_folder "/home/haziq/datasets/ipad/251201/sam_rgb_1764569695903/" \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt