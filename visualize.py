import os
import sys
import torch
import numpy as np
import open3d as o3d
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
NPZ_PATH = "/home/haziq/datasets/mocap/data/self/train/haziq/mhr/laptop_webcam/20260104_001543_mhr_outputs.npz"
MHR_ROOT = "/home/haziq/MHR"
FRAME_ID = 1337

sys.path.append(MHR_ROOT)
from mhr.mhr import MHR

# ------------------------------------------------------------------
# Load NPZ
# ------------------------------------------------------------------
npz = np.load(NPZ_PATH, allow_pickle=True)
smplx_params = {k: npz[k] for k in npz.files}

# ------------------------------------------------------------------
# NPZ vertices (reference)
# ------------------------------------------------------------------
verts_npz = npz["vertices"][FRAME_ID].astype(np.float64)

pcd_npz = o3d.geometry.PointCloud()
pcd_npz.points = o3d.utility.Vector3dVector(verts_npz)
pcd_npz.paint_uniform_color([1.0, 0.0, 0.0])  # RED
pcd_npz.estimate_normals()

# ------------------------------------------------------------------
# Load MHR model
# ------------------------------------------------------------------
mhr_model = MHR.from_files(
    folder=Path(f"{MHR_ROOT}/assets"),
    device=torch.device("cpu"),
    lod=1
)

# ------------------------------------------------------------------
# Prepare MHR inputs
# ------------------------------------------------------------------
shape            = smplx_params["shape_params"]       # [T, 45]
body_pose_params = smplx_params["body_pose_params"]   # [T, 133]
expr_params      = smplx_params["expr_params"]        # [T, 72]

# MHR expects 204-D pose vector:
# [root(6) | body(130) | hands+face(68)]
model_parameters = np.concatenate([
    np.zeros((body_pose_params.shape[0], 6), dtype=body_pose_params.dtype),
    body_pose_params[:, :130],
    np.zeros((body_pose_params.shape[0], 68), dtype=body_pose_params.dtype),
], axis=1)

# ------------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------------
with torch.no_grad():
    verts_mhr, skel_state = mhr_model(
        torch.tensor(shape, dtype=torch.float32),
        torch.tensor(model_parameters, dtype=torch.float32),
        torch.tensor(expr_params, dtype=torch.float32),
    )

# ------------------------------------------------------------------
# Post-processing (IMPORTANT)
# ------------------------------------------------------------------
verts_mhr = verts_mhr.cpu().numpy()

# mm → meters
verts_mhr /= 100.0

# axis convention (match your pipeline)
verts_mhr[..., [1, 2]] *= -1

verts_mhr_frame = verts_mhr[FRAME_ID].astype(np.float64)

pcd_mhr = o3d.geometry.PointCloud()
pcd_mhr.points = o3d.utility.Vector3dVector(verts_mhr_frame)
pcd_mhr.paint_uniform_color([0.0, 1.0, 0.0])  # GREEN
pcd_mhr.estimate_normals()

# ------------------------------------------------------------------
# Optional: offset one mesh so you can see both clearly
# ------------------------------------------------------------------
pcd_mhr.translate([0.0, 0.0, 0.0])  # 30 cm in +X

# ------------------------------------------------------------------
# Visualize together
# ------------------------------------------------------------------
o3d.visualization.draw_geometries(
    [pcd_npz, pcd_mhr],
    window_name="NPZ (red) vs MHR (green)",
    width=1280,
    height=720
)
