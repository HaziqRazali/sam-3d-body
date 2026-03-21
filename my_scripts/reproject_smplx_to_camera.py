"""
Reproject GT world-space SMPL-X onto a target camera view using fit3d calibration.

WHY the previous SAM3D-based approach was poor:
  SAM3D-Body uses a simplified pseudo-camera model (fx=fy≈1030, cx=W/2, cy=H/2).
  Its pred_cam_t is in that pseudo-camera space, not in the real fit3d calibrated
  camera frame.  Using pred_cam_t + fit3d extrinsics to reconstruct world space
  therefore introduces systematic errors in X/Y (≈0.35 m at ~4.5 m depth) and
  in global orientation.

THIS script uses the GT world-space SMPLX parameters from
  fit3d/smplx/<action>.json
which are already properly aligned in fit3d world coordinates, and projects them
to any target camera using calibrated fit3d extrinsics + intrinsics.

World → target-camera transform (row-vector convention used by fit3d):
  transl_cam        = (transl_world + pelvis - T_tgt) @ R_tgt^T - pelvis
  global_orient_cam = R_tgt @ global_orient_world

This is exactly the logic from SMPLXHelper.get_camera_smplx_params in smplx_util.py.

Usage (project GT squat SMPLX to camera 50591643):
  conda run -n mhr_new python /home/haziq/sam-3d-body/my_scripts/reproject_smplx_to_camera.py \\
    --gt_smplx_json /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/squat.json \\
    --tgt_cam_json  /home/haziq/datasets/mocap/data/fit3d/train/s03/camera_parameters/50591643/squat.json \\
    --tgt_video     /home/haziq/datasets/mocap/data/fit3d/train/s03/videos/50591643/squat.mp4 \\
    --out_video     /home/haziq/datasets/mocap/data/fit3d/train/s03/sam3d/50591643/squat_gt_reprojected.mp4 \\
    --max_frames 100
"""

import os
import sys
import json
import argparse
import multiprocessing

import cv2
import numpy as np
import torch
import smplx
import trimesh
from scipy.spatial.transform import Rotation as Rot

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    shape = rotmat.shape[:-2]
    aa = Rot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec()
    return aa.reshape(*shape, 3)


def _make_raymond_light_nodes():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis   = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes  = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z  = np.array([xp, yp, zp]); z /= np.linalg.norm(z)
        x  = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y  = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix,
        ))
    return nodes


def find_smplx_path():
    candidates = [
        os.path.expanduser("~/datasets/mocap/data/models_smplx_v1_1/models/smplx"),
        os.path.expanduser("/media/haziq/Haziq/mocap/data/models_smplx_v1_1/models/smplx"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------

def load_cam_params(json_path: str) -> dict:
    """Load fit3d camera calibration JSON."""
    with open(json_path) as f:
        cam = json.load(f)
    R  = np.array(cam["extrinsics"]["R"], dtype=np.float64)     # (3, 3)
    T  = np.array(cam["extrinsics"]["T"], dtype=np.float64)     # (1, 3)
    fx = cam["intrinsics_w_distortion"]["f"][0][0]
    fy = cam["intrinsics_w_distortion"]["f"][0][1]
    cx = cam["intrinsics_w_distortion"]["c"][0][0]
    cy = cam["intrinsics_w_distortion"]["c"][0][1]
    return dict(R=R, T=T, fx=fx, fy=fy, cx=cx, cy=cy)


# ---------------------------------------------------------------------------
# World → camera transform for SMPLX params
# Mirrors SMPLXHelper.get_camera_smplx_params from smplx_util.py exactly.
#
# fit3d row-vector convention:
#   v_cam = (v_world - T) @ R^T
#
# For translation (pelvis-relative SMPLX transl):
#   transl_cam = (transl_world + pelvis - T) @ R^T - pelvis
#
# For global_orient (rotation matrix, shape T×1×3×3):
#   go_cam = R_ext @ go_world
#   Computed as: (go_world^T @ R_ext^T)^T  (matches smplx_util.py formula)
# ---------------------------------------------------------------------------

def world_smplx_to_cam(
    transl_world:        np.ndarray,  # (T, 3)
    global_orient_world: np.ndarray,  # (T, 1, 3, 3)
    pelvis:              np.ndarray,  # (T, 3) T-pose pelvis joint
    R_ext:               np.ndarray,  # (3, 3)
    T_ext:               np.ndarray,  # (1, 3)
):
    """Transform SMPLX transl + global_orient from world to camera space."""
    # translation: (transl_world + pelvis - T) @ R^T - pelvis
    transl_cam = (transl_world + pelvis - T_ext) @ R_ext.T - pelvis   # (T, 3)

    # global_orient: (go_world^T @ R_ext^T)^T = R_ext @ go_world
    go_T              = global_orient_world.transpose(0, 1, 3, 2)      # R_world^T  (T,1,3,3)
    global_orient_cam = (go_T @ R_ext.T).transpose(0, 1, 3, 2)        # R_ext @ R_world  (T,1,3,3)

    return transl_cam.astype(np.float32), global_orient_cam.astype(np.float32)


# ---------------------------------------------------------------------------
# Render vertices already in target-camera OpenCV space.
# Camera at origin; 180° X-flip converts OpenCV → OpenGL for pyrender.
# Uses actual calibrated intrinsics (fx, fy, cx, cy).
# ---------------------------------------------------------------------------

def render_vertices_opencv(
    renderer:    pyrender.OffscreenRenderer,
    image_float: np.ndarray,    # (H, W, 3) float32 [0,1]
    v_cam:       np.ndarray,    # (V, 3) in target-cam OpenCV space
    faces:       np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
) -> np.ndarray:
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(*LIGHT_BLUE, 1.0),
    )

    mesh_tm = trimesh.Trimesh(v_cam.copy(), faces.copy())
    # OpenCV (y-down, z-forward) → OpenGL (y-up, z-backward)
    rot180x = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh_tm.apply_transform(rot180x)

    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tm, material=material)
    scene   = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh_pr, "mesh")

    # Camera at origin, looking in -Z (OpenGL).
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=1e12)
    scene.add(camera, pose=np.eye(4))

    for node in _make_raymond_light_nodes():
        scene.add_node(node)

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color    = color.astype(np.float32) / 255.0
    mask     = color[:, :, 3:4]
    return (color[:, :, :3] * mask + (1.0 - mask) * image_float).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Project GT world SMPL-X onto a target camera using "
            "fit3d calibrated extrinsics and intrinsics."
        )
    )
    parser.add_argument("--gt_smplx_json", required=True,
                        help="GT world-space SMPL-X JSON (e.g. fit3d/smplx/squat.json).")
    parser.add_argument("--tgt_cam_json",  required=True,
                        help="Target camera calibration JSON (fit3d format).")
    parser.add_argument("--tgt_video",     required=True,
                        help="Target-camera video to overlay the mesh on.")
    parser.add_argument("--out_video",     required=True,
                        help="Output video path.")
    parser.add_argument("--smplx_path",    default=find_smplx_path())
    parser.add_argument("--max_frames",    type=int, default=0,
                        help="Process at most this many frames (0 = all).")
    args = parser.parse_args()

    if args.smplx_path is None or not os.path.isdir(args.smplx_path):
        sys.exit("[ERROR] SMPL-X model not found. Pass --smplx_path explicitly.")

    # ------------------------------------------------------------------
    # 1) Load GT world SMPLX
    # ------------------------------------------------------------------
    print(f"[INFO] Loading GT world SMPL-X: {args.gt_smplx_json}")
    with open(args.gt_smplx_json) as f:
        data = json.load(f)

    transl_world_all    = np.array(data["transl"],          dtype=np.float32)  # (T, 3)
    global_orient_R_all = np.array(data["global_orient"],   dtype=np.float32)  # (T, 1, 3, 3)
    body_pose_R_all     = np.array(data["body_pose"],        dtype=np.float32)  # (T, 21, 3, 3)
    betas_all           = np.array(data["betas"],            dtype=np.float32)  # (T, 10)
    lhand_R_all         = np.array(data["left_hand_pose"],   dtype=np.float32)
    rhand_R_all         = np.array(data["right_hand_pose"],  dtype=np.float32)
    jaw_R_all           = np.array(data["jaw_pose"],         dtype=np.float32)
    leye_R_all          = np.array(data["leye_pose"],        dtype=np.float32)
    reye_R_all          = np.array(data["reye_pose"],        dtype=np.float32)
    expr_all            = np.array(data["expression"],       dtype=np.float32)

    T_data = transl_world_all.shape[0]
    print(f"[INFO] GT frames: {T_data}")

    # Convert rotation matrices → axis-angle for smplx model
    body_pose_aa = rotmat_to_aa(body_pose_R_all.reshape(T_data, 21, 3, 3))  # (T, 21, 3)
    lhand_aa     = rotmat_to_aa(lhand_R_all.reshape(T_data, 15, 3, 3))
    rhand_aa     = rotmat_to_aa(rhand_R_all.reshape(T_data, 15, 3, 3))
    jaw_aa       = rotmat_to_aa(jaw_R_all.reshape(T_data, 3, 3))
    leye_aa      = rotmat_to_aa(leye_R_all.reshape(T_data, 3, 3))
    reye_aa      = rotmat_to_aa(reye_R_all.reshape(T_data, 3, 3))

    # ------------------------------------------------------------------
    # 2) Load target camera parameters
    # ------------------------------------------------------------------
    print(f"[INFO] Loading target camera: {args.tgt_cam_json}")
    tgt_cam = load_cam_params(args.tgt_cam_json)
    R_tgt   = tgt_cam["R"]
    T_tgt   = tgt_cam["T"]
    print(f"[INFO] Target intrinsics: fx={tgt_cam['fx']:.1f} fy={tgt_cam['fy']:.1f} "
          f"cx={tgt_cam['cx']:.1f} cy={tgt_cam['cy']:.1f}")

    # ------------------------------------------------------------------
    # 3) SMPL-X model — needed to get T-pose pelvis joint per frame
    # ------------------------------------------------------------------
    device      = torch.device("cpu")
    smplx_model = smplx.SMPLX(
        model_path=args.smplx_path,
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)
    faces = np.asarray(smplx_model.faces, dtype=np.int32)

    # Compute per-frame T-pose pelvis (depends only on betas)
    print("[INFO] Computing T-pose pelvis for each frame ...")
    pelvis_all = []
    for i in range(T_data):
        with torch.no_grad():
            out = smplx_model(betas=torch.tensor(betas_all[i:i+1], dtype=torch.float32))
        pelvis_all.append(out.joints[0, 0].numpy())
    pelvis_all = np.stack(pelvis_all, axis=0)   # (T, 3)
    print(f"[INFO] Pelvis computed. Mean={pelvis_all.mean(0)}")

    # ------------------------------------------------------------------
    # 4) Transform GT world params → target-camera space
    # ------------------------------------------------------------------
    print("[INFO] Transforming world SMPLX params to target camera space ...")
    transl_cam_all, global_orient_cam_all = world_smplx_to_cam(
        transl_world_all,
        global_orient_R_all,
        pelvis_all,
        R_tgt,
        T_tgt,
    )
    global_orient_cam_aa = rotmat_to_aa(
        global_orient_cam_all.reshape(T_data, 3, 3)
    )  # (T, 3)
    print(f"[INFO] Sample transl_cam[0]        = {transl_cam_all[0]}")
    print(f"[INFO] Sample global_orient_cam[0] = {global_orient_cam_aa[0]}")

    # ------------------------------------------------------------------
    # 5) Open target video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(args.tgt_video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.tgt_video}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames = min(n_vid, T_data)
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)

    print(f"[INFO] Video: {vid_W}x{vid_H} @ {fps:.1f} fps  |  processing {n_frames} frames")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_video)), exist_ok=True)
    writer = cv2.VideoWriter(
        args.out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (vid_W, vid_H),
    )
    renderer = pyrender.OffscreenRenderer(viewport_width=vid_W, viewport_height=vid_H)

    # ------------------------------------------------------------------
    # 6) Per-frame loop
    # ------------------------------------------------------------------
    for frame_idx in range(n_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"[WARN] Video ended early at frame {frame_idx}")
            break

        # Forward-pass SMPLX with camera-space transl + orient
        with torch.no_grad():
            out = smplx_model(
                transl          = torch.tensor(transl_cam_all[frame_idx:frame_idx+1],          dtype=torch.float32),
                global_orient   = torch.tensor(global_orient_cam_aa[frame_idx:frame_idx+1],   dtype=torch.float32),
                body_pose       = torch.tensor(body_pose_aa[frame_idx:frame_idx+1].reshape(1, 63), dtype=torch.float32),
                betas           = torch.tensor(betas_all[frame_idx:frame_idx+1],               dtype=torch.float32),
                left_hand_pose  = torch.tensor(lhand_aa[frame_idx:frame_idx+1].reshape(1, 45), dtype=torch.float32),
                right_hand_pose = torch.tensor(rhand_aa[frame_idx:frame_idx+1].reshape(1, 45), dtype=torch.float32),
                jaw_pose        = torch.tensor(jaw_aa[frame_idx:frame_idx+1],                  dtype=torch.float32),
                leye_pose       = torch.tensor(leye_aa[frame_idx:frame_idx+1],                 dtype=torch.float32),
                reye_pose       = torch.tensor(reye_aa[frame_idx:frame_idx+1],                 dtype=torch.float32),
                expression      = torch.tensor(expr_all[frame_idx:frame_idx+1],                dtype=torch.float32),
            )

        # vertices in target-camera OpenCV space
        v_cam = out.vertices[0].numpy()   # (V, 3)

        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        result_rgb = render_vertices_opencv(
            renderer, frame_rgb, v_cam, faces,
            tgt_cam["fx"], tgt_cam["fy"],
            tgt_cam["cx"], tgt_cam["cy"],
        )
        result_bgr = cv2.cvtColor((result_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        cv2.putText(result_bgr, f"{frame_idx}/{n_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(result_bgr)

        if frame_idx % 50 == 0:
            print(f"  [{frame_idx}/{n_frames}]")

    cap.release()
    writer.release()
    renderer.delete()
    print(f"\n[DONE] Saved: {args.out_video}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
