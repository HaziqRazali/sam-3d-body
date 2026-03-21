"""
Overlay SMPL-X mesh (from mhr_to_smpl.py JSON) onto the original video frames.

Rendering matches demo.py Renderer exactly:
  - camera_translation[0] *= -1
  - 180 deg X-axis rotation on the mesh
  - pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length, cx=W/2, cy=H/2)
  - Raymond directional lights + RGBA alpha-blend

SMPL-X vertices are pelvis-centered (joints[0] subtracted) so the body
origin matches the convention used by demo.py with MHR vertices.
Camera placement uses pred_cam_t + focal_length from the original npz.

Usage:
    conda run -n mhr_new python /home/haziq/sam-3d-body/my_scripts/visualize_smplx.py \
        --video_path  <path/to/video.mp4> \
        --smplx_json  <path/to/squat_smplx.json> \
        --mhr_npz     <path/to/squat_mhr_outputs.npz> \
        --out_video   <path/to/output.mp4>
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
# Lighting
# ---------------------------------------------------------------------------

def _make_raymond_light_nodes():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis   = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])
    nodes = []
    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)
        z = np.array([xp, yp, zp]); z /= np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix,
        ))
    return nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    shape = rotmat.shape[:-2]
    aa = Rot.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec()
    return aa.reshape(*shape, 3)


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
# Renderer (matches demo.py Renderer.__call__)
# ---------------------------------------------------------------------------

def render_on_image(renderer, image_float, vertices, faces, cam_t, focal_length):
    """
    renderer    : persistent pyrender.OffscreenRenderer
    image_float : [H, W, 3] float32 in [0, 1]
    Returns     : [H, W, 3] float32 in [0, 1]
    """
    h, w = image_float.shape[:2]

    camera_translation = cam_t.copy()
    camera_translation[0] *= -1.0          # matches demo.py Renderer

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode="OPAQUE",
        baseColorFactor=(*LIGHT_BLUE, 1.0),
    )

    mesh_tm = trimesh.Trimesh(vertices.copy(), faces.copy())
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh_tm.apply_transform(rot)            # matches demo.py Renderer
    mesh_pr = pyrender.Mesh.from_trimesh(mesh_tm, material=material)

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh_pr, "mesh")

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = camera_translation
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=w / 2.0,     cy=h / 2.0,
        zfar=1e12,
    )
    scene.add(camera, pose=camera_pose)
    for node in _make_raymond_light_nodes():
        scene.add_node(node)

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

    color = color.astype(np.float32) / 255.0
    valid_mask = color[:, :, 3:4]
    output = color[:, :, :3] * valid_mask + (1.0 - valid_mask) * image_float
    return output.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overlay SMPL-X mesh onto original video frames."
    )
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--smplx_json", required=True,
                        help="JSON output from mhr_to_smpl.py")
    parser.add_argument("--mhr_npz",    required=True,
                        help="*_mhr_outputs.npz from demo.py (provides pred_cam_t + focal_length)")
    parser.add_argument("--out_video",  required=True)
    parser.add_argument("--smplx_path", default=find_smplx_path())
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    if args.smplx_path is None or not os.path.isdir(args.smplx_path):
        sys.exit("[ERROR] SMPL-X model path not found. Pass --smplx_path explicitly.")

    # ------------------------------------------------------------------
    # 1) Load SMPL-X JSON
    # ------------------------------------------------------------------
    print(f"[INFO] Loading SMPL-X JSON: {args.smplx_json}")
    with open(args.smplx_json) as f:
        data = json.load(f)

    transl_all      = np.array(data["transl"],          dtype=np.float32)  # [T, 3]
    global_orient_R = np.array(data["global_orient"],   dtype=np.float32)
    body_pose_R     = np.array(data["body_pose"],        dtype=np.float32)
    betas_all       = np.array(data["betas"],            dtype=np.float32)
    lhand_R         = np.array(data["left_hand_pose"],   dtype=np.float32)
    rhand_R         = np.array(data["right_hand_pose"],  dtype=np.float32)
    jaw_R           = np.array(data["jaw_pose"],         dtype=np.float32)
    leye_R          = np.array(data["leye_pose"],        dtype=np.float32)
    reye_R          = np.array(data["reye_pose"],        dtype=np.float32)
    expr_all        = np.array(data["expression"],       dtype=np.float32)

    T = transl_all.shape[0]
    print(f"[INFO] Total frames in JSON: {T}")

    global_orient_aa = rotmat_to_aa(global_orient_R.reshape(T, 3, 3))
    body_pose_aa     = rotmat_to_aa(body_pose_R.reshape(T, 21, 3, 3))
    lhand_aa         = rotmat_to_aa(lhand_R.reshape(T, 15, 3, 3))
    rhand_aa         = rotmat_to_aa(rhand_R.reshape(T, 15, 3, 3))
    jaw_aa           = rotmat_to_aa(jaw_R.reshape(T, 3, 3))
    leye_aa          = rotmat_to_aa(leye_R.reshape(T, 3, 3))
    reye_aa          = rotmat_to_aa(reye_R.reshape(T, 3, 3))

    # ------------------------------------------------------------------
    # 2) Load pred_cam_t and focal_length from the original npz
    # ------------------------------------------------------------------
    print(f"[INFO] Loading camera params from: {args.mhr_npz}")
    npz = np.load(args.mhr_npz, allow_pickle=True)
    pred_cam_t_all   = npz["pred_cam_t"].astype(np.float32)    # [T, 3]
    focal_length_all = npz["focal_length"].astype(np.float32)  # [T]

    # ------------------------------------------------------------------
    # 3) SMPL-X model  (transl=zeros — body centred at pelvis; we use pred_cam_t)
    # ------------------------------------------------------------------
    device = torch.device("cpu")
    smplx_model = smplx.SMPLX(
        model_path=args.smplx_path,
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)
    faces = np.asarray(smplx_model.faces, dtype=np.int32)

    # ------------------------------------------------------------------
    # 4) Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames = min(n_vid, T)
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
    # 5) Per-frame loop
    # ------------------------------------------------------------------
    for frame_idx in range(n_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            print(f"[WARN] Video ended early at frame {frame_idx}")
            break

        with torch.no_grad():
            out = smplx_model(
                transl          = torch.tensor(transl_all[frame_idx:frame_idx+1], dtype=torch.float32),
                global_orient   = torch.tensor(global_orient_aa[frame_idx:frame_idx+1], dtype=torch.float32),
                body_pose       = torch.tensor(body_pose_aa[frame_idx:frame_idx+1].reshape(1, 63), dtype=torch.float32),
                betas           = torch.tensor(betas_all[frame_idx:frame_idx+1], dtype=torch.float32),
                left_hand_pose  = torch.tensor(lhand_aa[frame_idx:frame_idx+1].reshape(1, 45), dtype=torch.float32),
                right_hand_pose = torch.tensor(rhand_aa[frame_idx:frame_idx+1].reshape(1, 45), dtype=torch.float32),
                jaw_pose        = torch.tensor(jaw_aa[frame_idx:frame_idx+1], dtype=torch.float32),
                leye_pose       = torch.tensor(leye_aa[frame_idx:frame_idx+1], dtype=torch.float32),
                reye_pose       = torch.tensor(reye_aa[frame_idx:frame_idx+1], dtype=torch.float32),
                expression      = torch.tensor(expr_all[frame_idx:frame_idx+1], dtype=torch.float32),
            )

        # transl in the JSON is already in metres (the converter stores it at
        # the same scale as the original MHR vertices), so no rescaling needed.
        verts = out.vertices[0].numpy()   # [V, 3]

        cam_t        = pred_cam_t_all[frame_idx]
        focal_length = float(focal_length_all[frame_idx])

        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        result_rgb = render_on_image(renderer, frame_rgb, verts, faces, cam_t, focal_length)
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
