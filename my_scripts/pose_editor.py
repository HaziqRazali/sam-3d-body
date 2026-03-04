"""
pose_editor.py  –  Interactive body-pose orientation editor

Usage:
conda run -n sam_3d_body python my_scripts/pose_editor.py \
--npz_path /path/to/video_mhr_outputs.npz \
--video_path /path/to/video.mp4 \
--frame 0 \
--checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
--mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

python my_scripts/pose_editor.py \
--npz_path /home/haziq/datasets/motion-x++/data/sam3d_new/animation/Ways_to_Go_to_Sleep_Watching_TV_clip1/Ways_to_Go_to_Sleep_Watching_TV_clip1_mhr_outputs.npz \
--video_path /home/haziq/datasets/motion-x++/data/video/animation/Ways_to_Go_to_Sleep_Watching_TV_clip1.mp4 \
--frame 0 \
--checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
--mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt

Controls
--------
  [ / ]             : cycle to previous / next joint (prints name to terminal)
  Up   / Down       : active joint  X+/-
  Left / Right      : active joint  Y+/-
  PgUp / PgDn       : active joint  Z+/-
  ,  / .  (comma/period) : active joint Z alternative (more reliable on Linux)
  R                 : reset deltas for ALL joints to zero
  S                 : save modified params to NPZ and re-render the full video
  Q / Esc           : quit

Joint layout  (MHR body_pose_params, 133-dim) — empirically verified:
  12 × 3DOF body joints  (spine1–4, neck, head, collars, shoulders, hips)
  26 × 1DOF body joints  (elbow/wrist/knee/ankle/foot hinge axes)
  -> left_hip  (joint index 11): params[53,54,55]  CONFIRMED
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

import pyrootutils
pyrootutils.setup_root(
    search_from=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from sam_3d_body import load_sam_3d_body
from sam_3d_body.visualization.renderer import Renderer

# ── constants ──────────────────────────────────────────────────────────────────
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

# ── MHR body_pose_params joint layout (133-dim) ────────────────────────────────
# Empirically verified by perturbing each group in T-pose and observing vertex movement.
# In model space: positive X = person's left, negative X = person's right.
#
# 12 × 3DOF body joints (X, Y, Z Euler angles stored at non-consecutive indices):
_BODY_3DOF = [
    # name                 (idx_X, idx_Y, idx_Z)   empirical evidence
    ("spine1",             ( 0,  2,  4)),   # symmetric; moves 80% of body
    ("spine2",             ( 6,  8, 10)),   # symmetric; moves 80% of body
    ("spine3",             (12, 13, 14)),   # symmetric; moves 78% of body
    ("spine4",             (15, 16, 17)),   # symmetric; moves 76% of body
    ("neck",               (18, 19, 20)),   # upper; moves 35%, centroid y=1.59
    ("head",               (21, 22, 23)),   # upper; moves 33%, centroid y=1.59
    ("right_collar",       (24, 25, 26)),   # right side (x<0); moves 26%
    ("right_shoulder",     (27, 28, 29)),   # right side (x=-0.39); moves 19%
    ("left_collar",        (34, 35, 36)),   # left side (x>0); moves 29%
    ("left_shoulder",      (37, 38, 39)),   # left side (x=+0.39); moves 19%
    ("right_hip",          (44, 45, 46)),   # lower body (x=-0.09)
    ("left_hip",           (53, 54, 55)),   # lower body (x=+0.09) — CONFIRMED
]

# 26 × 1DOF body joints (single rotation axis each):
_BODY_1DOF = [
    # Interleaved with spine groups: likely spinal twist / lateral coupling
    ("spine1_1dof_a",  1), ("spine1_1dof_b",  3), ("spine1_1dof_c",  5),
    ("spine2_1dof_a",  7), ("spine2_1dof_b",  9), ("spine2_1dof_c", 11),
    # Right arm hinge joints (elbow / wrist flexion)
    ("right_elbow_a", 30), ("right_elbow_b", 31),
    ("right_wrist_a", 32), ("right_wrist_b", 33),
    # Left arm hinge joints
    ("left_elbow_a",  40), ("left_elbow_b",  41),
    ("left_wrist_a",  42), ("left_wrist_b",  43),
    # Right leg hinge joints (knee / ankle / foot)
    ("right_knee_a",  47), ("right_knee_b",  48),
    ("right_ankle_a", 49), ("right_ankle_b", 50),
    ("right_foot_a",  51), ("right_foot_b",  52),
    # Left leg hinge joints
    ("left_knee_a",   56), ("left_knee_b",   57),
    ("left_ankle_a",  58), ("left_ankle_b",  59),
    ("left_foot_a",   60), ("left_foot_b",   61),
]

def _build_joints():
    """
    Build the editable joint list.
    3DOF joints first (most useful), then 1DOF joints.
    Each entry: (display_name, [idx_X, idx_Y, idx_Z])
    For 1DOF joints all three idx point to the same param (only Up/Down is meaningful).
    """
    joints = []
    for name, (xi, yi, zi) in _BODY_3DOF:
        joints.append((f"{name}  [{xi},{yi},{zi}]", [xi, yi, zi]))
    for name, idx in _BODY_1DOF:
        joints.append((f"{name} (1DOF)  [{idx}]", [idx, idx, idx]))
    return joints

JOINTS = _build_joints()          # list of (display_name, [idx_X, idx_Y, idx_Z])
DEFAULT_JOINT = 11                 # start on left_hip (index 11 in _BODY_3DOF)

# OpenCV key codes on Linux (after & 0xFF)
KEY_UP       = 82     # Arrow Up
KEY_DOWN     = 84     # Arrow Down
KEY_LEFT     = 81     # Arrow Left
KEY_RIGHT    = 83     # Arrow Right
KEY_PGUP     = 85     # Page Up  (may vary by keyboard/setup)
KEY_PGDN     = 86     # Page Down
KEY_COMMA    = ord(',')   # Z- alternative  (more reliable on Linux)
KEY_PERIOD   = ord('.')   # Z+ alternative
KEY_Q        = ord('q')
KEY_ESC      = 27
KEY_R        = ord('r')
KEY_S        = ord('s')
KEY_LBRACKET = ord('[')   # previous joint
KEY_RBRACKET = ord(']')   # next joint
KEY_TAB      = 9          # cycle active axis X->Y->Z
AXIS_NAMES   = ["X", "Y", "Z"]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_video_frame_rgb(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def npz_frame(npz: np.lib.npyio.NpzFile, t: int) -> dict:
    """
    Extract all stored per-frame params from the NPZ at temporal index t.
    body_pose_params (133,), hand_pose_params (108,),
    scale_params (28,), shape_params (45,), expr_params (72,),
    global_rot (3,), pred_cam_t (3,), focal_length (), bbox (4,).
    """
    def _f(key):
        return npz[key][t]

    d = {
        "body_pose_params": _f("body_pose_params").copy(),
        "hand_pose_params": _f("hand_pose_params").copy(),
        "scale_params":     _f("scale_params").copy(),
        "shape_params":     _f("shape_params").copy(),
        "expr_params":      _f("expr_params").copy(),
        "global_rot":       _f("global_rot").copy(),
        "pred_cam_t":       _f("pred_cam_t").copy(),
        "focal_length":     float(_f("focal_length")),
        "bbox":             _f("bbox").copy(),
    }
    # Sanity: warn on NaN
    if np.isnan(d["body_pose_params"]).any():
        print(f"[WARN] Frame t={t} has NaN in body_pose_params (no detection?)")
    return d


def to_tensor(arr, device, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)  # [1, ...]


def run_mhr_forward(model, params: dict, device) -> np.ndarray:
    """
    Re-run head_pose.mhr_forward with (possibly modified) params.
    Returns pred_vertices as float32 numpy array [18439, 3].
    """
    head = model.head_pose

    global_trans     = torch.zeros(1, 3, dtype=torch.float32, device=device)
    global_rot       = to_tensor(params["global_rot"],       device)     # [1, 3]
    body_pose_params = to_tensor(params["body_pose_params"], device)     # [1, 133]
    hand_pose_params = to_tensor(params["hand_pose_params"], device)     # [1, 108]
    scale_params     = to_tensor(params["scale_params"],     device)     # [1, 28]
    shape_params     = to_tensor(params["shape_params"],     device)     # [1, 45]
    expr_params      = to_tensor(params["expr_params"],      device)     # [1, 72]

    with torch.no_grad():
        verts = head.mhr_forward(
            global_trans     = global_trans,
            global_rot       = global_rot,
            body_pose_params = body_pose_params,
            hand_pose_params = hand_pose_params,
            scale_params     = scale_params,
            shape_params     = shape_params,
            expr_params      = expr_params,
            return_keypoints    = False,
            return_joint_coords = False,
            return_model_params = False,
            return_joint_rotations = False,
        )
    # Camera-system flip (matches what mhr_head.forward does after mhr_forward)
    verts = verts.clone()
    verts[..., [1, 2]] *= -1
    return verts[0].cpu().numpy().astype(np.float32)   # [18439, 3]


def render_frame(
    frame_rgb: np.ndarray,
    verts: np.ndarray,
    pred_cam_t: np.ndarray,
    focal_length: float,
    faces: np.ndarray,
    side_view: bool = False,
) -> np.ndarray:
    """
    Render mesh onto frame_rgb (RGB uint8).  Returns BGR uint8.
    frame_rgb must be uint8 RGB.
    """
    renderer = Renderer(focal_length=focal_length, faces=faces)

    # Front view: overlay on original frame
    img_overlay = (
        renderer(
            verts,
            pred_cam_t,
            frame_rgb.copy(),
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        )
        * 255
    ).astype(np.uint8)

    if side_view:
        white = np.ones_like(frame_rgb) * 255
        img_side = (
            renderer(
                verts,
                pred_cam_t,
                white,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)
        combined = np.concatenate([img_overlay, img_side], axis=1)
        return cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    return cv2.cvtColor(img_overlay, cv2.COLOR_RGB2BGR)


def draw_hud(img_bgr: np.ndarray, joint_deltas, active_joint, active_axis, step, frame_idx, t_idx):
    """Overlay text HUD on img_bgr (in-place)."""
    jname, jidx = JOINTS[active_joint][0], JOINTS[active_joint][1]
    d = joint_deltas.get(active_joint, np.zeros(3))
    axis_indicators = [f"[{AXIS_NAMES[i]}]" if i == active_axis else f" {AXIS_NAMES[i]} " for i in range(3)]
    lines = [
        f"Frame idx: {frame_idx}  (NPZ row: {t_idx})",
        f"step: {step:.4f}",
        "",
        f"  [ / ]        -> prev / next joint",
        f"  Tab          -> cycle axis  {' '.join(axis_indicators)}",
        f"  Active : {jname}",
        f"  Up/Down      -> {AXIS_NAMES[active_axis]}  delta: {d[active_axis]:+.4f}",
        f"  Left/Right   -> Y  delta: {d[1]:+.4f}",
        f"  PgUp/PgDn    -> Z  delta: {d[2]:+.4f}",
        "",
        "  R=reset all  S=save+rerender  Q/Esc=quit",
    ]
    x, y0, dy = 12, 28, 26
    for i, line in enumerate(lines):
        y = y0 + i * dy
        # shadow
        cv2.putText(img_bgr, line, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 2, cv2.LINE_AA)
        # main
        cv2.putText(img_bgr, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 200), 1, cv2.LINE_AA)


def find_npz_row(npz: np.lib.npyio.NpzFile, frame_idx: int) -> int:
    """
    Find the row in the NPZ that corresponds to frame_idx (video frame number).
    The NPZ stores frame_indices for the stride-processed frames.
    Falls back to frame_idx directly if frame_indices not stored or out of range.
    """
    if "frame_indices" in npz:
        fi = npz["frame_indices"]
        # Exact match
        matches = np.where(fi == frame_idx)[0]
        if len(matches) > 0:
            return int(matches[0])
        # Nearest
        nearest = int(np.argmin(np.abs(fi - frame_idx)))
        actual  = int(fi[nearest])
        print(f"[INFO] Frame {frame_idx} not in NPZ; using nearest: row {nearest} = frame {actual}")
        return nearest
    # No frame_indices key – assume row == frame_idx
    n = npz["body_pose_params"].shape[0]
    t = min(frame_idx, n - 1)
    return t


def save_to_npz(npz_path: str, npz: np.lib.npyio.NpzFile, t_idx: int, modified_body_pose: np.ndarray):
    """
    Save modified body_pose_params back to the NPZ file at row t_idx.
    All other arrays and rows are preserved as-is.
    """
    print(f"[SAVE] Writing body_pose_params row {t_idx} back to {npz_path} ...")
    data = {}
    for key in npz.files:
        arr = npz[key]
        if key == "body_pose_params":
            arr = arr.copy()
            arr[t_idx] = modified_body_pose
        data[key] = arr
    np.savez_compressed(npz_path, **data)
    print(f"[SAVE] Done.")


def recompute_vertices_in_npz(npz_path: str, model, device):
    """
    Re-run mhr_forward for every row in the NPZ using the stored
    (possibly modified) body_pose_params and save the resulting
    vertices array back into the same file.
    This keeps the `vertices` key in sync with `body_pose_params`
    so that external visualizers (e.g. visualize_mesh.py) see the
    edited geometry.
    """
    from tqdm import tqdm

    npz  = np.load(npz_path, allow_pickle=True)
    T    = npz["body_pose_params"].shape[0]
    verts_list = []
    print(f"[VERTICES] Recomputing vertices for {T} rows...")
    for row in tqdm(range(T), desc="Recomputing vertices"):
        p     = npz_frame(npz, row)
        verts = run_mhr_forward(model, p, device)  # [18439, 3]
        verts_list.append(verts)
    vertices_all = np.stack(verts_list, axis=0)    # [T, 18439, 3]

    data = {key: npz[key] for key in npz.files}
    data["vertices"] = vertices_all
    np.savez_compressed(npz_path, **data)
    print(f"[VERTICES] Saved updated vertices -> {npz_path}")


def rerender_video(
    npz_path: str,
    video_path: str,
    out_path: str,
    model,
    faces: np.ndarray,
    device,
):
    """
    Re-render the full video from the (freshly saved) NPZ onto the source video frames.
    Output: out_path  (side_view=False, overlay-only).
    """
    from tqdm import tqdm

    npz = np.load(npz_path, allow_pickle=True)
    T = npz["body_pose_params"].shape[0]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        fps = 30.0

    writer = None
    print(f"[RERENDER] Rendering {T} frames -> {out_path}")
    for row in tqdm(range(T), desc="Re-rendering"):
        p = npz_frame(npz, row)
        vid_frame_idx = int(npz["frame_indices"][row]) if "frame_indices" in npz else row
        frame_rgb = load_video_frame_rgb(video_path, vid_frame_idx)
        verts = run_mhr_forward(model, p, device)
        img_bgr = render_frame(
            frame_rgb, verts, p["pred_cam_t"], p["focal_length"], faces, side_view=False
        )
        if writer is None:
            h, w = img_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter at {out_path}")
        writer.write(img_bgr)

    if writer is not None:
        writer.release()
    print(f"[RERENDER] Saved -> {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[INFO] Device: {device}")

    # ── output rerender path derived from NPZ path (saves alongside the NPZ) ──────
    npz_dir  = os.path.dirname(args.npz_path)
    npz_stem = os.path.splitext(os.path.basename(args.npz_path))[0]
    # strip trailing _mhr_outputs suffix if present, then append _rerendered
    base = npz_stem.removesuffix("_mhr_outputs")
    _, video_ext = os.path.splitext(args.video_path)
    rerender_path = os.path.join(npz_dir, base + "_rerendered" + (video_ext or ".mp4"))
    print(f"[INFO] Re-render output: {rerender_path}")

    # ── load model ──────────────────────────────────────────────────────────────
    print("[INFO] Loading SAM-3D-Body model...")
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    model, _ = load_sam_3d_body(args.checkpoint_path, device=device, mhr_path=mhr_path)
    model.eval()
    faces = model.head_pose.faces.cpu().numpy()
    print("[INFO] Model loaded.")

    # ── load NPZ ────────────────────────────────────────────────────────────────
    npz = np.load(args.npz_path, allow_pickle=True)
    total_rows = npz["body_pose_params"].shape[0]
    print(f"[INFO] NPZ loaded. Total rows: {total_rows}")

    t_idx = find_npz_row(npz, args.frame)
    frame_idx = int(npz["frame_indices"][t_idx]) if "frame_indices" in npz else args.frame
    print(f"[INFO] Using NPZ row {t_idx} = video frame {frame_idx}")

    params = npz_frame(npz, t_idx)

    # ── load video frame ─────────────────────────────────────────────────────────
    frame_rgb = load_video_frame_rgb(args.video_path, frame_idx)
    print(f"[INFO] Video frame loaded: shape={frame_rgb.shape}")

    # ── interactive state ────────────────────────────────────────────────────────
    # Per-joint deltas dict: joint_index -> np.zeros(3)  (X, Y, Z)
    joint_deltas = {}   # type: dict[int, np.ndarray]
    active_joint = DEFAULT_JOINT  # type: int
    active_axis  = 0              # 0=X  1=Y  2=Z
    step = float(args.step)

    def _active_d() -> np.ndarray:
        """Return (creating if needed) the delta array for the active joint."""
        if active_joint not in joint_deltas:
            joint_deltas[active_joint] = np.zeros(3, dtype=np.float32)
        return joint_deltas[active_joint]

    def get_modified_params() -> dict:
        p = params.copy()
        p["body_pose_params"] = params["body_pose_params"].copy()
        for jidx, d in joint_deltas.items():
            if np.any(d != 0):
                xi, yi, zi = JOINTS[jidx][1]
                p["body_pose_params"][xi] += d[0]
                p["body_pose_params"][yi] += d[1]
                p["body_pose_params"][zi] += d[2]
        return p

    def rebuild():
        p     = get_modified_params()
        verts = run_mhr_forward(model, p, device)
        img   = render_frame(frame_rgb, verts, params["pred_cam_t"], params["focal_length"], faces, side_view=True)
        draw_hud(img, joint_deltas, active_joint, active_axis, step, frame_idx, t_idx)
        return img

    def print_active_joint():
        jname = JOINTS[active_joint][0]
        d = joint_deltas.get(active_joint, np.zeros(3))
        print(f"[JOINT] ({active_joint:02d}/{len(JOINTS)-1})  {jname}  |  axis={AXIS_NAMES[active_axis]}  |  deltas: X={d[0]:+.4f}  Y={d[1]:+.4f}  Z={d[2]:+.4f}")

    # ── window setup ─────────────────────────────────────────────────────────────
    win = "SAM-3D-Body Pose Editor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("[INFO] Building initial render...")
    display = rebuild()
    cv2.imshow(win, display)
    print_active_joint()

    print("[INFO] Window open.  [ / ] = cycle joints  |  arrows/PgUp/PgDn = edit  |  S = save+rerender  |  Q/Esc = quit")

    dirty = False

    while True:
        key = cv2.waitKey(30) & 0xFF

        if key in (KEY_Q, KEY_ESC):
            break

        # ── joint cycling ──────────────────────────────────────────────────────
        elif key == KEY_LBRACKET:
            active_joint = (active_joint - 1) % len(JOINTS)
            print_active_joint()
            dirty = True

        elif key == KEY_RBRACKET:
            active_joint = (active_joint + 1) % len(JOINTS)
            print_active_joint()
            dirty = True

        # ── axis cycling ───────────────────────────────────────────────────────
        elif key == KEY_TAB:
            active_axis = (active_axis + 1) % 3
            print_active_joint()
            dirty = True

        # ── active axis (Up/Down always follows active_axis) ───────────────────
        elif key == KEY_UP:
            _active_d()[active_axis] += step
            dirty = True
        elif key == KEY_DOWN:
            _active_d()[active_axis] -= step
            dirty = True

        # ── Y axis ────────────────────────────────────────────────────────────
        elif key == KEY_LEFT:
            _active_d()[1] -= step
            dirty = True
        elif key == KEY_RIGHT:
            _active_d()[1] += step
            dirty = True

        # ── Z axis ────────────────────────────────────────────────────────────
        elif key in (KEY_PGUP, KEY_PERIOD):
            _active_d()[2] += step
            dirty = True
        elif key in (KEY_PGDN, KEY_COMMA):
            _active_d()[2] -= step
            dirty = True

        # ── reset all deltas ──────────────────────────────────────────────────
        elif key == KEY_R:
            joint_deltas.clear()
            dirty = True
            print("[INFO] All joint deltas reset.")
            print_active_joint()

        # ── save to NPZ + rerender ────────────────────────────────────────────
        elif key == KEY_S:
            modified_bp = get_modified_params()["body_pose_params"]
            # Print summary of all edited joints
            edited = {jidx: d for jidx, d in joint_deltas.items() if np.any(d != 0)}
            if not edited:
                print("[SAVE] No changes to save.")
            else:
                print(f"[SAVE] Edited joints: {list(edited.keys())}")
                for jidx, d in edited.items():
                    jname = JOINTS[jidx][0]
                    print(f"  {jname}  deltas: X={d[0]:+.6f}  Y={d[1]:+.6f}  Z={d[2]:+.6f}")
                save_to_npz(args.npz_path, npz, t_idx, modified_bp)
                # close window temporarily so user sees progress in terminal
                cv2.destroyWindow(win)
                rerender_video(
                    args.npz_path, args.video_path, rerender_path, model, faces, device
                )
                recompute_vertices_in_npz(args.npz_path, model, device)
                # reopen window
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.imshow(win, display)

        if dirty:
            display = rebuild()
            cv2.imshow(win, display)
            dirty = False

    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive body-pose editor for SAM-3D-Body outputs")
    parser.add_argument("--npz_path",        required=True,  type=str,   help="Path to the per-video NPZ from demo.py")
    parser.add_argument("--video_path",      required=True,  type=str,   help="Original video path (to load the frame)")
    parser.add_argument("--frame",           default=0,      type=int,   help="Video frame index to edit (default: 0)")
    parser.add_argument("--checkpoint_path", required=True,  type=str,   help="SAM-3D-Body checkpoint .ckpt")
    parser.add_argument("--mhr_path",        default="",     type=str,   help="Path to MHR assets .pt file (or SAM3D_MHR_PATH env var)")
    parser.add_argument("--step",            default=0.02,   type=float, help="Rotation delta per keypress in radians (default: 0.02)")
    args = parser.parse_args()
    main(args)
