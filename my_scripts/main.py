#!/usr/bin/env python3
"""
main.py

Entry point for ROM visualization.
Faithful refactor of the original rom.py with:
- restored defaults
- no non-existent smplx helpers
- clean separation into utils_* modules
"""

import os
import cv2, threading, time
import json
import argparse
import numpy as np
import torch
import smplx
import open3d as o3d

from utils_math import (
    normalize,
    project_point_to_plane,
    project_vec_to_plane,
    signed_angle_in_plane,
    build_plane_basis_from_up_and_right,
)

from utils_vis import (
    make_plane_patch,
    make_arrow_from_to,
    make_body_material,
    make_joint_material,
    make_plane_material,
    make_arrow_material,
    create_visualizer,
    add_joint_spheres,
    run_visualizer,
)

from utils_rom_config import ROM_TASKS, JOINT_NAMES


# =============================
# JSON + SMPL-X helpers
# =============================

def to_torch(x, device, dtype=torch.float32):
    x = np.asarray(x)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def load_json_rotmats(path, device):
    d = json.load(open(path, "r"))
    params = {k: to_torch(d[k], device) for k in d.keys()}

    if params["transl"].ndim == 1:
        params["transl"] = params["transl"].unsqueeze(0)
    if params["betas"].ndim == 1:
        params["betas"] = params["betas"].unsqueeze(0)
    if params["expression"].ndim == 1:
        params["expression"] = params["expression"].unsqueeze(0)

    return params


def rotmat_to_axis_angle(R):
    R = R.to(dtype=torch.float32)

    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    r = torch.stack([rx, ry, rz], dim=-1)

    sin_theta = torch.sin(theta)
    denom = 2.0 * torch.clamp(sin_theta, min=1e-8)[..., None]
    axis = r / denom

    small = theta < 1e-6
    axis = torch.where(small[..., None], torch.zeros_like(axis), axis)

    return axis * theta[..., None]


def convert_rotmats_to_axis_angle_params(rot_params):
    B = rot_params["betas"].shape[0]

    def aa_single(name):
        R = rot_params[name]
        if R.ndim == 4:
            R = R[:, 0]
        return rotmat_to_axis_angle(R)

    def aa_multi(name, J):
        R = rot_params[name]
        aa = rotmat_to_axis_angle(R.reshape(-1, 3, 3)).reshape(B, J, 3)
        return aa.reshape(B, J * 3)

    return {
        "transl": rot_params["transl"],
        "betas": rot_params["betas"],
        "expression": rot_params["expression"],
        "global_orient": aa_single("global_orient"),
        "body_pose": aa_multi("body_pose", 21),
        "left_hand_pose": aa_multi("left_hand_pose", 15),
        "right_hand_pose": aa_multi("right_hand_pose", 15),
        "jaw_pose": aa_single("jaw_pose"),
        "leye_pose": aa_single("leye_pose"),
        "reye_pose": aa_single("reye_pose"),
    }


# =============================
# ROM computation (current task)
# =============================

def compute_hinge_flexion(
    joints,
    joint_name,
    proximal_name,
    distal_name,
    body_scale,
    zero_when_straight=True,
):
    p_joint = joints[JOINT_NAMES.index(joint_name)]
    p_prox  = joints[JOINT_NAMES.index(proximal_name)]
    p_dist  = joints[JOINT_NAMES.index(distal_name)]

    v_ref  = p_prox - p_joint
    v_main = p_dist - p_joint

    n_ref = np.linalg.norm(v_ref)
    n_main = np.linalg.norm(v_main)
    if n_ref < 1e-8 or n_main < 1e-8:
        return np.nan, {
            "plane": None,
            "vectors": {
                "raw": (p_joint, p_dist),
                "reference": (p_joint, p_prox),
            },
            "angle_pos": p_joint,
        }

    u = v_ref / n_ref
    v = v_main / n_main

    cosang = np.clip(np.dot(u, v), -1.0, 1.0)
    ang_rad = np.arccos(cosang)
    ang_deg = float(np.degrees(ang_rad))

    if zero_when_straight:
        ang_deg = 180.0 - ang_deg

    # Visualization
    v_ref_draw = u * n_main

    geom = {
        "plane": None,
        "vectors": {
            "raw": (p_joint, p_dist),
            "reference": (p_joint, p_joint + v_ref_draw),
            "projected": (p_joint, p_joint + v_main),
        },
        "angle_pos": p_joint + 0.02 * body_scale * normalize(v_main),
    }

    return ang_deg, geom

def compute_shoulder_sagittal_angle(joints, body_scale, plane_scale):
    # torso frame joints
    p_pelvis = joints[JOINT_NAMES.index("pelvis")]
    p_spine3 = joints[JOINT_NAMES.index("spine3")]
    p_lsho   = joints[JOINT_NAMES.index("left_shoulder")]
    p_rsho   = joints[JOINT_NAMES.index("right_shoulder")]
    p_wr     = joints[JOINT_NAMES.index("left_wrist")]

    # basis
    up_axis = normalize(p_spine3 - p_pelvis)
    right_guess = normalize(p_rsho - p_lsho)
    right_axis, forward, up_axis = build_plane_basis_from_up_and_right(up_axis, right_guess)

    # choose sign so flexion (forward) is positive
    plane_normal = -right_axis  # sagittal plane normal

    # vectors projected into sagittal plane
    v_main = p_wr - p_lsho
    v_main_proj = project_vec_to_plane(v_main, plane_normal)

    v_ref = p_pelvis - p_spine3  # DOWN reference (0° when arm down)
    v_ref_proj = project_vec_to_plane(v_ref, plane_normal)

    n_main = np.linalg.norm(v_main_proj)
    n_ref  = np.linalg.norm(v_ref_proj)
    if n_main < 1e-8 or n_ref < 1e-8:
        return np.nan, {
            "plane": {
                "origin": p_lsho,
                "right": forward,
                "forward": up_axis,
                "half": plane_scale * body_scale,
            },
            "vectors": {"raw": (p_lsho, p_wr)},
            "angle_pos": p_lsho,
        }

    # match ref length for drawing
    v_ref_proj = normalize(v_ref_proj) * n_main

    # signed angle: from ref -> main around plane_normal
    ang_rad = signed_angle_in_plane(v_main_proj, v_ref_proj, plane_normal)
    ang_deg = float(np.degrees(ang_rad)) if np.isfinite(ang_rad) else np.nan

    # draw endpoints
    p_main_end = p_lsho + v_main_proj
    p_ref_end  = p_lsho + v_ref_proj

    geom = {
        "plane": {
            "origin": p_lsho,
            "right": forward,
            "forward": up_axis,
            "half": plane_scale * body_scale,
        },
        "vectors": {
            "raw": (p_lsho, p_wr),
            "projected": (p_lsho, p_main_end),
            "reference": (p_lsho, p_ref_end),
        },
        "angle_pos": p_lsho + 0.02 * body_scale * forward,
    }

    return ang_deg, geom

def compute_task(task_name, joints, body_scale, plane_scale):
    """
    Compute ROM angle + debug geometry for a given task_name.
    """

    if task_name == "left_shoulder_internal_rotation":
        # ===== YOUR EXISTING CODE (unchanged) =====
        p_pelvis = joints[JOINT_NAMES.index("pelvis")]
        p_spine2 = joints[JOINT_NAMES.index("spine2")]
        p_lsho = joints[JOINT_NAMES.index("left_shoulder")]
        p_rsho = joints[JOINT_NAMES.index("right_shoulder")]

        p0 = 0.5 * (p_pelvis + p_spine2)
        up = normalize(p_spine2 - p_pelvis)
        right_guess = normalize(p_rsho - p_lsho)

        right, forward, up = build_plane_basis_from_up_and_right(up, right_guess)

        p_el = joints[JOINT_NAMES.index("left_elbow")]
        p_wr = joints[JOINT_NAMES.index("left_wrist")]

        v = p_wr - p_el

        p_el_proj = project_point_to_plane(p_el, p0, up)
        v_proj = project_vec_to_plane(v, up)
        p_wr_proj = p_el_proj + v_proj

        ref_len = np.linalg.norm(v_proj)
        f_plane = normalize(project_vec_to_plane(forward, up))
        v_ref = f_plane * ref_len
        p_ref_end = p_el_proj + v_ref

        ang_rad = signed_angle_in_plane(v_proj, v_ref, up)
        ang_deg = float(np.degrees(ang_rad)) if np.isfinite(ang_rad) else np.nan

        geom = {
            "plane": {"origin": p0, "right": right, "forward": forward, "half": plane_scale * body_scale},
            "vectors": {"raw": (p_el, p_wr), "projected": (p_el_proj, p_wr_proj), "reference": (p_el_proj, p_ref_end)},
            "angle_pos": p_el_proj + 0.02 * body_scale * right,
        }
        return ang_deg, geom

    elif task_name == "left_hip_internal_rotation":
        # ===== NEW HIP TASK =====
        # Plane axes (body-parallel)
        p_spine1 = joints[JOINT_NAMES.index("spine1")]
        p_spine2 = joints[JOINT_NAMES.index("spine2")]
        p_lhip = joints[JOINT_NAMES.index("left_hip")]
        p_rhip = joints[JOINT_NAMES.index("right_hip")]

        # Anchor everything at knee (as you requested)
        p_knee = joints[JOINT_NAMES.index("left_knee")]
        p_ankle = joints[JOINT_NAMES.index("left_ankle")]

        # In-plane axis 1: body vertical
        up_axis = normalize(p_spine2 - p_spine1)

        # In-plane axis 2: body left->right
        right_guess = normalize(p_rhip - p_lhip)

        # Build an orthonormal body frame
        right, forward, up_axis = build_plane_basis_from_up_and_right(up_axis, right_guess)

        # Define the plane using its normal.
        # Since 'up_axis' and 'right' are in-plane axes, the normal is +/- forward.
        plane_normal = forward

        # Main vector: knee -> ankle, projected to plane
        v_main = (p_ankle - p_knee)
        v_main_proj = project_vec_to_plane(v_main, plane_normal)

        # Reference vector direction: spine1 -> spine2, anchored at knee
        cfg = ROM_TASKS[task_name]
        ref = cfg["reference_vector"]
        p_from = joints[JOINT_NAMES.index(ref["from"])]
        p_to   = joints[JOINT_NAMES.index(ref["to"])]
        v_ref  = p_to - p_from
        v_ref_proj = project_vec_to_plane(v_ref, plane_normal)

        # Match lengths for nicer drawing
        main_len = np.linalg.norm(v_main_proj)
        if main_len < 1e-8:
            ang_deg = np.nan
            geom = {
                "plane": {"origin": p_knee, "right": right, "forward": up_axis, "half": plane_scale * body_scale},
                "vectors": {"raw": (p_knee, p_ankle)},
                "angle_pos": p_knee,
            }
            return ang_deg, geom

        v_ref_proj = normalize(v_ref_proj) * main_len

        # Endpoints for drawing (both start at knee)
        p_main_end = p_knee + v_main_proj
        p_ref_end = p_knee + v_ref_proj

        # Signed angle in plane around plane_normal
        ang_rad = signed_angle_in_plane(v_main_proj, v_ref_proj, plane_normal)
        ang_deg = float(np.degrees(ang_rad)) if np.isfinite(ang_rad) else np.nan

        geom = {
            "plane": {
                "origin": p_knee,
                # For the plane patch we need two in-plane directions:
                # use 'right' and 'up_axis' as the patch axes (both lie in the plane)
                "right": right,
                "forward": up_axis,
                "half": plane_scale * body_scale,
            },
            "vectors": {
                "raw": (p_knee, p_ankle),                  # original main (for optional draw)
                "projected": (p_knee, p_main_end),         # projected main
                "reference": (p_knee, p_ref_end),          # projected reference
            },
            "angle_pos": p_knee + 0.02 * body_scale * right,
        }
        return ang_deg, geom

    elif task_name == "left_knee_flexion":
        return compute_hinge_flexion(
            joints,
            joint_name="left_knee",
            proximal_name="left_hip",
            distal_name="left_ankle",
            body_scale=body_scale,
            zero_when_straight=True,
        )

    elif task_name == "left_elbow_flexion":
        return compute_hinge_flexion(
            joints,
            joint_name="left_elbow",
            proximal_name="left_shoulder",
            distal_name="left_wrist",
            body_scale=body_scale,
            zero_when_straight=False,
        )

    elif task_name == "left_shoulder_abduction":

        # ===== LEFT SHOULDER ABDUCTION (frontal plane) =====
        # Plane axes:
        #   up_axis    = pelvis -> spine3   (used to form plane)
        #   right_axis = left_shoulder -> right_shoulder
        # Frontal plane is spanned by (up_axis, right_axis)
        # Plane normal is forward = cross(up_axis, right_axis)
        #
        # Angle convention here:
        #   reference points DOWN (spine3 -> pelvis),
        #   so arm-down ≈ 0°, arm-horizontal ≈ 90°, arm-up ≈ 180°.

        # --- joints for torso frame ---
        p_pelvis = joints[JOINT_NAMES.index("pelvis")]
        p_spine3 = joints[JOINT_NAMES.index("spine3")]
        p_lsho   = joints[JOINT_NAMES.index("left_shoulder")]
        p_rsho   = joints[JOINT_NAMES.index("right_shoulder")]

        # --- plane basis ---
        up_axis = normalize(p_spine3 - p_pelvis)
        right_guess = normalize(p_rsho - p_lsho)
        right, forward, up_axis = build_plane_basis_from_up_and_right(up_axis, right_guess)

        # Frontal plane normal (front/back axis)
        plane_normal = forward

        # --- main vector: shoulder -> wrist, projected into frontal plane ---
        p_wr = joints[JOINT_NAMES.index("left_wrist")]
        v_main = p_wr - p_lsho
        v_main_proj = project_vec_to_plane(v_main, plane_normal)

        # --- reference vector: DOWN torso direction, projected into frontal plane ---
        # DOWN = spine3 -> pelvis  (this makes 0° correspond to arm-down)
        v_ref = p_pelvis - p_spine3
        v_ref_proj = project_vec_to_plane(v_ref, plane_normal)

        n_main = np.linalg.norm(v_main_proj)
        n_ref  = np.linalg.norm(v_ref_proj)
        if n_main < 1e-8 or n_ref < 1e-8:
            return np.nan, {
                "plane": {
                    "origin": p_lsho,
                    "right": right,
                    "forward": up_axis,  # in-plane axis for patch
                    "half": plane_scale * body_scale,
                },
                "vectors": {"raw": (p_lsho, p_wr)},
                "angle_pos": p_lsho,
            }

        # --- match ref length to main for visualization ---
        v_ref_proj = normalize(v_ref_proj) * n_main

        # --- unsigned angle between projected vectors ---
        u = v_ref_proj / np.linalg.norm(v_ref_proj)
        v = v_main_proj / np.linalg.norm(v_main_proj)
        cosang = np.clip(np.dot(u, v), -1.0, 1.0)
        ang_rad = np.arccos(cosang)
        ang_deg = float(np.degrees(ang_rad)) if np.isfinite(ang_rad) else np.nan

        # --- endpoints for drawing (both start at left shoulder) ---
        p_main_end = p_lsho + v_main_proj
        p_ref_end  = p_lsho + v_ref_proj

        geom = {
            "plane": {
                "origin": p_lsho,
                "right": right,
                "forward": up_axis,
                "half": plane_scale * body_scale,
            },
            "vectors": {
                "raw": (p_lsho, p_wr),
                "projected": (p_lsho, p_main_end),
                "reference": (p_lsho, p_ref_end),
            },
            "angle_pos": p_lsho + 0.02 * body_scale * right,
        }

        return ang_deg, geom
    
    elif task_name == "left_shoulder_flexion_extension":
        # ===== LEFT SHOULDER FLEXION / EXTENSION (sagittal plane) =====
        # Convention:
        #   0° at arm-down,
        #   + = flexion (arm forward),
        #   - = extension (arm backward).
        #
        # Torso frame:
        #   up_axis    = pelvis -> spine3
        #   right_axis = left_shoulder -> right_shoulder
        #   forward    = cross(up_axis, right_axis)
        #
        # Sagittal plane is spanned by (up_axis, forward)
        # Plane normal is +/- right_axis.
        # We use plane_normal = -right_axis so that flexion becomes positive.

        # --- joints ---
        p_pelvis = joints[JOINT_NAMES.index("pelvis")]
        p_spine3 = joints[JOINT_NAMES.index("spine3")]
        p_lsho   = joints[JOINT_NAMES.index("left_shoulder")]
        p_rsho   = joints[JOINT_NAMES.index("right_shoulder")]
        p_wr     = joints[JOINT_NAMES.index("left_wrist")]

        # --- plane basis (torso) ---
        up_axis = normalize(p_spine3 - p_pelvis)
        right_guess = normalize(p_rsho - p_lsho)
        right_axis, forward, up_axis = build_plane_basis_from_up_and_right(up_axis, right_guess)

        # Sagittal plane normal (choose sign so flexion is +)
        plane_normal = -right_axis

        # --- main vector: shoulder -> wrist (project into sagittal plane) ---
        v_main = p_wr - p_lsho
        v_main_proj = project_vec_to_plane(v_main, plane_normal)

        # --- reference vector: DOWN torso direction (spine3 -> pelvis), drawn from shoulder ---
        v_ref = p_pelvis - p_spine3  # DOWN
        v_ref_proj = project_vec_to_plane(v_ref, plane_normal)

        n_main = np.linalg.norm(v_main_proj)
        n_ref  = np.linalg.norm(v_ref_proj)
        if n_main < 1e-8 or n_ref < 1e-8:
            return np.nan, {
                "plane": {
                    "origin": p_lsho,
                    # sagittal plane patch axes are forward + up_axis
                    "right": forward,
                    "forward": up_axis,
                    "half": plane_scale * body_scale,
                },
                "vectors": {"raw": (p_lsho, p_wr)},
                "angle_pos": p_lsho,
            }

        # --- match ref length to main for visualization ---
        v_ref_proj = normalize(v_ref_proj) * n_main

        # --- signed angle in sagittal plane ---
        # signed_angle_in_plane(a, b, n): angle from b -> a around n
        ang_rad = signed_angle_in_plane(v_main_proj, v_ref_proj, plane_normal)
        ang_deg = float(np.degrees(ang_rad)) if np.isfinite(ang_rad) else np.nan

        # --- endpoints for drawing (both start at left shoulder) ---
        p_main_end = p_lsho + v_main_proj
        p_ref_end  = p_lsho + v_ref_proj

        geom = {
            "plane": {
                "origin": p_lsho,
                "right": forward,    # in-plane axis 1
                "forward": up_axis,  # in-plane axis 2
                "half": plane_scale * body_scale,
            },
            "vectors": {
                "raw": (p_lsho, p_wr),
                "projected": (p_lsho, p_main_end),
                "reference": (p_lsho, p_ref_end),
            },
            "angle_pos": p_lsho + 0.02 * body_scale * forward,
        }

        return ang_deg, geom

    elif task_name == "left_shoulder_flexion":
        ang, geom = compute_shoulder_sagittal_angle(joints, body_scale, plane_scale)
        return max(0.0, ang), geom

    elif task_name == "left_shoulder_extension":
        ang, geom = compute_shoulder_sagittal_angle(joints, body_scale, plane_scale)
        return max(0.0, -ang), geom

    else:
        raise ValueError(f"Unknown task: {task_name}")

# =============================
# Main
# =============================

# python main.py --filename "/home/haziq/datasets/telept/data/ipad/rgb_1764569430654/timestamps/ts_0000_00-21.744_f000423" --task "left_shoulder_flexion"
# python main.py --filename "/home/haziq/datasets/telept/data/ipad/rgb_1764569430654/timestamps/ts_0001_00-17.978_f000350" --task "left_shoulder_extension"
# python main.py --filename "/home/haziq/datasets/telept/data/ipad/rgb_1764569430654/timestamps/ts_0002_00-31.970_f000622" --task "left_elbow_flexion"
# python main.py --filename "/home/haziq/datasets/telept/data/ipad/rgb_1764569430654/timestamps/ts_0004_01-28.586_f001723" --task "left_shoulder_abduction"
# python main.py --filename "/home/haziq/datasets/telept/data/ipad/rgb_1764569430654/timestamps/ts_0003_01-38.357_f001913" --task "left_shoulder_internal_rotation"
# python main.py --filename "/home/haziq/datasets/telept/data/ipad/rgb_1764569971278/timestamps/ts_0000_00-20.347_f000399" --task "left_hip_internal_rotation"
# python main.py --filename "/home/haziq/datasets/telept/data/ipad/rgb_1764569695903/timestamps/ts_0001_00-31.270_f000622" --task "left_knee_flexion"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--filename",
        type=str,
        default=os.path.expanduser("~/datasets/telept/data/ipad/rgb_1764569430654/timestamps/ts_0001_01-38.357_f001913"),
    )
    parser.add_argument(
        "--smplx_models_path",
        type=str,
        default=os.path.expanduser(
            "~/datasets/mocap/data/models_smplx_v1_1/models/"
        ),
    )
    parser.add_argument("--gender", type=str, default="neutral")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--task",
        type=str,
        default="left_shoulder_internal_rotation",
    )
    parser.add_argument(
        "--plane_scale",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--plane_alpha",
        type=float,
        default=0.7,
    )

    args = parser.parse_args()
    args.filename = os.path.expanduser(args.filename)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- load SMPL-X ----
    rot_params = load_json_rotmats(args.filename+"_data.json", device)
    Rz180 = torch.tensor(
        [[-1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0]],
        device=device,
        dtype=rot_params["global_orient"].dtype,
    )
    Ry180 = torch.tensor(
        [[-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0, -1.0]],
        device=device,
        dtype=rot_params["global_orient"].dtype,
    )
    G = rot_params["global_orient"]
    if G.ndim == 4:      # (B, 1, 3, 3)
        G = G[:, 0]
        G = Ry180 @ (Rz180 @ G)
        rot_params["global_orient"] = G[:, None]
    elif G.ndim == 3:    # (B, 3, 3)
        rot_params["global_orient"] = Ry180 @ (Rz180 @ G)
    else:
        raise ValueError(f"Unexpected global_orient shape: {G.shape}")

    aa_params = convert_rotmats_to_axis_angle_params(rot_params)

    model = smplx.create(
        model_path=args.smplx_models_path,
        model_type="smplx",
        gender=args.gender,
        num_betas=aa_params["betas"].shape[-1],
        num_expression_coeffs=aa_params["expression"].shape[-1],
        use_pca=False,
        batch_size=1,
    ).to(device)

    with torch.no_grad():
        out = model(**aa_params)

    verts = out.vertices[0].cpu().numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int32)
    joints = out.joints[0].cpu().numpy()[: len(JOINT_NAMES)]

    body_scale = float(np.linalg.norm(np.ptp(verts, axis=0)))

    # ---- compute ROM ----
    angle_deg, geom = compute_task(args.task, joints, body_scale, args.plane_scale)
    print(f"[ANGLE] {args.task}: {angle_deg:.2f} deg")

    # ---- visualization ----
    app, w = create_visualizer("ROM Visualizer")

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    w.add_geometry("body", mesh, make_body_material(alpha=args.plane_alpha))

    add_joint_spheres(
        w,
        joints,
        body_scale,
        make_joint_material(),
        JOINT_NAMES,
    )

    # ---- plane (optional) ----
    if geom.get("plane") is not None:
        plane = make_plane_patch(
            geom["plane"]["origin"],
            geom["plane"]["right"],
            geom["plane"]["forward"],
            geom["plane"]["half"],
            geom["plane"]["half"],
        )
        w.add_geometry("plane", plane, make_plane_material(alpha=args.plane_alpha))

    arrow_mats = {
        "raw": make_arrow_material((0.0, 0.0, 0.0)),
        "projected": make_arrow_material((1.0, 0.5, 0.0)),
        "reference": make_arrow_material((0.0, 0.7, 0.0)),
    }

    for name, (p0, p1) in geom["vectors"].items():
        arrow = make_arrow_from_to(p0, p1, body_scale)
        if arrow is not None:
            w.add_geometry(f"vec_{name}", arrow, arrow_mats[name])

    #w.add_3d_label(geom["angle_pos"], f"theta = {angle_deg:.1f}°")

    # ---- camera setup (force upright + facing you) ----
    # bbox = mesh.get_axis_aligned_bounding_box()
    # center = bbox.get_center()

    # eye = center + np.array([0.0, -0.4, -0.7]) * body_scale  # in front
    # up  = np.array([0.0, -1.0, 0.0])                        # Y-up (try)

    # w.scene.camera.look_at(center, eye, up)

    def show_img_loop(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("[WARN] Could not read:", path)
            return

        # If RGBA, drop alpha for display
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        cv2.namedWindow("RGB Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("RGB Frame", img)

        # Keep pumping events until window closed
        while cv2.getWindowProperty("RGB Frame", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.waitKey(16)
            time.sleep(0.01)

    img_path = args.filename + "_render.jpg"
    if os.path.exists(img_path):
        threading.Thread(target=show_img_loop, args=(img_path,), daemon=True).start()

    run_visualizer(app, w, True)


if __name__ == "__main__":
    main()
