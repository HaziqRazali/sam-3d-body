#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np

import torch
import smplx
import open3d as o3d
import open3d.visualization as vis
import open3d.visualization.rendering as rendering


# -----------------------------
# Joint names (FIRST 22 BODY JOINTS)
# -----------------------------
JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


# -----------------------------
# Utils
# -----------------------------
def to_torch(x, device, dtype=torch.float32):
    x = np.asarray(x)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def print_shapes(tag, dct):
    print(f"\n=== {tag} ===")
    for k, v in dct.items():
        if torch.is_tensor(v):
            print(f"{k:16s}: {tuple(v.shape)} dtype={v.dtype} device={v.device}")
        else:
            a = np.asarray(v)
            print(f"{k:16s}: np {a.shape} dtype={a.dtype}")
    print("=" * (len(tag) + 8))


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


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def project_point_to_plane(x, p0, n):
    """Project point x onto plane defined by (p0, normal n)."""
    x = np.asarray(x, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    n = normalize(n)
    return x - np.dot(x - p0, n) * n


def project_vec_to_plane(v, n):
    """Project vector v onto plane with normal n (i.e., remove normal component)."""
    v = np.asarray(v, dtype=np.float64)
    n = normalize(n)
    return v - np.dot(v, n) * n


def make_plane_patch(p0, right, forward, half_w, half_h):
    """
    Make a rectangular plane patch centered at p0, spanning (right, forward),
    with size (2*half_w) x (2*half_h). Returns an Open3D TriangleMesh.
    """
    c00 = p0 - half_w * right - half_h * forward
    c10 = p0 + half_w * right - half_h * forward
    c11 = p0 + half_w * right + half_h * forward
    c01 = p0 - half_w * right + half_h * forward

    vertices = np.stack([c00, c10, c11, c01], axis=0).astype(np.float64)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(vertices)
    plane.triangles = o3d.utility.Vector3iVector(triangles)
    plane.compute_vertex_normals()
    plane.compute_triangle_normals()
    return plane


def rotation_matrix_from_a_to_b(a, b, eps=1e-8):
    """
    Rodrigues-like rotation taking vector a to vector b (both 3D).
    Returns 3x3 rotation matrix.
    """
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < eps:
        # parallel or anti-parallel
        if c < 0:
            # 180 deg: pick any orthogonal axis
            axis = normalize(np.cross(a, np.array([1.0, 0.0, 0.0])))
            if np.linalg.norm(axis) < eps:
                axis = normalize(np.cross(a, np.array([0.0, 1.0, 0.0])))
            return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * np.pi)
        return np.eye(3)

    vx = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]],
        dtype=np.float64,
    )
    R = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s * s))
    return R


def make_arrow_from_to(p_start, p_end, body_scale,
                       cylinder_radius=None, cone_radius=None, cone_height=None):
    """
    Create an arrow mesh from p_start to p_end.
    Default arrow is along +Z; we rotate it to the direction (p_end - p_start).
    """
    p_start = np.asarray(p_start, dtype=np.float64)
    p_end = np.asarray(p_end, dtype=np.float64)
    v = p_end - p_start
    length = float(np.linalg.norm(v))
    if length < 1e-8:
        return None

    if cylinder_radius is None:
        cylinder_radius = 0.004 * body_scale
    if cone_radius is None:
        cone_radius = 0.010 * body_scale
    if cone_height is None:
        cone_height = 0.06 * body_scale

    cyl_h = max(length - cone_height, 1e-6)

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cyl_h,
        cone_height=cone_height,
        resolution=20,
        cylinder_split=4,
        cone_split=1,
    )
    arrow.compute_vertex_normals()

    # Rotate +Z to direction v
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R = rotation_matrix_from_a_to_b(z, v)
    arrow.rotate(R, center=np.zeros(3))
    arrow.translate(p_start)
    return arrow


def signed_angle_in_plane(a, b, plane_normal, eps=1e-8):
    """
    Signed angle from reference b -> a, measured in the plane with normal plane_normal.
    Returns angle in radians in (-pi, pi].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = normalize(plane_normal)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan

    a_u = a / na
    b_u = b / nb

    # unsigned angle
    c = float(np.clip(np.dot(b_u, a_u), -1.0, 1.0))
    ang = float(np.arccos(c))

    # sign using right-hand rule around plane normal
    s = float(np.dot(n, np.cross(b_u, a_u)))
    if abs(s) < 1e-12:
        return ang  # nearly colinear, sign doesn't matter
    return np.sign(s) * ang


# -----------------------------
# Load JSON
# -----------------------------
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


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        type=str,
        default="/home/haziq/datasets/telept/data/ipad/rgb_1764569430654/timestamps/ts_0001_01-38.357_f001913_data.json",
    )
    parser.add_argument(
        "--smplx_models_path",
        type=str,
        default=os.path.expanduser("~/datasets/mocap/data/models_smplx_v1_1/models/"),
    )
    parser.add_argument("--gender", type=str, default="neutral")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--plane_scale", type=float, default=0.30, help="plane half-size as fraction of body_scale")
    parser.add_argument("--plane_alpha", type=float, default=0.25, help="plane transparency alpha [0..1]")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rot_params = load_json_rotmats(args.filename, device)
    print_shapes("RAW ROTMAT PARAMS (TORCH)", rot_params)

    aa_params = convert_rotmats_to_axis_angle_params(rot_params)
    print_shapes("CONVERTED AXIS-ANGLE PARAMS (TORCH)", aa_params)

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

    verts = out.vertices[0].detach().cpu().numpy().astype(np.float64)
    faces = np.asarray(model.faces, dtype=np.int32)
    joints = out.joints[0].detach().cpu().numpy().astype(np.float64)[: len(JOINT_NAMES)]

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    # joint spheres
    body_scale = float(np.linalg.norm(np.ptp(verts, axis=0)))
    radius = 0.008 * body_scale

    joint_spheres = []
    for j in joints:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
        s.translate(j)
        s.paint_uniform_color([1, 0, 0])
        s.compute_vertex_normals()
        joint_spheres.append(s)

    # -----------------------------
    # Transverse plane (torso-fixed)
    # p0 = (pelvis + spine2)/2
    # up = normalize(spine2 - pelvis)
    # right = normalize(right_shoulder - left_shoulder)
    # forward = normalize(cross(up, right))
    # re-orthogonalize right = normalize(cross(forward, up))
    # -----------------------------
    idx_pelvis = JOINT_NAMES.index("pelvis")
    idx_spine2 = JOINT_NAMES.index("spine2")
    idx_lsho = JOINT_NAMES.index("left_shoulder")
    idx_rsho = JOINT_NAMES.index("right_shoulder")

    pelvis = joints[idx_pelvis]
    spine2 = joints[idx_spine2]
    lsho = joints[idx_lsho]
    rsho = joints[idx_rsho]

    p0 = 0.5 * (pelvis + spine2)
    up = normalize(spine2 - pelvis)
    right = normalize(rsho - lsho)

    forward = normalize(np.cross(up, right))
    right = normalize(np.cross(forward, up))  # re-orthogonalize

    half = args.plane_scale * body_scale
    plane_patch = make_plane_patch(p0, right, forward, half_w=half, half_h=half)

    # -----------------------------
    # Vectors / arrows:
    # 1) elbow -> wrist (3D)
    # 2) elbow -> wrist projected onto the transverse plane (origin on plane, direction in plane)
    # 3) forward reference vector originating from elbow projected onto plane (flat on plane)
    # 4) signed angle between (2) and (3) in the transverse plane
    # -----------------------------
    idx_lelbow = JOINT_NAMES.index("left_elbow")
    idx_lwrist = JOINT_NAMES.index("left_wrist")

    p_el = joints[idx_lelbow]
    p_wr = joints[idx_lwrist]

    # (1) raw arrow
    arrow_raw = make_arrow_from_to(p_el, p_wr, body_scale)

    # (2) projected forearm arrow: origin AND direction flattened onto plane
    p_el_proj = project_point_to_plane(p_el, p0, up)
    v = p_wr - p_el
    v_proj = project_vec_to_plane(v, up)
    p_wr_proj = p_el_proj + v_proj
    arrow_proj = make_arrow_from_to(p_el_proj, p_wr_proj, body_scale)

    # (3) forward reference arrow from elbow projected onto plane, flat on plane
    # Make the reference arrow the same length as the projected forearm (so it’s easy to compare visually).
    f_plane = project_vec_to_plane(forward, up)
    f_plane = normalize(f_plane)
    ref_len = float(np.linalg.norm(v_proj))
    v_fwd_ref = f_plane * ref_len
    p_fwd_end = p_el_proj + v_fwd_ref
    arrow_fwd = make_arrow_from_to(p_el_proj, p_fwd_end, body_scale)

    # (4) signed angle: reference (forward) -> projected forearm
    ang_rad = signed_angle_in_plane(v_proj, v_fwd_ref, up)
    ang_deg = float(np.degrees(ang_rad)) if np.isfinite(ang_rad) else np.nan
    print(f"[ANGLE] signed angle (forward -> forearm_proj) = {ang_deg:.2f} deg")

    # -----------------------------
    # Visualization
    # -----------------------------
    app = vis.gui.Application.instance
    app.initialize()

    w = vis.O3DVisualizer("SMPL-X + Joints + Plane + Vectors", 1024, 1024)
    w.show_settings = True
    w.scene.set_background([1, 1, 1, 1])

    # body material (translucent)
    mat_body = rendering.MaterialRecord()
    mat_body.shader = "defaultLitTransparency"
    mat_body.base_color = (0.8, 0.8, 0.8, 0.25)

    # joints material (opaque red)
    mat_joint = rendering.MaterialRecord()
    mat_joint.shader = "defaultLit"
    mat_joint.base_color = (1.0, 0.0, 0.0, 1.0)

    # plane material (translucent blue)
    mat_plane = rendering.MaterialRecord()
    mat_plane.shader = "defaultLitTransparency"
    mat_plane.base_color = (0.2, 0.6, 1.0, float(args.plane_alpha))

    # arrow materials
    mat_arrow_raw = rendering.MaterialRecord()
    mat_arrow_raw.shader = "defaultLit"
    mat_arrow_raw.base_color = (0.0, 0.0, 0.0, 1.0)  # black

    mat_arrow_proj = rendering.MaterialRecord()
    mat_arrow_proj.shader = "defaultLit"
    mat_arrow_proj.base_color = (1.0, 0.5, 0.0, 1.0)  # orange (forearm projected)

    mat_arrow_fwd = rendering.MaterialRecord()
    mat_arrow_fwd.shader = "defaultLit"
    mat_arrow_fwd.base_color = (0.0, 0.7, 0.0, 1.0)  # green (forward reference)

    w.add_geometry("body", mesh, mat_body)
    w.add_geometry("transverse_plane", plane_patch, mat_plane)

    for i, (s, name) in enumerate(zip(joint_spheres, JOINT_NAMES)):
        w.add_geometry(f"joint_{name}", s, mat_joint)
        w.add_3d_label(joints[i], f"{i}: {name}")

    # Label p0
    w.add_3d_label(p0, "p0 (mid pelvis-spine2)")

    # Add arrows if valid
    if arrow_raw is not None:
        w.add_geometry("vec_elbow_to_wrist", arrow_raw, mat_arrow_raw)
        w.add_3d_label(p_el, "L elbow → wrist (3D)")
    else:
        print("[WARN] Raw elbow→wrist vector too small; skipping arrow_raw.")

    if arrow_proj is not None and np.linalg.norm(v_proj) > 1e-8:
        w.add_geometry("vec_elbow_to_wrist_proj", arrow_proj, mat_arrow_proj)
        w.add_3d_label(p_el_proj, "Forearm proj (orange)")
    else:
        print("[WARN] Projected elbow→wrist vector too small; skipping arrow_proj.")

    if arrow_fwd is not None and np.linalg.norm(v_fwd_ref) > 1e-8:
        w.add_geometry("vec_forward_ref", arrow_fwd, mat_arrow_fwd)
        w.add_3d_label(p_el_proj, "Forward ref (green)")
    else:
        print("[WARN] Forward ref vector too small; skipping arrow_fwd.")

    # Label angle near projected elbow (slightly offset so it doesn't overlap)
    angle_label_pos = p_el_proj + 0.02 * body_scale * right
    if np.isfinite(ang_deg):
        w.add_3d_label(angle_label_pos, f"θ = {ang_deg:.1f}°")
    else:
        w.add_3d_label(angle_label_pos, "θ = NaN")

    w.reset_camera_to_default()
    app.add_window(w)
    app.run()


if __name__ == "__main__":
    main()
