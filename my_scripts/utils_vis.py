"""
utils_vis.py

Open3D visualization utilities.
- ALL Open3D code lives here
- NO math / angle computation
- Consumes numeric geometry (points, vectors, planes)

This file provides:
- arrows
- plane patches
- materials
- viewer setup helpers
"""

import numpy as np
import open3d as o3d
import open3d.visualization as vis
import open3d.visualization.rendering as rendering

from utils_math import rotation_matrix_from_a_to_b


# =============================
# Geometry creation helpers
# =============================

def make_plane_patch(p0, right, forward, half_w, half_h):
    """
    Create a rectangular plane patch centered at p0, spanning (right, forward).

    Args:
        p0: plane origin (3,)
        right: unit right direction
        forward: unit forward direction
        half_w: half width
        half_h: half height
    Returns:
        o3d.geometry.TriangleMesh
    """
    p0 = np.asarray(p0, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)

    c00 = p0 - half_w * right - half_h * forward
    c10 = p0 + half_w * right - half_h * forward
    c11 = p0 + half_w * right + half_h * forward
    c01 = p0 - half_w * right + half_h * forward

    vertices = np.stack([c00, c10, c11, c01], axis=0)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(vertices)
    plane.triangles = o3d.utility.Vector3iVector(triangles)
    plane.compute_vertex_normals()
    plane.compute_triangle_normals()
    return plane


def make_arrow_from_to(
    p_start,
    p_end,
    body_scale,
    cylinder_radius=None,
    cone_radius=None,
    cone_height=None,
):
    """
    Create an arrow mesh from p_start to p_end.
    Default arrow is along +Z and rotated to match direction.
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


# =============================
# Materials
# =============================

def make_body_material(alpha=0.25):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = (0.8, 0.8, 0.8, float(alpha))
    return mat


def make_joint_material():
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = (1.0, 0.0, 0.0, 1.0)
    return mat


def make_plane_material(alpha=0.25):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLitTransparency"
    mat.base_color = (0.2, 0.6, 1.0, float(alpha))
    return mat


def make_arrow_material(color):
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = (*color, 1.0)
    return mat


# =============================
# Viewer helpers
# =============================

def create_visualizer(
    title="ROM Visualizer",
    width=1024,
    height=1024,
    background=(1, 1, 1, 1),
):
    """
    Create and initialize an Open3D O3DVisualizer.
    """
    app = vis.gui.Application.instance
    app.initialize()

    w = vis.O3DVisualizer(title, width, height)
    w.show_settings = True
    w.scene.set_background(background)

    return app, w


def add_joint_spheres(
    visualizer,
    joints,
    body_scale,
    material,
    joint_names=None,
    add_labels=True,
):
    """
    Add joint spheres to the visualizer.
    - Geometry names (checkbox list) will use joint_names if provided.
    - 3D labels on the body are optional via add_labels.
    """
    import open3d as o3d
    import numpy as np

    radius = 0.008 * body_scale

    for i, p in enumerate(joints):
        p = np.asarray(p, dtype=np.float64)

        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
        s.translate(p)
        s.compute_vertex_normals()

        # ✅ checkbox name
        if joint_names is not None:
            gname = f"joint/{joint_names[i]}"
        else:
            gname = f"joint/{i}"

        visualizer.add_geometry(gname, s, material)

        # ✅ 3D text label on the model
        #if add_labels and joint_names is not None:
        #    visualizer.add_3d_label(p, f"{i}: {joint_names[i]}")
        #elif add_labels:
        #    visualizer.add_3d_label(p, f"{i}")

def run_visualizer(app, visualizer, reset_camera=False):
    if reset_camera:
        visualizer.reset_camera_to_default()
    app.add_window(visualizer)
    app.run()
