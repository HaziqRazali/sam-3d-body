"""
utils_math.py

Pure geometry & math utilities for ROM computation.
- NO Open3D
- NO SMPL-X
- NO visualization
- NumPy only (Torch only where unavoidable for rotmats)

This file provides:
- vector normalization
- projections (point/vector to plane)
- plane basis construction
- signed angles in planes
- rotation utilities (math only)
"""

import numpy as np
import torch


# =============================
# Basic vector utilities
# =============================

def normalize(v, eps=1e-8):
    """Normalize a 3D vector. Returns zero vector if norm is too small."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


# =============================
# Projection utilities
# =============================

def project_point_to_plane(x, p0, n):
    """
    Project point x onto a plane defined by (p0, normal n).
    """
    x = np.asarray(x, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    n = normalize(n)
    return x - np.dot(x - p0, n) * n


def project_vec_to_plane(v, n):
    """
    Project vector v onto a plane with normal n
    (i.e., remove the normal component).
    """
    v = np.asarray(v, dtype=np.float64)
    n = normalize(n)
    return v - np.dot(v, n) * n


# =============================
# Plane / basis construction
# =============================

def build_plane_basis_from_up_and_right(up, right):
    """
    Build an orthonormal (right, forward, up) basis for a plane.

    - up: plane normal direction
    - right: approximate right direction (will be re-orthogonalized)

    Returns:
        right_u, forward_u, up_u
    """
    up_u = normalize(up)
    right_u = normalize(right)

    # Forward is perpendicular to up and right
    forward_u = normalize(np.cross(up_u, right_u))

    # Re-orthogonalize right to ensure exact orthogonality
    right_u = normalize(np.cross(forward_u, up_u))

    return right_u, forward_u, up_u


# =============================
# Angle computations
# =============================

def signed_angle_in_plane(a, b, plane_normal, eps=1e-8):
    """
    Signed angle from reference vector b -> vector a,
    measured in the plane with normal plane_normal.

    Args:
        a: target vector
        b: reference vector
        plane_normal: plane normal defining orientation
    Returns:
        angle in radians, in (-pi, pi]
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

    # Unsigned angle
    c = float(np.clip(np.dot(b_u, a_u), -1.0, 1.0))
    ang = float(np.arccos(c))

    # Sign using right-hand rule around plane normal
    s = float(np.dot(n, np.cross(b_u, a_u)))
    if abs(s) < 1e-12:
        return ang  # nearly colinear

    return np.sign(s) * ang


# =============================
# Rotation utilities (math only)
# =============================

def rotation_matrix_from_a_to_b(a, b, eps=1e-8):
    """
    Compute rotation matrix that rotates vector a to vector b.

    Uses a Rodrigues-like formulation.
    Returns a 3x3 numpy rotation matrix.
    """
    a = normalize(a)
    b = normalize(b)

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < eps:
        # Parallel or anti-parallel
        if c < 0.0:
            # 180-degree rotation: choose any orthogonal axis
            axis = normalize(np.cross(a, np.array([1.0, 0.0, 0.0])))
            if np.linalg.norm(axis) < eps:
                axis = normalize(np.cross(a, np.array([0.0, 1.0, 0.0])))
            return axis_angle_to_matrix(axis * np.pi)
        return np.eye(3)

    vx = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]],
        dtype=np.float64,
    )

    R = np.eye(3) + vx + (vx @ vx) * ((1.0 - c) / (s * s))
    return R


def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle vector to rotation matrix.
    axis_angle: (3,) vector = axis * angle
    """
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angle = np.linalg.norm(axis_angle)

    if angle < 1e-8:
        return np.eye(3)

    axis = axis_angle / angle
    x, y, z = axis

    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c

    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ], dtype=np.float64)

    return R


# =============================
# Torch-specific helpers (kept here, math only)
# =============================

def rotmat_to_axis_angle(R):
    """
    Convert rotation matrices to axis-angle (Torch).

    Args:
        R: (..., 3, 3) torch tensor
    Returns:
        (..., 3) axis-angle torch tensor
    """
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
