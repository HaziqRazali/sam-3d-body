"""
utils_rom_config.py

Declarative ROM task definitions.
- NO math
- NO Open3D
- NO SMPL-X
- Just describes WHAT to measure and WHAT to visualize

Each task is a dictionary consumed by:
- compute_task(...)  (math layer)
- draw_task(...)     (vis layer)
"""


# =============================
# Joint name reference
# =============================

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


def jid(name: str) -> int:
    """Helper to get joint index by name."""
    return JOINT_NAMES.index(name)


# =============================
# ROM task definitions
# =============================

ROM_TASKS = {

    # -------------------------------------------------
    # Left forearm yaw in torso transverse plane
    # (your current implementation)
    # -------------------------------------------------
    "left_forearm_yaw_transverse": {

        # -------- Plane definition --------
        "plane": {
            "type": "torso_transverse",
            "origin": {
                "type": "midpoint",
                "joints": ["pelvis", "spine2"],
            },
            "up": {
                "type": "vector",
                "from": "pelvis",
                "to": "spine2",
            },
            "right": {
                "type": "vector",
                "from": "left_shoulder",
                "to": "right_shoulder",
            },
        },

        # -------- Main vector (measured) --------
        "main_vector": {
            "from": "left_elbow",
            "to": "left_wrist",
            "project_origin_to_plane": True,
            "project_direction_to_plane": True,
        },

        # -------- Reference vector --------
        "reference_vector": {
            "type": "plane_forward",     # derived from plane basis
            "origin": "left_elbow",
            "match_length_to": "main",   # same length as projected forearm
        },

        # -------- Angle computation --------
        "angle": {
            "type": "signed_in_plane",
            "unit": "degrees",
        },

        # -------- Visualization hints --------
        "viz": {
            "draw_plane": True,
            "draw_raw_vector": True,
            "draw_projected_vector": True,
            "draw_reference_vector": True,
            "label_angle": True,
        },
    },

    "left_hip_internal_rotation": {

        # -------- Plane definition --------
        "plane": {
            "type": "torso_transverse",
            "plane": {
                "type": "body_parallel",   # (we’ll interpret this in compute_task later)
                "origin": {"type": "joint", "name": "left_knee"},
                "up": {"type": "vector", "from": "spine1", "to": "spine2"},
                "right": {"type": "vector", "from": "left_hip", "to": "right_hip"},  # stable right axis
            },
        },

        # -------- Main vector (measured) --------
        "main_vector": {
            "from": "left_knee",
            "to": "left_ankle",
            "project_direction_to_plane": True,
        },

        # -------- Reference vector --------
        "reference_vector": {
            "type": "vector",
            "from": "spine2",
            "to": "spine1",
            "origin": "left_knee",
            "project_direction_to_plane": True,
            "match_length_to": "main",
        },

        # -------- Angle computation --------
        "angle": {
            "type": "signed_in_plane",
            "unit": "degrees",
        },

        # -------- Visualization hints --------
        "viz": {
            "draw_plane": True,
            "draw_raw_vector": True,
            "draw_projected_vector": True,
            "draw_reference_vector": True,
            "label_angle": True,
        },
    },

    "left_knee_flexion": {

        # -------- No plane needed --------
        "plane": None,

        # -------- Vectors (both anchored at knee) --------
        # Thigh direction: knee -> hip
        "reference_vector": {
            "type": "vector",
            "from": "left_knee",
            "to": "left_hip",
            "origin": "left_knee",          # for drawing
            "match_length_to": "main",
        },

        # Shank direction: knee -> ankle
        "main_vector": {
            "from": "left_knee",
            "to": "left_ankle",
            "project_direction_to_plane": False,
        },

        # -------- Angle computation --------
        "angle": {
            "type": "unsigned_between_vectors",  # plain arccos(dot)
            "unit": "degrees",

            # Optional: define flexion = 0 when straight
            # (straight leg gives ~180 deg between vectors)
            "postprocess": {
                "type": "one_eighty_minus",      # flexion = 180 - angle
            },
        },

        # -------- Visualization hints --------
        "viz": {
            "draw_plane": False,
            "draw_raw_vector": True,
            "draw_projected_vector": False,
            "draw_reference_vector": True,
            "label_angle": True,
        },
    },

}
