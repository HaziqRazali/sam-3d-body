"""Stable output identities for SAM-3D-Body parquet rows."""

import hashlib
import posixpath
import re


_CAMERATED_DATASETS = frozenset(("harmony4d", "egohumans", "egoexo4d"))
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9.-]+")


def _safe_component(value: str) -> str:
    value = _SAFE_COMPONENT.sub("_", value).strip("_")
    return value or "unknown"


def output_identity(dataset: str, image: str, subject_idx: int, person_id: int) -> tuple[str, str]:
    """Return collision-safe ``(subject, action)`` names for one parquet row.

    The subject preserves the source sequence/group. For datasets whose image
    paths include a camera, the camera is moved into the action name so all
    views of one source sequence remain in one loader subject while individual
    rows stay unique.
    """
    normalized = posixpath.normpath(str(image)).lstrip("/")
    parts = normalized.split("/") if normalized not in ("", ".") else []
    filename = parts[-1] if parts else "sample.jpg"
    parent = parts[:-1]

    if not parent or parent == ["images"]:
        group_parts = []
        source_camera = ""
    elif dataset in _CAMERATED_DATASETS and len(parent) >= 2:
        group_parts = parent[:-1]
        source_camera = parent[-1]
    else:
        group_parts = parent
        source_camera = ""

    if len(group_parts) == 1:
        subject = _safe_component(group_parts[0])
    elif group_parts:
        group = "/".join(group_parts)
        readable = "__".join(_safe_component(part) for part in group_parts)
        digest = hashlib.sha1(group.encode("utf-8")).hexdigest()[:10]
        subject = f"{readable}__{digest}"
    else:
        subject = _safe_component(dataset)

    stem = _safe_component(posixpath.splitext(filename)[0])
    camera_prefix = f"{_safe_component(source_camera)}__" if source_camera else ""
    action = f"{camera_prefix}{stem}_s{int(subject_idx)}_p{int(person_id)}"
    return subject, action
