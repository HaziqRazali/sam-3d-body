#!/usr/bin/env python3
"""Run SAM-3D-Body on the person boxes in an MMPose staging manifest.

The staging script creates one manifest record per person, but several records
can refer to the same source image.  This runner groups those records and calls
SAM-3D once per image with all of its MMPose boxes.  The returned person i is
written to the NPZ path belonging to manifest record i.

This intentionally does not run a detector and does not select a centre
person.  The MMPose annotation boxes are the prompts supplied to SAM-3D.

The saved NPZ schema follows the compact image-dataset schema already consumed
by Synthium's MHR loader: parameters have a leading time dimension of one,
``scale_params`` remains the raw 28-dimensional SAM-3D coefficient vector,
and no large vertex array is stored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_STAGING_ROOT = Path(
    "/data/haziq/mmpose/splits/mmpose_70_30"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/sam-3d-body-dinov3/model.ckpt"
)
DEFAULT_MHR = REPO_ROOT / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
REQUIRED_OUTPUT_KEYS = (
    "shape_params",
    "global_rot",
    "body_pose_params",
    "expr_params",
    "scale_params",
    "pred_cam_t",
    "focal_length",
    "bbox",
)


@dataclass
class ImageGroup:
    key: str
    image_path: Path
    records: list[dict[str, Any]] = field(default_factory=list)


def read_manifest(
    manifest_path: Path,
    datasets: set[str] | None,
    splits: set[str],
) -> tuple[OrderedDict[str, ImageGroup], dict[str, int]]:
    groups: OrderedDict[str, ImageGroup] = OrderedDict()
    stats: dict[str, int] = {
        "manifest_records": 0,
        "ready_records": 0,
        "filtered_records": 0,
        "nonready_records": 0,
        "missing_staged_images": 0,
        "invalid_records": 0,
    }

    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid manifest JSON at {manifest_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Manifest line {line_number} is not an object")

            stats["manifest_records"] += 1
            if record.get("status") != "ready":
                stats["nonready_records"] += 1
                continue
            if datasets is not None and record.get("dataset") not in datasets:
                stats["filtered_records"] += 1
                continue
            if record.get("split") not in splits:
                stats["filtered_records"] += 1
                continue

            staged_value = record.get("staged_image")
            bbox = record.get("bbox_xyxy")
            output_value = record.get("sam3d_output")
            if not staged_value or not bbox or not output_value:
                stats["invalid_records"] += 1
                continue

            staged_path = Path(staged_value)

            try:
                bbox_array = np.asarray(bbox, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                stats["invalid_records"] += 1
                continue
            if bbox_array.size < 4 or not np.isfinite(bbox_array[:4]).all():
                stats["invalid_records"] += 1
                continue
            if bbox_array[2] <= bbox_array[0] or bbox_array[3] <= bbox_array[1]:
                stats["invalid_records"] += 1
                continue

            # The source path is the stable grouping key.  Fall back to the
            # staged link so a moved/rewritten manifest can still be used.
            source_value = record.get("source_image") or staged_value
            # The staging manifest writes the same normalized source string for
            # duplicate person records.  Avoid Path.resolve() here: resolving
            # every symlink in a 100k+ record manifest makes even --dry-run
            # unnecessarily slow.  The actual image is checked once per group
            # below.
            group_key = os.path.normpath(str(source_value))
            group = groups.get(group_key)
            if group is None:
                group = ImageGroup(key=group_key, image_path=staged_path)
                groups[group_key] = group
            elif group.image_path != staged_path:
                # This should only happen if equivalent links were generated
                # with different spellings.  Keep the first path but retain all
                # person records in the same inference batch.
                pass

            record["_bbox_array"] = bbox_array[:4]
            group.records.append(record)
            stats["ready_records"] += 1

    return groups, stats


def keep_existing_image_groups(
    groups: list[ImageGroup], stats: dict[str, int]
) -> list[ImageGroup]:
    """Check selected image links once per group.

    This is intentionally after shard/limit selection: a small smoke test
    should not stat every image in a large manifest just to process one image.
    A full run still checks every selected group.
    """
    kept: list[ImageGroup] = []
    for group in groups:
        if group.image_path.is_file():
            kept.append(group)
        else:
            stats["missing_staged_images"] += len(group.records)
    return kept


def output_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return all(key in data for key in REQUIRED_OUTPUT_KEYS)
    except (OSError, ValueError, EOFError):
        return False


def _time_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.reshape(1)
    return array[None, ...]


def _required_output(output: dict[str, Any], key: str) -> Any:
    if key not in output or output[key] is None:
        raise KeyError(f"SAM-3D output is missing {key!r}")
    return output[key]


def make_npz_payload(
    output: dict[str, Any],
    record: dict[str, Any],
    image_path: Path,
    inference_type: str,
) -> dict[str, Any]:
    """Convert one estimator person result to Synthium's compact NPZ format."""
    # These are the fields consumed by mocap_mainloader.py.  The estimator
    # returns one-dimensional arrays for one image; Synthium expects [T, ...].
    payload: dict[str, Any] = {
        "shape_params": _time_array(_required_output(output, "shape_params")),
        "global_rot": _time_array(_required_output(output, "global_rot")),
        "body_pose_params": _time_array(
            _required_output(output, "body_pose_params")
        ),
        "expr_params": _time_array(
            output.get("expr_params", np.zeros(72, dtype=np.float32))
        ),
        "scale_params": _time_array(_required_output(output, "scale_params")),
        "pred_cam_t": _time_array(_required_output(output, "pred_cam_t")),
        "focal_length": _time_array(_required_output(output, "focal_length")),
        "bbox": _time_array(_required_output(output, "bbox")),
        "frame_indices": np.zeros((1,), dtype=np.int32),
        "meta": np.array(
            [
                {
                    "mode": "sam3dbody_mmpose_image",
                    "dataset": record.get("dataset"),
                    "split": record.get("split"),
                    "annotation_id": record.get("annotation_id"),
                    "image_id": record.get("image_id"),
                    "person_id": record.get("person_id"),
                    "file_name": record.get("file_name"),
                    "image_path": str(image_path),
                    "input_w": record.get("image_width"),
                    "input_h": record.get("image_height"),
                    "subject": record.get("subject"),
                    "action": record.get("action"),
                    "inference_type": inference_type,
                }
            ],
            dtype=object,
        ),
    }

    # These are useful for diagnostics and inexpensive compared with vertices.
    for key in (
        "pred_keypoints_3d",
        "pred_keypoints_2d",
        "pred_joint_coords",
        "pred_global_rots",
        "mhr_model_params",
        "hand_pose_params",
    ):
        if output.get(key) is not None:
            payload[key] = _time_array(output[key])

    return payload


def atomic_save_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp.npz", dir=str(path.parent)
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with temp_path.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def build_estimator(args: argparse.Namespace):
    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    if not torch.cuda.is_available():
        raise RuntimeError("SAM-3D-Body requires CUDA; no CUDA device is available")

    device = torch.device("cuda")
    model, model_cfg = load_sam_3d_body(
        checkpoint_path=str(args.checkpoint_path),
        device=device,
        mhr_path=str(args.mhr_path),
    )

    fov_estimator = None
    if not args.no_fov:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name=args.fov_name,
            device=device,
            path=str(args.fov_path) if args.fov_path is not None else "",
        )

    human_segmentor = None
    if args.use_mask:
        from tools.build_sam import HumanSegmentor

        segmentor_path = args.segmentor_path
        if segmentor_path is None:
            segmentor_path = os.environ.get("SAM3D_SEGMENTOR_PATH", "")
        human_segmentor = HumanSegmentor(
            name=args.segmentor_name,
            device=device,
            path=str(segmentor_path),
        )

    # Deliberately no detector: annotation boxes are the authoritative person
    # prompts for this dataset conversion.  SAM2 is optional and only built
    # when --use-mask is explicitly requested.
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )


def write_error(error_path: Path, group: ImageGroup, exc: Exception) -> None:
    error_path.parent.mkdir(parents=True, exist_ok=True)
    item = {
        "image": str(group.image_path),
        "group_key": group.key,
        "records": [
            {
                "dataset": r.get("dataset"),
                "split": r.get("split"),
                "action": r.get("action"),
                "sam3d_output": r.get("sam3d_output"),
            }
            for r in group.records
        ],
        "error": repr(exc),
    }
    with error_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")


def process_group(
    estimator: Any,
    group: ImageGroup,
    args: argparse.Namespace,
) -> tuple[int, int]:
    boxes = np.stack([record["_bbox_array"] for record in group.records]).astype(
        np.float32
    )
    outputs = estimator.process_one_image(
        str(group.image_path),
        bboxes=boxes,
        use_mask=args.use_mask,
        inference_type=args.inference_type,
    )
    if not isinstance(outputs, (list, tuple)):
        raise RuntimeError(f"Unexpected SAM-3D result type: {type(outputs).__name__}")
    if len(outputs) != len(group.records):
        raise RuntimeError(
            f"SAM-3D returned {len(outputs)} persons for {len(group.records)} boxes "
            f"({group.image_path})"
        )

    saved = 0
    skipped = 0
    for record, output in zip(group.records, outputs):
        if not isinstance(output, dict):
            raise RuntimeError(
                f"Unexpected person result type: {type(output).__name__}"
            )
        output_path = Path(record["sam3d_output"])
        if not args.force and output_is_valid(output_path):
            skipped += 1
            continue
        payload = make_npz_payload(
            output=output,
            record=record,
            image_path=group.image_path,
            inference_type=args.inference_type,
        )
        atomic_save_npz(output_path, payload)
        saved += 1
    return saved, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM-3D-Body on MMPose person boxes from a staging manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_STAGING_ROOT / "mmpose_staging_manifest.jsonl",
    )
    parser.add_argument(
        "--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--mhr-path", type=Path, default=DEFAULT_MHR)
    parser.add_argument("--fov-name", default="moge2")
    parser.add_argument("--fov-path", type=Path, default=None)
    parser.add_argument("--segmentor-name", default="sam2")
    parser.add_argument("--segmentor-path", type=Path, default=None)
    parser.add_argument(
        "--no-fov",
        action="store_true",
        help="Use SAM-3D's fallback camera instead of the MoGe2 FOV estimator.",
    )
    parser.add_argument(
        "--inference-type",
        choices=("body", "full"),
        default="body",
        help="body is sufficient for Synthium MHR targets; full also refines hands.",
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val"), default=["train", "val"]
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Process at most this many unique images; 0 means all.",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--force", action="store_true", help="Recompute valid existing NPZ files."
    )
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Ask SAM-3D to generate SAM2 masks from the supplied boxes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/group the manifest without loading the model or writing NPZs.",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop at the first inference error."
    )
    parser.add_argument(
        "--log-every", type=int, default=10, help="Print progress every N images."
    )
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--error-log", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.manifest = args.manifest.expanduser().resolve()
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    args.mhr_path = args.mhr_path.expanduser().resolve()
    if args.fov_path is not None:
        args.fov_path = args.fov_path.expanduser()
    if args.segmentor_path is not None:
        args.segmentor_path = args.segmentor_path.expanduser()
    if args.summary_path is not None:
        args.summary_path = args.summary_path.expanduser()
    if args.error_log is not None:
        args.error_log = args.error_log.expanduser()

    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.max_images < 0:
        raise ValueError("--max-images must be >= 0")
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise ValueError("Require 0 <= --shard < --num-shards and --num-shards >= 1")
    if args.log_every < 1:
        raise ValueError("--log-every must be >= 1")

    datasets = set(args.datasets) if args.datasets else None
    groups, manifest_stats = read_manifest(args.manifest, datasets, set(args.splits))
    ordered_groups = list(groups.values())
    shard_groups = [
        group
        for index, group in enumerate(ordered_groups)
        if index % args.num_shards == args.shard
    ]
    if args.max_images:
        shard_groups = shard_groups[: args.max_images]
    shard_groups = keep_existing_image_groups(shard_groups, manifest_stats)

    records_selected = sum(len(group.records) for group in shard_groups)
    records_existing = sum(
        1
        for group in shard_groups
        for record in group.records
        if output_is_valid(Path(record["sam3d_output"]))
    )
    print(f"[INFO] Manifest: {args.manifest}", flush=True)
    print(
        f"[INFO] Groups: total={len(ordered_groups)} selected={len(shard_groups)} "
        f"records_selected={records_selected} existing_valid={records_existing}",
        flush=True,
    )
    print(f"[INFO] Manifest stats: {manifest_stats}", flush=True)
    print(
        f"[INFO] shard={args.shard}/{args.num_shards} "
        f"inference_type={args.inference_type} use_mask={args.use_mask}",
        flush=True,
    )

    if args.dry_run:
        print("[DRY-RUN] No model loaded and no NPZ files written", flush=True)
        return 0

    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not args.mhr_path.is_file():
        raise FileNotFoundError(f"MHR asset not found: {args.mhr_path}")

    estimator = build_estimator(args)
    shard_tag = f"_shard{args.shard}_of_{args.num_shards}" if args.num_shards > 1 else ""
    error_path = args.error_log or (
        args.manifest.parent / f"sam3d_mmpose_errors{shard_tag}.jsonl"
    )
    started = time.time()
    groups_done = 0
    groups_failed = 0
    records_saved = 0
    records_skipped = 0

    for group in shard_groups:
        try:
            saved, skipped = process_group(estimator, group, args)
            records_saved += saved
            records_skipped += skipped
        except Exception as exc:  # keep a long conversion resumable
            groups_failed += 1
            print(f"[ERROR] {group.image_path}: {exc!r}", flush=True)
            write_error(error_path, group, exc)
            if args.fail_fast:
                traceback.print_exc()
                raise
        groups_done += 1
        if groups_done == 1 or groups_done % args.log_every == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[PROGRESS] images={groups_done}/{len(shard_groups)} "
                f"saved={records_saved} skipped={records_skipped} "
                f"failed_groups={groups_failed} rate={groups_done / elapsed:.2f}/s",
                flush=True,
            )

    summary = {
        "manifest": str(args.manifest),
        "checkpoint_path": str(args.checkpoint_path),
        "mhr_path": str(args.mhr_path),
        "datasets": sorted(datasets) if datasets is not None else None,
        "splits": args.splits,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "inference_type": args.inference_type,
        "use_mask": bool(args.use_mask),
        "groups_selected": len(shard_groups),
        "records_selected": records_selected,
        "records_saved": records_saved,
        "records_skipped_existing": records_skipped,
        "groups_failed": groups_failed,
        "elapsed_seconds": time.time() - started,
        "error_log": str(error_path) if groups_failed else None,
    }
    summary_path = args.summary_path or (
        args.manifest.parent / f"sam3d_mmpose_run_summary{shard_tag}.json"
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"[DONE] {summary}", flush=True)
    print(f"[DONE] Summary: {summary_path}", flush=True)
    return 0 if groups_failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
