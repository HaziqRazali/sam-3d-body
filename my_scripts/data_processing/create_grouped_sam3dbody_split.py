#!/usr/bin/env python
"""Build a leakage-safe, symlink-only SAM-3D-Body train/val view.

The converted SAM-3D-Body source tree is never modified. Each output view
contains only SAM-3D-Body dataset directories, a manifest, and relative
symlinks to source images, MHR NPZs, and MMPose JSON files. Sequence-grouped
datasets use one directory symlink per group; image-grouped datasets use file
symlinks.

Create a different output root to try another validation ratio later. The
output is intentionally source-specific: mocap datasets are not added here.

Examples:
    python create_grouped_sam3dbody_split.py \
        --source-root /data/haziq/sam3dbody \
        --output-root /data/haziq/sam3dbody/splits/sam3dbody_60_40 \
        --val-ratio 0.40 --seed 20260825 --dry-run

    python create_grouped_sam3dbody_split.py \
        --source-root /data/haziq/sam3dbody \
        --output-root /data/haziq/sam3dbody/splits/sam3dbody_60_40 \
        --val-ratio 0.40 --seed 20260825
"""

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq

from sam3dbody_identity import output_identity


DEFAULT_DATASETS = ("coco", "mpii", "aic", "sa1b", "3dpw", "harmony4d", "egohumans")
DEFAULT_MMPOSE_MODEL = "rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122"


def _valid_row(row: dict) -> bool:
    if not bool(row.get("mhr_valid", False)):
        return False
    bbox = np.asarray(row["bbox"], dtype=np.float64).reshape(-1)
    return (
        row.get("bbox_format") == "xyxy"
        and bbox.size >= 4
        and np.isfinite(bbox[:4]).all()
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _group_key(dataset: str, image: str) -> str:
    """Return the unit that must stay in one split."""
    parts = PurePosixPath(str(image)).parts
    if dataset in ("coco", "mpii", "aic", "sa1b"):
        return str(image)
    if dataset == "3dpw":
        return parts[0]
    if dataset in ("harmony4d", "egohumans"):
        if len(parts) < 3:
            raise ValueError(f"Unexpected {dataset} image path: {image}")
        return "/".join(parts[:-2])
    raise ValueError(f"Unsupported dataset: {dataset}")


def _group_order(dataset: str, group: str, seed: int) -> tuple[str, str]:
    token = f"{seed}:{dataset}:{group}".encode("utf-8")
    return hashlib.sha256(token).hexdigest(), group


def _split_groups(
    dataset: str, group_counts: dict[str, int], val_ratio: float, seed: int
) -> dict[str, str]:
    target_val = int(round(sum(group_counts.values()) * val_ratio))
    ordered = sorted(
        group_counts,
        key=lambda group: _group_order(dataset, group, seed),
    )

    assignments: dict[str, str] = {}
    val_rows = 0
    seen_rows = 0
    total_rows = sum(group_counts.values())
    for group in ordered:
        count = group_counts[group]
        remaining_after = total_rows - seen_rows - count
        target_after = target_val - val_rows
        if target_after <= 0:
            use_val = False
        elif target_after >= remaining_after + count:
            use_val = True
        else:
            use_val = abs(val_rows + count - target_val) <= abs(val_rows - target_val)
        assignments[group] = "val" if use_val else "train"
        if use_val:
            val_rows += count
        seen_rows += count
    return assignments


def _symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve(strict=False) == source.resolve(strict=False):
            return
        raise RuntimeError(f"Conflicting existing symlink: {destination}")
    if destination.exists():
        raise RuntimeError(f"Conflicting existing file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def _source_paths(
    source_root: Path,
    dataset: str,
    subject: str,
    action: str,
    mmpose_model: str,
) -> dict[str, Path]:
    base = source_root / dataset / "train" / subject
    return {
        "image": base / "videos" / "cam0" / f"{action}.jpg",
        "npz": base / "sam3d" / "cam0" / f"{action}_mhr_outputs.npz",
        "json": base / "mmpose" / mmpose_model / "cam0" / f"{action}.json",
    }


def _iter_parquet_rows(annotation_root: Path, dataset: str):
    annotation_dir = annotation_root / "annotations" / f"{dataset}_train"
    parquets = sorted(annotation_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found: {annotation_dir}")
    for parquet_path in parquets:
        table = pq.read_table(
            parquet_path,
            columns=["image", "subject_idx", "person_id", "mhr_valid", "bbox", "bbox_format"],
        )
        yield from table.to_pylist()


def _scan_dataset(annotation_root: Path, dataset: str, seed: int, val_ratio: float) -> dict:
    group_counts: Counter[str] = Counter()
    group_subjects: dict[str, str] = {}
    total_rows = 0
    skipped_rows = 0
    identities: set[tuple[str, str]] = set()
    for row in _iter_parquet_rows(annotation_root, dataset):
        total_rows += 1
        if not _valid_row(row):
            skipped_rows += 1
            continue
        group = _group_key(dataset, row["image"])
        group_counts[group] += 1
        identity = output_identity(
            dataset, row["image"], row["subject_idx"], row["person_id"]
        )
        group_subjects.setdefault(group, identity[0])
        if identity in identities:
            raise RuntimeError(f"Duplicate output identity for {dataset}: {identity}")
        identities.add(identity)

    assignments = _split_groups(dataset, group_counts, val_ratio, seed)
    train_rows = sum(count for group, count in group_counts.items() if assignments[group] == "train")
    val_rows = sum(count for group, count in group_counts.items() if assignments[group] == "val")
    return {
        "total_annotation_rows": total_rows,
        "valid_rows": sum(group_counts.values()),
        "skipped_rows": skipped_rows,
        "groups": len(group_counts),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "group_counts": dict(group_counts),
        "group_subjects": group_subjects,
        "assignments": assignments,
    }


def _materialize_dataset(
    source_root: Path,
    output_root: Path,
    dataset: str,
    scan: dict,
    mmpose_model: str,
) -> Counter:
    stats: Counter[str] = Counter()
    if dataset in ("3dpw", "harmony4d", "egohumans"):
        for group, split in scan["assignments"].items():
            subject = scan["group_subjects"][group]
            source_subject = source_root / dataset / "train" / subject
            if not source_subject.is_dir():
                raise FileNotFoundError(f"Missing source subject directory: {source_subject}")
            _symlink(source_subject, output_root / dataset / split / subject)
            stats[split] += scan["group_counts"][group]
        return stats

    for row in _iter_parquet_rows(source_root, dataset):
        if not _valid_row(row):
            stats["skipped_invalid"] += 1
            continue
        group = _group_key(dataset, row["image"])
        split = scan["assignments"][group]
        subject, action = output_identity(
            dataset, row["image"], row["subject_idx"], row["person_id"]
        )
        sources = _source_paths(source_root, dataset, subject, action, mmpose_model)
        missing = [name for name, path in sources.items() if not path.exists()]
        if missing:
            # Skip (do not abort): a row may lack assets when its source image
            # was missing from the upstream archive (e.g. SA1B gold-archive gaps,
            # MMPose 'missing_image'), so no MMPose JSON could be produced.
            stats["skipped_missing_assets"] += 1
            print(
                f"  [skip-missing] {dataset} {row['image']} -> {subject}/{action}: "
                f"missing {missing}",
                flush=True,
            )
            continue

        destination_base = output_root / dataset / split / subject
        _symlink(sources["image"], destination_base / "videos" / "cam0" / f"{action}.jpg")
        _symlink(sources["npz"], destination_base / "sam3d" / "cam0" / f"{action}_mhr_outputs.npz")
        _symlink(
            sources["json"],
            destination_base / "mmpose" / mmpose_model / "cam0" / f"{action}.json",
        )
        stats[split] += 1
    return stats


def _write_manifest(
    output_root: Path,
    args: argparse.Namespace,
    scans: OrderedDict,
) -> None:
    manifest = {
        "format": "sam3dbody_grouped_split_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(Path(args.source_root).resolve()),
        "source_subset": "train",
        "output_root": str(output_root.resolve()),
        "datasets": list(scans),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "mmpose_model": args.mmpose_model,
        "link_mode": "relative_symlink (directory for sequence datasets, files for image datasets)",
        "grouping": {
            "coco": "unique image",
            "mpii": "unique image",
            "aic": "unique image",
            "sa1b": "unique image",
            "3dpw": "sequence",
            "harmony4d": "capture sequence excluding camera",
            "egohumans": "sequence excluding camera",
        },
        "datasets_stats": {},
    }
    for dataset, scan in scans.items():
        manifest["datasets_stats"][dataset] = {
            key: scan[key]
            for key in (
                "total_annotation_rows",
                "valid_rows",
                "skipped_rows",
                "groups",
                "train_rows",
                "val_rows",
            )
        }
        manifest["datasets_stats"][dataset]["groups_manifest"] = [
            {
                "group": group,
                "rows": scan["group_counts"][group],
                "split": scan["assignments"][group],
            }
            for group in sorted(scan["group_counts"])
        ]
    (output_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--val-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--mmpose-model", default=DEFAULT_MMPOSE_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be strictly between 0 and 1")
    datasets = tuple(name.strip() for name in args.datasets.split(",") if name.strip())
    if not datasets:
        parser.error("--datasets must not be empty")

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.is_dir():
        parser.error(f"source root does not exist: {source_root}")

    scans: OrderedDict[str, dict] = OrderedDict()
    for dataset in datasets:
        print(f"Scanning {dataset} annotations ...", flush=True)
        scans[dataset] = _scan_dataset(source_root, dataset, args.seed, args.val_ratio)
        scan = scans[dataset]
        print(
            f"  rows={scan['valid_rows']:,} groups={scan['groups']:,} "
            f"train={scan['train_rows']:,} val={scan['val_rows']:,}",
            flush=True,
        )

    if args.dry_run:
        print("DRY RUN: source tree unchanged; no split view created.", flush=True)
        return

    if output_root.exists():
        parser.error(f"output root already exists; choose a new root: {output_root}")

    staging_root = output_root.parent / f".{output_root.name}.partial"
    if staging_root.exists():
        parser.error(f"staging root already exists; remove or rename it: {staging_root}")
    staging_root.mkdir(parents=True)
    try:
        for dataset, scan in scans.items():
            print(f"Materializing {dataset} symlinks ...", flush=True)
            stats = _materialize_dataset(
                source_root, staging_root, dataset, scan, args.mmpose_model
            )
            print(
                f"  train={stats['train']:,} val={stats['val']:,} "
                f"skipped_invalid={stats['skipped_invalid']:,} "
                f"skipped_missing_assets={stats['skipped_missing_assets']:,}",
                flush=True,
            )
        _write_manifest(staging_root, args, scans)
        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    print(f"Created split view: {output_root}", flush=True)


if __name__ == "__main__":
    main()
