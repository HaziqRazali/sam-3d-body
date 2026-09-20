#!/usr/bin/env python3
"""
Mirror a split's cam0 directory onto NVMe so the cache builder stops doing
per-action HDD metadata lookups on the 2.4M-entry split dir (sda is the
bottleneck; source files already live on NVMe).

For each (dataset, split, kind) pair we:
  1. list the split's cam0 entries (filenames only, fast via scandir)
  2. create a mirror dir on NVMe containing ABSOLUTE symlinks to the
     NVMe source files (split entries are symlinks into the source tree,
     which itself is now a symlink to NVMe)
  3. verify entry count matches
  4. rename the HDD split cam0 -> cam0.hdd_split (backup) and symlink
     split cam0 -> NVMe mirror

USAGE:
  python mirror_split_cam0_to_nvme.py --dataset sa1b --kind mmpose [--dry-run]
  python mirror_split_cam0_to_nvme.py --all
"""
import argparse
import os
import shutil
import sys
import time

SPLIT_ROOT = "/data/haziq/sam3dbody/splits/sam3dbody_70_30_sa1b"
NVME_SRC   = "/home/haziq/nvme_stage/sam3dbody"          # source tree mirror on NVMe
NVME_MIR   = "/home/haziq/nvme_stage/split_mirror"       # split-cam0 mirrors
MODEL      = "rtmw-dw-l-m_simcc-cocktail14_270e-256x192-20231122"

# dataset -> split layout note: sa1b uses subject dir "sa1b" inside train/val;
# other datasets use per-sequence subject dirs (handled separately if needed).
DATASETS = ["coco", "mpii", "aic", "3dpw", "harmony4d", "egohumans", "sa1b"]


def split_cam0_for(dataset, split, kind):
    """Return the split cam0 dir for a (dataset, split, kind). sa1b-specific."""
    if kind == "mmpose":
        return os.path.join(
            SPLIT_ROOT, dataset, split, dataset, "mmpose", MODEL, "cam0")
    elif kind == "sam3d":
        return os.path.join(
            SPLIT_ROOT, dataset, split, dataset, "sam3d", "cam0")
    raise ValueError(kind)


def nvme_source_cam0_for(dataset, kind):
    """The NVMe source cam0 that split entries resolve to (sa1b layout)."""
    if kind == "mmpose":
        return os.path.join(
            NVME_SRC, dataset, "train", dataset, "mmpose", MODEL, "cam0")
    elif kind == "sam3d":
        return os.path.join(
            NVME_SRC, dataset, "train", dataset, "sam3d", "cam0")
    raise ValueError(kind)


def mirror_one(dataset, split, kind, dry_run=False):
    src_dir = split_cam0_for(dataset, split, kind)
    nvme_src = nvme_source_cam0_for(dataset, kind)
    if not os.path.isdir(src_dir):
        print(f"[skip] {dataset}/{split}/{kind}: no split dir {src_dir}")
        return 0
    if not os.path.isdir(nvme_src):
        print(f"[FAIL] {dataset}/{split}/{kind}: no nvme source {nvme_src}")
        return -1

    # mirror path mirrors the split path under NVME_MIR
    rel = os.path.relpath(src_dir, SPLIT_ROOT)
    mir_dir = os.path.join(NVME_MIR, rel)
    print(f"[mirror] {dataset}/{split}/{kind}\n  src={src_dir}\n  dst={mir_dir}\n  -> {nvme_src}")

    os.makedirs(mir_dir, exist_ok=True)
    t0 = time.time()
    n = 0
    with os.scandir(src_dir) as it:
        for e in it:
            if not (e.is_symlink() or e.is_file(follow_symlinks=False)):
                continue
            link = os.path.join(mir_dir, e.name)
            if os.path.islink(link):
                os.unlink(link)
            os.symlink(os.path.join(nvme_src, e.name), link)
            n += 1
            if n % 500000 == 0:
                print(f"  ... {n:,} symlinks ({time.time()-t0:.0f}s)", flush=True)
    print(f"  created {n:,} symlinks in {time.time()-t0:.0f}s")

    # verify count
    n_src = sum(1 for e in os.scandir(src_dir)
                if e.is_symlink() or e.is_file(follow_symlinks=False))
    n_mir = sum(1 for e in os.scandir(mir_dir)
                if e.is_symlink() or e.is_file(follow_symlinks=False))
    print(f"  verify: src={n_src:,} mirror={n_mir:,}")
    if n_src != n_mir:
        print(f"[FAIL] count mismatch src={n_src} mir={n_mir}")
        return -1

    if dry_run:
        print("[dry-run] not swapping")
        return 0

    # swap: rename HDD cam0 -> cam0.hdd_split, symlink cam0 -> mirror
    backup = src_dir + ".hdd_split"
    if os.path.exists(backup):
        print(f"[WARN] backup exists, removing: {backup}")
        shutil.rmtree(backup)
    os.rename(src_dir, backup)
    os.symlink(mir_dir, src_dir)
    print(f"[swap] {src_dir} -> {mir_dir}  (backup: {backup})")

    # verify through-symlink
    sample = next((e.name for e in os.scandir(src_dir) if True), None)
    if sample:
        p = os.path.join(src_dir, sample)
        ok = os.path.isfile(p)
        print(f"[verify] {p} -> {'OK' if ok else 'FAIL'}  ({os.path.realpath(p)})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--kind", choices=["mmpose", "sam3d"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = []
    if args.all:
        for ds in DATASETS:
            for split in ("train", "val"):
                for kind in ("mmpose", "sam3d"):
                    jobs.append((ds, split, kind))
    else:
        if not args.dataset or not args.kind:
            print("need --dataset and --kind, or --all")
            sys.exit(1)
        for split in ("train", "val"):
            jobs.append((args.dataset, split, args.kind))

    rc = 0
    for ds, split, kind in jobs:
        r = mirror_one(ds, split, kind, dry_run=args.dry_run)
        if r != 0:
            rc = 1
    print("=== done ===")
    sys.exit(rc)


if __name__ == "__main__":
    main()
