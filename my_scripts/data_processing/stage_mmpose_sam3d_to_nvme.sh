#!/bin/bash
# Stage mmpose/ + sam3d/ dirs from the HDD source tree to NVMe, then swap to
# symlinks so the cache build reads from NVMe (IOPS-bound bottleneck).
#
# PHASE 1 (--copy):   cp -a each mmpose/sam3d dir -> /home/haziq/nvme_stage/...
#                     verify file counts match, log to a manifest.
# PHASE 2 (--swap):   for each dir in manifest: rename source to <dir>.hdd_bak
#                     and symlink source -> NVMe.  (backup kept until verified)
# PHASE 3 (--restore): undo a swap by removing symlink and renaming .hdd_bak back.
#
# Usage:
#   bash stage_mmpose_sam3d_to_nvme.sh --copy
#   bash stage_mmpose_sam3d_to_nvme.sh --swap
#   bash stage_mmpose_sam3d_to_nvme.sh --restore
#
set -uo pipefail

SRC_ROOT="/data/haziq/sam3dbody"
DST_ROOT="/home/haziq/nvme_stage/sam3dbody"
MANIFEST="/home/haziq/nvme_stage/manifest.txt"
LOG="/home/haziq/nvme_stage/stage.log"
DATASETS=(coco mpii aic 3dpw harmony4d egohumans sa1b)

mkdir -p "$DST_ROOT"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

collect_dirs() {
  # echo all mmpose/sam3d dirs (rel to SRC_ROOT), excluding cached_data
  for ds in "${DATASETS[@]}"; do
    find "$SRC_ROOT/$ds" -type d \( -name mmpose -o -name sam3d \) \
      -not -path '*/cached_data/*' 2>/dev/null
  done
}

count_files() { find "$1" -type f 2>/dev/null | wc -l; }

cmd_copy() {
  log "=== PHASE 1: copy ==="
  : > "$MANIFEST"   # fresh manifest only for a fresh copy pass
  local n=0
  while IFS= read -r src; do
    rel="${src#$SRC_ROOT/}"
    dst="$DST_ROOT/$rel"
    n=$((n+1))
    if [ -d "$dst" ]; then
      local sc dc
      sc=$(count_files "$src"); dc=$(count_files "$dst")
      if [ "$sc" = "$dc" ]; then
        log "[skip] $rel ($sc files already on nvme)"
        echo "$src" >> "$MANIFEST"
        continue
      fi
      log "[partial] $rel: src=$sc dst=$dc -> recopying"
      rm -rf "$dst"
    fi
    log "[copy] $rel"
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst" || { log "  FAILED copy $rel"; exit 1; }
    local sc dc
    sc=$(count_files "$src"); dc=$(count_files "$dst")
    if [ "$sc" != "$dc" ]; then
      log "  MISMATCH $rel: src=$sc dst=$dc"; exit 1
    fi
    log "  ok: $sc files"
    echo "$src" >> "$MANIFEST"
  done < <(collect_dirs)
  log "=== copy phase done: $n dirs (see $MANIFEST) ==="
}

cmd_swap() {
  log "=== PHASE 2: swap to symlinks (backup kept as .hdd_bak) ==="
  # If the manifest is empty (e.g. an interrupted --copy or a wiped file),
  # rebuild it from the directory scan so swap can still run safely.
  if [ ! -s "$MANIFEST" ]; then
    log "  manifest empty/missing -> rebuilding from directory scan"
    collect_dirs > "$MANIFEST"
  fi
  local n=0
  while IFS= read -r src; do
    [ -z "$src" ] && continue
    rel="${src#$SRC_ROOT/}"
    dst="$DST_ROOT/$rel"
    n=$((n+1))
    if [ -L "$src" ]; then
      log "[skip-symlink] $rel already a symlink"
      continue
    fi
    # safety: NVMe copy must exist and match
    sc=$(count_files "$src"); dc=$(count_files "$dst")
    if [ ! -d "$dst" ] || [ "$sc" != "$dc" ]; then
      log "  ABORT $rel: nvme copy missing/mismatch (src=$sc dst=$dc). Run --copy first."
      exit 1
    fi
    bak="$src.hdd_bak"
    if [ -e "$bak" ]; then log "  backup exists $bak -> skipping"; fi
    mv "$src" "$bak" || { log "  FAILED rename $rel"; exit 1; }
    ln -s "$dst" "$src" || { log "  FAILED symlink $rel"; exit 1; }
    # verify resolution
    local sample resolved
    sample=$(find "$src" -type f 2>/dev/null | head -1)
    if [ -n "$sample" ] && [ -f "$sample" ]; then
      resolved=$(readlink -f "$sample")
      log "  ok $rel -> $resolved"
    else
      log "  WARN: no readable file through symlink $rel"
    fi
  done < "$MANIFEST"
  log "=== swap phase done: $n entries ==="
}

cmd_restore() {
  log "=== PHASE 3: restore from .hdd_bak ==="
  while IFS= read -r src; do
    [ -z "$src" ] && continue
    bak="$src.hdd_bak"
    if [ -L "$src" ] && [ -d "$bak" ]; then
      rm -f "$src"
      mv "$bak" "$src"
      log "restored $src"
    fi
  done < "$MANIFEST"
  log "=== restore done ==="
}

case "${1:-}" in
  --copy)    cmd_copy ;;
  --swap)    cmd_swap ;;
  --restore) cmd_restore ;;
  *) echo "usage: $0 {--copy|--swap|--restore}"; exit 1 ;;
esac
