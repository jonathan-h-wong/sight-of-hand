#!/usr/bin/env bash
# Pull what the amodal mask renderer needs from s3://xray-hands/raw (labels and meshes only, no images).
#   bash scripts/prepare_data.sh DATA dexycb     # per-camera labels_*.npz + meta.yml, calibration, YCB meshes (~6 GB)
#   bash scripts/prepare_data.sh DATA graspxl    # the audited GraspXL v2 sample (same seed, 3000 objects per archive)
#   bash scripts/prepare_data.sh DATA mano
# graspxl and mano reuse sight-of-hand-eval's prepare script: set EVAL=<sight-of-hand-eval checkout> (default ../sight-of-hand-eval).
set -euo pipefail
DATA=${1:?data dir}; WHAT=${2:?dexycb|graspxl|mano}
mkdir -p "$DATA"; DATA=$(cd "$DATA" && pwd)  # absolute: the eval script runs from its own checkout
B=s3://xray-hands/raw
EVAL=${EVAL:-$(cd "$(dirname "$0")/../../sight-of-hand-eval" 2>/dev/null && pwd)}
mkdir -p "$DATA"

case $WHAT in
  dexycb)
    X="$DATA/dexycb"; mkdir -p "$X"
    aws s3 cp --quiet $B/dexycb/calibration.tar.gz - | tar -xzf - -C "$X"
    aws s3 cp --quiet $B/dexycb/models.tar.gz - | tar -xzf - -C "$X" --wildcards '*textured_simple.obj'
    # each subject tarball is ~12 GB (images); stream it and keep only the per-camera labels + meta (5 in parallel)
    aws s3 ls $B/dexycb/ | grep -oE '20[0-9]{6}-subject-[0-9]+\.tar\.gz' | \
      xargs -P 5 -I{} sh -c "aws s3 cp --quiet $B/dexycb/{} - | tar -xzf - -C '$X' --wildcards --no-anchored 'labels_*.npz' meta.yml && echo done {}"
    ;;
  graspxl)
    (cd "$EVAL" && GRASPXL_OBJECTS=${GRASPXL_OBJECTS:-3000} bash scripts/prepare_data.sh "$DATA" graspxl) ;;
  mano)
    (cd "$EVAL" && bash scripts/prepare_data.sh "$DATA" mano) ;;
esac
