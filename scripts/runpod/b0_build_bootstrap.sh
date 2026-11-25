#!/usr/bin/env bash
# RunPod (CPU) job: build the B0 DexYCB s0 sample set (scripts/b0/build_b0_dataset.py) into
# s3://xray-hands/train/b0_dexycb_s0_v1/. Env: AWS_*, optional BUILD_ARGS. Logs in $W/status.
# Removes itself on success, stops on failure.
#   start command: bash -c "pip install -q awscli && aws s3 cp s3://xray-hands/work/b0/build/bootstrap.sh /b.sh && bash /b.sh"
set -uo pipefail
W=${WORK:-s3://xray-hands/work/b0/build}
R=/workspace/b0_build; mkdir -p $R/status && cd $R
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/status/pod.log; }
push_status() { aws s3 sync --quiet $R/status $W/status; }
finish() {
  log "$1"; echo "$1" > $R/status/STATE; push_status
  local m; [ "$1" = DONE ] && m=podTerminate || m=podStop
  if command -v runpodctl > /dev/null; then
    if [ "$1" = DONE ]; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi
  elif [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -s -H 'Content-Type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
      -d "{\"query\":\"mutation { $m(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" > /dev/null
  fi
  sleep infinity
}
log "start; vCPU $(nproc)"; echo RUNNING > $R/status/STATE; push_status
( while true; do sleep 120; push_status; done ) &
{ apt-get -qq update && apt-get -qq install -y unzip git; } > $R/status/apt.log 2>&1
pip install -q numpy pandas pyarrow pillow pyyaml boto3 trimesh scipy > $R/status/pip.log 2>&1 || finish FAILED
mkdir -p code && aws s3 cp --quiet $W/code_sight_of_hand.tar.gz - | tar -xzf - -C code || finish FAILED
git clone -q --depth 1 https://github.com/jonathan-h-wong/sight-of-hand-eval $R/eval || finish FAILED
D=$R/data; mkdir -p $D/dexycb $D/mano
log "prepare"; push_status
aws s3 cp --quiet s3://xray-hands/raw/dexycb/calibration.tar.gz - | tar -xzf - -C $D/dexycb || finish FAILED
aws s3 cp --quiet s3://xray-hands/raw/dexycb/models.tar.gz - | tar -xzf - -C $D/dexycb --wildcards '*textured_simple.obj' || finish FAILED
aws s3 cp --quiet s3://xray-hands/raw/mano/mano_v1_2.zip $D/mano/ && unzip -q -o $D/mano/mano_v1_2.zip -d $D/mano || finish FAILED
aws s3 cp --quiet $W/amodal_index_cams.parquet $D/amodal_index_cams.parquet || finish FAILED
cd code
log "build"; push_status
PYTHONPATH=$R/code:$R/eval python scripts/b0/build_b0_dataset.py --data $D --workers 10 ${BUILD_ARGS:-} \
  > $R/status/build.log 2>&1 || finish FAILED
log "build done"; finish DONE
