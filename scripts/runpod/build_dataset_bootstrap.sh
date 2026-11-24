#!/usr/bin/env bash
# RunPod (CPU) job: build the DexYCB s0 amodal training set (scripts/build_amodal_dataset.py) into
# s3://xray-hands/train/amodal_dexycb_s0_v1/. Env: AWS_*. Logs in $W/status. Removes itself on success, stops on failure.
#   start command: bash -c "pip install -q awscli && aws s3 cp s3://xray-hands/work/amodal_ds/bootstrap.sh /b.sh && bash /b.sh"
set -uo pipefail
W=${WORK:-s3://xray-hands/work/amodal_ds}
R=/workspace/amodal_ds; mkdir -p $R/status && cd $R
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
pip install -q numpy pandas pyarrow pillow pyyaml boto3 trimesh > $R/status/pip.log 2>&1 || finish FAILED
mkdir -p code && aws s3 cp --quiet $W/code_sight_of_hand.tar.gz - | tar -xzf - -C code || finish FAILED
cd code
( while true; do sleep 120; push_status; done ) &
log "build"; push_status
PYTHONPATH=$R/code python scripts/build_amodal_dataset.py --subjects all --workers 10 > $R/status/build.log 2>&1 || finish FAILED
log "build done"; finish DONE
