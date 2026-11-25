#!/usr/bin/env bash
# RunPod (CPU) job: run one sight-of-hand script ($CMD, from the repo root) with logs in $W/status.
# Env: AWS_*, CMD, WORK. Removes itself on success, stops on failure.
#   start command: bash -c "pip install -q awscli && aws s3 cp $WORK/bootstrap.sh /b.sh && bash /b.sh"
set -uo pipefail
W=${WORK:?WORK}
R=/workspace/cpujob; mkdir -p $R/status && cd $R
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
log "run: $CMD"; push_status
PYTHONPATH=$R/code bash -c "$CMD" > $R/status/job.log 2>&1 || finish FAILED
log "job done"; finish DONE
