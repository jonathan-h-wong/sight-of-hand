#!/usr/bin/env bash
# Generic B0 GPU job. Env: AWS_*, WANDB_API_KEY, JOB (a name under s3://xray-hands/work/b0/jobs/), DEADLINE_S
# (hard self-removal, default 12 h). The job script s3://xray-hands/work/b0/jobs/$JOB/job.sh runs after the
# environment is set up, from the code checkout, with $OUT = /workspace/out; $OUT and the status dir are synced to
# s3://xray-hands/work/b0/jobs/$JOB/ every 2 minutes and at the end. The pod removes itself on success, failure or
# deadline, so a stuck job cannot keep billing.
#   start: aws s3 cp s3://xray-hands/work/b0/jobs/$JOB/bootstrap.sh /b.sh && bash /b.sh
set -uo pipefail
J=s3://xray-hands/work/b0/jobs/$JOB
R=/workspace/run; mkdir -p $R/status /workspace/out && cd $R
export OUT=/workspace/out
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/status/pod.log; }
push() { aws s3 sync --quiet $R/status $J/status; aws s3 sync --quiet $OUT $J/out --exclude "*/renders_cond/*" --exclude "*_latents/*" --exclude "data/*"; }
die() {
  log "$1"; echo "$1" > $R/status/STATE; push
  if command -v runpodctl > /dev/null; then runpodctl remove pod "$RUNPOD_POD_ID"
  elif [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -s -H 'Content-Type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
      -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" > /dev/null
  fi
  sleep 300; kill 1
}
( sleep ${DEADLINE_S:-43200}; die "DEADLINE ${DEADLINE_S:-43200}s" ) &
( while true; do sleep 120; push; done ) &
log "start $JOB; $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); vCPU $(nproc); RAM $(free -g | awk '/Mem/{print $2}') GB"
echo RUNNING > $R/status/STATE; push
aws s3 cp --quiet s3://xray-hands/work/b0/code/b0_gpu_setup.sh $R/setup.sh && bash $R/setup.sh > $R/status/setup.log 2>&1 || die FAILED_SETUP
source /root/b0env.sh
mkdir -p /root/code && aws s3 cp --quiet s3://xray-hands/work/b0/code/code_sight_of_hand.tar.gz - | tar -xzf - -C /root/code || die FAILED_CODE
export PYTHONPATH=/root/code:$PX:$EVAL
log "setup ok ($ATTN_BACKEND)"; push
aws s3 cp --quiet $J/job.sh $R/job.sh || die FAILED_JOB_FETCH
cd /root/code && bash $R/job.sh > $R/status/job.log 2>&1 && die DONE || die FAILED
