#!/usr/bin/env bash
# RunPod (GPU, H100/H200: the natten wheel ships sm_90 kernels only) job: one v2 experiment
# (docs/amodal_branch_spec_v2.md) on the DexYCB s0 shards, logging to W&B, then overlays (~50 per occlusion bin).
# Env: AWS_*, WANDB_API_KEY, EXP (E1, E2, ...), TRAIN_ARGS (train_amodal_v2.py flags for this experiment).
# Control and logs: $W/status. A 30-step smoke test (W&B off; includes the NAF backward check and an overlay dump)
# runs first. On success the pod removes itself; on failure it stops.
#   start command: bash -c "apt-get -qq update && apt-get -qq install -y python3-pip curl >/dev/null && pip install -q awscli && aws s3 cp s3://xray-hands/work/amodal_v2/bootstrap.sh /b.sh && bash /b.sh"
set -uo pipefail
ROOT=s3://xray-hands/work/amodal_v2
W=$ROOT/$EXP
DATA_PREFIX=${DATA_PREFIX:-s3://xray-hands/train/amodal_dexycb_s0_v1}
R=/workspace/amodal_v2; mkdir -p $R/status && cd $R
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/status/pod.log; }
push_status() { aws s3 sync --quiet $R/status $W/status; }
finish() {
  log "$1"; echo "$1" > $R/status/STATE; push_status; aws s3 sync --quiet $R/wandb $W/wandb 2>/dev/null
  local m; [ "$1" = DONE ] && m=podTerminate || m=podStop
  if command -v runpodctl > /dev/null; then
    if [ "$1" = DONE ]; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi
  elif [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -s -H 'Content-Type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" https://api.runpod.io/graphql \
      -d "{\"query\":\"mutation { $m(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) }\"}" > /dev/null
  fi
  sleep infinity
}
export WANDB_DIR=$R
if [ -z "${WANDB_API_KEY:-}" ]; then export WANDB_MODE=offline; fi
mkdir -p $R/wandb
syncer() { while true; do sleep 120; push_status; aws s3 sync --quiet $R/wandb $W/wandb; done; }

log "exp $EXP; args: ${TRAIN_ARGS:-}; W&B mode: ${WANDB_MODE:-online}"
log "start; $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); vCPU $(nproc)"
echo RUNNING > $R/status/STATE; push_status
syncer &
apt-get -qq install -y git >/dev/null 2>&1
{
  pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 &&
  pip install -q "https://github.com/LDYang694/Storages/releases/download/20260430/natten-0.21.0+torch2.6cu124-cp310-cp310-linux_x86_64.whl" &&
  pip install -q transformers==4.57.3 safetensors einops timm wandb pillow pandas pyarrow boto3 numpy
} > $R/status/pip.log 2>&1 || finish FAILED
git clone -q --depth 1 https://github.com/valeoai/NAF $R/NAF || finish FAILED
export NAF_DIR=$R/NAF
mkdir -p code && aws s3 cp --quiet $ROOT/code_sight_of_hand.tar.gz - | tar -xzf - -C code || finish FAILED
log "download shards"; push_status
mkdir -p $R/data/shards
aws s3 sync --quiet $DATA_PREFIX/shards $R/data/shards --exclude "*" --include "train_*" --include "val_*" || finish FAILED
aws s3 cp --quiet $DATA_PREFIX/index.parquet $R/data/index.parquet || finish FAILED
aws s3 cp --quiet $DATA_PREFIX/joints_2d.parquet $R/data/joints_2d.parquet || finish FAILED
log "data: $(ls $R/data/shards | wc -l) shards, $(du -sh $R/data/shards | cut -f1)"
cd code
NW=$(( $(nproc) > 16 ? 16 : $(nproc) ))
log "smoke test"; push_status
{ WANDB_MODE=disabled python3 scripts/train_amodal_v2.py --data $R/data --exp $EXP ${TRAIN_ARGS:-} --steps 30 \
    --val-every 30 --val-images 64 --no-full-val --workers $NW --s3-prefix "" --run-name smoke &&
  python3 scripts/dump_overlays.py --data $R/data --ckpt runs/smoke/last.pt --out $R/smoke_ovl --per-bin 1; } \
  > $R/status/smoke.log 2>&1 || finish FAILED
grep -E "NAF backward|trainable|step 30" $R/status/smoke.log | while read -r l; do log "smoke: $l"; done
RUN=v2_${EXP}_$(date -u +%m%d_%H%M)
log "train $RUN"; push_status
python3 scripts/train_amodal_v2.py --data $R/data --exp $EXP --run-name $RUN --workers $NW ${TRAIN_ARGS:-} \
  > $R/status/train.log 2>&1 || finish FAILED
log "overlays"; push_status
python3 scripts/dump_overlays.py --data $R/data --ckpt runs/$RUN/best.pt --out $R/overlays --per-bin 50 \
  --s3-prefix work/amodal_v2/$EXP/overlays/$RUN > $R/status/overlays.log 2>&1 || finish FAILED
finish DONE
