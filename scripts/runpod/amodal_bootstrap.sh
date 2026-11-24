#!/usr/bin/env bash
# RunPod (CPU) job: render amodal + modal hand/object masks for DexYCB and GraspXL into s3://xray-hands/masks/amodal_v1/.
# Control and logs go through S3: $W/status. Code comes from $W/code_sight_of_hand.tar.gz and $W/code_eval.tar.gz
# (git archives of sight-of-hand and sight-of-hand-eval). Needs AWS_* in the pod env. Resumable: finished shards are
# skipped. On success the pod removes itself; on failure it stops.
#   start command: bash -c "pip install -q awscli && aws s3 cp s3://xray-hands/work/amodal_v1/bootstrap.sh /b.sh && bash /b.sh"
set -uo pipefail
W=${WORK:-s3://xray-hands/work/amodal_v1}
R=/workspace/amodal; mkdir -p $R/status && cd $R
DATASETS=${DATASETS:-"dexycb graspxl"}
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/status/pod.log; }
push_status() { aws s3 sync --quiet $R/status $W/status; }
finish() {  # $1 = DONE|FAILED
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

NW=$(nproc)
log "start; vCPU: $NW; RAM: $(free -g | awk '/Mem/{print $2}') GB"
echo RUNNING > $R/status/STATE; push_status
pip install -q libigl==2.6.3 trimesh embreex pillow pandas pyarrow scipy pyyaml boto3 >> $R/status/pip.log 2>&1 || finish FAILED
apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y unzip >/dev/null 2>&1
mkdir -p sight-of-hand sight-of-hand-eval
aws s3 cp --quiet $W/code_sight_of_hand.tar.gz - | tar -xzf - -C sight-of-hand || finish FAILED
aws s3 cp --quiet $W/code_eval.tar.gz - | tar -xzf - -C sight-of-hand-eval || finish FAILED
export PYTHONPATH=$R/sight-of-hand:$R/sight-of-hand-eval EVAL=$R/sight-of-hand-eval
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd sight-of-hand
bash scripts/prepare_data.sh $R/data mano > $R/status/prep_mano.log 2>&1 || finish FAILED
syncer() { while true; do sleep 300; push_status; done; }
syncer &

for ds in $DATASETS; do
  [ -f $R/status/$ds.done ] && continue
  log "prepare $ds"; push_status
  bash scripts/prepare_data.sh $R/data $ds > $R/status/prep_$ds.log 2>&1 || finish FAILED
  log "render $ds"; push_status
  python -W ignore scripts/render_amodal.py --dataset $ds --root $R/data/$ds --out $R/out \
    --mano-dir $R/data/mano/mano_v1_2/models --workers $NW > $R/status/render_$ds.log 2>&1 || finish FAILED
  cp $R/out/$ds/summary.json $R/status/summary_$ds.json
  touch $R/status/$ds.done; log "render $ds done -> s3://xray-hands/masks/amodal_v1/$ds/"; push_status
done
finish DONE
