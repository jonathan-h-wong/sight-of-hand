# One LoRA stage with coordinate augmentation and/or the decoded-occupancy loss. Env: STAGE (ss64|shape512), STEPS,
# RUN_NAME, TRAIN_ARGS (train_b0.py flags), HAND=1 to build or fetch the train hand cells (gen_hand_vox.py; cached at
# s3://xray-hands/work/b0/hand_vox64/), SMOKE_CONFIGS (';'-separated "STAGE|TRAIN_ARGS" for a short smoke run of each).
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
W=s3://xray-hands/work/b0
J=$W/jobs/$JOB
mkdir -p /workspace/b0 /workspace/hand_vox64 $OUT
aws s3 cp --quiet $E/train.tar - | tar -xf - -C /workspace/b0
if [ "${HAND:-0}" = 1 ]; then
  if [ "$(aws s3 ls $W/hand_vox64.tar)" ]; then
    aws s3 cp --quiet $W/hand_vox64.tar - | tar -xf - -C /workspace
  elif [ "$(aws s3 ls $W/hand_vox64/DONE)" ]; then
    aws s3 sync --quiet $W/hand_vox64 /workspace/hand_vox64
    tar -cf - -C /workspace hand_vox64 | aws s3 cp - $W/hand_vox64.tar
  else
    mkdir -p /workspace/mano && aws s3 cp --quiet s3://xray-hands/raw/mano/mano_v1_2.zip /workspace/mano/ \
      && unzip -q -o /workspace/mano/mano_v1_2.zip -d /workspace/mano
    python3 /root/code/scripts/b0/gen_hand_vox.py --mano /workspace/mano/mano_v1_2/models --out /workspace/hand_vox64 --split train --workers 48
    aws s3 sync --quiet /workspace/hand_vox64 $W/hand_vox64 && echo ok | aws s3 cp - $W/hand_vox64/DONE
  fi
  ls /workspace/hand_vox64 | wc -l
fi
cd $PX
if [ -n "${SMOKE_CONFIGS:-}" ]; then
  IFS=';' read -ra CF <<< "$SMOKE_CONFIGS"
  for i in "${!CF[@]}"; do
    st=${CF[$i]%%|*}; ar=${CF[$i]#*|}
    python3 /root/code/scripts/b0/train_b0.py --stage $st --data /workspace/b0 --out $OUT/smoke$i --steps 30 --log-every 5 $ar
  done
  exit 0
fi
aws s3 sync --quiet $J/out/$STAGE $OUT/$STAGE || true
python3 /root/code/scripts/b0/train_b0.py --stage $STAGE --data /workspace/b0 --out $OUT/$STAGE --steps ${STEPS:-8000} \
  --resume --wandb --run-name ${RUN_NAME:-b0_$STAGE_aug} ${TRAIN_ARGS:-}
ls -la $OUT/$STAGE
