# One LoRA stage on the train encoding. Env: STAGE (ss64|shape512|shape1024), STEPS, TRAIN_ARGS (extra flags).
# Resumes from the LoRA checkpoints already synced to this job's S3 out/ (a restarted pod picks up where it stopped).
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
J=s3://xray-hands/work/b0/jobs/$JOB
mkdir -p /workspace/b0 && aws s3 cp --quiet $E/train.tar - | tar -xf - -C /workspace/b0
aws s3 sync --quiet $J/out/$STAGE $OUT/$STAGE || true
cd $PX
python3 /root/code/scripts/b0/train_b0.py --stage $STAGE --data /workspace/b0 --out $OUT/$STAGE --steps ${STEPS:-20000} \
  --resume --wandb --run-name ${RUN_NAME:-b0_$STAGE} ${TRAIN_ARGS:-}
ls -la $OUT/$STAGE
