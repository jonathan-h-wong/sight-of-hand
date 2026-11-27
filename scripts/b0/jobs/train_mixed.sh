# Stage 2 (512) LoRA mixed training: same recipe as B0's train_shape512 (batch 16 split 2, LR 2e-4, 8K steps),
# with --pred-coords from the gen_pred_* jobs. Env: GEN_JOBS (space-separated job names), PRED_FRAC, EXTRA, STEPS,
# SMOKE=1 (use whatever predictions are uploaded so far, without waiting for the jobs to finish), INIT_FROM (a job whose
# out/shape512 checkpoints to continue from), TRAIN_EXTRA (extra train_b0.py flags).
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
W=s3://xray-hands/work/b0/jobs
J=$W/$JOB
mkdir -p /workspace/b0 /workspace/pred && aws s3 cp --quiet $E/train.tar - | tar -xf - -C /workspace/b0
for g in $GEN_JOBS; do
  [ "${SMOKE:-0}" = 1 ] || until [ "$(aws s3 cp $W/$g/status/STATE - 2>/dev/null)" = "DONE" ]; do sleep 60; done
  aws s3 sync --quiet $W/$g/out/pred/coords /workspace/pred
done
ls /workspace/pred | wc -l
python3 - <<'PY' > $OUT/pred_stats.txt
import numpy as np, glob, os
r = "/workspace/b0/train/shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"
iou = []
for f in sorted(glob.glob("/workspace/pred/*.npz")):
    g = os.path.join(r, os.path.basename(f)[:-4], "view00.npz")
    if not os.path.exists(g):
        continue
    P = {tuple(x) for x in np.load(f)["coords"].astype(int).tolist()}
    G = {tuple(x) for x in np.load(g)["coords"].astype(int).tolist()}
    iou.append((len(P & G) / max(len(P | G), 1), len(P - G) / max(len(P), 1), len(G - P) / max(len(G), 1)))
a = np.array(iou)
print("train preds", len(a), "voxel IoU median", np.median(a[:, 0]), "p10", np.percentile(a[:, 0], 10),
      "extra share", np.median(a[:, 1]), "missed share", np.median(a[:, 2]))
PY
cat $OUT/pred_stats.txt
aws s3 sync --quiet $J/out/shape512 $OUT/shape512 || true
if [ -n "${INIT_FROM:-}" ] && ! ls $OUT/shape512/ckpts/misc_step*.pt >/dev/null 2>&1; then
  aws s3 sync --quiet $W/$INIT_FROM/out/shape512/ckpts $OUT/shape512/ckpts
fi
cd $PX
python3 /root/code/scripts/b0/train_b0.py --stage shape512 --data /workspace/b0 --out $OUT/shape512 --steps ${STEPS:-8000} \
  --batch 16 --batch-split 2 --resume --wandb --run-name ${RUN_NAME:-b0_shape512_mixed} \
  --pred-coords /workspace/pred --pred-frac ${PRED_FRAC:-0.5} --extra-target ${EXTRA:-mean} ${TRAIN_EXTRA:-}
ls -la $OUT/shape512
