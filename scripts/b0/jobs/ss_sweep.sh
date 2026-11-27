# Stage 1 inference sweep (K-averaging, threshold, CFG, steps) + cascade. Env: EVAL_JOBS, ROW, LORA_SS, LORA_512, N, SHARD.
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
W=s3://xray-hands/work/b0/jobs
mkdir -p /workspace/b0 /workspace/runs /workspace/lora $OUT
aws s3 cp --quiet $E/val.tar - | tar -xf - -C /workspace/b0 --wildcards 'val/shape_latents/*'
aws s3 cp --quiet $LORA_SS /workspace/lora/ss.pt && aws s3 cp --quiet $LORA_512 /workspace/lora/512.pt
R=""
for j in $EVAL_JOBS; do
  aws s3 sync --quiet --exclude "*" --include "vis/*/gt_soup.npz" $W/$j/out/eval_${ROW}/ /workspace/runs/$j
  R="$R /workspace/runs/$j"
done
cd $PX
python3 /root/code/scripts/b0/ss_sweep.py --val /workspace/b0/val --vis-runs $R --lora-ss /workspace/lora/ss.pt \
  --lora-512 /workspace/lora/512.pt --n ${N:-90} --shard ${SHARD:-0/1} --out $OUT/ss.csv
