# Stage 2 coordinate ablation on frames saved by eval_b0 jobs. Env: EVAL_JOBS, ROW, LORA_SS, LORA_512, N.
set -ex
W=s3://xray-hands/work/b0/jobs
mkdir -p /workspace/lora /workspace/runs $OUT
aws s3 cp --quiet $LORA_SS /workspace/lora/ss.pt && aws s3 cp --quiet $LORA_512 /workspace/lora/512.pt
R=""
for j in $EVAL_JOBS; do
  aws s3 sync --quiet --exclude "*" --include "vis/*/gt_soup.npz" --include "vis/*/coords32__*" $W/$j/out/eval_${ROW}/ /workspace/runs/$j
  R="$R /workspace/runs/$j"
done
cd $PX
python3 /root/code/scripts/b0/coord_ablation.py --vis-runs $R --lora-ss /workspace/lora/ss.pt --lora-512 /workspace/lora/512.pt \
  --row $ROW --n ${N:-60} --out $OUT/ablation.csv
