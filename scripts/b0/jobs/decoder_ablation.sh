# Shape-decoder sensitivity to voxel-set changes (no flow model). Env: EVAL_JOBS, ROW, N.
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
W=s3://xray-hands/work/b0/jobs
mkdir -p /workspace/b0 /workspace/runs $OUT
aws s3 cp --quiet $E/val.tar - | tar -xf - -C /workspace/b0 --wildcards 'val/shape_latents/*512*'
R=""
for j in $EVAL_JOBS; do
  aws s3 sync --quiet --exclude "*" --include "vis/*/gt_soup.npz" --include "vis/*/coords32__*" $W/$j/out/eval_${ROW}/ /workspace/runs/$j
  R="$R /workspace/runs/$j"
done
cd $PX
python3 /root/code/scripts/b0/decoder_ablation.py --val /workspace/b0/val --vis-runs $R --row $ROW --n ${N:-60} --out $OUT/decoder.csv
