# MANO fits (GPU, batched) to the hands in eval_b0 outputs. Env: EVAL_JOBS (space-separated job names whose
# out/eval_<row>/ to fit), ROWS (comma-separated eval rows), BATCH (targets per batch, default 32).
set -ex
W=s3://xray-hands/work/b0/jobs
mkdir -p /workspace/mano /workspace/runs $OUT
aws s3 cp --quiet s3://xray-hands/raw/mano/mano_v1_2.zip /workspace/mano/ && unzip -q -o /workspace/mano/mano_v1_2.zip -d /workspace/mano
aws s3 cp --quiet s3://xray-hands/train/b0_dexycb_s0_v1/index.parquet /workspace/index.parquet
R=""
for j in $EVAL_JOBS; do
  until [ "$(aws s3 cp $W/$j/status/STATE - 2>/dev/null)" = "DONE" ]; do sleep 60; done
  for d in $(aws s3 ls $W/$j/out/ | awk '/eval_/{print $2}'); do
    aws s3 sync --quiet --exclude "*" --include "meshes/*" --include "vis/*/gt_soup.npz" $W/$j/out/$d /workspace/runs/$j/$d
    R="$R /workspace/runs/$j/$d"
  done
done
python3 /root/code/scripts/b0/fit_mano.py --runs $R --mano /workspace/mano/mano_v1_2/models --index /workspace/index.parquet \
  --out $OUT/mano.csv --rows ${ROWS:-b0} --batch ${BATCH:-32} --save-dir $OUT/fits
