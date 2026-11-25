# Evaluation on s0 val. Env: ROW (zeroshot|b0), LORA_SS / LORA_512 / LORA_1024 (S3 URIs, optional), MODES, EVAL_ARGS.
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
mkdir -p /workspace/b0 /workspace/lora && aws s3 cp --quiet $E/val.tar - | tar -xf - -C /workspace/b0
aws s3 cp --quiet s3://xray-hands/masks/amodal_v1/dexycb/index.parquet /workspace/amodal_index.parquet
pip install -q git+https://github.com/microsoft/MoGe.git
L=""
for v in SS 512 1024; do
  n=LORA_$v; u=${!n:-}
  if [ -n "$u" ]; then f=/workspace/lora/$v.pt; aws s3 cp --quiet $u $f; L="$L --lora-$(echo $v | tr A-Z a-z) $f"; fi
done
cd $PX
python3 /root/code/scripts/b0/eval_b0.py --val /workspace/b0/val --out $OUT/eval_${ROW} --row $ROW \
  --amodal-index /workspace/amodal_index.parquet --modes ${MODES:-ss,gt512,cascade} --moge $L ${EVAL_ARGS:-}
