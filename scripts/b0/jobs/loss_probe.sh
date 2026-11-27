# Denoising loss of stage 2 LoRAs on GT vs stage-1 coordinates (val). Env: EVAL_JOBS, ROW, LORAS (NAME=s3uri ...), N.
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
W=s3://xray-hands/work/b0/jobs
mkdir -p /workspace/b0 /workspace/runs /workspace/lora $OUT
aws s3 cp --quiet $E/val.tar - | tar -xf - -C /workspace/b0 --wildcards 'val/shape_latents/*512*'
L=""
for nv in $LORAS; do n=${nv%%=*}; u=${nv#*=}; aws s3 cp --quiet $u /workspace/lora/$n.pt; L="$L $n=/workspace/lora/$n.pt"; done
R=""
for j in $EVAL_JOBS; do
  aws s3 sync --quiet --exclude "*" --include "vis/*/coords32__*" --include "vis/*/gt_soup.npz" $W/$j/out/eval_${ROW}/ /workspace/runs/$j
  R="$R /workspace/runs/$j"
done
cd $PX
python3 /root/code/scripts/b0/loss_probe.py --val /workspace/b0/val --vis-runs $R --row $ROW --lora $L --n ${N:-60} --out $OUT/probe.csv
