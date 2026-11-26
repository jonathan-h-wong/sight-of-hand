# Stage 1 predictions on a train subset for stage 2 mixed training. Env: LORA_SS (S3 URI), N, SHARD (i/n),
# TEST_EMPTY=1 to first run test_empty_latent.py on 20 val frames (needs val.tar).
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
mkdir -p /workspace/lora $OUT
aws s3 cp --quiet $LORA_SS /workspace/lora/ss.pt
cd $PX
if [ "${TEST_EMPTY:-0}" = 1 ]; then
  mkdir -p /workspace/b0 && aws s3 cp --quiet $E/val.tar - | tar -xf - -C /workspace/b0 --wildcards 'val/shape_latents/*512*'
  python3 /root/code/scripts/b0/test_empty_latent.py --val /workspace/b0/val --n 20 --out $OUT/empty_latent.json
fi
python3 /root/code/scripts/b0/gen_pred_coords.py --lora-ss /workspace/lora/ss.pt --out $OUT/pred --n ${N:-8000} --shard ${SHARD:-0/1}
