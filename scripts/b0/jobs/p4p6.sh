# P4 (overfit 32 samples per stage) and P6 (throughput, 100 steps per stage at the training batch) on the val encoding.
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
mkdir -p /workspace/b0 && aws s3 cp --quiet $E/val.tar - | tar -xf - -C /workspace/b0
cd $PX
python3 /root/code/scripts/b0/p5_lora_check.py --out $OUT/p5 || echo "P5 FAILED"
ls /workspace/b0/val/ss_latents/*/ | head -3
python3 - <<P
import pandas as pd
m = pd.read_csv('/workspace/b0/val/ss_latents/ss_enc_conv3d_16l8_fp16_64_view/metadata.csv')
open('/workspace/sub32.txt', 'w').write('\n'.join(m.sha256.sample(32, random_state=0)))
P
T=/root/code/scripts/b0/train_b0.py
for st in ${STAGES:-ss64 shape512 shape1024}; do
  python3 $T --stage $st --data /workspace/b0 --split val --out $OUT/p6_$st --steps 100 --save-every 100000 --log-every 10 \
    --wandb --run-name p6_$st 2>&1 | tail -40
  python3 $T --stage $st --data /workspace/b0 --split val --subset /workspace/sub32.txt --out $OUT/p4_$st \
    --steps ${P4_STEPS:-600} --save-every 100000 --log-every 10 --ema 0.99 --wandb --run-name p4_$st 2>&1 | tail -40
done
