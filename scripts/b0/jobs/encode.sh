# Encode val (with P1 round trip + P2 overlays), then train; each split is tarred to
# s3://xray-hands/train/b0_dexycb_s0_v1/encoded/<split>.tar as soon as it is done. Run by b0_gpu_bootstrap.sh.
set -ex
E=s3://xray-hands/train/b0_dexycb_s0_v1/encoded
cd $PX
nvidia-smi
python3 /root/code/scripts/b0/p5_lora_check.py --out $OUT/p5 || echo "P5 FAILED (continuing with encode)"
W=$(( $(nproc) - 4 ))
for split in ${SPLITS:-val train}; do
  extra=""; [ $split = val ] && extra="--roundtrip 64 --overlays 16"
  python3 /root/code/scripts/b0/encode_b0.py --split $split --out /workspace/b0 --data /workspace/data --workers $W $extra
  cp /workspace/b0/$split/encode_summary.json $OUT/encode_summary_$split.json
  [ -f /workspace/b0/$split/p1_roundtrip.json ] && cp /workspace/b0/$split/p1_roundtrip.json $OUT/ && cp -r /workspace/b0/$split/p2 $OUT/p2
  tar -cf /workspace/$split.tar -C /workspace/b0 $split
  aws s3 cp --quiet /workspace/$split.tar $E/$split.tar
  aws s3 cp --quiet $OUT/encode_summary_$split.json $E/encode_summary_$split.json
  rm /workspace/$split.tar
done
