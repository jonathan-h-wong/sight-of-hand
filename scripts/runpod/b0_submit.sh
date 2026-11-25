#!/usr/bin/env bash
# Submit a B0 GPU job: pack this checkout, upload it with the setup and bootstrap scripts, upload the job script,
# and start a pod (scripts/runpod/launch.py) that runs it under b0_gpu_bootstrap.sh.
#   bash scripts/runpod/b0_submit.sh JOB job.sh [GPU] [DEADLINE_S] [DISK_GB] [KEY=VALUE ...]   (extra pod env)
#   GPU defaults to "NVIDIA H100 80GB HBM3"; status/output land in s3://xray-hands/work/b0/jobs/JOB/
set -euo pipefail
JOB=$1; JS=$2; GPU=${3:-NVIDIA H100 80GB HBM3}; DL=${4:-43200}; DISK=${5:-200}; shift $(( $# < 5 ? $# : 5 ))
EXTRA=(); for kv in "$@"; do EXTRA+=(--env "$kv"); done
CLOUD=${CLOUD:-ALL}
cd "$(dirname "$0")/../.."
C=s3://xray-hands/work/b0/code; J=s3://xray-hands/work/b0/jobs/$JOB
T=$(mktemp -d)
tar -czf $T/code.tgz --exclude=.git --exclude=__pycache__ --exclude=data --exclude=out .
aws s3 cp --quiet $T/code.tgz $C/code_sight_of_hand.tar.gz
aws s3 cp --quiet scripts/runpod/b0_gpu_setup.sh $C/b0_gpu_setup.sh
aws s3 cp --quiet scripts/runpod/b0_gpu_bootstrap.sh $J/bootstrap.sh
aws s3 cp --quiet "$JS" $J/job.sh
aws s3 rm --quiet --recursive $J/status || true
python3 scripts/runpod/launch.py gpu --name "b0-$JOB" --gpu "$GPU" --disk $DISK --bootstrap $J/bootstrap.sh \
  --env JOB=$JOB --env DEADLINE_S=$DL --cloud $CLOUD --min-vcpu ${MIN_VCPU:-8} --min-mem ${MIN_MEM:-60} "${EXTRA[@]}"
