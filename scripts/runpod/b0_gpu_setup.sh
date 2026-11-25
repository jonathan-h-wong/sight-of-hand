#!/usr/bin/env bash
# Pixal3D training environment on a RunPod GPU pod (image nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04, Python 3.10):
# torch 2.6.0 + cu124, the prebuilt natten 0.21.0 wheel (NAF, stage 2), and TRELLIS.2's CUDA extensions (FlexGEMM,
# CuMesh, o-voxel, nvdiffrast) compiled for this GPU's arch. Compiled wheels are cached at
# s3://xray-hands/work/b0/wheels/sm80_90/ so later pods skip the build. Sourced by b0_gpu_bootstrap.sh.
# Run as a subprocess; it writes /root/b0env.sh (PX, EVAL, ATTN_BACKEND, ...) for the caller to source.
set -e
WH=s3://xray-hands/work/b0/wheels
# One wheel set for A100 (8.0) and H100/H200 (9.0). MAX_JOBS is kept low: 32 parallel nvcc jobs got the container
# OOM-killed on the first attempt (the pod died silently mid-compile).
ARCH=80_90
export TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=${MAX_JOBS:-6}
apt-get -qq install -y git libgl1 libglib2.0-0 unzip > /dev/null 2>&1 || true
python3 -m pip install -q --upgrade pip
pip install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -q "https://github.com/LDYang694/Storages/releases/download/20260430/natten-0.21.0+torch2.6cu124-cp310-cp310-linux_x86_64.whl"
pip install -q ninja trimesh scipy rtree easydict pandas pyarrow pyyaml boto3 huggingface_hub hf_transfer zstandard \
  plyfile tqdm opencv-python-headless==4.12.0.88 imageio imageio-ffmpeg kornia timm transformers==4.57.3 \
  safetensors einops wandb pillow lpips "numpy<2.3"
pip install -q https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl
pip install -q "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" || true
mkdir -p /root/ext /root/wheels && cd /root/ext
if aws s3 ls $WH/sm$ARCH/DONE > /dev/null 2>&1; then
  aws s3 sync --quiet $WH/sm$ARCH /root/wheels
else
  git clone -q --depth 1 --recursive https://github.com/microsoft/TRELLIS.2
  git clone -q --recursive https://github.com/JeffreyXiang/FlexGEMM.git
  git clone -q --recursive https://github.com/JeffreyXiang/CuMesh.git
  git clone -q -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git
  for p in FlexGEMM CuMesh nvdiffrast TRELLIS.2/o-voxel; do
    pip wheel -q --no-build-isolation --no-deps -w /root/wheels ./$p
  done
  aws s3 sync --quiet /root/wheels $WH/sm$ARCH && touch /tmp/DONE && aws s3 cp --quiet /tmp/DONE $WH/sm$ARCH/DONE
fi
pip install -q --no-deps /root/wheels/*.whl
cd /root
[ -d Pixal3D ] || git clone -q https://github.com/TencentARC/Pixal3D && git -C Pixal3D checkout -q f7cf384
[ -d sight-of-hand-eval ] || git clone -q --depth 1 https://github.com/jonathan-h-wong/sight-of-hand-eval
AB=flash_attn; python3 -c "import flash_attn" 2>/dev/null || AB=sdpa
cat > /root/b0env.sh <<E
export PX=/root/Pixal3D EVAL=/root/sight-of-hand-eval ATTN_BACKEND=$AB SPARSE_ATTN_BACKEND=$AB HF_HUB_ENABLE_HF_TRANSFER=1
E
source /root/b0env.sh
# MANO + YCB meshes (soup construction)
D=/workspace/data; mkdir -p $D/dexycb $D/mano
[ -d $D/mano/mano_v1_2 ] || { aws s3 cp --quiet s3://xray-hands/raw/mano/mano_v1_2.zip $D/mano/ && unzip -q -o $D/mano/mano_v1_2.zip -d $D/mano; }
[ -d $D/dexycb/models ] || aws s3 cp --quiet s3://xray-hands/raw/dexycb/models.tar.gz - | tar -xzf - -C $D/dexycb --wildcards '*textured_simple.obj'
cd $PX && python3 -c "import torch, o_voxel, flex_gemm, cumesh, natten; import pixal3d.models; print('env ok', torch.__version__, torch.cuda.get_device_name())"
