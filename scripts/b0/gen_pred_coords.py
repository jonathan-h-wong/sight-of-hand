"""Stage 1 predictions on training frames, for stage 2 mixed training (GPU; run from a Pixal3D checkout).

For --n training frames (seeded subset of the filtered train split), sample the B0 stage 1 exactly as eval_b0.py's
cascade does (oracle matte, same camera, sample_sparse_structure(cond, 32, 1)) and save the predicted 32^3
coordinates to <out>/coords/<key>.npz (uint8, N x 3). Frames already present are skipped (restartable).

    python gen_pred_coords.py --lora-ss ss64.pt --out /workspace/out/pred --n 8000 --shard 0/4
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sight_of_hand.b0 import geometry as G  # noqa: E402
from encode_b0 import apply_filters  # noqa: E402
from eval_b0 import BUCKET, build_pipeline, load_sample_raw, padded_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="train/b0_dexycb_s0_v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-ss", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-inside", type=float, default=0.5)
    ap.add_argument("--d-min", type=float)
    ap.add_argument("--d-max", type=float)
    a = ap.parse_args()
    a.lora_512 = a.lora_1024 = None

    import boto3
    import torch
    sys.path.insert(0, os.environ.get("PX", "."))
    torch.set_grad_enabled(False)
    s3 = boto3.client("s3")
    out = Path(a.out) / "coords"
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{a.prefix}/index.parquet")["Body"].read()))
    kept, _ = apply_filters(idx, "train", a)
    keys = sorted(kept.key)
    keys = list(np.random.default_rng(a.seed).choice(keys, size=min(a.n, len(keys)), replace=False))
    si, sn = map(int, a.shard.split("/"))
    keys = [k for k in sorted(keys)[si::sn] if not (out / f"{k}.npz").exists()]
    print(f"{len(keys)} frames to do", flush=True)
    pipe = build_pipeline(a)
    v = idx.set_index("key").loc[keys]
    t0, done = time.time(), 0
    for sh, ks in v.groupby("shard").apply(lambda g: list(g.index)).to_dict().items():
        raw = load_sample_raw(s3, a.prefix, sh, set(ks))
        for key in ks:
            if key not in raw:
                continue
            row = v.loc[key]
            image, _ = padded_image(raw[key], row, "oracle")
            cam = dict(camera_angle_x=float(row.camera_angle_x), distance=G.D_N, mesh_scale=1.0 / float(row.L_n))
            torch.manual_seed(a.seed + done)
            cond = pipe.get_proj_cond_ss([image], **cam)
            coords = pipe.sample_sparse_structure(cond, 32, 1)
            np.savez_compressed(out / f"{key}.npz", coords=coords[:, 1:].cpu().numpy().astype(np.uint8))
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(keys)}  {(time.time() - t0) / done:.2f} s/frame", flush=True)
    print(f"done {done} in {(time.time() - t0) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
