"""Denoising loss of stage 2 LoRAs on GT vs predicted coordinates, held-out val frames (GPU; Pixal3D checkout).

Same flow-matching loss as training (x_t = (1-t) x0 + (s + (1-s) t) eps, target v = (1-s) eps - x0, normalized
latents, conditional branch only), at fixed t values and fixed noise per frame, identical across LoRAs:
  gt     GT 32^3 coordinates, GT latent
  pred   stage 1 coordinates (eval visual dump): GT latent where predicted AND GT ("tp" tokens), the mean latent
         (0 normalized) at predicted-only voxels ("extra" tokens) -- the mixed-training target
Reports mean MSE per (lora, coords, token type, t).

    python loss_probe.py --val /workspace/b0/val --vis-runs runs/... --row b0 --lora NAME=path ... --n 60 --out probe.csv
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sight_of_hand.b0 import geometry as G  # noqa: E402
from sight_of_hand.b0 import lora  # noqa: E402
from eval_b0 import BUCKET, build_pipeline, load_sample_raw, padded_image  # noqa: E402

TS = (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--vis-runs", nargs="+", required=True)
    ap.add_argument("--prefix", default="train/b0_dexycb_s0_v1")
    ap.add_argument("--row", default="b0")
    ap.add_argument("--lora", nargs="+", required=True, help="NAME=path of a stage 2 (512) LoRA")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--noise-seeds", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import boto3
    import torch
    sys.path.insert(0, os.environ.get("PX", "."))
    from pixal3d.modules.sparse import SparseTensor
    torch.set_grad_enabled(False)
    a.lora_ss = a.lora_512 = a.lora_1024 = None
    pipe = build_pipeline(a)
    fm = pipe.models["shape_slat_flow_model_512"]
    lora.inject(fm, a.rank, a.alpha)
    fm.to("cuda")
    loras = {k: torch.load(v, map_location="cuda") for k, v in (x.split("=", 1) for x in a.lora)}
    mean = torch.tensor(pipe.shape_slat_normalization["mean"]).float().cuda()
    std = torch.tensor(pipe.shape_slat_normalization["std"]).float().cuda()
    s3 = boto3.client("s3")
    idx = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{a.prefix}/index.parquet")["Body"].read()))
    idx = idx.set_index("key")
    lat = Path(a.val) / "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"
    vis = {}
    for r in map(Path, a.vis_runs):
        for v in sorted((r / "vis").iterdir()):
            if (v / f"coords32__cascade512__{a.row}.npz").exists() and (lat / v.name / "view00.npz").exists():
                vis[v.name] = v
    keys = sorted(vis)
    keys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(a.n, len(keys))).round().astype(int)]
    res = []
    sig = 1e-5
    for sh, ks in idx.loc[keys].groupby("shard").apply(lambda g: list(g.index)).to_dict().items():
        raw = load_sample_raw(s3, a.prefix, sh, set(ks))
        for key in ks:
            row = idx.loc[key]
            image, _ = padded_image(raw[key], row, "oracle")
            cam = dict(camera_angle_x=float(row.camera_angle_x), distance=G.D_N, mesh_scale=1.0 / float(row.L_n))
            zl = np.load(lat / key / "view00.npz")
            gc = zl["coords"].astype(int)
            gf = (torch.from_numpy(zl["feats"].astype(np.float32)).cuda() - mean) / std
            lut = {tuple(c): i for i, c in enumerate(gc.tolist())}
            pr = np.load(vis[key] / f"coords32__cascade512__{a.row}.npz")["pred"].astype(int)
            j = torch.tensor([lut.get(tuple(c), -1) for c in pr.tolist()], device="cuda")
            pf = torch.where((j >= 0)[:, None], gf[j.clamp(min=0)], torch.zeros(len(pr), gf.shape[1], device="cuda"))
            sets = {"gt": (gc, gf, torch.ones(len(gc), dtype=torch.bool, device="cuda")), "pred": (pr, pf, j >= 0)}
            for cname, (c, x0f, tp) in sets.items():
                co = torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), torch.from_numpy(c.astype(np.int32))], 1).cuda()
                cond = pipe.get_proj_cond_shape(pipe.image_cond_model_shape_512, [image], co, **cam)["cond"]
                x0 = SparseTensor(feats=x0f, coords=co)
                for ns in range(a.noise_seeds):
                    g = torch.Generator(device="cuda").manual_seed(1000 * ns + 7)
                    eps = torch.randn(x0f.shape, generator=g, device="cuda")
                    for t in TS:
                        xt = x0.replace((1 - t) * x0f + (sig + (1 - sig) * t) * eps)
                        v = (1 - sig) * eps - x0f
                        for lname, sd in loras.items():
                            lora.load_lora(fm, sd)
                            pred = fm(xt, torch.full((1,), 1000.0 * t, device="cuda"), cond)
                            se = ((pred.feats.float() - v) ** 2).mean(1)
                            for tok, m in (("all", torch.ones_like(tp)), ("tp", tp), ("extra", ~tp)):
                                if m.any():
                                    res.append({"key": key, "lora": lname, "coords": cname, "tokens": tok, "t": t,
                                                "noise": ns, "mse": float(se[m].mean()), "n": int(m.sum())})
            print(key, flush=True)
        pd.DataFrame(res).to_csv(a.out, index=False)
    df = pd.DataFrame(res)
    print(df.groupby(["coords", "tokens", "lora"]).mse.mean().unstack().round(4).to_string())
    print(df[df.tokens != "extra"].groupby(["coords", "tokens", "t", "lora"]).mse.mean().unstack().round(4).to_string())


if __name__ == "__main__":
    main()
