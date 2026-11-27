"""Stage 1 inference settings without training (GPU; run from a Pixal3D checkout): K-sample probability averaging,
decoder threshold, CFG scale and step count; stage 1 metrics and the end-to-end 512 cascade.

Per val frame (eval_b0 visual dumps for the GT soup, val.tar latents for the GT 64^3 / 32^3 coordinates):
  for each sampler setting (CFG, steps): K samples; decoder logits -> sigmoid; mean over the first k in {1,2,4,8}
  for each (k, threshold): 64^3 IoU / precision / recall, the 32^3 set after the pipeline's max-pool, its IoU, and
  hand / object voxel recall at 64^3 and 32^3 (GT voxels labelled by the GT hand / object surface they contain)
  for selected settings: stage 2 (512, LoRA) on the 32^3 set -> soup / hand / object F@{2,5} vs the GT soup
Single-sample threshold 0.5 on the sigmoid = logit 0 = the released pipeline.

    python ss_sweep.py --val /workspace/b0/val --vis-runs runs/... --lora-ss ss.pt --lora-512 512.pt --n 60 --out ss.csv
"""
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sight_of_hand.b0 import geometry as G  # noqa: E402
from coord_ablation import fsc, surf  # noqa: E402
from eval_b0 import BUCKET, build_pipeline, load_sample_raw, mesh_points, padded_image  # noqa: E402

SAMPLERS = {"cfg7.5_s12": dict(), "cfg1_s12": dict(guidance_strength=1.0), "cfg3_s12": dict(guidance_strength=3.0),
            "cfg12_s12": dict(guidance_strength=12.0), "cfg7.5_s25": dict(steps=25)}
KS = (1, 2, 4, 8)
THRS = (0.3, 0.4, 0.5, 0.6, 0.7)
STAGE2 = [("cfg7.5_s12", 1, 0.5), ("cfg7.5_s12", 8, 0.5), ("cfg7.5_s12", 8, 0.35), ("cfg7.5_s12", 8, 0.65),
          ("cfg3_s12", 1, 0.5), ("cfg12_s12", 1, 0.5), ("cfg7.5_s25", 1, 0.5), ("cfg7.5_s12", 4, 0.5)]


def vox(p, d, L, R):
    return np.clip(np.floor((G.cam_to_grid(p, d, L) + 0.5) * R).astype(int), 0, R - 1)


def grid(c, R):
    g = np.zeros((R,) * 3, bool)
    if len(c):
        g[tuple(np.asarray(c, int).T)] = True
    return g


def pool(g):
    return g.reshape(32, 2, 32, 2, 32, 2).any((1, 3, 5))


def stats(P, Gt, H, O, tag, out):
    i = (P & Gt).sum()
    out[f"{tag}_iou"] = i / max((P | Gt).sum(), 1)
    out[f"{tag}_prec"] = i / max(P.sum(), 1)
    out[f"{tag}_rec"] = i / max(Gt.sum(), 1)
    out[f"{tag}_hand_rec"] = (P & H).sum() / max(H.sum(), 1)
    out[f"{tag}_obj_rec"] = (P & O).sum() / max(O.sum(), 1)
    out[f"{tag}_n"] = int(P.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--vis-runs", nargs="+", required=True)
    ap.add_argument("--prefix", default="train/b0_dexycb_s0_v1")
    ap.add_argument("--lora-ss", required=True)
    ap.add_argument("--lora-512", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.lora_1024 = None
    import boto3
    import torch
    sys.path.insert(0, os.environ.get("PX", "."))
    torch.set_grad_enabled(False)
    pipe = build_pipeline(a)
    fm, dec = pipe.models["sparse_structure_flow_model"], pipe.models["sparse_structure_decoder"]
    s3 = boto3.client("s3")
    idx = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{a.prefix}/index.parquet")["Body"].read()))
    idx = idx.set_index("key")
    lat = Path(a.val) / "shape_latents"
    vis = {}
    for r in map(Path, a.vis_runs):
        for v in sorted((r / "vis").iterdir()):
            if (v / "gt_soup.npz").exists() and (lat / "shape_enc_next_dc_f16c32_fp16_1024_view" / v.name).exists():
                vis[v.name] = v
    keys = sorted(vis)
    keys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(a.n, len(keys))).round().astype(int)]
    si, sn = map(int, a.shard.split("/"))
    keys = keys[si::sn]
    res, res2 = [], []
    for sh, ks in idx.loc[keys].groupby("shard").apply(lambda g: list(g.index)).to_dict().items():
        raw = load_sample_raw(s3, a.prefix, sh, set(ks))
        for key in ks:
            row = idx.loc[key]
            image, _ = padded_image(raw[key], row, "oracle")
            cam = dict(camera_angle_x=float(row.camera_angle_x), distance=G.D_N, mesh_scale=1.0 / float(row.L_n))
            z = np.load(vis[key] / "gt_soup.npz")
            d, L = float(z["d"]), float(z["L"])
            gh, go = surf(z["hv"], z["hf"], 20000, 0), surf(z["ov"], z["of"], 20000, 1)
            g = np.r_[gh, go]
            G64 = grid(np.load(lat / "shape_enc_next_dc_f16c32_fp16_1024_view" / key / "view00.npz")["coords"], 64)
            G32 = grid(np.load(lat / "shape_enc_next_dc_f16c32_fp16_512_view" / key / "view00.npz")["coords"], 32)
            H64, O64 = grid(vox(gh, d, L, 64), 64) & G64, grid(vox(go, d, L, 64), 64) & G64
            H32, O32 = grid(vox(gh, d, L, 32), 32) & G32, grid(vox(go, d, L, 32), 32) & G32
            cond = pipe.get_proj_cond_ss([image], **cam)
            probs = {}
            for sname, sp in SAMPLERS.items():
                params = {**pipe.sparse_structure_sampler_params, **sp}
                pk = []
                for k in range(max(KS)):
                    torch.manual_seed(a.seed + k)
                    noise = torch.randn(1, fm.in_channels, *[fm.resolution] * 3, device="cuda")
                    zs = pipe.sparse_structure_sampler.sample(fm, noise, **cond, **params).samples
                    pk.append(torch.sigmoid(dec(zs).float())[0, 0].cpu().numpy())
                pk = np.stack(pk)
                for k in KS:
                    pm = pk[:k].mean(0)
                    probs[(sname, k)] = pm
                    for thr in THRS:
                        P64 = pm > thr
                        out = {"key": key, "sampler": sname, "k": k, "thr": thr}
                        stats(P64, G64, H64, O64, "v64", out)
                        stats(pool(P64), G32, H32, O32, "v32", out)
                        res.append(out)
            for sname, k, thr in STAGE2:
                P32 = pool(probs[(sname, k)] > thr)
                c = np.argwhere(P32)
                out = {"key": key, "sampler": sname, "k": k, "thr": thr, "n32": len(c)}
                stats(P32, G32, H32, O32, "v32", out)
                if len(c):
                    co = torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), torch.from_numpy(c.astype(np.int32))], 1).cuda()
                    cl = pipe.get_proj_cond_shape(pipe.image_cond_model_shape_512, [image], co, **cam)
                    torch.manual_seed(a.seed)
                    slat = pipe.sample_shape_slat(cl, pipe.models["shape_slat_flow_model_512"], co)
                    m = pipe.decode_shape_slat(slat, 512)
                    mesh = m[0][0] if isinstance(m, tuple) else m[0]
                    Pm = G.grid_to_cam(mesh_points(mesh, 20000, 2), d, L)
                    if len(Pm):
                        hm = cKDTree(gh).query(Pm)[0] <= cKDTree(go).query(Pm)[0]
                        for tau in (2, 5):
                            out[f"soup_f{tau}"] = fsc(Pm, g, tau / 1000)
                            out[f"hand_f{tau}"] = fsc(Pm[hm], gh, tau / 1000) if hm.sum() > 50 else 0.0
                            out[f"obj_f{tau}"] = fsc(Pm[~hm], go, tau / 1000) if (~hm).sum() > 50 else 0.0
                res2.append(out)
            print(key, {f"{r['sampler']}/k{r['k']}/t{r['thr']}": round(r.get("soup_f2", 0), 3) for r in res2 if r["key"] == key},
                  flush=True)
            pd.DataFrame(res).to_csv(a.out, index=False)
            pd.DataFrame(res2).to_csv(a.out.replace(".csv", "_cascade.csv"), index=False)
    df = pd.DataFrame(res)
    print(df.groupby(["sampler", "k", "thr"])[["v64_iou", "v64_hand_rec", "v64_obj_rec", "v32_iou", "v32_hand_rec"]]
          .median().round(3).to_string())
    print(pd.DataFrame(res2).groupby(["sampler", "k", "thr"]).median(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
