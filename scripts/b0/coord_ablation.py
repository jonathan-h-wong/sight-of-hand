"""Which voxel errors hurt stage 2? (GPU; run from a Pixal3D checkout.)

For frames with saved eval visuals (vis/<key>/gt_soup.npz and coords32__cascade512__<row>.npz from eval_b0.py),
run stage 2 at 512 through ONE code path (the gt512 one) on different 32^3 coordinate sets, and score soup / hand
F@{2,5} against the saved GT soup:
  gt, gt_seed2   GT coordinates, two noise seeds (sampling variance)
  pred           stage 1's predicted coordinates (as in the cascade)
  inter          pred AND gt (missed voxels removed, no extra ones)
  union          pred OR gt (extra voxels added, none missed)
  gt_swap15      GT with ~15% of voxels swapped at the boundary (random drop + random 6-neighbour add)

    python coord_ablation.py --vis-runs runs/... --lora-ss ss.pt --lora-512 shape512.pt --row b0 --n 60 --out abl.csv
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
from eval_b0 import BUCKET, build_pipeline, load_sample_raw, mesh_points, padded_image  # noqa: E402


def fsc(P, Q, tau):
    p = (cKDTree(Q).query(P)[0] <= tau).mean()
    r = (cKDTree(P).query(Q)[0] <= tau).mean()
    return 2 * p * r / (p + r) if p + r else 0.0


def surf(V, F, n, seed):
    import trimesh
    return np.asarray(trimesh.Trimesh(V, F, process=False).sample(n, seed=seed))


def swap(gt, frac, rng):
    G_ = {tuple(x) for x in gt.tolist()}
    keep = [x for x in G_ if rng.random() > frac]
    nb = {tuple(np.array(x) + o) for x in G_ for o in np.r_[np.eye(3, dtype=int), -np.eye(3, dtype=int)]}
    nb = [x for x in nb if x not in G_ and all(0 <= v < 32 for v in x)]
    add = [nb[i] for i in rng.choice(len(nb), size=min(len(nb), len(G_) - len(keep)), replace=False)]
    return np.array(keep + add, int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vis-runs", nargs="+", required=True)
    ap.add_argument("--prefix", default="train/b0_dexycb_s0_v1")
    ap.add_argument("--lora-ss")
    ap.add_argument("--lora-512")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--row", default="b0")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.lora_1024 = None
    import boto3
    import torch
    sys.path.insert(0, os.environ.get("PX", "."))
    torch.set_grad_enabled(False)
    s3 = boto3.client("s3")
    idx = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{a.prefix}/index.parquet")["Body"].read()))
    idx = idx.set_index("key")
    vis = {}
    for r in map(Path, a.vis_runs):
        for v in sorted((r / "vis").iterdir()):
            if (v / f"coords32__cascade512__{a.row}.npz").exists():
                vis[v.name] = v
    keys = sorted(vis)
    keys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(a.n, len(keys))).round().astype(int)]
    pipe = build_pipeline(a)
    rng = np.random.default_rng(0)
    res = []
    for sh, ks in idx.loc[keys].groupby("shard").apply(lambda g: list(g.index)).to_dict().items():
        raw = load_sample_raw(s3, a.prefix, sh, set(ks))
        for key in ks:
            row = idx.loc[key]
            image, _ = padded_image(raw[key], row, "oracle")
            cam = dict(camera_angle_x=float(row.camera_angle_x), distance=G.D_N, mesh_scale=1.0 / float(row.L_n))
            z = np.load(vis[key] / "gt_soup.npz")
            d, L = float(z["d"]), float(z["L"])
            gh = surf(z["hv"], z["hf"], 10000, 0)
            go = surf(z["ov"], z["of"], 10000, 1)
            g = np.r_[gh, go]
            o = np.load(vis[key] / f"coords32__cascade512__{a.row}.npz")
            pr, gt = o["pred"].astype(int), o["gt"].astype(int)
            Ps, Gs = {tuple(x) for x in pr.tolist()}, {tuple(x) for x in gt.tolist()}
            sets = {"gt": (gt, a.seed), "gt_seed2": (gt, a.seed + 1), "pred": (pr, a.seed),
                    "inter": (np.array(sorted(Ps & Gs), int), a.seed), "union": (np.array(sorted(Ps | Gs), int), a.seed),
                    "gt_swap15": (swap(gt, 0.15, rng), a.seed)}
            for name, (c, sd) in sets.items():
                out = {"key": key, "variant": name, "n_vox": len(c),
                       "vox_iou": len({tuple(x) for x in c.tolist()} & Gs) / max(len({tuple(x) for x in c.tolist()} | Gs), 1)}
                if len(c) == 0:
                    res.append(out)
                    continue
                coords = torch.cat([torch.zeros(len(c), 1, dtype=torch.int32),
                                    torch.from_numpy(c.astype(np.int32))], 1).cuda()
                cond = pipe.get_proj_cond_shape(pipe.image_cond_model_shape_512, [image], coords, **cam)
                torch.manual_seed(sd)
                slat = pipe.sample_shape_slat(cond, pipe.models["shape_slat_flow_model_512"], coords)
                m = pipe.decode_shape_slat(slat, 512)
                mesh = m[0][0] if isinstance(m, tuple) else m[0]
                P = G.grid_to_cam(mesh_points(mesh, 20000, 2), d, L)
                hm = cKDTree(gh).query(P)[0] <= cKDTree(go).query(P)[0]
                for tau in (2, 5):
                    t = tau / 1000
                    out[f"soup_f{tau}"] = fsc(P, g, t)
                    out[f"hand_f{tau}"] = fsc(P[hm], gh, t) if hm.sum() > 50 else 0.0
                res.append(out)
            print(key, {r["variant"]: round(r.get("soup_f2", 0), 3) for r in res if r["key"] == key}, flush=True)
            pd.DataFrame(res).to_csv(a.out, index=False)
    df = pd.DataFrame(res)
    print(df.groupby("variant")[["vox_iou", "soup_f2", "hand_f2", "soup_f5", "hand_f5"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
