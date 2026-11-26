"""Is the frozen shape decoder sensitive to voxel-set changes? (GPU; run from a Pixal3D checkout.)

No flow model: decode the GT 512 shape latent on perturbed coordinate sets and score soup F@{2,5} vs the GT soup
(saved by eval_b0 as vis/<key>/gt_soup.npz, with the stage 1 prediction in coords32__cascade512__<row>.npz):
  dec_gt         GT latent, GT coordinates (the P1 round trip)
  dec_inter      GT latent restricted to pred AND gt (missed voxels dropped)
  dec_union      GT latent + the dataset-mean latent at pred-only voxels (extra voxels added)
  dec_pred       GT latent where pred AND gt, mean latent at pred-only voxels (the stage 1 voxel set)
  dec_swap15     GT with ~15% boundary voxels swapped (dropped; mean latent at the added ones)
If these stay near dec_gt, the decoder is robust and the cascade loss is the flow model's; if they fall like the
flow-model ablation, the decoder limits any stage 2 that keeps GT-like latents.

    python decoder_ablation.py --val /workspace/b0/val --vis-runs runs/... --row b0 --n 60 --out dec.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sight_of_hand.b0 import geometry as G  # noqa: E402
from coord_ablation import fsc, surf, swap  # noqa: E402
from eval_b0 import build_pipeline, mesh_points  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--vis-runs", nargs="+", required=True)
    ap.add_argument("--row", default="b0")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    sys.path.insert(0, os.environ.get("PX", "."))
    from pixal3d.modules.sparse import SparseTensor
    torch.set_grad_enabled(False)
    a.lora_ss = a.lora_512 = a.lora_1024 = None
    a.rank, a.alpha = 32, 32
    pipe = build_pipeline(a)
    mean = np.asarray(pipe.shape_slat_normalization["mean"], np.float32)
    lat = Path(a.val) / "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"
    vis = {}
    for r in map(Path, a.vis_runs):
        for v in sorted((r / "vis").iterdir()):
            if (v / f"coords32__cascade512__{a.row}.npz").exists() and (lat / v.name / "view00.npz").exists():
                vis[v.name] = v
    keys = sorted(vis)
    keys = [keys[i] for i in np.linspace(0, len(keys) - 1, min(a.n, len(keys))).round().astype(int)]
    rng = np.random.default_rng(0)
    res = []
    for key in keys:
        z = np.load(vis[key] / "gt_soup.npz")
        d, L = float(z["d"]), float(z["L"])
        gh, go = surf(z["hv"], z["hf"], 10000, 0), surf(z["ov"], z["of"], 10000, 1)
        g = np.r_[gh, go]
        zl = np.load(lat / key / "view00.npz")
        gc, gf = zl["coords"].astype(int), zl["feats"].astype(np.float32)
        lut = {tuple(c): i for i, c in enumerate(gc.tolist())}
        pr = np.load(vis[key] / f"coords32__cascade512__{a.row}.npz")["pred"].astype(int)
        P = {tuple(x) for x in pr.tolist()}
        Gs = set(lut)
        sw = {tuple(x) for x in swap(gc, 0.15, rng).tolist()}
        sets = {"dec_gt": Gs, "dec_inter": P & Gs, "dec_union": P | Gs, "dec_pred": P, "dec_swap15": sw}
        for name, cs in sets.items():
            cs = sorted(cs)
            out = {"key": key, "variant": name, "n_vox": len(cs), "vox_iou": len(set(cs) & Gs) / max(len(set(cs) | Gs), 1)}
            if cs:
                f = np.stack([gf[lut[c]] if c in lut else mean for c in cs])
                co = torch.cat([torch.zeros(len(cs), 1, dtype=torch.int32), torch.tensor(cs, dtype=torch.int32)], 1)
                m = pipe.decode_shape_slat(SparseTensor(feats=torch.from_numpy(f).cuda(), coords=co.cuda()), 512)
                mesh = m[0][0] if isinstance(m, tuple) else m[0]
                Pm = G.grid_to_cam(mesh_points(mesh, 20000, 2), d, L)
                if len(Pm):
                    hm = cKDTree(gh).query(Pm)[0] <= cKDTree(go).query(Pm)[0]
                    for tau in (2, 5):
                        out[f"soup_f{tau}"] = fsc(Pm, g, tau / 1000)
                        out[f"hand_f{tau}"] = fsc(Pm[hm], gh, tau / 1000) if hm.sum() > 50 else 0.0
            res.append(out)
        print(key, {r["variant"]: round(r.get("soup_f2", 0), 3) for r in res if r["key"] == key}, flush=True)
        pd.DataFrame(res).to_csv(a.out, index=False)
    df = pd.DataFrame(res)
    print(df.groupby("variant")[["vox_iou", "soup_f2", "hand_f2", "soup_f5", "hand_f5"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
