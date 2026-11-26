"""Does a zero (mean) normalized latent decode to empty space? (GPU; run from a Pixal3D checkout.)

Stage 2 mixed training puts predicted coordinates that are not in GT (extra voxels) in the input. They have no GT
latent. If the dataset-mean latent at such a voxel decodes to no surface, it can serve as their training target.
For val frames: GT 512 latent + all 6-neighbour voxels of GT not in GT, filled with the mean latent; decode; report
the share of mesh vertices that land in the added voxels, and the GT-voxel decode's change (vertex count).

    python test_empty_latent.py --val /workspace/b0/val --n 20
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--add-frac", type=float, default=0.15, help="added voxels as a share of GT voxels")
    ap.add_argument("--out")
    a = ap.parse_args()
    import torch
    sys.path.insert(0, os.environ.get("PX", "."))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pixal3d.modules.sparse import SparseTensor
    from eval_b0 import build_pipeline
    torch.set_grad_enabled(False)
    a.lora_ss = a.lora_512 = a.lora_1024 = None
    a.rank, a.alpha = 32, 32
    pipe = build_pipeline(a)
    mean = torch.tensor(pipe.shape_slat_normalization["mean"]).float().cuda()
    root = Path(a.val) / "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"
    rng = np.random.default_rng(0)
    res = []
    for d in sorted(root.iterdir())[:a.n]:
        z = np.load(d / "view00.npz")
        c, f = z["coords"].astype(int), z["feats"].astype(np.float32)
        G = {tuple(x) for x in c.tolist()}
        nb = {tuple(x + o) for x in c for o in np.r_[np.eye(3, dtype=int), -np.eye(3, dtype=int)]}
        nb = [x for x in nb if x not in G and all(0 <= v < 32 for v in x)]
        nb = [nb[i] for i in rng.choice(len(nb), size=min(len(nb), int(len(c) * a.add_frac)), replace=False)]
        out = {"key": d.name, "n_gt": len(c), "n_add": len(nb)}
        for name, cc, ff in (("gt", c, torch.from_numpy(f).cuda()),
                             ("plus_mean", np.r_[c, np.array(nb, int).reshape(-1, 3)],
                              torch.cat([torch.from_numpy(f).cuda(), mean[None].expand(len(nb), -1)]))):
            co = torch.cat([torch.zeros(len(cc), 1, dtype=torch.int32), torch.from_numpy(cc.astype(np.int32))], 1)
            slat = SparseTensor(feats=ff.float(), coords=co.cuda())
            m = pipe.decode_shape_slat(slat, 512)
            mesh = m[0][0] if isinstance(m, tuple) else m[0]
            v = mesh.vertices.float().cpu().numpy()
            vox = np.clip(np.floor((v + 0.5) * 32).astype(int), 0, 31)
            added = {tuple(x) for x in nb}
            out[f"{name}_nverts"] = len(v)
            out[f"{name}_in_added"] = float(np.mean([tuple(x) in added for x in vox.tolist()])) if len(v) else 0.0
        res.append(out)
        print(json.dumps(out), flush=True)
    s = {k: float(np.median([r[k] for r in res])) for k in res[0] if k != "key"}
    print("MEDIAN", json.dumps(s), flush=True)
    if a.out:
        json.dump({"frames": res, "median": s}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
