"""Fit MANO to B0's predicted hand surface and score the fit (CPU).

For each frame with saved meshes (eval_b0 --save-every), targets:
  gt          the GT hand surface (checks the fitter: a perfect fit gives F@2 = 1, MPJPE = 0)
  gt512       B0 stage 2 on GT voxels, predicted hand points
  cascade512  B0 cascade, predicted hand points
Predicted hand points = predicted surface samples nearer the GT hand than the GT object (the oracle split the hand
F-scores use; B0 has no label head). Each fit starts from 24 random orientations at the mean pose (no GT pose used).
Per target: fit_to_target F@2/F@5 (does a MANO hand explain the prediction?), target_vs_gt and fit_vs_gt F@2/F@5
(does the MANO projection keep, lose or improve accuracy?), MPJPE of the fit's 21 joints vs the GT hand's (mm).

    python scripts/b0/fit_mano.py --runs DIR_B0 ... --mano mano_v1_2/models --index index.parquet --out mano.csv
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sight_of_hand.b0.mano_fit import Mano, fit  # noqa: E402


def surf(V, F, n, seed=0):
    return np.asarray(trimesh.Trimesh(V, F, process=False).sample(n, seed=seed))


def fsc(a, b, tau):
    p = (cKDTree(b).query(a)[0] <= tau).mean()
    r = (cKDTree(a).query(b)[0] <= tau).mean()
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--mano", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", default="b0")
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--save-dir", help="write fitted MANO meshes here (<key>__<target>.ply)")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    side = pd.read_parquet(a.index, columns=["key", "side"]).set_index("key").side.to_dict()
    manos = {s: Mano(a.mano, s) for s in ("right", "left")}
    done = set()
    res = []
    if Path(a.out).exists():
        old = pd.read_csv(a.out)
        res = old.to_dict("records")
        done = set(zip(old.key, old.target))
    sd = Path(a.save_dir) if a.save_dir else None
    if sd:
        sd.mkdir(parents=True, exist_ok=True)
    for run in map(Path, a.runs):
        for v in sorted((run / "vis").iterdir()):
            key = v.name
            if a.keys and key not in a.keys:
                continue
            z = np.load(v / "gt_soup.npz")
            hv, hf, ov, of = z["hv"], z["hf"], z["ov"], z["of"]
            m = manos[side[key]]
            gth = surf(hv, hf, 8000, 0)
            gto = surf(ov, of, 8000, 1)
            gtj = m.joints_of(hv)
            tgts = {"gt": gth}
            for mode in ("gt512", "cascade512"):
                p = run / "meshes" / f"{key}__{mode}__{a.row}.ply"
                if p.exists():
                    mm = trimesh.load(p, process=False)
                    pts = surf(np.asarray(mm.vertices), np.asarray(mm.faces), 16000, 2)
                    hand = cKDTree(gth).query(pts)[0] <= cKDTree(gto).query(pts)[0]
                    if hand.sum() >= 200:
                        tgts[mode] = pts[hand]
            for name, T in tgts.items():
                if (key, name) in done:
                    continue
                t0 = time.time()
                r = fit(m, T)
                fv = surf(r["verts"], m.faces, 8000, 3)
                row = {"key": key, "target": name, "side": side[key], "n_pts": len(T), "seconds": time.time() - t0,
                       "loss": r["loss"], "mpjpe_mm": float(np.linalg.norm(r["joints"] - gtj, axis=1).mean() * 1000),
                       "tip_err_mm": float(np.linalg.norm(r["joints"][16:] - gtj[16:], axis=1).mean() * 1000)}
                for tau in (2, 5, 10):
                    row[f"fit_to_target_f{tau}"] = fsc(fv, T, tau / 1000)
                    row[f"target_vs_gt_f{tau}"] = fsc(T, gth, tau / 1000)
                    row[f"fit_vs_gt_f{tau}"] = fsc(fv, gth, tau / 1000)
                res.append(row)
                if sd:
                    trimesh.Trimesh(r["verts"], m.faces, process=False).export(sd / f"{key}__{name}.ply")
                print(f"{key} {name}: fit->tgt F@2 {row['fit_to_target_f2']:.2f}  tgt F@2 {row['target_vs_gt_f2']:.2f}"
                      f"  fit F@2 {row['fit_vs_gt_f2']:.2f}  MPJPE {row['mpjpe_mm']:.1f} mm  {row['seconds']:.0f}s",
                      flush=True)
                pd.DataFrame(res).to_csv(a.out, index=False)
    df = pd.DataFrame(res)
    print(df.groupby("target").median(numeric_only=True).round(3).T.to_string())


if __name__ == "__main__":
    main()
