"""Is the 512 cascade limited by stage 1, or does stage 2 mishandle predicted occupancy? (CPU, from eval visual dumps)

Per frame with a cascade512 dump (vis/<key>/coords32__cascade512__<row>.npz, meshes/<key>__cascade512__<row>.ply):
  vox_recall     share of the GT surface lying inside predicted 32^3 voxels
  ceiling_f2     F@2mm of a perfect stage 2 given the predicted voxels: the GT surface clipped to them
                 (precision 1; extra voxels left empty)
  actual_f2      F@2mm of the cascade mesh (same sampling)
  pred_in_fp     share of the predicted surface lying in voxels stage 1 added (not in GT); stage 2 trained on GT
                 coords has only seen voxels that all contain surface, so it may fill these
  prec_in_fp / prec_in_tp   F@2mm precision of predicted surface inside added vs correct voxels
and, for reference, the GT-coords mesh (gt512) scored the same way (its ceiling is 1).

    python scripts/b0/cascade_bound.py --runs DIR_ZS DIR_B0 --out bound.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sight_of_hand.b0 import geometry as G  # noqa: E402

R = 32
TAU = 2.0  # mm


def surf(V, F, n, seed=0):
    return np.asarray(trimesh.Trimesh(V, F, process=False).sample(n, seed=seed))


def vox(p_cam, d, L):
    g = G.cam_to_grid(p_cam, d, L)
    return np.clip(np.floor((g + 0.5) * R).astype(int), 0, R - 1)


def fscore(pred, gt, tau=TAU):
    if len(pred) == 0:
        return 0.0, 0.0, 0.0, np.zeros(0, bool)
    dp = cKDTree(gt).query(pred)[0] <= tau
    dg = cKDTree(pred).query(gt)[0] <= tau
    p, r = dp.mean(), dg.mean()
    return (2 * p * r / (p + r) if p + r else 0.0), p, r, dp


def keyset(c):
    return {tuple(x) for x in np.asarray(c).astype(int).tolist()}


def frame(run, key, row):
    z = np.load(run / "vis" / key / "gt_soup.npz")
    d, L = float(z["d"]), float(z["L"])
    gt = surf(np.r_[z["hv"], z["ov"]], np.r_[z["hf"], z["of"] + len(z["hv"])], 20000) * 1000.0
    o = np.load(run / "vis" / key / f"coords32__cascade512__{row}.npz")
    P, Gv = keyset(o["pred"]), keyset(o["gt"])
    gvox = [tuple(v) for v in vox(gt / 1000.0, d, L).tolist()]
    in_gt = np.array([v in Gv for v in gvox])
    in_pred = np.array([v in P for v in gvox])
    out = {"key": key, "row": row, "n_pred_vox": len(P), "n_gt_vox": len(Gv),
           "vox_iou": len(P & Gv) / max(len(P | Gv), 1),
           "gt_surface_in_gt_vox": in_gt.mean(),  # sanity check of the voxel mapping, should be ~1
           "vox_recall": in_pred.mean()}
    out["ceiling_f2"] = fscore(gt[in_pred], gt)[0]
    for mode in ("cascade512", "gt512"):
        m = run / "meshes" / f"{key}__{mode}__{row}.ply"
        if not m.exists():
            continue
        mm = trimesh.load(m, process=False)
        pr = surf(np.asarray(mm.vertices), np.asarray(mm.faces), 20000) * 1000.0
        f, p, r, ok = fscore(pr, gt)
        out[f"{mode}_f2"], out[f"{mode}_prec"], out[f"{mode}_rec"] = f, p, r
        if mode == "cascade512":
            pv = [tuple(v) for v in vox(pr / 1000.0, d, L).tolist()]
            fp = np.array([v not in Gv for v in pv])
            out["pred_in_fp"] = fp.mean()
            out["prec_in_fp"] = ok[fp].mean() if fp.any() else np.nan
            out["prec_in_tp"] = ok[~fp].mean() if (~fp).any() else np.nan
            out["share_of_ceiling"] = f / out["ceiling_f2"] if out["ceiling_f2"] else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for run in map(Path, a.runs):
        for f in sorted(run.glob("vis/*/coords32__cascade512__*.npz")):
            key, row = f.parent.name, f.stem.split("__")[-1]
            try:
                res.append(frame(run, key, row))
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {key} {row}: {e}", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(a.out, index=False)
    cols = ["gt_surface_in_gt_vox", "vox_iou", "vox_recall", "ceiling_f2", "cascade512_f2", "share_of_ceiling",
            "gt512_f2", "pred_in_fp", "prec_in_fp", "prec_in_tp", "cascade512_prec", "cascade512_rec"]
    print(df.groupby("row")[[c for c in cols if c in df]].median().round(3).T.to_string())
    print(df.groupby("row").size().to_string())


if __name__ == "__main__":
    main()
