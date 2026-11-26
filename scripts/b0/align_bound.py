"""How much of the error is a global transform? (CPU, from eval visual dumps)

For each saved prediction (meshes/<key>__<mode>__<row>.ply), align it to the GT soup by trimmed ICP under nested
transform families, then re-score F@{1,2,5} (soup / hand / object, oracle part split as in the eval):
  none        as predicted
  trans       translation only
  scale       uniform scale + translation (includes the scaling about the camera centre that a wrong depth d causes)
  sim         rotation + uniform scale + translation
  affine      full 3x4 affine
  sim_parts   a separate similarity for the hand and for the object (relative placement freed)
Each is an oracle bound: it uses GT to pick the transform (kept only if it raises soup F@2), i.e. what a perfect learned correction of that family
could recover. Also: the best integer shift of the stage 1 voxels (32^3, within +-2) and its IoU.

    python scripts/b0/align_bound.py --runs DIR ... --out align.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

TAUS = (1, 2, 5)


def surf(V, F, n, seed=0):
    return np.asarray(trimesh.Trimesh(V, F, process=False).sample(n, seed=seed))


def fit_tf(P, Q, kind):
    """least-squares transform mapping P -> Q (paired), returns 4x4"""
    T = np.eye(4)
    if kind == "trans":
        T[:3, 3] = (Q - P).mean(0)
    elif kind == "scale":  # q = s p + t
        pc, qc = P - P.mean(0), Q - Q.mean(0)
        s = (pc * qc).sum() / (pc * pc).sum()
        T[:3, :3] = s * np.eye(3)
        T[:3, 3] = Q.mean(0) - s * P.mean(0)
    elif kind == "sim":  # Umeyama
        mp, mq = P.mean(0), Q.mean(0)
        pc, qc = P - mp, Q - mq
        U, S, Vt = np.linalg.svd(qc.T @ pc / len(P))
        D = np.eye(3)
        D[2, 2] = np.sign(np.linalg.det(U @ Vt))
        R = U @ D @ Vt
        s = np.trace(np.diag(S) @ D) / (pc ** 2).sum(1).mean()
        T[:3, :3], T[:3, 3] = s * R, mq - s * R @ mp
    elif kind == "affine":
        A = np.c_[P, np.ones(len(P))]
        X = np.linalg.lstsq(A, Q, rcond=None)[0]
        T[:3, :] = X.T
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def icp(P, G, kind, iters=40, trim=0.9):
    """trimmed point-to-point ICP of P onto G; returns 4x4"""
    tree = cKDTree(G)
    T = np.eye(4)
    for _ in range(iters):
        X = apply(T, P)
        d, j = tree.query(X)
        keep = d <= np.quantile(d, trim)
        dT = fit_tf(X[keep], G[j[keep]], kind)
        T = dT @ T
        if np.abs(dT - np.eye(4)).max() < 1e-6:
            break
    return T


def fsc(P, G, tau):
    p = (cKDTree(G).query(P)[0] <= tau).mean()
    r = (cKDTree(P).query(G)[0] <= tau).mean()
    return 2 * p * r / (p + r) if p + r else 0.0


def score(P, hand_mask, gh, go, out, tag):
    g = np.r_[gh, go]
    for t in TAUS:
        out[f"{tag}/soup_f{t}"] = fsc(P, g, t)
        out[f"{tag}/hand_f{t}"] = fsc(P[hand_mask], gh, t) if hand_mask.sum() > 50 else 0.0
        out[f"{tag}/obj_f{t}"] = fsc(P[~hand_mask], go, t) if (~hand_mask).sum() > 50 else 0.0


def tf_size(T, c):
    """translation of the soup centroid (mm), scale change (%), rotation (deg)"""
    M = T[:3, :3]
    s = np.cbrt(abs(np.linalg.det(M)))
    R = M / s
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return np.linalg.norm(apply(T, c[None])[0] - c), (s - 1) * 100, ang


def best_shift(pred, gt):
    P = {tuple(x) for x in pred.astype(int).tolist()}
    G = {tuple(x) for x in gt.astype(int).tolist()}
    best = (len(P & G) / max(len(P | G), 1), (0, 0, 0))
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            for dz in range(-2, 3):
                S = {(a + dx, b + dy, c + dz) for a, b, c in P}
                iou = len(S & G) / max(len(S | G), 1)
                if iou > best[0] + 1e-9:
                    best = (iou, (dx, dy, dz))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12000)
    a = ap.parse_args()
    res = []
    for run in map(Path, a.runs):
        for m in sorted((run / "meshes").glob("*.ply")):
            mode, row = m.stem.split("__")[-2:]
            key = "__".join(m.stem.split("__")[:-2])
            z = np.load(run / "vis" / key / "gt_soup.npz")
            gh, go = surf(z["hv"], z["hf"], a.n // 2, 0) * 1000, surf(z["ov"], z["of"], a.n // 2, 1) * 1000  # mm
            g = np.r_[gh, go]
            mm = trimesh.load(m, process=False)
            P = surf(np.asarray(mm.vertices), np.asarray(mm.faces), a.n, 2) * 1000
            out = {"key": key, "mode": mode, "row": row}
            hm = cKDTree(gh).query(P)[0] <= cKDTree(go).query(P)[0]  # oracle split, fixed before alignment
            score(P, hm, gh, go, out, "none")
            c = P.mean(0)
            for kind in ("trans", "scale", "sim", "affine"):
                T = icp(P, g, kind)
                Q = apply(T, P)
                tmp = {}
                score(Q, hm, gh, go, tmp, kind)
                if tmp[f"{kind}/soup_f2"] < out["none/soup_f2"]:  # ICP minimises distance, not F: a perfect
                    T = np.eye(4)                                  # correction would keep the prediction instead
                    score(P, hm, gh, go, tmp, kind)
                out.update(tmp)
                out[f"{kind}/shift_mm"], out[f"{kind}/scale_pct"], out[f"{kind}/rot_deg"] = tf_size(T, c)
            Q = P.copy()
            for part, gp in ((hm, gh), (~hm, go)):
                if part.sum() > 50:
                    Q[part] = apply(icp(P[part], gp, "sim"), P[part])
            tmp = {}
            score(Q, hm, gh, go, tmp, "sim_parts")
            if tmp["sim_parts/soup_f2"] < out["none/soup_f2"]:
                score(P, hm, gh, go, tmp, "sim_parts")
            out.update(tmp)
            cz = run / "vis" / key / f"coords32__{mode}__{row}.npz"
            if cz.exists():
                o = np.load(cz)
                iou, sh = best_shift(o["pred"], o["gt"])
                P0 = {tuple(x) for x in o["pred"].astype(int).tolist()}
                G0 = {tuple(x) for x in o["gt"].astype(int).tolist()}
                out["vox_iou"], out["vox_iou_best_shift"] = len(P0 & G0) / max(len(P0 | G0), 1), iou
                out["vox_best_shift"] = str(sh)
            res.append(out)
            print(key, mode, row, f"soup F@2 {out['none/soup_f2']:.2f} -> trans {out['trans/soup_f2']:.2f} "
                  f"sim {out['sim/soup_f2']:.2f} affine {out['affine/soup_f2']:.2f} parts {out['sim_parts/soup_f2']:.2f}",
                  flush=True)
            pd.DataFrame(res).to_csv(a.out, index=False)
    df = pd.DataFrame(res)
    cols = [c for c in df.columns if c.endswith(("soup_f2", "hand_f2", "obj_f2", "shift_mm", "scale_pct", "rot_deg"))
            or c.startswith("vox_iou")]
    print(df.groupby(["row", "mode"])[cols].median().round(3).T.to_string())


if __name__ == "__main__":
    main()
