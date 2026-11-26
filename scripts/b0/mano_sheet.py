"""Side-by-side renders of MANO fits (CPU): per frame, GT hand | predicted hand (oracle split) | MANO fit to it,
camera view and side view, for gt512 and cascade512.

    python scripts/b0/mano_sheet.py --runs DIR_B0 ... --fits fits/ --csv mano.csv --keys K1 K2 --out mano.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import trimesh  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_vis import HAND, draw, shade, side  # noqa: E402

FIT = np.array([0.45, 0.70, 0.45])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--fits", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.csv).set_index(["key", "target"])
    cols = [("gt", "GT hand"), ("gt512", "B0 on GT voxels, hand"), ("fit:gt512", "MANO fit to it"),
            ("cascade512", "B0 cascade, hand"), ("fit:cascade512", "MANO fit to it")]
    fig, ax = plt.subplots(2 * len(a.keys), len(cols), figsize=(3 * len(cols), 5.6 * len(a.keys)))
    for i, key in enumerate(a.keys):
        run = next(Path(r) for r in a.runs if (Path(r) / "vis" / key / "gt_soup.npz").exists())
        z = np.load(run / "vis" / key / "gt_soup.npz")
        hv, hf, ov, K = z["hv"], z["hf"], z["ov"], z["K"]
        M = {"gt": (hv, hf)}
        for mode in ("gt512", "cascade512"):
            m = trimesh.load(run / "meshes" / f"{key}__{mode}__b0.ply", process=False)
            V, F = np.asarray(m.vertices), np.asarray(m.faces)
            c = V[F].mean(1)
            keep = cKDTree(hv).query(c)[0] <= cKDTree(ov).query(c)[0]
            M[mode] = (V, F[keep])
            f = trimesh.load(Path(a.fits) / f"{key}__{mode}.ply", process=False)
            M[f"fit:{mode}"] = (np.asarray(f.vertices), np.asarray(f.faces))
        u, v = K[0, 0] * hv[:, 0] / hv[:, 2] + K[0, 2], K[1, 1] * hv[:, 1] / hv[:, 2] + K[1, 2]
        cx, cy, half = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2, max(np.ptp(u), np.ptp(v)) * 0.65 + 10
        c0 = hv.mean(0)
        sv = side(hv, c0)[0]
        lo, hi = sv.min(0) - 0.02, sv.max(0) + 0.02
        for j, (k, t) in enumerate(cols):
            V, F = M[k]
            base = FIT if k.startswith("fit") else HAND
            col = np.repeat(base[None], len(F), 0)
            a0, a1 = ax[2 * i, j], ax[2 * i + 1, j]
            V2 = np.c_[K[0, 0] * V[:, 0] / V[:, 2] + K[0, 2], K[1, 1] * V[:, 1] / V[:, 2] + K[1, 2]]
            draw(a0, V2, V[:, 2], F, shade(V, F, col))
            a0.set_xlim(cx - half, cx + half), a0.set_ylim(cy + half, cy - half), a0.set_aspect("equal")
            S2, dep = side(V, c0)
            draw(a1, S2, -dep, F, shade(V, F, col, light=np.array([1.0, -0.5, 0.0])))
            a1.set_xlim(lo[0], hi[0]), a1.set_ylim(hi[1], lo[1]), a1.set_aspect("equal")
            tt = t
            if k.startswith("fit"):
                r = df.loc[(key, k[4:])]
                tt += f"\nF@2 vs GT {r.fit_vs_gt_f2:.2f}  MPJPE {r.mpjpe_mm:.0f} mm"
            elif k != "gt":
                tt += f"\nF@2 vs GT {df.loc[(key, k)].target_vs_gt_f2:.2f}"
            a0.set_title(tt, fontsize=8)
            for q in (a0, a1):
                q.set_xticks([]), q.set_yticks([])
        ax[2 * i, 0].set_ylabel(key[-28:], fontsize=7)
    fig.suptitle("MANO fits to B0's predicted hand (camera view, then side view). Fit starts from random "
                 "orientations at the mean pose; no GT pose used.", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=100)


if __name__ == "__main__":
    main()
