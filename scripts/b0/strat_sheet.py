"""Failure-mode sheets by stage 1 IoU (CPU): one PNG per IoU bin, a row per frame, B0 only.

Columns: input (oracle matte) | GT | B0 stage 2 on GT voxels | B0 cascade, from the camera; the same three from the
side; stage 1 voxels at 32^3 from the side (grey both, red missed, blue extra). Faces are coloured by the nearest
GT part (hand tan, object blue-grey): an oracle split, the same one the hand / object F-scores use.

    python scripts/b0/strat_sheet.py --runs DIR_B0 ... --table strat.csv --out sheets/ --per-bin 3
strat.csv: key, ss/iou, vox_iou, gt512, cascade512 (hand F@2), bin
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
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sight_of_hand.b0 import geometry as G  # noqa: E402
from render_vis import draw, part_colors, shade, side  # noqa: E402


def run_of(runs, key):
    for r in runs:
        if (r / "vis" / key / "gt_soup.npz").exists():
            return r


def sheet(rows, runs, title, out):
    n = len(rows)
    fig, axes = plt.subplots(n, 8, figsize=(8 * 2.4, n * 2.6))
    axes = np.atleast_2d(axes)
    heads = ["frame", "GT", "B0 on GT voxels", "B0 cascade", "GT side", "GT voxels side", "cascade side",
             "stage 1 voxels 32³, side"]
    for i, (_, rw) in enumerate(rows.iterrows()):
        key = rw.key
        run = run_of(runs, key)
        z = np.load(run / "vis" / key / "gt_soup.npz")
        hv, hf, ov, of, K = z["hv"], z["hf"], z["ov"], z["of"], z["K"]
        frame = np.asarray(Image.open(run / "vis" / key / "frame.jpg"))
        M = {"gt": (np.r_[hv, ov], np.r_[hf, of + len(hv)])}
        for mode in ("gt512", "cascade512"):
            p = run / "meshes" / f"{key}__{mode}__b0.ply"
            if p.exists():
                m = trimesh.load(p, process=False)
                M[mode] = (np.asarray(m.vertices), np.asarray(m.faces))
        Vg = M["gt"][0]
        u, v = K[0, 0] * Vg[:, 0] / Vg[:, 2] + K[0, 2], K[1, 1] * Vg[:, 1] / Vg[:, 2] + K[1, 2]
        cx, cy, half = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2, max(np.ptp(u), np.ptp(v)) * 0.75 + 15
        c0 = Vg.mean(0)
        sv = side(Vg, c0)[0]
        lo, hi = sv.min(0) - 0.03, sv.max(0) + 0.03
        ax = axes[i, 0]
        ax.imshow(frame), ax.set_xlim(cx - half, cx + half), ax.set_ylim(cy + half, cy - half)
        ax.set_ylabel(f"stage 1 IoU {rw['ss/iou']:.2f}\nvoxel IoU 32³ {rw.vox_iou:.2f}", fontsize=8)
        for j, k in enumerate(("gt", "gt512", "cascade512")):
            if k not in M:
                continue
            V, F = M[k]
            col = part_colors(V, F, hv, ov)
            ax = axes[i, 1 + j]
            ax.imshow(frame, alpha=0.35)
            V2 = np.c_[K[0, 0] * V[:, 0] / V[:, 2] + K[0, 2], K[1, 1] * V[:, 1] / V[:, 2] + K[1, 2]]
            draw(ax, V2, V[:, 2], F, shade(V, F, col))
            ax.set_xlim(cx - half, cx + half), ax.set_ylim(cy + half, cy - half)
            ax = axes[i, 4 + j]
            S2, dep = side(V, c0)
            draw(ax, S2, -dep, F, shade(V, F, col, light=np.array([1.0, -0.5, 0.0])))
            ax.set_xlim(lo[0], hi[0]), ax.set_ylim(hi[1], lo[1]), ax.set_aspect("equal")
            if k != "gt":
                f2 = rw[k]
                axes[i, 1 + j].set_title(f"hand F@2 {f2:.2f}", fontsize=8)
        o = np.load(run / "vis" / key / "coords32__cascade512__b0.npz")
        g = {tuple(x) for x in o["gt"].astype(int).tolist()}
        p = {tuple(x) for x in o["pred"].astype(int).tolist()}
        ax = axes[i, 7]
        # voxel centres -> camera metres with the eval's own grid mapping, then the same side view as the meshes;
        # drawn far to near so the nearest voxel along each line of sight is the one seen
        pts, cols = [], []
        for s, c in ((g & p, (0.6, 0.6, 0.6)), (g - p, (0.84, 0.15, 0.16)), (p - g, (0.12, 0.47, 0.71))):
            for x in s:
                pts.append(x), cols.append(c)
        cam = G.grid_to_cam((np.array(pts, float) + 0.5) / 32 - 0.5, float(z["d"]), float(z["L"]))
        S2, dep = side(cam, c0)
        o_ = np.argsort(dep)  # viewer on +x: small x is far, drawn first (as draw(..., -dep) does for meshes)
        ax.scatter(S2[o_, 0], S2[o_, 1], s=70, c=np.array(cols)[o_], marker="s", edgecolors="white", linewidths=0.5)
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(hi[1], lo[1]), ax.set_aspect("equal")
        ax.set_title(f"both {len(g & p)}  missed {len(g - p)}  extra {len(p - g)}", fontsize=7)
        for ax in axes[i]:
            ax.set_xticks([]), ax.set_yticks([])
    for j, h in enumerate(heads):
        axes[0, j].set_title((h + "\n" + axes[0, j].get_title()).strip(), fontsize=9)
    fig.suptitle(title + "\nVoxels: grey = predicted and in GT, red = missed (GT only), blue = extra (predicted only). Colour = nearest GT part (oracle).", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bin", type=int, default=3)
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]
    t = pd.read_csv(a.table)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for b, g in t.groupby("bin", sort=True):
        g = g.sort_values("ss/iou")
        pick = g.iloc[np.linspace(0, len(g) - 1, min(a.per_bin, len(g))).round().astype(int)]
        name = str(b).replace("(", "").replace("]", "").replace(", ", "-")
        title = (f"Stage 1 IoU {name}: {len(g)} frames, median IoU {g['ss/iou'].median():.2f}, "
                 f"hand F@2 on GT voxels {g.gt512.median():.2f}, cascade {g.cascade512.median():.2f}")
        sheet(pick, runs, title, out / f"iou_{name}.png")
        print(name, list(pick.key), flush=True)


if __name__ == "__main__":
    main()
