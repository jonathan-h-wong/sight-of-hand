"""Render eval_b0.py visual dumps (CPU, matplotlib): per frame, GT vs zero-shot vs B0 for stage 2 on GT coords and the
512 cascade, from the camera (over the frame) and from the side, plus the stage 1 occupancy (32^3) seen from the side.

    python scripts/b0/render_vis.py --runs DIR_ZS DIR_B0 ... --rows rows.parquet --out vis_png/

Each run dir is an eval_b0 output (meshes/<key>__<mode>__<row>.ply, vis/<key>/...); keys present in every run are drawn.
Predicted faces are coloured by the nearest GT part (hand tan, object blue-grey).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

HAND, OBJ = np.array([0.86, 0.66, 0.52]), np.array([0.55, 0.62, 0.72])
COLS = [("gt", "GT"), ("gt512__zeroshot", "Zero-shot, stage 2 on GT coords"), ("gt512__b0", "B0, stage 2 on GT coords"),
        ("cascade512__zeroshot", "Zero-shot cascade (1 -> 512)"), ("cascade512__b0", "B0 cascade (1 -> 512)")]


def shade(V, F, base, light=np.array([0.3, -0.5, -0.8])):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    lam = np.abs(n @ (light / np.linalg.norm(light)))
    return np.clip(base * (0.35 + 0.65 * lam[:, None]), 0, 1)


def draw(ax, V2, depth, F, col, alpha=1.0):
    """painter's algorithm: far faces first"""
    o = np.argsort(-depth[F].mean(1))
    pc = PolyCollection(V2[F[o]], facecolors=col[o], edgecolors="none", alpha=alpha, antialiased=False)
    ax.add_collection(pc)


def part_colors(V, F, hv, ov):
    c = V[F].mean(1)
    dh, do = cKDTree(hv).query(c)[0], cKDTree(ov).query(c)[0]
    return np.where((dh <= do)[:, None], HAND, OBJ)


def side(V, c0):
    """rotate 90 deg about the camera's vertical axis around c0: view from the right; returns (2D, depth)"""
    p = V - c0
    return np.c_[-p[:, 2], p[:, 1]], p[:, 0]


def render_key(key, runs, rows, out):
    gt = np.load(runs[0] / "vis" / key / "gt_soup.npz")
    hv, hf, ov, of, K = gt["hv"], gt["hf"], gt["ov"], gt["of"], gt["K"]
    frame = np.asarray(Image.open(runs[0] / "vis" / key / "frame.jpg"))
    meshes = {"gt": (np.r_[hv, ov], np.r_[hf, of + len(hv)])}
    for r in runs:
        for p in (r / "meshes").glob(f"{key}__*.ply"):
            m = trimesh.load(p, process=False)
            mode, row = p.stem.split("__")[-2:]
            meshes[f"{mode}__{row}"] = (np.asarray(m.vertices), np.asarray(m.faces))
    cols = [c for c in COLS if c[0] in meshes]
    c0 = meshes["gt"][0].mean(0)
    allv = side(meshes["gt"][0], c0)[0]
    lo, hi = allv.min(0) - 0.03, allv.max(0) + 0.03
    u = (K[0, 0] * meshes["gt"][0][:, 0] / meshes["gt"][0][:, 2] + K[0, 2])
    v = (K[1, 1] * meshes["gt"][0][:, 1] / meshes["gt"][0][:, 2] + K[1, 2])
    cx, cy, half = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2, max(np.ptp(u), np.ptp(v)) * 0.8 + 20
    occ = {c: np.load(f) for c in ("zeroshot", "b0") for f in runs_vis(runs, key, c)}
    fig, axes = plt.subplots(3, len(cols), figsize=(3.3 * len(cols), 10.2))
    for j, (k, title) in enumerate(cols):
        V, F = meshes[k]
        col = part_colors(V, F, hv, ov)
        sub = ""
        if k != "gt":
            mode, row = k.split("__")
            rr = rows[(rows.key == key) & (rows["mode"] == mode) & (rows["row"] == row) & (rows.matte == "oracle")]
            if len(rr):
                sub = f"\nsoup F@2 {rr['gt_d/soup/f@2'].iloc[0]:.2f}  hand F@2 {rr['gt_d/hand/f@2'].iloc[0]:.2f}"
        ax = axes[0, j]
        ax.imshow(frame)
        V2 = np.c_[K[0, 0] * V[:, 0] / V[:, 2] + K[0, 2], K[1, 1] * V[:, 1] / V[:, 2] + K[1, 2]]
        draw(ax, V2, V[:, 2], F, shade(V, F, col), alpha=0.85)
        ax.set_xlim(cx - half, cx + half), ax.set_ylim(cy + half, cy - half)
        ax.set_title(title + sub, fontsize=9)
        ax = axes[1, j]
        S2, dep = side(V, c0)
        draw(ax, S2, -dep, F, shade(V, F, col, light=np.array([1.0, -0.5, 0.0])))
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(hi[1], lo[1]), ax.set_aspect("equal")
        ax.set_title("side view (camera at left)", fontsize=8)
        ax = axes[2, j]
        row = k.split("__")[-1]
        if k.startswith("cascade") and row in occ:
            o = occ[row]
            g = {tuple(x) for x in o["gt"].astype(int).tolist()}  # coords are stored as uint8
            p = {tuple(x) for x in o["pred"].astype(int).tolist()}
            for s, c, lab in ((g & p, "0.6", "both"), (g - p, "tab:red", "GT only (missed)"),
                              (p - g, "tab:blue", "predicted only")):
                if s:
                    a = np.array(sorted(s))
                    # grid axes: 0 = x, 1 = up, 2 = toward the camera; side view = (depth, up)
                    ax.scatter(-a[:, 2], a[:, 1], s=9, c=c, marker="s", label=f"{lab} ({len(s)})", alpha=0.6)
            ax.set_aspect("equal"), ax.legend(fontsize=7, loc="lower left")
            ax.set_title("stage 1 occupancy, 32^3, side view", fontsize=8)
        else:
            ax.axis("off")
        for a_ in axes[:2, j]:
            a_.set_xticks([]), a_.set_yticks([])
    hid = rows.loc[rows.key == key, "hand_hidden"]
    fig.suptitle(f"{key}   hand hidden {float(hid.iloc[0]) * 100:.0f}%" if len(hid) else key, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{key}.png", dpi=110)
    plt.close(fig)


def runs_vis(runs, key, row):
    for r in runs:
        f = r / "vis" / key / f"coords32__cascade512__{row}.npz"
        if f.exists():
            yield f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", nargs="*")
    a = ap.parse_args()
    runs = [Path(r) for r in a.runs]
    rows = pd.read_parquet(a.rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = a.keys or sorted({p.name for r in runs for p in (r / "vis").glob("*") if (p / "gt_soup.npz").exists()})
    for k in keys:
        render_key(k, [r for r in runs if (r / "vis" / k).exists()] + [r for r in runs if not (r / "vis" / k).exists()],
                   rows, out)
        print(k, flush=True)


if __name__ == "__main__":
    main()
