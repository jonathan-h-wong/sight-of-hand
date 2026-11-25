"""Encode B0 samples into Pixal3D's view-aligned toolkit layout, plus the P1 round trip and P2 alignment checks
(docs/baseline_b0_training_spec.md, sections 2.5-2.7 and 3). GPU; run from a Pixal3D checkout (PYTHONPATH has
Pixal3D, this repo and sight-of-hand-eval).

Per kept sample (filters below), with D_n = 1 and mesh_scale = total_scale = 1 / L_n:
  renders_cond/<key>/view00.png       RGBA, the frame padded to the square centred on the principal point (S x S,
                                      native pixels; the dataset resizes), alpha = modal hand + grasped object
  renders_cond/<key>/transforms.json  camera_angle_x, one frame with the canonical front camera at distance 1
  shape_latents/shape_enc_next_dc_f16c32_fp16_{512,1024}_view/<key>/view00.npz (+ view00_scale.json)
  ss_latents/ss_enc_conv3d_16l8_fp16_64_view/<key>/view00.npz (+ view00_scale.json)   from the 1024 latent coords
  <sub-root>/metadata.csv             the columns the stock datasets filter on
The soup (MANO hand + grasped YCB object, concatenated, no union) goes straight to the grid frame with
sight_of_hand.b0.geometry.cam_to_grid: no sphere normalization, no per-mesh box scale, no transform_mesh.

Filters (spec 2.2-2.4; the penetration and approach selection was done by build_b0_dataset.py):
  contained (soup inside the cube, 2/1024 margin) | hand and object amodal >= --min-inside inside the frame |
  d within [--d-min, --d-max] (defaults: the 0.5 / 99.5 percentiles of train) | some modal pixel of the hand or object.

    python encode_b0.py --split train --out /workspace/b0 [--limit N] [--roundtrip 64 --overlays 16]
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sight_of_hand.b0 import geometry as G  # noqa: E402

BUCKET = "xray-hands"
SHAPE_ENC = "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
SHAPE_DEC = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
SS_ENC = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
SS_DEC = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16"
SHAPE_NAME = "shape_enc_next_dc_f16c32_fp16_{res}_view"
SS_NAME = "ss_enc_conv3d_16l8_fp16_64_view"
TAUS_MM = (1.0, 2.0, 5.0)


def apply_filters(idx, split, a):
    tr = idx[idx.split == "train"]
    d_min = a.d_min if a.d_min is not None else float(np.percentile(tr.d, 0.5))
    d_max = a.d_max if a.d_max is not None else float(np.percentile(tr.d, 99.5))
    x = idx[idx.split == split].copy()
    steps = {"total": len(x)}
    for name, m in [("contained", x.contained),
                    ("hand_inside", x.hand_inside_frac >= a.min_inside),
                    ("obj_inside", x.obj_inside_frac >= a.min_inside),
                    ("depth_range", (x.d >= d_min) & (x.d <= d_max)),
                    ("modal_visible", (x.hand_modal_px + x.obj_modal_px) > 0)]:
        x = x[m.loc[x.index]]
        steps[name] = len(x)
    return x, {"d_min": d_min, "d_max": d_max, "steps": steps}


class Meshes:
    def __init__(self, data):
        from hoiaudit.sources import common
        from sight_of_hand.amodal.dexycb import YCB
        common.MANO_DIR = Path(data) / "mano" / "mano_v1_2" / "models"
        self.faces = {s: np.asarray(common.mano(s, False).faces, np.int64) for s in ("left", "right")}
        self.ycb, self.YCB, self.root = {}, YCB, Path(data) / "dexycb" / "models"

    def obj(self, yid):
        if yid not in self.ycb:
            import trimesh
            m = trimesh.load(self.root / self.YCB[yid] / "textured_simple.obj", process=False, force="mesh")
            self.ycb[yid] = (np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64))
        return self.ycb[yid]

    def soup(self, z):
        hv = z["hand_v"].astype(np.float64)
        V0, F0 = self.obj(int(z["ycb"]))
        P = z["obj_pose"].astype(np.float64)
        ov = V0 @ P[:, :3].T + P[:, 3]
        fh = self.faces[str(z["side"])]
        V = np.concatenate([hv, ov])
        F = np.concatenate([fh, F0 + len(hv)])
        part = np.r_[np.zeros(len(hv), np.int8), np.ones(len(ov), np.int8)]
        return V, F, part, (hv, fh), (ov, F0)


def dual_grid(o_voxel, torch, g, F, res):
    v = torch.from_numpy(g).float().contiguous()
    f = torch.from_numpy(F).long().contiguous()
    idx, dv, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        vertices=v, faces=f, grid_size=res, aabb=[[-0.5] * 3, [0.5] * 3],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2, timing=False)
    dv = (dv.float() * res - idx.float())
    assert torch.all(dv >= -1e-3) and torch.all(dv <= 1 + 1e-3)
    dv = (dv.clamp(0, 1) * 255).to(torch.uint8)
    inter = (inter[:, 0:1] + 2 * inter[:, 1:2] + 4 * inter[:, 2:3]).to(torch.uint8)
    return idx, dv, inter


def to_sparse(sp, torch, idx, dv, inter):
    coords = torch.cat([torch.zeros_like(idx[:, :1]), idx], -1).int()
    vertices = sp.SparseTensor((dv.float() / 255.0), coords)
    inter = inter.long()
    intersected = vertices.replace(torch.cat([inter % 2, inter // 2 % 2, inter // 4 % 2], -1).bool())
    return vertices.cuda(), intersected.cuda()


def rgba(jpg, lab, row):
    img = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
    lm = np.asarray(Image.open(io.BytesIO(lab)))
    a = ((lm > 0) * 255).astype(np.uint8)
    im = np.concatenate([img, a[..., None]], -1)
    return G.pad_image(im, int(row.S), int(row.pad_x), int(row.pad_y))


def surface(V, F, n, seed=0):
    import trimesh
    pts, fi = trimesh.sample.sample_surface(trimesh.Trimesh(V, F, process=False), n, seed=seed)
    return pts, fi


def fscores(pred, gt, taus=TAUS_MM):
    from scipy.spatial import cKDTree
    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    out = {"chamfer_mm": float((d_pg.mean() + d_gp.mean()) / 2)}
    for t in taus:
        p, r = (d_pg <= t).mean(), (d_gp <= t).mean()
        out[f"f@{t:g}mm"] = float(2 * p * r / (p + r)) if p + r > 0 else 0.0
    return out


def splat(uv, S, r=1):
    m = np.zeros((S, S), bool)
    u = np.round(uv).astype(int)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            x, y = u[:, 0] + dx, u[:, 1] + dy
            ok = (x >= 0) & (x < S) & (y >= 0) & (y < S)
            m[y[ok], x[ok]] = True
    return m


_W = {}


def _winit(data, res_list, rc):
    import torch
    torch.set_num_threads(1)
    _W.update(meshes=Meshes(data), res=res_list, rc=Path(rc))


def prep(job):
    """CPU side of one sample (pool worker): soup -> grid -> dual grids; RGBA image + transforms.json on disk."""
    import torch
    import o_voxel
    key, f, row = job
    try:
        z = np.load(io.BytesIO(f["npz"]))
        V, F, part, hand, obj = _W["meshes"].soup(z)
        d, L = float(row["d"]), float(row["L_n"])
        g = G.cam_to_grid(V, d, L)
        if not G.contained(g, 1.0 / 1024):
            return key, None, f"outside cube {np.abs(g).max():.4f}"
        rc = _W["rc"] / key
        rc.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba(f["jpg"], f["png"], pd.Series(row)), "RGBA").save(rc / "view00.png")
        cax = float(row["camera_angle_x"])
        (rc / "transforms.json").write_text(json.dumps({
            "camera_angle_x": cax, "mesh_scale": 1.0 / L,
            "frames": [{"file_path": "view00.png", "camera_angle_x": cax,
                        "transform_matrix": G.front_transform_matrix(G.D_N)}]}))
        grids = {r: tuple(t.numpy() for t in dual_grid(o_voxel, torch, g, F, r)) for r in _W["res"]}
        return key, {"grids": grids, "V": V, "g": g, "hand": hand, "obj": obj, "K": z["K"]}, None
    except Exception as e:  # recorded, not fatal
        return key, None, repr(e)[:300]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--prefix", default="train/b0_dexycb_s0_v1")
    ap.add_argument("--data", default="/workspace/data", help="mano/ and dexycb/models")
    ap.add_argument("--out", default="/workspace/b0")
    ap.add_argument("--res", default="512,1024")
    ap.add_argument("--min-inside", type=float, default=0.5)
    ap.add_argument("--d-min", type=float)
    ap.add_argument("--d-max", type=float)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--roundtrip", type=int, default=0, help="P1: decode this many samples, score vs GT soup")
    ap.add_argument("--overlays", type=int, default=0, help="P2: write this many alignment overlays")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    import boto3
    import torch
    import o_voxel
    import pixal3d.models as models
    import pixal3d.modules.sparse as sp
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import project_points_to_image_batch
    torch.set_grad_enabled(False)
    s3 = boto3.client("s3")
    res_list = [int(r) for r in a.res.split(",")]
    out = Path(a.out) / a.split
    rc = out / "renders_cond"
    rc.mkdir(parents=True, exist_ok=True)

    idx = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{a.prefix}/index.parquet")["Body"].read()))
    keep, fstats = apply_filters(idx, a.split, a)
    if a.limit:
        keep = keep.sample(min(a.limit, len(keep)), random_state=0) if a.split != "train" else keep.head(a.limit)
    print(json.dumps(fstats), flush=True)
    rows = keep.set_index("key")
    from multiprocessing import get_context
    pool = get_context("fork").Pool(a.workers, initializer=_winit, initargs=(a.data, res_list, str(rc)))
    enc = models.from_pretrained(SHAPE_ENC).eval().cuda()
    ss_enc = models.from_pretrained(SS_ENC).eval().cuda()
    dec = models.from_pretrained(SHAPE_DEC).eval().cuda() if a.roundtrip else None
    rt_keys = set(keep.key.sample(min(a.roundtrip, len(keep)), random_state=1)) if a.roundtrip else set()
    ov_keys = set(list(rt_keys)[:a.overlays])
    rt_rows, recs, t0, n, errs = [], {"rc": [], "ss": [], **{r: [] for r in res_list}}, time.time(), 0, []
    (out / "p2").mkdir(exist_ok=True)
    saver = ThreadPoolExecutor(8)

    def save_npz(path, **kw):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **kw)

    def save_json(path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj))

    def jobs():
        for sh in sorted(rows.shard.unique()):
            body = io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{a.prefix}/shards/{sh}")["Body"].read())
            mem = {}
            with tarfile.open(fileobj=body) as tf:
                for m in tf:
                    k, ext = m.name.rsplit(".", 1)
                    if k in rows.index:
                        mem.setdefault(k, {})[ext] = tf.extractfile(m).read()
            for key, f in mem.items():
                yield key, f, rows.loc[key].to_dict()

    for key, P, err in pool.imap_unordered(prep, jobs(), chunksize=4):
        if P is None:
            errs.append((key, err))
            print(f"[skip] {key}: {err}", flush=True)
            continue
        row = rows.loc[key]
        d, L = float(row.d), float(row.L_n)
        g = P["g"]
        scale = {"total_scale": 1.0 / L, "L_n": L, "d": d, "D_n": G.D_N, "camera_angle_x": float(row.camera_angle_x)}
        lat = {}
        for r in res_list:
            vi, dv, it = (torch.from_numpy(x) for x in P["grids"][r])
            vert, inter = to_sparse(sp, torch, vi, dv, it)
            zz = enc(vert, inter)
            if not torch.isfinite(zz.feats).all():
                errs.append((key, f"non-finite latent at {r}"))
                lat = None
                break
            lat[r] = (zz, len(vi))
        if lat is None:
            continue
        for r, (zz, nv) in lat.items():
            pack = {"feats": zz.feats.float().cpu().numpy(), "coords": zz.coords[:, 1:].cpu().numpy().astype(np.uint8)}
            dd = out / "shape_latents" / SHAPE_NAME.format(res=r) / key
            saver.submit(save_npz, dd / "view00.npz", **pack)
            save_json(dd / "view00_scale.json", scale)
            recs[r].append({"sha256": key, "shape_latent_view00_encoded": True,
                            "shape_latent_tokens": int(len(pack["coords"])), "dual_grid_voxels": int(nv)})
        recs["rc"].append({"sha256": key, "cond_rendered": True})
        # stage 1 target: the 1024 latent cells (64^3) -> ss VAE
        if 1024 in lat:
            c64 = lat[1024][0].coords[:, 1:].long()
            ss = torch.zeros(1, 1, 64, 64, 64, device="cuda")
            ss[0, 0, c64[:, 0], c64[:, 1], c64[:, 2]] = 1
            zs = ss_enc(ss, sample_posterior=False)
            dd = out / "ss_latents" / SS_NAME / key
            saver.submit(save_npz, dd / "view00.npz", z=zs[0].float().cpu().numpy())
            save_json(dd / "view00_scale.json", scale)
            recs["ss"].append({"sha256": key, "ss_latent_view00_encoded": True, "ss_cells_64": int(len(c64))})
        # P1 round trip / P2 alignment
        if key in rt_keys:
            (hv, hf), (ov, of) = P["hand"], P["obj"]
            gt_h, _ = surface(hv, hf, 20000, 0)
            gt_o, _ = surface(ov, of, 20000, 1)
            r_row = {"key": key, "d": d, "L_n": L, "kind": row.kind, "grasp_obj": row.grasp_obj,
                     "bbox_gt": [g.min(0).round(4).tolist(), g.max(0).round(4).tolist()]}
            from scipy.spatial import cKDTree
            for r in res_list:
                dec.set_resolution(r)
                mo = dec(lat[r][0])
                mo = mo[0] if isinstance(mo, (list, tuple)) else mo
                mo = mo[0] if isinstance(mo, (list, tuple)) else mo
                dv_ = mo.vertices.float().cpu().numpy()
                df_ = mo.faces.long().cpu().numpy()
                r_row[f"bbox_dec_{r}"] = [dv_.min(0).round(4).tolist(), dv_.max(0).round(4).tolist()]
                pc, _ = surface(G.grid_to_cam(dv_, d, L), df_, 40000, 2)
                dh, do = cKDTree(gt_h).query(pc)[0], cKDTree(gt_o).query(pc)[0]
                ph, po = pc[dh <= do], pc[dh > do]
                for nm, pr, gt in (("soup", pc, np.r_[gt_h, gt_o]), ("hand", ph, gt_h), ("obj", po, gt_o)):
                    if len(pr):
                        for kk, vv in fscores(pr * 1000, gt * 1000).items():
                            r_row[f"{nm}_{kk}_{r}"] = vv
                r_row[f"tokens_{r}"] = int(lat[r][0].feats.shape[0])
                if key in ov_keys and r == max(res_list):
                    # P2: decoded vertices through ProjGrid's fixed front camera vs GT soup through the real K
                    Sx = int(row.S)
                    w = torch.tensor(dv_ @ G.R_PROJ.T * L, dtype=torch.float32)[None]
                    T = torch.tensor(G.front_transform_matrix(G.D_N))[None]
                    uv = project_points_to_image_batch(w, T, torch.tensor([float(row.camera_angle_x)]), Sx)[0][0].numpy()
                    V, K = P["V"], P["K"]
                    uv_gt = (V @ K.T)[:, :2] / V[:, 2:3] + [row.pad_x, row.pad_y]
                    sd, sg = splat(uv, Sx, 2), splat(uv_gt, Sx, 2)
                    r_row["p2_iou"] = float((sd & sg).sum() / max((sd | sg).sum(), 1))
                    vis = np.asarray(Image.open(rc / key / "view00.png").convert("RGB")).copy()
                    vis[sg] = (0.5 * vis[sg] + [0, 120, 0]).astype(np.uint8)
                    vis[sd] = (0.5 * vis[sd] + [120, 0, 120]).astype(np.uint8)
                    Image.fromarray(vis).save(out / "p2" / f"{key}.jpg", quality=90)
            rt_rows.append(r_row)
            print(json.dumps({k: v for k, v in r_row.items() if "f@2" in k or "chamfer" in k or k == "p2_iou"}),
                  flush=True)
        n += 1
        if n % 500 == 0:
            print(f"{n} samples, {time.time() - t0:.0f}s", flush=True)
    pool.close()
    saver.shutdown(wait=True)
    # metadata.csv per sub-root
    pd.DataFrame(recs["rc"]).to_csv(rc / "metadata.csv", index=False)
    for r in res_list:
        if recs[r]:
            pd.DataFrame(recs[r]).to_csv(out / "shape_latents" / SHAPE_NAME.format(res=r) / "metadata.csv", index=False)
    if recs["ss"]:
        pd.DataFrame(recs["ss"]).to_csv(out / "ss_latents" / SS_NAME / "metadata.csv", index=False)
    summ = {"split": a.split, "filters": fstats, "encoded": n, "skipped": len(errs), "skip_examples": errs[:20], "seconds": time.time() - t0}
    for r in res_list:
        if recs[r]:
            t = pd.DataFrame(recs[r]).shape_latent_tokens
            summ[f"tokens_{r}"] = {f"p{p}": float(np.percentile(t, p)) for p in (1, 50, 99, 100)}
    if recs["ss"]:
        t = pd.DataFrame(recs["ss"]).ss_cells_64
        summ["ss_cells_64"] = {f"p{p}": float(np.percentile(t, p)) for p in (1, 50, 99, 100)}
    if rt_rows:
        rt = pd.DataFrame(rt_rows)
        rt.to_json(out / "p1_roundtrip.json", orient="records", indent=1)
        num = rt.select_dtypes("number")
        summ["p1_median"] = num.median().round(4).to_dict()
    (out / "encode_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
