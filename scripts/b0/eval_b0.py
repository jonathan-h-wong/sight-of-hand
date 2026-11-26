"""B0 evaluation on s0 val (spec sections 5-6; P3 is this script with no LoRA). GPU; run from a Pixal3D checkout.

Rows: `zeroshot` (released weights) and `b0` (LoRA per stage, --lora-ss / --lora-512 / --lora-1024, raw or EMA
weights, whichever files are given). Every row uses this cube and camera: padded square, camera_angle_x from the
real intrinsics, distance D_n = 1, mesh_scale = 1 / L_n. The image is NOT cropped (no stock preprocess_image).

Modes (--modes):
  ss       stage 1: sample the 16^3 latent, decode to 64^3 occupancy, IoU / precision / recall vs the GT cells
  gt512    stage 2 at 512 on the GT 32^3 coordinates -> mesh
  gt1024   stage 2 at 1024 on the GT 64^3 coordinates -> mesh
  cascade  stage 1 -> 32^3 (max-pool, as the released pipeline) -> 512 -> decoder upsample -> 1024 -> mesh
Mesh metrics (decoded mesh surface vs GT soup surface, 40K/20K+20K samples):
  scale-free  normalized units x 1000 (per-mille of the wrist depth): Chamfer, F@{1,2,5} (thresholds in mm at the
              GT d, i.e. tau/d per-mille), per part (predicted points split by the nearer GT part: an oracle split,
              since B0 has no label head), hand-object centroid offset error, contact agreement
  metric      the same after multiplying by the GT d ("gt_d", the model's error) and by the MoGe-2 d ("moge_d")
MoGe-2 d: MoGe-2 (Ruicheng/moge-2-vitl, real fov_x) metric depth on the unpadded frame; d = median over the
predicted hand's front-most surface points that project inside the modal hand mask of moge_z / z_n (a 1-D fit at
the generated hand; the spec's "depth at the generated wrist" needs a MANO fit, which B0 does not have).
Mattes (--mattes): oracle (alpha = GT modal hand + object) and raw (alpha = 1, the full frame).
Eval set: --n-strat frames stratified by hand occlusion bin (0-10, 10-30, 30-60, >60 % hidden), plus --n-paired
(sequence, frame) groups with all 8 cameras for the paired occlusion gap, plus --n-stress stress frames (hand mostly
hidden or outside the frame; these failed the training filters). Per-sample rows go to rows.parquet; summary.json
holds the medians per row, mode, matte and stratum.

    python eval_b0.py --val /workspace/b0/val --out /workspace/out/eval_zeroshot --row zeroshot
    python eval_b0.py ... --row b0 --lora-ss ss64/lora_final_ema0.999.pt --lora-512 shape512/lora_final_ema0.999.pt
"""
import argparse
import io
import json
import math
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sight_of_hand.b0 import geometry as G  # noqa: E402
from sight_of_hand.b0 import lora  # noqa: E402
from encode_b0 import Meshes, apply_filters, surface  # noqa: E402

BUCKET = "xray-hands"
TAUS_MM = (1.0, 2.0, 5.0)
OCC_BINS = [0, 0.1, 0.3, 0.6, 1.01]
OCC_LABELS = ["0-10", "10-30", "30-60", ">60"]


def pick_eval_set(idx, val_keys, a):
    """Stratified + paired + stress keys from the val split."""
    amodal = pd.read_parquet(a.amodal_index) if a.amodal_index else None
    v = idx[idx.split == "val"].copy()
    if amodal is not None:
        am = amodal.assign(serial=amodal.cam.astype(str))[["seq", "serial", "frame", "hand_visible_frac",
                                                           "obj_occluded_by_hand_frac", "hand_occluded_by_obj_frac"]]
        v = v.merge(am, on=["seq", "serial", "frame"], how="left")
    v["hand_hidden"] = 1 - v.hand_visible_frac.fillna(1 - (v.hand_modal_px > 0))
    v["occ_bin"] = pd.cut(v.hand_hidden, OCC_BINS, labels=OCC_LABELS, right=False)
    ok = v[v.key.isin(val_keys)]
    rng = np.random.default_rng(0)
    strat = []
    for b in OCC_LABELS:
        g = ok[(ok.occ_bin == b) & (ok.kind == "contact")]
        strat += list(g.key.iloc[rng.permutation(len(g))[:a.n_strat // 4]])
    grp = ok[ok.kind == "contact"].groupby(["seq", "frame"]).key.apply(list)
    grp = grp[grp.apply(len) == 8]
    paired = sum([grp.iloc[i] for i in rng.permutation(len(grp))[:a.n_paired]], [])
    stress = v[~v.key.isin(val_keys) & ((v.hand_modal_px == 0) | (v.hand_inside_frac < 0.5)) & v.contained]
    stress = list(stress.key.iloc[rng.permutation(len(stress))[:a.n_stress]])
    sets = {k: "strat" for k in strat}
    for k in paired:
        sets[k] = "strat+paired" if k in sets else "paired"
    for k in stress:
        sets[k] = "stress"
    return v.set_index("key"), sets


def load_sample_raw(s3, prefix, shard, keys):
    body = io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{prefix}/shards/{shard}")["Body"].read())
    out = {}
    with tarfile.open(fileobj=body) as tf:
        for m in tf:
            k, ext = m.name.rsplit(".", 1)
            if k in keys:
                out.setdefault(k, {})[ext] = tf.extractfile(m).read()
    return out


def padded_image(f, row, matte):
    img = np.asarray(Image.open(io.BytesIO(f["jpg"])).convert("RGB")).astype(np.float32)
    lm = np.asarray(Image.open(io.BytesIO(f["png"])))
    a = (lm > 0).astype(np.float32) if matte == "oracle" else np.ones(lm.shape, np.float32)
    im = G.pad_image((img * a[..., None]).astype(np.uint8), int(row.S), int(row.pad_x), int(row.pad_y))
    return Image.fromarray(im), lm


class _NoRembg:
    """Stand-in for BiRefNet (briaai/RMBG-2.0 is gated for us): B0 never calls preprocess_image; mattes are the GT
    modal mask (oracle) or none (raw)."""
    def __init__(self, *a, **k):
        pass

    def to(self, *a, **k):
        return self

    cuda = cpu = to

    def __call__(self, *a, **k):
        raise RuntimeError("background removal is not used in B0 evaluation")


def build_pipeline(a):
    import pixal3d.pipelines.rembg as rembg
    rembg.BiRefNet = _NoRembg
    import inference as inf  # Pixal3D/inference.py
    pipe = inf.Pixal3DImageTo3DPipeline.from_pretrained("TencentARC/Pixal3D")
    pipe.image_cond_model_ss = inf.build_image_cond_model(inf.IMAGE_COND_CONFIGS["ss"])
    pipe.image_cond_model_shape_512 = inf.build_image_cond_model(inf.IMAGE_COND_CONFIGS["shape_512"])
    pipe.image_cond_model_shape_1024 = inf.build_image_cond_model(inf.IMAGE_COND_CONFIGS["shape_1024"])
    pipe.image_cond_model_tex_1024 = None
    pipe.low_vram = False
    pipe.cuda()
    for m in (pipe.image_cond_model_ss, pipe.image_cond_model_shape_512, pipe.image_cond_model_shape_1024):
        m.cuda()
        if getattr(m, "use_naf_upsample", False):
            m._load_naf()
    import torch
    for key, path in (("sparse_structure_flow_model", a.lora_ss), ("shape_slat_flow_model_512", a.lora_512),
                      ("shape_slat_flow_model_1024", a.lora_1024)):
        if path:
            sd = torch.load(path, map_location="cuda")
            m = pipe.models[key]
            lora.inject(m, a.rank, a.alpha)
            lora.load_lora(m, sd)  # kept unmerged: the pipeline models are bf16 and W + BA would round away
            m.to("cuda")
            print(f"LoRA loaded (unmerged) into {key} from {path}", flush=True)
    return pipe


def mesh_points(mesh, n, seed):
    V = mesh.vertices.float().cpu().numpy()
    F = mesh.faces.long().cpu().numpy()
    if len(F) == 0:
        return np.zeros((0, 3))
    return surface(V, F, n, seed)[0]


def fs(pred, gt, taus, names=None):
    """Chamfer and F-scores; keys f@<name>, names default to the thresholds themselves."""
    names = names or [f"{t:g}" for t in taus]
    if len(pred) == 0:
        return {"chamfer": np.nan, **{f"f@{n}": 0.0 for n in names}}
    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    o = {"chamfer": float((d_pg.mean() + d_gp.mean()) / 2)}
    for t, n in zip(taus, names):
        p, r = (d_pg <= t).mean(), (d_gp <= t).mean()
        o[f"f@{n}"] = float(2 * p * r / (p + r)) if p + r > 0 else 0.0
    return o


def mesh_metrics(pg, gt_h, gt_o, d_gt, d_moge):
    """pg: predicted surface points in the grid frame. GT parts in camera metres. Returns a flat dict."""
    out = {}
    gt = np.r_[gt_h, gt_o]
    for tag, d in (("gt_d", d_gt), ("moge_d", d_moge)):
        if d is None or not np.isfinite(d):
            continue
        pc = pg["to_cam"](d) * 1000.0  # mm
        gh, go = gt_h * 1000.0, gt_o * 1000.0
        dh, do = cKDTree(gh).query(pc)[0], cKDTree(go).query(pc)[0]
        ph, po = pc[dh <= do], pc[dh > do]
        for nm, pr, g in (("soup", pc, gt * 1000.0), ("hand", ph, gh), ("obj", po, go)):
            for k, v in fs(pr, g, TAUS_MM).items():
                out[f"{tag}/{nm}/{k}"] = v
        if len(ph) and len(po):
            out[f"{tag}/rel_offset_mm"] = float(np.linalg.norm((ph.mean(0) - po.mean(0)) - (gh.mean(0) - go.mean(0))))
            out[f"{tag}/pred_hand_obj_min_mm"] = float(cKDTree(po).query(ph)[0].min())
        out[f"{tag}/centroid_err_mm"] = float(np.linalg.norm(pc.mean(0) - np.r_[gh, go].mean(0)))
    # scale-free: normalized units x 1000, thresholds tau / d_gt
    pn = pg["to_cam"](1.0) * 1000.0
    gn = gt * 1000.0 / d_gt
    # keyed by the mm threshold (f@2 = 2 mm / d_gt) so the column is the same for every frame
    for k, v in fs(pn, gn, tuple(t / d_gt for t in TAUS_MM), [f"{t:g}" for t in TAUS_MM]).items():
        out[f"norm/soup/{k}"] = v
    return out


def moge_scale(moge, frame_rgb, fov_x, pts_norm, K, pad, hand_mask):
    """median of MoGe depth / normalized depth over front-most predicted points inside the modal hand mask."""
    import torch
    with torch.no_grad():
        o = moge.infer(torch.from_numpy(frame_rgb).float().permute(2, 0, 1).cuda() / 255.0, fov_x=math.degrees(fov_x))
    depth = o["depth"].float().cpu().numpy()
    z = pts_norm[:, 2]
    uv = (pts_norm @ K.T)[:, :2] / z[:, None]
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    H, W = depth.shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    u, v, z = u[ok], v[ok], z[ok]
    zbuf = np.full((H, W), np.inf)
    np.minimum.at(zbuf, (v, u), z)
    front = np.abs(z - zbuf[v, u]) < 1e-3 * 5
    sel = front & hand_mask[v, u] & np.isfinite(depth[v, u])
    if sel.sum() < 20:
        return np.nan, int(sel.sum())
    return float(np.median(depth[v[sel], u[sel]] / z[sel])), int(sel.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True, help="encode_b0.py output for the val split")
    ap.add_argument("--prefix", default="train/b0_dexycb_s0_v1")
    ap.add_argument("--data", default="/workspace/data")
    ap.add_argument("--amodal-index", help="local parquet of s3://xray-hands/masks/amodal_v1/dexycb/index.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--row", default="zeroshot")
    ap.add_argument("--lora-ss")
    ap.add_argument("--lora-512")
    ap.add_argument("--lora-1024")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--modes", default="ss,gt512,cascade")
    ap.add_argument("--mattes", default="oracle,raw")
    ap.add_argument("--n-strat", type=int, default=200)
    ap.add_argument("--n-paired", type=int, default=25)
    ap.add_argument("--n-stress", type=int, default=24)
    ap.add_argument("--min-inside", type=float, default=0.5)
    ap.add_argument("--d-min", type=float)
    ap.add_argument("--d-max", type=float)
    ap.add_argument("--moge", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-meshes", type=int, default=12, help="save meshes for the first N frames")
    ap.add_argument("--save-every", type=int, default=0, help="also save visuals for every k-th frame (0 = off)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: evaluate every n-th frame starting at i (split across pods)")
    a = ap.parse_args()

    import boto3
    import torch
    sys.path.insert(0, os.environ.get("PX", "."))
    from pixal3d.modules.sparse import SparseTensor
    torch.set_grad_enabled(False)
    s3 = boto3.client("s3")
    out = Path(a.out)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{a.prefix}/index.parquet")["Body"].read()))
    kept, _ = apply_filters(idx, "val", a)
    vidx, sets = pick_eval_set(idx, set(kept.key), a)
    keys = list(sets)
    if a.limit:
        keys = keys[:a.limit]
    si, sn = map(int, a.shard.split("/"))
    keys = keys[si::sn]
    print(f"eval set: {pd.Series(sets).value_counts().to_dict()}", flush=True)
    meshes = Meshes(a.data)
    pipe = build_pipeline(a)
    moge = None
    if a.moge:
        from moge.model.v2 import MoGeModel
        moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").cuda().eval()
    modes = a.modes.split(",")
    rows = []
    t0 = time.time()
    shards = vidx.loc[keys].groupby("shard").apply(lambda g: list(g.index)).to_dict()
    done = 0
    for sh, ks in shards.items():
        raw = load_sample_raw(s3, a.prefix, sh, set(ks))
        for key in ks:
            if key not in raw:
                continue
            row = vidx.loc[key]
            f = raw[key]
            z = np.load(io.BytesIO(f["npz"]))
            V, F, part, (hv, hf), (ov, of) = meshes.soup(z)
            gt_h, gt_o = surface(hv, hf, 20000, 0)[0], surface(ov, of, 20000, 1)[0]
            d, L, cax = float(row.d), float(row.L_n), float(row.camera_angle_x)
            K = z["K"]
            cam = dict(camera_angle_x=cax, distance=G.D_N, mesh_scale=1.0 / L)
            base = {"key": key, "set": sets[key], "seq": row.seq, "serial": row.serial, "frame": int(row.frame),
                    "kind": row.kind, "grasp_obj": row.grasp_obj, "d": d, "L_n": L,
                    "occ_bin": str(row.occ_bin), "hand_hidden": float(row.hand_hidden),
                    "pos_offset_n": float(math.hypot(row.lat_x_n, row.lat_y_n)), "hand_px": float(row.hand_px),
                    "row": a.row}
            gt_dir = Path(a.val)
            frame_rgb = np.asarray(Image.open(io.BytesIO(f["jpg"])).convert("RGB"))
            for matte in a.mattes.split(","):
                image, lm = padded_image(f, row, matte)
                hand_mask = lm == 1
                torch.manual_seed(a.seed)
                cond_ss = pipe.get_proj_cond_ss([image], **cam)
                for mode in modes:
                    r = {**base, "matte": matte, "mode": mode}
                    tt = time.time()
                    try:
                        if mode == "ss":
                            gtc = np.load(gt_dir / "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view" / key / "view00.npz")["coords"]
                            torch.manual_seed(a.seed)
                            fm = pipe.models["sparse_structure_flow_model"]
                            noise = torch.randn(1, fm.in_channels, *[fm.resolution] * 3, device="cuda")
                            zs = pipe.sparse_structure_sampler.sample(fm, noise, **cond_ss,
                                                                      **pipe.sparse_structure_sampler_params).samples
                            occ = (pipe.models["sparse_structure_decoder"](zs) > 0)[0, 0].cpu().numpy()
                            gtg = np.zeros((64, 64, 64), bool)
                            gtg[gtc[:, 0], gtc[:, 1], gtc[:, 2]] = True
                            inter = (occ & gtg).sum()
                            r.update({"ss/iou": float(inter / max((occ | gtg).sum(), 1)),
                                      "ss/precision": float(inter / max(occ.sum(), 1)),
                                      "ss/recall": float(inter / max(gtg.sum(), 1)),
                                      "ss/n_pred": int(occ.sum()), "ss/n_gt": int(gtg.sum())})
                            rows.append(r)
                            continue
                        if mode in ("gt512", "gt1024"):
                            res = int(mode[2:])
                            gtc = np.load(gt_dir / f"shape_latents/shape_enc_next_dc_f16c32_fp16_{res}_view" / key / "view00.npz")["coords"]
                            coords = torch.cat([torch.zeros(len(gtc), 1, dtype=torch.int32),
                                                torch.from_numpy(gtc.astype(np.int32))], 1).cuda()
                            cm = pipe.image_cond_model_shape_512 if res == 512 else pipe.image_cond_model_shape_1024
                            cond = pipe.get_proj_cond_shape(cm, [image], coords, **cam)
                            torch.manual_seed(a.seed)
                            slat = pipe.sample_shape_slat(cond, pipe.models[f"shape_slat_flow_model_{res}"], coords)
                        elif mode == "cascade512":  # stage 1 -> stage 2 at 512 -> decode at 512 (the B0 path)
                            torch.manual_seed(a.seed)
                            coords = pipe.sample_sparse_structure(cond_ss, 32, 1)
                            cl = pipe.get_proj_cond_shape(pipe.image_cond_model_shape_512, [image], coords, **cam)
                            slat, res = pipe.sample_shape_slat(cl, pipe.models["shape_slat_flow_model_512"], coords), 512
                            r["cascade/n_coords32"] = int(len(coords))
                        else:  # cascade: stage 1 -> 512 -> stock-style 1024 refinement (untuned unless --lora-1024)
                            torch.manual_seed(a.seed)
                            coords = pipe.sample_sparse_structure(cond_ss, 32, 1)
                            cl = pipe.get_proj_cond_shape(pipe.image_cond_model_shape_512, [image], coords, **cam)
                            lr = pipe.sample_shape_slat(cl, pipe.models["shape_slat_flow_model_512"], coords)
                            hr = pipe.models["shape_slat_decoder"].upsample(lr, upsample_times=4)
                            q = torch.cat([hr[:, :1], ((hr[:, 1:] + 0.5) / 512 * 63).round().int()], 1).unique(dim=0)
                            ch = pipe.get_proj_cond_shape(pipe.image_cond_model_shape_1024, [image], q, **cam,
                                                          grid_resolution_override=64)
                            noise = SparseTensor(feats=torch.randn(len(q), 32, device="cuda"), coords=q)
                            hs = pipe.shape_slat_sampler.sample(pipe.models["shape_slat_flow_model_1024"], noise, **ch,
                                                                **pipe.shape_slat_sampler_params).samples
                            std = torch.tensor(pipe.shape_slat_normalization["std"])[None].cuda()
                            mean = torch.tensor(pipe.shape_slat_normalization["mean"])[None].cuda()
                            slat, res = hs * std + mean, 1024
                            r["cascade/n_coords32"] = int(len(coords))
                        meshes_out = pipe.decode_shape_slat(slat, res)
                        mesh = meshes_out[0][0] if isinstance(meshes_out, tuple) else meshes_out[0]
                        pg_grid = mesh_points(mesh, 40000, 2)
                        r["n_tokens"] = int(slat.feats.shape[0])
                        to_cam = lambda dd, g=pg_grid: G.grid_to_cam(g, dd, L)  # noqa: E731
                        d_moge, n_moge = (np.nan, 0)
                        if moge is not None and len(pg_grid):
                            pn = to_cam(1.0)
                            # predicted hand points by the oracle split, in normalized camera coordinates
                            dh = cKDTree(gt_h / d).query(pn)[0]
                            do = cKDTree(gt_o / d).query(pn)[0]
                            d_moge, n_moge = moge_scale(moge, frame_rgb, 2 * math.atan(640 / (2 * K[0, 0])),
                                                        pn[dh <= do], K, (row.pad_x, row.pad_y), hand_mask)
                        r.update({"d_moge": d_moge, "n_moge_px": n_moge})
                        if len(pg_grid):
                            r.update(mesh_metrics({"to_cam": to_cam}, gt_h, gt_o, d, d_moge))
                        save = done < a.save_meshes or (a.save_every and done % a.save_every == 0)
                        if save and matte == "oracle":
                            import trimesh
                            trimesh.Trimesh(G.grid_to_cam(mesh.vertices.float().cpu().numpy(), d, L),
                                            mesh.faces.cpu().numpy(), process=False).export(
                                out / "meshes" / f"{key}__{mode}__{a.row}.ply")
                            vis = out / "vis" / key
                            vis.mkdir(parents=True, exist_ok=True)
                            if not (vis / "gt_soup.npz").exists():  # GT parts (camera metres), input, intrinsics
                                np.savez_compressed(vis / "gt_soup.npz", hv=hv, hf=hf, ov=ov, of=of, K=K, d=d, L=L,
                                                    pad=np.array([row.pad_x, row.pad_y, row.S]))
                                image.save(vis / "input_oracle.png")
                                Image.fromarray(frame_rgb).save(vis / "frame.jpg")
                            if mode.startswith("cascade"):
                                gtc = gt_dir / "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view" / key / "view00.npz"
                                np.savez_compressed(vis / f"coords32__{mode}__{a.row}.npz",
                                                    pred=coords[:, 1:].cpu().numpy(),
                                                    gt=np.load(gtc)["coords"] if gtc.exists() else np.zeros((0, 3)))
                    except Exception as e:  # a sample that fails is recorded, not fatal
                        r["error"] = repr(e)[:300]
                        print(f"[error] {key} {mode} {matte}: {e}", flush=True)
                    r["seconds"] = time.time() - tt
                    rows.append(r)
            done += 1
            if done % 10 == 0:
                print(f"{done}/{len(keys)} samples, {time.time() - t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_parquet(out / "rows.parquet")
    df = pd.DataFrame(rows)
    df.to_parquet(out / "rows.parquet")
    summ = summarize(df)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ["overall"], indent=1))


def fix_norm_cols(df):
    """Rows written before fs() took names keyed norm/soup F-scores by the per-frame threshold (f@<tau/d>);
    fold them into f@<tau> by rank (the thresholds are TAUS_MM / d_gt in order)."""
    bad = [c for c in df.columns if c.startswith("norm/soup/f@") and c[len("norm/soup/f@"):] not in
           {f"{t:g}" for t in TAUS_MM}]
    if not bad:
        return df
    df = df.copy()
    thr = {c: float(c[len("norm/soup/f@"):]) for c in bad}
    vals = np.full((len(df), len(TAUS_MM)), np.nan)
    for i, (_, r) in enumerate(df[bad].iterrows()):
        got = sorted((thr[c], r[c]) for c in bad if pd.notna(r[c]))
        if len(got) == len(TAUS_MM):
            vals[i] = [v for _, v in got]
    df = df.drop(columns=bad)
    for j, t in enumerate(TAUS_MM):
        df[f"norm/soup/f@{t:g}"] = vals[:, j]
    return df


def summarize(df):
    """Medians per row/mode/matte/stratum. A mesh mode that produced no mesh (empty stage 1, decode error) is a
    failure, not a missing value: its F-scores count as 0 and `failed` reports the rate (Chamfer stays NaN)."""
    df = fix_norm_cols(df)
    mesh = df["mode"] != "ss"
    probe = "gt_d/soup/f@2" if "gt_d/soup/f@2" in df else None
    failed = mesh & (df["error"].notna() if "error" in df else False)
    if probe:
        failed = failed | (mesh & df[probe].isna())
    df["failed"] = failed.astype(float)
    for c in [c for c in df.columns if "/f@" in c]:
        df.loc[failed, c] = 0.0
    num = [c for c in df.columns if ("/" in c or c == "failed") and df[c].dtype != object]
    main = ["failed", "ss/iou", "gt_d/soup/f@2", "gt_d/soup/chamfer", "gt_d/hand/f@2", "gt_d/obj/f@2", "gt_d/rel_offset_mm",
            "moge_d/soup/f@2", "norm/soup/f@2"]
    main = [c for c in main if c in df.columns]
    nonstress = df[df.set != "stress"]
    out = {"overall": nonstress.groupby(["row", "mode", "matte"])[num].median().round(4).reset_index()
           .to_dict("records"),
           "failure_rate": nonstress.groupby(["row", "mode", "matte"]).failed.mean().round(4).reset_index()
           .to_dict("records"),
           "n": nonstress.groupby(["row", "mode", "matte"]).size().rename("n").reset_index().to_dict("records")}
    for s in ("occ_bin", "kind"):
        out[f"by_{s}"] = nonstress.groupby(["row", "mode", "matte", s])[main].median().round(4).reset_index() \
            .to_dict("records")
    for col, bins in (("d", 3), ("pos_offset_n", 3), ("hand_px", 3)):
        b = pd.qcut(nonstress[col], bins, duplicates="drop").astype(str)
        out[f"by_{col}"] = nonstress.assign(_b=b).groupby(["row", "mode", "matte", "_b"])[main].median().round(4) \
            .reset_index().to_dict("records")
    # paired occlusion gap: least vs most occluded camera of the same grasp
    p = df[df.set.str.contains("paired") & (df.matte == "oracle")]
    gaps = []
    for (rw, mode, seq, fr), g in p.groupby(["row", "mode", "seq", "frame"]):
        if "gt_d/soup/f@2" in g and len(g) >= 2:
            g = g.sort_values("hand_hidden")
            gaps.append({"row": rw, "mode": mode, "gap_f@2": float(g["gt_d/soup/f@2"].iloc[0] - g["gt_d/soup/f@2"].iloc[-1]),
                         "gap_hand_f@2": float(g.get("gt_d/hand/f@2", pd.Series([np.nan] * len(g))).iloc[0]
                                               - g.get("gt_d/hand/f@2", pd.Series([np.nan] * len(g))).iloc[-1])})
    if gaps:
        out["paired_gap"] = pd.DataFrame(gaps).groupby(["row", "mode"]).median().round(4).reset_index().to_dict("records")
    st = df[df.set == "stress"]
    if len(st) and "gt_d/hand/f@5" in st:
        desc = st.groupby(["row", "mode", "matte"])[["gt_d/hand/f@5", "gt_d/soup/f@5"]].describe().round(3)
        desc.columns = ["/".join(c) for c in desc.columns]
        out["stress"] = desc.reset_index().to_dict("records")
    return out


if __name__ == "__main__":
    main()
