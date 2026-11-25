"""Build the B0 DexYCB s0 sample set (docs/baseline_b0_training_spec.md, section 2, steps 1-5; CPU only).

Frame selection (per hand-frame, then all 8 cameras):
  contact   grounded contact frames of s3://xray-hands/eval/dexycb/grounded_v1/original_pen2_gap2 (penetration
            <= 2 mm, gap <= 2 mm), every --stride-th frame (train), every frame (val, test)
  approach  non-contact frames of the penetration audit with the hand within --approach-mm of the object, sampled
            per split so they are about --approach-frac of the selected frames (seeded)
Splits: dex-ycb-toolkit s0 (scripts/build_amodal_dataset.py, split_of).

Per (sequence, frame, camera) the shard stores the raw inputs, not the derived files, so the cube rule can change
without a rebuild:
  <key>.jpg   the DexYCB colour frame, bytes as released (640x480)
  <key>.png   8-bit label map: 0 background, 1 hand (seg 255), 2 grasped object (seg = its YCB id); other objects 0
  <key>.npz   hand_v (778,3) and joints (21,3) in the camera frame (m), obj_pose (3,4), ycb (int), K (3,3), side
and the index row holds the geometry (spec section 1) and every filter's inputs and verdicts:
  d (wrist z), L_n, S, camera_angle_x, grid bounds of the soup (gmin, gmax), contained, hand/object amodal inside-frame
  fraction, modal pixel counts, penetration, contact, kind (contact|approach), split.

Output: s3://xray-hands/train/b0_dexycb_s0_v1/{shards/<split>_<subject>_<k>.tar, index.parquet, p0_summary.json}

    python scripts/b0/build_b0_dataset.py --workers 10
"""
import argparse
import io
import json
import os
import sys
import tarfile
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_amodal_dataset import SUBJECTS, ResumableS3, split_of  # noqa: E402
from sight_of_hand.b0 import geometry as G  # noqa: E402

BUCKET = "xray-hands"
GROUNDED = "eval/dexycb/grounded_v1/original_pen2_gap2/index.parquet"
AUDIT = "eval/dexycb/penetration_v1/frames.parquet"
PER_SHARD = 2000
H, W = 480, 640
N_SURF = 4000


def s3_parquet(s3, key):
    return pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()))


def select_frames(s3, stride, approach_frac, approach_mm, seed=0):
    """-> DataFrame seq, frame, kind, split, + audit columns."""
    aud = s3_parquet(s3, AUDIT)
    grd = s3_parquet(s3, GROUNDED)
    seqs = {}
    for s in SUBJECTS:
        seqs[s] = sorted(aud.seq[aud.seq.str.startswith(s + "/")].unique())
    split = {q: split_of(SUBJECTS.index(s), i) for s, qs in seqs.items() for i, q in enumerate(qs)}
    grd = grd.assign(split=grd.seq.map(split), kind="contact")
    keep_c = (grd.split != "train") | (grd.frame % stride == 0)
    con = grd[keep_c]
    app = aud[(~aud.contact) & (aud.min_sd_mm <= approach_mm) & aud.grasp_obj_tracked].assign(kind="approach")
    app = app.assign(split=app.seq.map(split))
    rng = np.random.default_rng(seed)
    parts = [con]
    for sp, g in app.groupby("split"):
        n_c = int((con.split == sp).sum())
        n_a = int(round(n_c * approach_frac / (1 - approach_frac)))
        parts.append(g.iloc[np.sort(rng.choice(len(g), min(n_a, len(g)), replace=False))])
    cols = ["seq", "frame", "kind", "split", "min_sd_mm", "pen_max_mm", "contact", "grasp_obj"]
    return pd.concat(parts, ignore_index=True)[cols]


def surface_samples(V, F, n, rng):
    import trimesh
    m = trimesh.Trimesh(V, F, process=False)
    pts, _ = trimesh.sample.sample_surface(m, n, seed=int(rng.integers(1 << 31)))
    return pts


def inside_frac(P, K):
    z = P[:, 2]
    uv = (P @ K.T)[:, :2] / np.maximum(z[:, None], 1e-6)
    ok = (z > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return float(ok.mean())


def process_subject(args):
    subject_i, sel, prefix, data_dir, local_tar = args
    import boto3
    import yaml
    from sight_of_hand.amodal.dexycb import YCB, _Loader, _mesh
    from hoiaudit.sources import common
    common.MANO_DIR = Path(data_dir) / "mano" / "mano_v1_2" / "models"
    s3 = boto3.client("s3")
    subj = SUBJECTS[subject_i]
    sel = sel[sel.seq.str.startswith(subj + "/")]
    want = {(r.seq, int(r.frame)): r for r in sel.itertuples()}
    seqset = set(sel.seq)
    intr = pd.read_parquet(Path(data_dir) / "amodal_index_cams.parquet").set_index("cam")
    droot = Path(data_dir) / "dexycb"
    rng = np.random.default_rng(subject_i)
    meta, pending, writers, rows, t0 = {}, {}, {}, [], time.time()
    work = Path(f"/tmp/b0_ds/{subj}")
    work.mkdir(parents=True, exist_ok=True)

    def writer(split):
        w = writers.get(split)
        if w is None or w["n"] >= PER_SHARD:
            if w is not None:
                w["tar"].close()
                upload(w)
            k = 0 if w is None else w["k"] + 1
            name = f"{split}_{subj}_{k:03d}.tar"
            w = {"name": name, "k": k, "n": 0, "path": work / name}
            w["tar"] = tarfile.open(w["path"], "w")
            writers[split] = w
        return w

    def upload(w):
        s3.upload_file(str(w["path"]), BUCKET, f"{prefix}/shards/{w['name']}")
        w["path"].unlink()

    def add(tf, name, data):
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))

    def emit(seq, serial, fr, jpg, lab):
        from PIL import Image
        r = want[(seq, fr)]
        mt = meta[seq]
        z = np.load(io.BytesIO(lab))
        pm = z["pose_m"][0]
        P = z["pose_y"][mt["g"]]
        if not np.any(pm != 0) or not np.any(P[:, :3]):
            return
        K = np.array([[intr.fx[int(serial)], 0, intr.cx[int(serial)]], [0, intr.fy[int(serial)], intr.cy[int(serial)]],
                      [0, 0, 1]])
        hv = mt["m"](mt["betas"], pm[None, :3], mt["m"].pca_to_pose(pm[None, 3:48]), pm[None, 48:51])[0]
        V0, F0 = _mesh(droot, mt["ycb"])
        ov = V0 @ P[:, :3].T + P[:, 3]
        joints = z["joint_3d"][0].astype(np.float64)
        d = float(joints[0, 2])
        geo = G.frame_geometry(K, H, W, d)
        soup = np.concatenate([hv, ov])
        g = G.cam_to_grid(soup, d, geo["L_n"])
        seg = z["seg"]
        hm, om = seg == 255, seg == mt["ycb_id"]
        lm = np.zeros(seg.shape, np.uint8)
        lm[hm] = 1
        lm[om] = 2
        hs = surface_samples(hv, mt["m"].faces, N_SURF, rng)
        os_ = surface_samples(ov, F0, N_SURF, rng)
        c = soup.mean(0)
        key = f"{seq.replace('/', '__')}__{serial}__{fr:06d}"
        row = {"key": key, "seq": seq, "serial": serial, "frame": fr, "split": r.split, "kind": r.kind,
               "grasp_obj": YCB[mt["ycb_id"]], "side": mt["side"], "min_sd_mm": r.min_sd_mm,
               "pen_max_mm": r.pen_max_mm, "contact": bool(r.contact),
               "d": d, "S": geo["S"], "pad_x": geo["pad_x"], "pad_y": geo["pad_y"], "fx": geo["fx"], "fy": geo["fy"],
               "camera_angle_x": geo["camera_angle_x"], "L_n": geo["L_n"],
               "gmin_x": g[:, 0].min(), "gmin_y": g[:, 1].min(), "gmin_z": g[:, 2].min(),
               "gmax_x": g[:, 0].max(), "gmax_y": g[:, 1].max(), "gmax_z": g[:, 2].max(),
               "g_absmax": float(np.abs(g).max()), "contained": G.contained(g, 2.0 / 1024),
               "lat_x_n": float(c[0] / c[2]), "lat_y_n": float(c[1] / c[2]), "soup_z_n": float(c[2] / d),
               "hand_inside_frac": inside_frac(hs, K), "obj_inside_frac": inside_frac(os_, K),
               "hand_modal_px": int(hm.sum()), "obj_modal_px": int(om.sum()),
               "hand_extent_m": float(np.linalg.norm(hv.max(0) - hv.min(0))),
               "hand_px": float(np.ptp((hv @ K.T)[:, :2] / hv[:, 2:3], axis=0).max())}
        w = writer(r.split)
        add(w["tar"], key + ".jpg", jpg)
        b = io.BytesIO()
        Image.fromarray(lm).save(b, "PNG")
        add(w["tar"], key + ".png", b.getvalue())
        b = io.BytesIO()
        np.savez_compressed(b, hand_v=hv.astype(np.float32), joints=joints.astype(np.float32),
                            obj_pose=P.astype(np.float32), ycb=mt["ycb_id"], K=K, side=mt["side"])
        add(w["tar"], key + ".npz", b.getvalue())
        w["n"] += 1
        rows.append({**row, "shard": w["name"]})

    def load_meta(seq, raw):
        mt = yaml.load(raw, Loader=_Loader)
        gi = int(mt["ycb_grasp_ind"])
        side = mt["mano_sides"][0]
        calib = mt["mano_calib"][0]
        betas = np.array(yaml.load(open(droot / "calibration" / f"mano_{calib}" / "mano.yml"), Loader=_Loader)["betas"])
        meta[seq] = {"g": gi, "ycb_id": int(mt["ycb_ids"][gi]), "ycb": int(mt["ycb_ids"][gi]), "side": side,
                     "m": common.mano(side, False), "betas": betas}

    body = open(local_tar, "rb") if local_tar else io.BufferedReader(ResumableS3(s3, f"raw/dexycb/{subj}.tar.gz"), 1 << 22)
    with tarfile.open(fileobj=body, mode="r|gz") as tf:
        for m in tf:
            p = m.name.split("/")
            if not m.isfile() or len(p) < 3:
                continue
            seq = f"{p[0]}/{p[1]}"
            if seq not in seqset:
                continue
            if p[-1] == "meta.yml":
                load_meta(seq, tf.extractfile(m).read())
                for k in [k for k, v in pending.items() if k[0] == seq and len(v) == 2]:
                    emit(*k, pending[k]["jpg"], pending[k]["npz"])
                    del pending[k]
                continue
            if len(p) != 4 or not (p[3].startswith("color_") or p[3].startswith("labels_")):
                continue
            fr = int(p[3].split("_")[-1].split(".")[0])
            if (seq, fr) not in want:
                continue
            k = (seq, p[2], fr)
            dd = pending.setdefault(k, {})
            dd["jpg" if p[3].startswith("color_") else "npz"] = tf.extractfile(m).read()
            if len(dd) == 2 and seq in meta:
                emit(seq, p[2], fr, dd["jpg"], dd["npz"])
                del pending[k]
    for w in writers.values():
        w["tar"].close()
        upload(w)
    df = pd.DataFrame(rows)
    part = work / f"index_{subj}.parquet"
    df.to_parquet(part)
    s3.upload_file(str(part), BUCKET, f"{prefix}/parts/index_{subj}.parquet")
    return subj, len(df), len(pending), time.time() - t0


def p0_summary(idx):
    """Spec P0: depth, lateral offsets, cube fit, filter pass rates, hand size (token counts come from encoding)."""
    q = lambda s: {f"p{p}": float(np.nanpercentile(s, p)) for p in (1, 5, 25, 50, 75, 95, 99)}  # noqa: E731
    out = {"images": int(len(idx)), "by_split_kind": idx.groupby(["split", "kind"]).size().rename("n").reset_index()
           .to_dict("records"),
           "d_m": q(idx.d), "L_n": q(idx.L_n), "S_px": q(idx.S), "g_absmax": q(idx.g_absmax),
           "lat_x_n": q(idx.lat_x_n), "lat_y_n": q(idx.lat_y_n), "soup_z_n": q(idx.soup_z_n),
           "hand_px": q(idx.hand_px), "hand_extent_m": q(idx.hand_extent_m),
           "cube_side_m": q(idx.d * idx.L_n),
           "pass": {"contained": float(idx.contained.mean()),
                    "hand_inside_ge_0.5": float((idx.hand_inside_frac >= 0.5).mean()),
                    "obj_inside_ge_0.5": float((idx.obj_inside_frac >= 0.5).mean()),
                    "hand_modal_gt0": float((idx.hand_modal_px > 0).mean())}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--stride", type=int, default=2, help="train contact frames: keep frame % stride == 0")
    ap.add_argument("--approach-frac", type=float, default=0.3)
    ap.add_argument("--approach-mm", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--data", default="data", help="holds dexycb/{calibration,models}, mano/, amodal_index_cams")
    ap.add_argument("--out-prefix", default="train/b0_dexycb_s0_v1")
    ap.add_argument("--max-frames", type=int, default=0, help="test: first N selected frames per subject")
    ap.add_argument("--local-tar")
    a = ap.parse_args()
    import boto3
    s3 = boto3.client("s3")
    sel = select_frames(s3, a.stride, a.approach_frac, a.approach_mm)
    print(sel.groupby(["split", "kind"]).size().to_string(), flush=True)
    subs = list(range(10)) if a.subjects == "all" else [int(x) for x in a.subjects.split(",")]
    if a.max_frames:
        sel = pd.concat([sel[sel.seq.str.startswith(SUBJECTS[i] + "/")].sort_values(["seq", "frame"]).head(a.max_frames)
                         for i in subs])
    jobs = [(i, sel, a.out_prefix, a.data, a.local_tar) for i in subs]
    with Pool(min(a.workers, len(subs))) as pool:
        for subj, n, orphan, dt in pool.imap_unordered(process_subject, jobs):
            print(f"{subj}: {n} images, {orphan} unpaired, {dt:.0f}s", flush=True)
    parts = []
    for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{a.out_prefix}/parts/").get("Contents", []):
        parts.append(s3_parquet(s3, o["Key"]))
    idx = pd.concat(parts, ignore_index=True)
    b = io.BytesIO()
    idx.to_parquet(b)
    s3.put_object(Bucket=BUCKET, Key=f"{a.out_prefix}/index.parquet", Body=b.getvalue())
    summ = p0_summary(idx)
    s3.put_object(Bucket=BUCKET, Key=f"{a.out_prefix}/p0_summary.json", Body=json.dumps(summ, indent=1).encode())
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
