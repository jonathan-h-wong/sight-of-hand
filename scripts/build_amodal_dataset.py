"""Build the DexYCB s0 training set for the amodal mask predictor (docs/amodal_branch_spec.md, section 3).

For every s0 frame at a fixed stride (default 5) and every camera:
  image        the DexYCB colour frame
  modal masks  from DexYCB's own segmentation: hand = 255, object = the grasped YCB id (other objects are ignored)
  amodal masks from the published mask store (s3://xray-hands/masks/amodal_v1/dexycb: each mesh ray-cast alone),
               then amodal |= modal so modal is a subset of amodal; the pixels that had to be added are counted
               per frame (hand_violation_px, obj_violation_px) as a pose-error signal
  crop         square, centred on the union box of the two amodal masks. The stored CONTEXT crop is 1.6x the box
               side (the training crop is 1.2x, so jitter of +-15% scale and +-10% shift stays inside it), resized
               to 640x640; pixels outside the image are black / mask 0. Crop parameters are stored.
Frames whose hand and object amodal masks are both empty are skipped.

Splits follow dex-ycb-toolkit (dex_ycb.py, setup s0; checked): per subject, its 100 sequences in sorted order;
train = index % 5 != 4 (all 10 subjects); val = index % 5 == 4 for subjects 1-2; test = index % 5 == 4 for subjects
3-10. Train sequences get a fold (A/B, alternating in sorted order within a subject) for the cross-fitted heads.

Output (s3://xray-hands/train/amodal_dexycb_s0_v1/ unless --out-prefix):
  shards/<split>_<subject>_<k>.tar   <key>.jpg (640x640 RGB, q95), <key>.png (640x640 RGBA: amodal hand, amodal object,
                                     modal hand, modal object as 0/255), <key>.json (metadata)
  index.parquet                       one row per sample (key, shard, split, fold, seq, serial, frame, crop, occlusion)

    python scripts/build_amodal_dataset.py --subjects all --workers 10
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from sight_of_hand.amodal import raycast, store  # noqa: E402

BUCKET = "xray-hands"
SUBJECTS = ["20200709-subject-01", "20200813-subject-02", "20200820-subject-03", "20200903-subject-04",
            "20200908-subject-05", "20200918-subject-06", "20200928-subject-07", "20201002-subject-08",
            "20201015-subject-09", "20201022-subject-10"]
CTX_SCALE = 1.6   # stored context crop side / amodal union box side
OUT_RES = 640
PER_SHARD = 2000


class ResumableS3(io.RawIOBase):
    """Sequential reader over an S3 object that re-requests from the current byte after a dropped connection."""

    def __init__(self, s3, key, tries=20):
        self.s3, self.key, self.pos, self.tries = s3, key, 0, tries
        self.size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        self.body = None

    def readable(self):
        return True

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        for t in range(self.tries):
            try:
                if self.body is None:
                    self.body = self.s3.get_object(Bucket=BUCKET, Key=self.key, Range=f"bytes={self.pos}-")["Body"]
                data = self.body.read(len(b))
                if not data and self.pos < self.size:
                    raise IOError("short read")
                b[:len(data)] = data
                self.pos += len(data)
                return len(data)
            except Exception:
                self.body = None
                time.sleep(min(30, 2 ** t))
        raise IOError(f"{self.key}: gave up at byte {self.pos}")


def split_of(subject_i, seq_i):
    if seq_i % 5 != 4:
        return "train"
    return "val" if subject_i in (0, 1) else "test"


def crop(img, masks, box, H, W):
    """Square context crop around box (x0, y0, x1, y1) -> (img OUT_RES^2, masks (4, OUT_RES, OUT_RES), params)."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0, 16) * CTX_SCALE
    a, b = int(round(cx - side / 2)), int(round(cy - side / 2))
    n = int(round(side))
    canvas = np.zeros((n, n, 3), np.uint8)
    mcanvas = np.zeros((4, n, n), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, a), max(0, b), min(W, a + n), min(H, b + n)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - b:sy1 - b, sx0 - a:sx1 - a] = img[sy0:sy1, sx0:sx1]
        mcanvas[:, sy0 - b:sy1 - b, sx0 - a:sx1 - a] = masks[:, sy0:sy1, sx0:sx1]
    im = Image.fromarray(canvas).resize((OUT_RES, OUT_RES), Image.BILINEAR)
    ms = np.stack([np.asarray(Image.fromarray(m * 255).resize((OUT_RES, OUT_RES), Image.NEAREST)) for m in mcanvas])
    return im, ms, {"crop_x0": a, "crop_y0": b, "crop_side": n}


def process_subject(args):
    subject_i, stride, prefix, max_seqs, local_tar = args
    import boto3
    s3 = boto3.client("s3")
    subj = SUBJECTS[subject_i]
    idx = store.read_index("dexycb")
    idx = idx[idx.seq.str.startswith(subj + "/")]
    seqs = sorted(idx.seq.unique())
    assert len(seqs) == 100, (subj, len(seqs))
    info = {}
    tr = [s for i, s in enumerate(seqs) if split_of(subject_i, i) == "train"]
    for i, s in enumerate(seqs):
        sp = split_of(subject_i, i)
        info[s] = {"split": sp, "fold": ("A" if tr.index(s) % 2 == 0 else "B") if sp == "train" else ""}
    if max_seqs:
        keep = set(seqs[:max_seqs])
        info = {k: v for k, v in info.items() if k in keep}
    rows_by = {k: g for k, g in idx[idx.seq.isin(info)].groupby(["seq", "cam"])}
    meta_cache, npz_cache = {}, {}
    pending = {}  # (seq, serial, frame) -> {"jpg": bytes, "npz": bytes}
    out_rows, writers, t0 = [], {}, time.time()
    work = Path(f"/tmp/amodal_ds/{subj}")
    work.mkdir(parents=True, exist_ok=True)

    def writer(split):
        w = writers.get(split)
        if w is None or w["n"] >= PER_SHARD:
            if w is not None:
                w["tar"].close()
                upload(w)
            k = (w["k"] + 1) if w else 0
            name = f"{split}_{subj}_{k:03d}.tar"
            w = {"name": name, "path": work / name, "n": 0, "k": k}
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
        key = (seq, serial)
        if key not in npz_cache:
            npz_cache.clear()
            g = rows_by.get(key)
            if g is None:
                return
            z = store.read_npz("dexycb", g.iloc[0], s3)
            npz_cache[key] = (z["frames"], {k: z[k] for k in store.NAMES}, json.loads(str(z["meta"]))["W"])
        frames, packed, W = npz_cache[key]
        hit = np.where(frames == fr)[0]
        if not len(hit):
            return
        j = int(hit[0])
        am = {k: raycast.unpack(packed[k][j], W) for k in ("hand_amodal", "obj_amodal")}
        z = np.load(io.BytesIO(lab))
        seg = z["seg"]
        gid = meta_cache[seq]["grasp_id"]
        hm, om = seg == 255, seg == gid
        hv, ov = int((hm & ~am["hand_amodal"]).sum()), int((om & ~am["obj_amodal"]).sum())
        ha, oa = am["hand_amodal"] | hm, am["obj_amodal"] | om
        u = ha | oa
        if not u.any():
            return
        ys, xs = np.where(u)
        img = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
        Hh, Ww = seg.shape
        im, ms, cp = crop(img, np.stack([ha, oa, hm, om]).astype(np.uint8), (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1),
                          Hh, Ww)
        nha, noa = int(ha.sum()), int(oa.sum())
        meta = {"seq": seq, "serial": serial, "frame": fr, **info[seq], **cp,
                "hand_amodal_px": nha, "obj_amodal_px": noa,
                "hand_hidden_frac": float((ha & ~hm).sum() / nha) if nha else float("nan"),
                "obj_hidden_frac": float((oa & ~om).sum() / noa) if noa else float("nan"),
                "hand_violation_px": hv, "obj_violation_px": ov, "image_hw": [Hh, Ww]}
        w = writer(info[seq]["split"])
        k = f"{seq.replace('/', '__')}__{serial}__{fr:06d}"
        b = io.BytesIO()
        im.save(b, "JPEG", quality=95)
        add(w["tar"], k + ".jpg", b.getvalue())
        b = io.BytesIO()
        Image.fromarray(np.moveaxis(ms, 0, -1), "RGBA").save(b, "PNG")
        add(w["tar"], k + ".png", b.getvalue())
        add(w["tar"], k + ".json", json.dumps(meta).encode())
        w["n"] += 1
        out_rows.append({"key": k, "shard": w["name"], **meta})

    body = open(local_tar, "rb") if local_tar else io.BufferedReader(ResumableS3(s3, f"raw/dexycb/{subj}.tar.gz"), 1 << 22)
    with tarfile.open(fileobj=body, mode="r|gz") as tf:
        for m in tf:
            p = m.name.split("/")
            if not m.isfile() or len(p) < 3:
                continue
            seq = f"{p[0]}/{p[1]}"
            if seq not in info:
                continue
            if p[-1] == "meta.yml":
                from sight_of_hand.amodal.dexycb import _Loader
                import yaml
                mt = yaml.load(tf.extractfile(m).read(), Loader=_Loader)
                meta_cache[seq] = {"grasp_id": int(mt["ycb_ids"][int(mt["ycb_grasp_ind"])])}
                for k in [k for k, d in pending.items() if k[0] == seq and len(d) == 2]:  # frames seen before meta
                    emit(*k, pending[k]["jpg"], pending[k]["npz"])
                    del pending[k]
                continue
            if len(p) != 4 or not (p[3].startswith("color_") or p[3].startswith("labels_")):
                continue
            fr = int(p[3].split("_")[-1].split(".")[0])
            if fr % stride:
                continue
            key = (seq, p[2], fr)
            d = pending.setdefault(key, {})
            d["jpg" if p[3].startswith("color_") else "npz"] = tf.extractfile(m).read()
            if len(d) == 2 and seq in meta_cache:
                emit(seq, p[2], fr, d["jpg"], d["npz"])
                del pending[key]
    for w in writers.values():
        w["tar"].close()
        upload(w)
    df = pd.DataFrame(out_rows)
    part = work / f"index_{subj}.parquet"
    df.to_parquet(part)
    s3.upload_file(str(part), BUCKET, f"{prefix}/parts/index_{subj}.parquet")
    return subj, len(df), len(pending), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default="all", help="'all' or comma list of indices 0-9")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-prefix", default="train/amodal_dexycb_s0_v1")
    ap.add_argument("--max-seqs", type=int, default=0, help="test: only the first N sequences per subject")
    ap.add_argument("--local-tar", help="test: read this local .tar.gz instead of the subject archive on S3")
    a = ap.parse_args()
    subs = list(range(10)) if a.subjects == "all" else [int(x) for x in a.subjects.split(",")]
    with Pool(min(a.workers, len(subs))) as pool:
        for subj, n, orphan, dt in pool.imap_unordered(process_subject, [(i, a.stride, a.out_prefix, a.max_seqs, a.local_tar) for i in subs]):
            print(f"{subj}: {n} samples, {orphan} unpaired, {dt:.0f}s", flush=True)
    import boto3
    s3 = boto3.client("s3")
    parts = [pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=o["Key"])["Body"].read()))
             for o in s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{a.out_prefix}/parts/").get("Contents", [])]
    idx = pd.concat(parts, ignore_index=True)
    if idx.empty:
        sys.exit("no samples were produced")
    b = io.BytesIO()
    idx.to_parquet(b)
    s3.put_object(Bucket=BUCKET, Key=f"{a.out_prefix}/index.parquet", Body=b.getvalue())
    summ = idx.groupby("split").agg(samples=("key", "size"), seqs=("seq", "nunique")).to_dict("index")
    summ["violation_frames"] = {p: int((idx[f"{p}_violation_px"] > 0).sum()) for p in ("hand", "obj")}
    s3.put_object(Bucket=BUCKET, Key=f"{a.out_prefix}/summary.json", Body=json.dumps(summ, indent=1).encode())
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
