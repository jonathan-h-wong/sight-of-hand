"""Render amodal + modal hand/object masks for a dataset into the S3 mask store (sight_of_hand/amodal/store.py).

    python scripts/render_amodal.py --dataset dexycb  --root data/dexycb  --workers 32
    python scripts/render_amodal.py --dataset graspxl --root data/graspxl --workers 32
    python scripts/render_amodal.py --dataset dexycb  --root data/dexycb  --limit 2 --no-upload --out /tmp/amodal  # local test

Sequences are split into shards (--shard-size); each shard is rendered by one worker, written as a tar of npz files plus
a parquet of per-image rows, and uploaded at once. A rerun skips shards already in S3, so the job is resumable.
At the end the shard parquets are merged into <dataset>/index.parquet and summary.json.
Needs hoiaudit (sight-of-hand-eval) on PYTHONPATH and the MANO models (--mano-dir).
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one thread per worker process
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sight_of_hand.amodal import store  # noqa: E402

_G = {}


def _tasks(dataset, root):
    if dataset == "dexycb":
        from sight_of_hand.amodal import dexycb
        return dexycb.list_tasks(root)
    from hoiaudit.sources import graspxl
    return graspxl.list_tasks(root)


def _seq_key(dataset, task, root):
    p = Path(task)
    if dataset == "dexycb":
        return f"{p.parent.name}/{p.name}"
    return str(p.relative_to(Path(root) / "recorded").with_suffix(""))


def _init(dataset, mano_dir, out, upload):
    from hoiaudit.sources import common
    common.MANO_DIR = Path(mano_dir)
    _G.update(dataset=dataset, out=Path(out), upload=upload)
    if upload:
        import boto3
        _G["s3"] = boto3.client("s3")


def _render_shard(args):
    shard, tasks, root = args
    ds = _G["dataset"]
    mod = __import__(f"sight_of_hand.amodal.{ds}", fromlist=["render_sequence"])
    H, W = mod.H, mod.W
    tar_path = _G["out"] / "shards" / f"{shard:05d}.tar"
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    rows, errs, t0 = [], [], time.time()
    with tarfile.open(tar_path, "w", format=tarfile.PAX_FORMAT) as tf:
        for task in tasks:
            seq = _seq_key(ds, task, root)
            try:
                for item in mod.render_sequence(task):
                    cam, frames, masks, K, rr = item[:5]
                    extra = item[5] if len(item) > 5 else None
                    meta = {"dataset": ds, "seq": seq, "cam": str(cam), "H": H, "W": W,
                            "axes": "OpenCV camera (x right, y down, z forward)"
                                    + ("; X_cam = R X_world + t, metres" if extra else "; dataset camera frame")}
                    data = store.npz_bytes(frames, masks, K, meta, extra)
                    ti = tarfile.TarInfo(f"{seq}/{cam}.npz")
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
                    Ks = K if K.ndim == 3 else np.broadcast_to(K, (len(frames), 3, 3))
                    for j, r in enumerate(rr):
                        rows.append({"dataset": ds, "seq": seq, "cam": str(cam), "row": j, "shard": shard,
                                     "member": ti.name, "size": ti.size,
                                     "fx": Ks[j][0, 0], "fy": Ks[j][1, 1], "cx": Ks[j][0, 2], "cy": Ks[j][1, 2],
                                     "H": H, "W": W, **r})
            except Exception as e:  # keep going; reported in the shard's error list
                import traceback
                errs.append(f"{seq}: {e!r}\n{traceback.format_exc(limit=3)}")
    # byte offsets of each npz inside the tar (addfile stores a copy of the header, so read them back)
    with tarfile.open(tar_path, "r") as tf:
        off = {m.name: m.offset_data for m in tf.getmembers()}
    df = pd.DataFrame(rows)
    if len(df):
        df["offset"] = df.member.map(off).astype("int64")
    pq = tar_path.with_suffix(".parquet")
    df.to_parquet(pq)
    (tar_path.with_suffix(".errors.json")).write_text(json.dumps(errs))
    if _G["upload"]:
        base = f"{store.PREFIX}/{ds}/shards/{shard:05d}"
        s3 = _G["s3"]
        s3.upload_file(str(tar_path), store.BUCKET, base + ".tar")
        s3.upload_file(str(tar_path.with_suffix(".errors.json")), store.BUCKET, base + ".errors.json")
        s3.upload_file(str(pq), store.BUCKET, base + ".parquet")  # last: marks the shard done
        tar_path.unlink()
    return shard, len(rows), len(errs), time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["dexycb", "graspxl"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out/amodal"))
    ap.add_argument("--mano-dir", default=os.environ.get("MANO_DIR", "data/mano/mano_v1_2/models"))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--shard-size", type=int, help="sequences per shard (default: dexycb 10, graspxl 200)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N sequences")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()
    upload = not a.no_upload
    tasks = _tasks(a.dataset, a.root)
    if a.limit:
        tasks = tasks[: a.limit]
    size = a.shard_size or (10 if a.dataset == "dexycb" else 200)
    shards = [(i, tasks[i * size:(i + 1) * size], str(a.root)) for i in range((len(tasks) + size - 1) // size)]
    out = a.out / a.dataset
    prefix = f"{store.PREFIX}/{a.dataset}"
    done = set()
    if upload:
        import boto3
        s3 = boto3.client("s3")
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=store.BUCKET, Prefix=prefix + "/shards/"):
            done |= {int(Path(o["Key"]).stem) for o in page.get("Contents", []) if o["Key"].endswith(".parquet")}
    todo = [s for s in shards if s[0] not in done]
    print(f"{a.dataset}: {len(tasks)} sequences in {len(shards)} shards, {len(done)} done, {len(todo)} to go, "
          f"{a.workers} workers", flush=True)
    t0, n_img, n_err = time.time(), 0, 0
    with Pool(a.workers, initializer=_init, initargs=(a.dataset, a.mano_dir, out, upload)) as pool:
        for k, (shard, n, ne, dt) in enumerate(pool.imap_unordered(_render_shard, todo)):
            n_img += n
            n_err += ne
            print(f"  shard {shard}: {n} images, {ne} errors, {dt:.0f}s  [{k + 1}/{len(todo)}, "
                  f"{time.time() - t0:.0f}s total]", flush=True)
    # merge
    if upload:
        parts = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=store.BUCKET, Prefix=prefix + "/shards/"):
            for o in page.get("Contents", []):
                if o["Key"].endswith(".parquet"):
                    parts.append(pd.read_parquet(io.BytesIO(s3.get_object(Bucket=store.BUCKET, Key=o["Key"])["Body"].read())))
        errs = []
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=store.BUCKET, Prefix=prefix + "/shards/"):
            for o in page.get("Contents", []):
                if o["Key"].endswith(".errors.json"):
                    errs += json.loads(s3.get_object(Bucket=store.BUCKET, Key=o["Key"])["Body"].read())
    else:
        parts = [pd.read_parquet(p) for p in sorted((out / "shards").glob("*.parquet"))]
        errs = sum((json.loads(p.read_text()) for p in sorted((out / "shards").glob("*.errors.json"))), [])
    idx = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    def q(c):
        x = idx[c].dropna()
        return {f"p{p}": round(float(np.percentile(x, p)), 4) for p in (10, 50, 90)} if len(x) else None
    summ = {"dataset": a.dataset, "sequences": int(idx.seq.nunique()) if len(idx) else 0, "images": int(len(idx)),
            "cameras_per_sequence": round(float(idx.groupby("seq").cam.nunique().mean()), 2) if len(idx) else 0,
            "obj_occluded_by_hand_frac": q("obj_occluded_by_hand_frac"),
            "hand_occluded_by_obj_frac": q("hand_occluded_by_obj_frac"),
            "hand_amodal_px": q("hand_amodal_px"), "obj_amodal_px": q("obj_amodal_px"),
            "hand_truncated_frac": round(float(idx.hand_truncated.mean()), 4) if len(idx) else None,
            "obj_truncated_frac": round(float(idx.obj_truncated.mean()), 4) if len(idx) else None,
            "errors": len(errs), "first_errors": errs[:5]}
    if a.dataset == "dexycb" and len(idx):
        summ |= {"iou_hand_seg": q("iou_hand_seg"), "iou_obj_seg": q("iou_obj_seg")}
    out.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(out / "index.parquet")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    if upload:
        s3.upload_file(str(out / "index.parquet"), store.BUCKET, prefix + "/index.parquet")
        s3.upload_file(str(out / "summary.json"), store.BUCKET, prefix + "/summary.json")
    print(json.dumps({k: v for k, v in summ.items() if k != "first_errors"}, indent=1))
    for e in errs[:5]:
        print("ERROR", e)


if __name__ == "__main__":
    main()
