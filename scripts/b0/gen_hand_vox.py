"""Hand cells per frame for the decoded-occupancy loss (CPU): the 64^3 grid cells (B0 cube, wrist-depth normalized)
that the GT hand surface passes through, from the raw shards (hand_v + MANO faces).

    python gen_hand_vox.py --mano mano_v1_2/models --out /workspace/hand_vox64 --split train --workers 64
Writes <out>/<key>.npz with coords (N,3) uint8.
"""
import argparse
import io
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sight_of_hand.b0 import geometry as G  # noqa: E402
from sight_of_hand.b0.mano_fit import _load_pkl  # noqa: E402

BUCKET = "xray-hands"
A = {}


def init(mano, prefix, out, rows):
    A["faces"] = {s: np.asarray(_load_pkl(Path(mano) / f"MANO_{s.upper()}.pkl")["f"], np.int64) for s in ("left", "right")}
    A["prefix"], A["out"], A["rows"] = prefix, Path(out), rows


def surface(V, F, n, rng):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    m = u + v > 1
    u[m], v[m] = 1 - u[m], 1 - v[m]
    return a[f] + u[:, None] * (b[f] - a[f]) + v[:, None] * (c[f] - a[f])


def shard(name):
    import boto3
    s3 = boto3.client("s3")
    body = io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{A['prefix']}/shards/{name}")["Body"].read())
    rng = np.random.default_rng(0)
    n = 0
    with tarfile.open(fileobj=body) as tf:
        for m in tf:
            if not m.name.endswith(".npz"):
                continue
            key = m.name[:-4]
            r = A["rows"].get(key)
            if r is None:
                continue
            z = np.load(io.BytesIO(tf.extractfile(m).read()))
            p = surface(z["hand_v"].astype(np.float64), A["faces"][str(z["side"])], 30000, rng)
            g = np.floor((G.cam_to_grid(p, r[0], r[1]) + 0.5) * 64).astype(int)
            g = np.unique(g[((g >= 0) & (g < 64)).all(1)], axis=0)
            np.savez_compressed(A["out"] / f"{key}.npz", coords=g.astype(np.uint8))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="train/b0_dexycb_s0_v1")
    ap.add_argument("--mano", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    import boto3
    s3 = boto3.client("s3")
    idx = pd.read_parquet(io.BytesIO(s3.get_object(Bucket=BUCKET, Key=f"{a.prefix}/index.parquet")["Body"].read()))
    idx = idx[idx.split == a.split]
    rows = {k: (float(d), float(L)) for k, d, L in zip(idx.key, idx.d, idx.L_n)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    shards = sorted(idx.shard.unique())
    with Pool(min(a.workers, len(shards)), initializer=init, initargs=(a.mano, a.prefix, a.out, rows)) as p:
        n = sum(p.map(shard, shards))
    print(f"{n} frames -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
