"""Amodal mask store: tar shards of per-(sequence, camera) npz files + one index parquet per dataset.

    s3://xray-hands/masks/amodal_v1/<dataset>/shards/<shard:05d>.tar      uncompressed tar
        <seq>/<cam>.npz    frames (N,), hand_amodal, obj_amodal, hand_modal, obj_modal: (N, H, ceil(W/8)) uint8
                           (np.packbits along W; unpack with raycast.unpack(x, W)), K (3,3) or (N,3,3),
                           [R (N,3,3), t (N,3): world -> camera, GraspXL], meta (json: dataset, seq, cam, H, W, axes)
    s3://xray-hands/masks/amodal_v1/<dataset>/index.parquet   one row per image: seq, cam, frame, row (index into
                           the npz arrays), shard, member, offset, size (byte range of the npz inside the tar) + the
                           per-image statistics (mask areas, visible and occluded fractions, truncation, ...)
    s3://xray-hands/masks/amodal_v1/<dataset>/summary.json

Masks: <part>_amodal = the part's mesh rendered alone (full silhouette, nothing occludes it); <part>_modal = the
pixels where that part is the first surface hit with every mesh present (what is visible).
"""
import io
import json

import numpy as np

BUCKET = "xray-hands"
PREFIX = "masks/amodal_v1"
NAMES = ("hand_amodal", "obj_amodal", "hand_modal", "obj_modal")


def npz_bytes(frames, masks, K, meta, extra=None):
    from .raycast import pack
    buf = io.BytesIO()
    np.savez_compressed(buf, frames=np.asarray(frames), K=K, meta=json.dumps(meta),
                        **pack({k: masks[k] for k in NAMES}), **(extra or {}))
    return buf.getvalue()


def read_npz(dataset, row, s3=None):
    """Fetch one (seq, cam) npz by its byte range in the shard."""
    import boto3
    s3 = s3 or boto3.client("s3")
    key = f"{PREFIX}/{dataset}/shards/{int(row['shard']):05d}.tar"
    a, n = int(row["offset"]), int(row["size"])
    body = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={a}-{a + n - 1}")["Body"].read()
    return np.load(io.BytesIO(body))


def masks_for(dataset, row, s3=None):
    """One index row -> dict of the four (H, W) bool masks for that image."""
    from .raycast import unpack
    z = read_npz(dataset, row, s3)
    W = json.loads(str(z["meta"]))["W"]
    i = int(row["row"])
    return {k: unpack(z[k][i], W) for k in NAMES}


def read_index(dataset):
    import boto3
    import pandas as pd
    body = boto3.client("s3").get_object(Bucket=BUCKET, Key=f"{PREFIX}/{dataset}/index.parquet")["Body"].read()
    return pd.read_parquet(io.BytesIO(body))
