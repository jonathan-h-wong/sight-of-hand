"""DexYCB 2D hand joints for the amodal training frames (spec v2, section 4b).

Streams each subject archive once and reads labels_<frame>.npz `joint_2d` (1, 21, 2): the 21 joints projected into
that camera, in image pixels, for every joint including occluded ones (dex-ycb-toolkit; checked in the files:
keys seg, pose_y, pose_m, joint_3d, joint_2d). Frames without a hand (pose_m all zero) get valid = False.
Rows use the same key and stride as scripts/build_amodal_dataset.py, so they join to its index; the data loader maps
the pixels into the stored context crop with crop_x0/crop_y0/crop_side.

    python scripts/extract_dexycb_joints.py --workers 10      -> s3://xray-hands/<prefix>/joints_2d.parquet
"""
import argparse
import io
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import boto3  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_amodal_dataset import BUCKET, SUBJECTS, ResumableS3  # noqa: E402


def subject(args):
    i, stride, local_tar = args
    s3 = boto3.client("s3")
    subj = SUBJECTS[i]
    body = open(local_tar, "rb") if local_tar else io.BufferedReader(ResumableS3(s3, f"raw/dexycb/{subj}.tar.gz"), 1 << 22)
    rows = []
    with tarfile.open(fileobj=body, mode="r|gz") as tf:
        for m in tf:
            p = m.name.split("/")
            if not m.isfile() or len(p) != 4 or not p[3].startswith("labels_"):
                continue
            fr = int(p[3].split("_")[1].split(".")[0])
            if fr % stride:
                continue
            z = np.load(io.BytesIO(tf.extractfile(m).read()))
            j = z["joint_2d"][0].astype(np.float32)
            valid = bool(np.any(z["pose_m"] != 0)) and bool(np.all(np.isfinite(j))) and bool(np.any(j > 0))
            rows.append({"key": f"{p[0]}__{p[1]}__{p[2]}__{fr:06d}", "valid": valid, "joints": j.reshape(-1).tolist()})
    return subj, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--subjects", default="all")
    ap.add_argument("--prefix", default="train/amodal_dexycb_s0_v1")
    ap.add_argument("--local-tar")
    a = ap.parse_args()
    subs = list(range(10)) if a.subjects == "all" else [int(x) for x in a.subjects.split(",")]
    out = []
    with Pool(min(a.workers, len(subs))) as pool:
        for subj, rows in pool.imap_unordered(subject, [(i, a.stride, a.local_tar) for i in subs]):
            print(f"{subj}: {len(rows)} frames, {sum(r['valid'] for r in rows)} with a hand", flush=True)
            out += rows
    df = pd.DataFrame(out)
    b = io.BytesIO()
    df.to_parquet(b)
    boto3.client("s3").put_object(Bucket=BUCKET, Key=f"{a.prefix}/joints_2d.parquet", Body=b.getvalue())
    print(f"{len(df)} rows -> s3://{BUCKET}/{a.prefix}/joints_2d.parquet")


if __name__ == "__main__":
    main()
