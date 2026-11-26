"""Merge eval_b0.py rows.parquet files (zero-shot, B0, sharded runs) and write one summary.json with eval_b0.summarize.

    python scripts/b0/summarize_eval.py --out results/summary.json  s3://.../eval_zeroshot/rows.parquet  rows_b0_*.parquet

Stress frames are out-of-filter frames that were never encoded, so the ss and gt512 modes have no GT latent there
(FileNotFoundError rows); only the cascade is scored on them.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_b0 import summarize  # noqa: E402

MAIN = ["row", "mode", "matte", "ss/iou", "gt_d/soup/f@2", "gt_d/hand/f@2", "gt_d/obj/f@2", "gt_d/soup/chamfer",
        "gt_d/rel_offset_mm", "moge_d/soup/f@2"]


def read(p):
    if p.startswith("s3://"):
        import boto3
        b, k = p[5:].split("/", 1)
        return pd.read_parquet(io.BytesIO(boto3.client("s3").get_object(Bucket=b, Key=k)["Body"].read()))
    return pd.read_parquet(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merged", help="also write the merged rows parquet here")
    a = ap.parse_args()
    df = pd.concat([read(p) for p in a.rows], ignore_index=True)
    dup = df.duplicated(["row", "mode", "matte", "key"])
    if dup.any():
        sys.exit(f"{dup.sum()} duplicate (row, mode, matte, key) rows")
    if a.merged:
        df.to_parquet(a.merged)
    s = summarize(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(s, indent=1))
    o = pd.DataFrame(s["overall"])
    print(o[[c for c in MAIN if c in o]].merge(pd.DataFrame(s["n"]), on=["row", "mode", "matte"]).to_string())


if __name__ == "__main__":
    main()
