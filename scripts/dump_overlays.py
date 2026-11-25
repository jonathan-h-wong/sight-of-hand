"""Qualitative review: predictions of a trained amodal head on s0 val, binned by occlusion (spec section 5).

For each part (hand, object) and occlusion bin (hidden share of the part's amodal mask: 0-10, 10-30, 30-60, >60%),
samples --per-bin val images and renders one row per image:
  input | ground truth | prediction | hidden-region errors
  ground truth / prediction: hand red, object blue; solid fill = visible (modal), hatched light fill = hidden part of
    the amodal mask, outline = amodal boundary (prediction thresholded at 0.5; fill alpha follows the probability)
  hidden-region errors for the binned part: green = hidden pixels found, magenta = hidden pixels missed,
    yellow = predicted hidden where there is none
Writes OUT/<part>_<bin>.jpg contact sheets and OUT/samples.csv (per-image hidden IoU), optionally to S3.

    python scripts/dump_overlays.py --data /data/amodal --ckpt last.pt --out overlays --per-bin 8
"""
import argparse
import csv
import io
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from sight_of_hand.amodal_net import data as D  # noqa: E402
from sight_of_hand.amodal_net.model import AmodalMaskNet  # noqa: E402

BINS = ((0.0, 0.1, "00-10"), (0.1, 0.3, "10-30"), (0.3, 0.6, "30-60"), (0.6, 1.01, "60-100"))
RED, BLUE = np.array([235, 60, 60]), np.array([60, 110, 245])


def outline(m):
    e = np.zeros_like(m)
    e[1:-1, 1:-1] = m[1:-1, 1:-1] & ~(m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    return e


def paint(img, amodal, modal, prob=None):
    """img HxWx3 float; amodal/modal [2,H,W] bool (hand, obj); prob [2,H,W] amodal probabilities (prediction)."""
    out = img * 0.55
    H, W = modal.shape[1:]
    hatch = ((np.add.outer(np.arange(H), np.arange(W)) // 6) % 2 == 0)
    for k, col in ((1, BLUE), (0, RED)):
        hid = amodal[k] & ~modal[k]
        a = 0.45 if prob is None else 0.45 * np.clip(prob[k], 0, 1)
        sel = hid & hatch
        out[sel] = out[sel] * (1 - a[sel, None] if prob is not None else 1 - a) + col * (a[sel, None] if prob is not None else a)
        out[modal[k]] = out[modal[k]] * 0.35 + col * 0.65
        out[outline(amodal[k])] = col * 0.9 + 255 * 0.1
    return out


def errors(img, gt_am, gt_mo, pr_am, k):
    out = img * 0.35
    gh = gt_am[k] & ~gt_mo[k]
    ph = pr_am[k] & ~gt_mo[k]
    out[gh & ph] = (40, 200, 80)
    out[gh & ~ph] = (220, 40, 200)
    out[ph & ~gh] = (240, 210, 40)
    out[outline(gt_mo[k])] = (255, 255, 255)
    return out


def label(tile, text):
    im = Image.fromarray(tile.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, default=Path("overlays"))
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s3-prefix", default="")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.get("version") == 2:  # v2 checkpoints (scripts/train_amodal_v2.py): forward returns a dict
        from sight_of_hand.amodal_net.model_v2 import from_checkpoint
        model = from_checkpoint(ck).to(dev).eval()
    else:
        model = AmodalMaskNet(T=cfg["T"], rgb_skip=cfg["rgb_skip"]).to(dev).eval()
        model.head.load_state_dict(ck["head"])
    items = D.index_shards(sorted((a.data / "shards").glob("val_*.tar")))
    rng = random.Random(a.seed)
    picks = []
    for part, key in (("hand", "hand_hidden_frac"), ("obj", "obj_hidden_frac")):
        for lo, hi, name in BINS:
            pool = [i for i, it in enumerate(items)
                    if it["meta"][f"{part}_amodal_px"] > 400 and lo <= it["meta"][key] < hi]
            picks += [(part, name, i) for i in rng.sample(pool, min(a.per_bin, len(pool)))]
    ds = D.AmodalShards(items, False)
    a.out.mkdir(parents=True, exist_ok=True)
    rows, sheets = [], {}
    T = a.tile
    with torch.no_grad():
        for part, name, i in picks:
            x, y, meta = ds[i]
            o = model(x[None].to(dev))
            p = torch.sigmoid(o["logits"] if isinstance(o, dict) else o)[0].float().cpu().numpy()
            if p.shape[-1] != y.shape[-1]:
                y = torch.nn.functional.interpolate(y[None], size=p.shape[-2:], mode="nearest")[0]
            yb = y.numpy() > 0.5
            pb = p > 0.5
            img = np.asarray(Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize(p.shape[-2:][::-1]),
                             np.float32)
            k = 0 if part == "hand" else 1
            hi_h = iou(pb[0] & ~yb[2], yb[0] & ~yb[2])
            hi_o = iou(pb[1] & ~yb[3], yb[1] & ~yb[3])
            occ = items[i]["meta"][f"{part}_hidden_frac"]
            tiles = [label(img, f"{part} {occ:.0%} hidden"),
                     label(paint(img, yb[:2], yb[2:]), "ground truth"),
                     label(paint(img, pb[:2], pb[2:], p[:2]), "prediction"),
                     label(errors(img, yb[:2], yb[2:], pb[:2], k), f"{part} hidden IoU {[hi_h, hi_o][k]:.2f}")]
            row = np.concatenate([np.asarray(Image.fromarray(t.clip(0, 255).astype(np.uint8)).resize((T, T))) for t in tiles], 1)
            sheets.setdefault((part, name), []).append(row)
            m = items[i]["meta"]
            rows.append({"part": part, "bin": name, "key": items[i]["key"], "seq": m["seq"], "serial": m["serial"],
                         "frame": m["frame"], "hidden_frac": round(occ, 3), "hidden_iou_hand": round(hi_h, 3),
                         "hidden_iou_obj": round(hi_o, 3), "iou_amodal_hand": round(iou(pb[0], yb[0]), 3),
                         "iou_amodal_obj": round(iou(pb[1], yb[1]), 3)})
    for (part, name), rr in sheets.items():
        Image.fromarray(np.concatenate(rr, 0)).save(a.out / f"{part}_{name}.jpg", quality=88)
    with open(a.out / "samples.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if a.s3_prefix:
        import boto3
        s3 = boto3.client("s3")
        for p in a.out.iterdir():
            s3.upload_file(str(p), "xray-hands", f"{a.s3_prefix}/{p.name}")
    print(f"{len(rows)} samples -> {a.out}")


if __name__ == "__main__":
    main()
