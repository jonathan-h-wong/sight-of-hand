"""Dataset for the amodal mask predictor: the tar shards written by scripts/build_amodal_dataset.py.

Each sample is a 640x640 context crop (1.6x the amodal union box) with four 0/255 mask channels. Training takes a
square sub-crop of nominal side 480 (= 1.2x the box, the spec's crop) with scale jitter +-15% and shift +-10% of the
side, applied identically to image and masks, resized to 512 (bilinear image, nearest masks); plus colour jitter,
Gaussian blur and JPEG re-compression. No horizontal flip (it would change handedness). Evaluation takes the centred
480 crop without augmentation.
"""
import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

CTX, NOMINAL, OUT = 640, 480, 512


def index_shards(paths):
    """-> list of dicts {path, jpg (offset, size), png (offset, size), meta}."""
    out = []
    for p in paths:
        with tarfile.open(p, "r") as tf:
            by = {}
            for m in tf.getmembers():
                stem, ext = m.name.rsplit(".", 1)
                by.setdefault(stem, {})[ext] = m
            for stem, d in by.items():
                if {"jpg", "png", "json"} <= set(d):
                    meta = json.loads(tf.extractfile(d["json"]).read())
                    out.append({"path": str(p), "key": stem, "jpg": (d["jpg"].offset_data, d["jpg"].size),
                                "png": (d["png"].offset_data, d["png"].size), "meta": meta})
    return out


def _read(path, rng):
    with open(path, "rb") as f:
        f.seek(rng[0])
        return f.read(rng[1])


class AmodalShards(Dataset):
    def __init__(self, items, train, out=OUT, seed=0):
        self.items, self.train, self.out = items, train, out
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def _box(self):
        if not self.train:
            s = NOMINAL
            return ((CTX - s) / 2, (CTX - s) / 2, s)
        r = random
        s = NOMINAL * r.uniform(0.85, 1.15)
        cx = CTX / 2 + r.uniform(-0.1, 0.1) * s
        cy = CTX / 2 + r.uniform(-0.1, 0.1) * s
        # keep the box inside the stored context (max side 1.15 * 480 = 552 < 640, so only a shift is needed)
        x0 = min(max(0.0, cx - s / 2), CTX - s)
        y0 = min(max(0.0, cy - s / 2), CTX - s)
        return (x0, y0, s)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(io.BytesIO(_read(it["path"], it["jpg"]))).convert("RGB")
        msk = Image.open(io.BytesIO(_read(it["path"], it["png"])))  # RGBA: amodal hand/obj, modal hand/obj
        x0, y0, s = self._box()
        box = (x0, y0, x0 + s, y0 + s)
        img = img.resize((self.out, self.out), Image.BILINEAR, box=box)
        msk = msk.resize((self.out, self.out), Image.NEAREST, box=box)
        if self.train:
            r = random
            from torchvision.transforms import ColorJitter
            img = ColorJitter(0.3, 0.3, 0.3, 0.05)(img)
            if r.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.5)))
            if r.random() < 0.5:
                b = io.BytesIO()
                img.save(b, "JPEG", quality=r.randint(40, 95))
                img = Image.open(io.BytesIO(b.getvalue())).convert("RGB")
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy((np.asarray(msk) > 127).astype(np.float32)).permute(2, 0, 1)
        y[0] = torch.maximum(y[0], y[2])  # modal subset of amodal (holds by construction; keeps resizing exact)
        y[1] = torch.maximum(y[1], y[3])
        m = it["meta"]
        return x, y, {"hand_hidden_frac": m["hand_hidden_frac"], "obj_hidden_frac": m["obj_hidden_frac"], "idx": i}


def collate(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.stack([b[1] for b in batch])
    meta = {k: torch.tensor([float(b[2][k]) for b in batch]) for k in batch[0][2]}
    return x, y, meta


def local_shards(root, split):
    return sorted(Path(root).glob(f"{split}_*.tar"))
