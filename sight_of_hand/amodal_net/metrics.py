"""Loss and evaluation for the amodal mask predictor (docs/amodal_branch_spec.md, sections 4-5).

Channels: 0 amodal hand, 1 amodal object, 2 modal hand, 3 modal object. Part p uses amodal channel p, modal p + 2.
Hidden region of part p = GT amodal p AND NOT GT modal p.
"""
import torch
import torch.nn.functional as F

PARTS = ("hand", "obj")
CHANNELS = ("amodal_hand", "amodal_obj", "modal_hand", "modal_obj")
BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01))


def loss_fn(logits, y, hidden_weight=5.0):
    """Per-channel BCE + soft Dice. For the amodal channels, BCE on hidden-region pixels is up-weighted."""
    w = torch.ones_like(y)
    for p in range(2):
        hidden = y[:, p] * (1 - y[:, p + 2])
        w[:, p] = 1 + (hidden_weight - 1) * hidden
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean(dim=(0, 2, 3))
    pr = torch.sigmoid(logits)
    inter = (pr * y).sum(dim=(2, 3))
    dice = 1 - (2 * inter + 1) / (pr.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) + 1)
    per = bce + dice.mean(0)
    return per.sum(), {f"loss/{c}": per[i].item() for i, c in enumerate(CHANNELS)}


def _iou(a, b):
    """a, b bool [N,H,W] -> per-image IoU [N], NaN where the union is empty."""
    inter = (a & b).flatten(1).sum(1).float()
    union = (a | b).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.full_like(union, float("nan")))


class Evaluator:
    """Accumulates per-image metrics; .result() averages them (images with an empty union are skipped)."""

    def __init__(self):
        self.rows = []

    @torch.no_grad()
    def add(self, logits, y, hidden_frac):
        pr = torch.sigmoid(logits.float())
        pb, yb = pr > 0.5, y > 0.5
        out = {}
        for i, c in enumerate(CHANNELS):
            out[f"iou/{c}"] = _iou(pb[:, i], yb[:, i])
        for p, part in enumerate(PARTS):
            gt_hidden = yb[:, p] & ~yb[:, p + 2]
            out[f"hidden_iou/{part}"] = _iou(pb[:, p] & ~yb[:, p + 2], gt_hidden)
            out[f"baseline_modal_gt/hidden_iou/{part}"] = _iou(torch.zeros_like(gt_hidden), gt_hidden)
            out[f"baseline_own_modal/hidden_iou/{part}"] = _iou(pb[:, p + 2] & ~yb[:, p + 2], gt_hidden)
            out[f"baseline_modal_gt/iou/amodal_{part}"] = _iou(yb[:, p + 2], yb[:, p])
            # calibration on the hidden-candidate pixels (outside the GT modal mask)
            cand = ~yb[:, p + 2]
            q, t = pr[:, p][cand], yb[:, p][cand].float()
            bins = (q * 10).clamp(max=9.999).long()  # 10 equal-width bins; per-bin sums, not every pixel
            ece = torch.stack([torch.bincount(bins, minlength=10).float(),
                               torch.bincount(bins, weights=q, minlength=10), torch.bincount(bins, weights=t, minlength=10)])
            n = cand.flatten(1).sum(1).clamp(min=1).float()
            out[f"brier_hidden/{part}"] = ((pr[:, p] - yb[:, p].float()) ** 2 * cand).flatten(1).sum(1) / n
            out[f"_ece_counts/{part}"] = ece
            out[f"occ/{part}"] = hidden_frac[part]
        self.rows.append(out)

    def result(self):
        res = {}
        keys = [k for k in self.rows[0] if not k.startswith(("_ece", "occ/"))]
        for k in keys:
            v = torch.cat([r[k].cpu() for r in self.rows])
            res[k] = float(torch.nanmean(v)) if (~torch.isnan(v)).any() else float("nan")
        for part in PARTS:
            occ = torch.cat([r[f"occ/{part}"].cpu() for r in self.rows])
            hi = torch.cat([r[f"hidden_iou/{part}"].cpu() for r in self.rows])
            for lo, up in BINS:
                sel = (occ >= lo) & (occ < up)
                if sel.any():
                    res[f"hidden_iou_by_occ/{part}_{int(lo * 100)}-{min(100, int(up * 100))}"] = float(torch.nanmean(hi[sel]))
                    res[f"count_by_occ/{part}_{int(lo * 100)}-{min(100, int(up * 100))}"] = int(sel.sum())
            e = sum(r[f"_ece_counts/{part}"].cpu() for r in self.rows)  # [3,10]: count, sum q, sum t
            nz = e[0] > 0
            ece = float(((e[1][nz] - e[2][nz]).abs()).sum() / e[0].sum()) if nz.any() else float("nan")
            res[f"ece_hidden/{part}"] = float(ece)
        res["hidden_iou/mean"] = (res["hidden_iou/hand"] + res["hidden_iou/obj"]) / 2
        return res
