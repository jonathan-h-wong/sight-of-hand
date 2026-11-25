"""Loss and extra evaluation for the v2 head (docs/amodal_branch_spec_v2.md, sections 4b, 5, 6)."""
import torch
import torch.nn.functional as F

from .metrics import PARTS, Evaluator, _iou, loss_fn

DS_WEIGHT = 0.3     # deep supervision on the 32x32 logits
JOINT_WEIGHT = 0.5  # joint-heatmap MSE
SIGMA = 1.0         # heatmap sigma, in tokens (16 px)


def heatmaps(joints, size=32, out=512, sigma=SIGMA):
    """joints [B,21,2] output pixels -> Gaussian heatmaps [B,21,size,size] (token centres at (i+0.5)*out/size)."""
    c = (torch.arange(size, device=joints.device, dtype=torch.float32) + 0.5) * (out / size)
    s = sigma * out / size
    gx = torch.exp(-((c[None, None, :] - joints[..., 0:1]) ** 2) / (2 * s * s))  # [B,21,W]
    gy = torch.exp(-((c[None, None, :] - joints[..., 1:2]) ** 2) / (2 * s * s))  # [B,21,H]
    return gy[..., :, None] * gx[..., None, :]


def loss_v2(out, y, meta=None, hidden_weight=5.0):
    total, parts = loss_fn(out["logits"], y, hidden_weight)
    soft = F.avg_pool2d(y, kernel_size=y.shape[-1] // 32)  # area-downsampled soft GT at 32x32
    ds = F.binary_cross_entropy_with_logits(out["logits32"], soft)
    total = total + DS_WEIGHT * ds
    parts["loss/deep_sup"] = ds.item()
    if "joints" in out and meta is not None and "joints" in meta:
        v = meta["jvalid"].to(y.device)
        if v.sum() > 0:
            hm = heatmaps(meta["joints"].to(y.device), out=y.shape[-1])
            mse = ((out["joints"] - hm) ** 2).mean(dim=(1, 2, 3))
            jl = (mse * v).sum() / v.sum()
            total = total + JOINT_WEIGHT * jl
            parts["loss/joints"] = jl.item()
    return total, parts


def _dilate(m, r):
    return F.max_pool2d(m.float()[:, None], 2 * r + 1, stride=1, padding=r)[:, 0] > 0


def tolerant(a, b, r):
    """Boundary-tolerant agreement: a pixel of either mask counts as matched if the other mask is within r px.
    -> (|a & D_r(b)| + |b & D_r(a)|) / (|a| + |b|) per image (a Dice-style score; NaN when both are empty)."""
    num = (a & _dilate(b, r)).flatten(1).sum(1) + (b & _dilate(a, r)).flatten(1).sum(1)
    den = a.flatten(1).sum(1) + b.flatten(1).sum(1)
    return torch.where(den > 0, num.float() / den.clamp(min=1), torch.full_like(den, float("nan"), dtype=torch.float))


class EvaluatorV2(Evaluator):
    """v1 metrics + boundary-tolerant hidden-region scores (1, 2, 3 px) + hidden IoU on frames >= 10% occluded."""

    @torch.no_grad()
    def add(self, logits, y, hidden_frac):
        super().add(logits, y, hidden_frac)
        row = self.rows[-1]
        pb, yb = torch.sigmoid(logits.float()) > 0.5, y > 0.5
        for p, part in enumerate(PARTS):
            gt_h, pr_h = yb[:, p] & ~yb[:, p + 2], pb[:, p] & ~yb[:, p + 2]
            for r in (1, 2, 3):
                row[f"hidden_tol{r}px/{part}"] = tolerant(pr_h, gt_h, r)
            occ = hidden_frac[part]
            hi = row[f"hidden_iou/{part}"].clone()
            hi[occ < 0.1] = float("nan")
            row[f"hidden_iou_occ10/{part}"] = hi

    def result(self):
        res = super().result()
        res["hidden_iou_occ10/mean"] = (res["hidden_iou_occ10/hand"] + res["hidden_iou_occ10/obj"]) / 2
        res["modal_iou/mean"] = (res["iou/modal_hand"] + res["iou/modal_obj"]) / 2
        b = [res.get(f"hidden_iou_by_occ/{p}_{r}") for p in PARTS for r in ("10-30", "30-60")]
        b = [v for v in b if v is not None]
        res["hidden_iou_10to60/mean"] = sum(b) / len(b) if b else float("nan")
        return res


@torch.no_grad()
def resolution_ceiling(y, image, naf_up=None):
    """IoU of GT masks area-downsampled to 32x32 then upsampled back (bilinear; NAF when given) vs the originals:
    the upper bound for a head that only sees 32x32 information. -> {name: per-image IoU [B]} for the 4 channels."""
    T = y.shape[-1]
    low = F.avg_pool2d(y, kernel_size=T // 32)
    out = {}
    ups = {"bilinear": F.interpolate(low, size=T, mode="bilinear", align_corners=False)}
    if naf_up is not None:
        ups["naf"] = naf_up(low, image, T)
    names = ("amodal_hand", "amodal_obj", "modal_hand", "modal_obj")
    for k, u in ups.items():
        for c, n in enumerate(names):
            out[f"ceiling_{k}/{n}"] = _iou(u[:, c] > 0.5, y[:, c] > 0.5)
    return out
