"""Amodal mask predictor v2 (docs/amodal_branch_spec_v2.md): DINO taps + transformer head, NAF on logits, optional
RGB-stem U-Net decoder and joint-heatmap auxiliary head.

    predict_masks(image [B,3,512,512] in [0,1]) -> [B,4,T,T] sigmoid probabilities (0 amodal hand, 1 amodal object,
                                                                                    2 modal hand, 3 modal object)
Variants (Config.decoder):
  "naf"   v2a: transformer logits [B,4,32,32] -> NAF to TxT
  "lite"  v2b-lite: RGB stem (s1..s8) + decoder 32 -> 64 -> 128, 1x1 -> logits at 128, -> NAF to TxT
  "full"  v2b-full: decoder continues to 256 and 512 with stem skips; no NAF (T must be 512)
Config.joints adds a Linear 384 -> 21 joint-heatmap head on the transformer tokens (training only).

DINOv3 runs once, frozen, under no_grad in bf16; block outputs are collected at Config.taps (0-indexed layers of the
24-layer ViT-L, DPT convention) from the same forward pass Pixal3D uses (that final output is unchanged). Intermediate
tap outputs are raw block outputs (no final norm); each tap gets its own learnable LayerNorm.

NAF upsamples with gradients enabled (its weights are frozen, the op is differentiable in its input). NAF splits the
value channels over its 4 attention heads, so the 4 logits are placed at the start of 4 zero-padded 32-channel heads
(one logit per head, a head dim the fused natten kernels accept) and read back after upsampling. Because each head has
its own attention weights, NAF on logits is close to, but not exactly, NAF on the features followed by the linear
layer the spec assumes.
"""
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import DINO, MEAN, STD

PATCH = 16
GRID = 32  # 512 / 16


@dataclass
class Config:
    taps: tuple = (5, 11, 17, 23)
    width: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    drop_path: float = 0.1
    pos_emb: str = "learned"  # learned | sinusoidal | none
    decoder: str = "naf"  # naf | lite | full
    joints: bool = False
    T: int = 512
    naf_head_dim: int = 32

    def to_dict(self):
        d = asdict(self)
        d["taps"] = list(self.taps)
        return d


def load_naf():
    import os
    naf_dir = os.environ.get("NAF_DIR")
    if naf_dir:
        m = torch.hub.load(naf_dir, "naf", source="local", pretrained=True, device="cpu", trust_repo=True)
    else:
        m = torch.hub.load("valeoai/NAF", "naf", pretrained=True, device="cpu", trust_repo=True)
    return m.eval().requires_grad_(False)


class DinoTaps(nn.Module):
    def __init__(self, taps, dino=DINO):
        super().__init__()
        from transformers import DINOv3ViTModel
        self.dino = DINOv3ViTModel.from_pretrained(dino).eval().requires_grad_(False)
        self.taps = tuple(taps)
        self.n_reg = getattr(self.dino.config, "num_register_tokens", 4)
        self.dim = self.dino.config.hidden_size
        assert max(self.taps) < len(self.dino.layer), (self.taps, len(self.dino.layer))
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, image):
        x = ((image - self.mean) / self.std).to(self.dino.embeddings.patch_embeddings.weight.dtype)
        h = self.dino.embeddings(x, bool_masked_pos=None)
        pos = self.dino.rope_embeddings(x)
        out = []
        for i, layer in enumerate(self.dino.layer):
            h = layer(h, position_embeddings=pos)
            if i in self.taps:
                out.append(h[:, 1 + self.n_reg:])  # [B, 1024 patches, D]
            if i >= max(self.taps):
                break
        return out


def sincos_2d(n, dim):
    """[n*n, dim] fixed 2D sine-cosine position embedding."""
    y, x = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
    q = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(q, dtype=torch.float32) / q))
    out = [torch.sin(x.flatten()[:, None] * omega), torch.cos(x.flatten()[:, None] * omega),
           torch.sin(y.flatten()[:, None] * omega), torch.cos(y.flatten()[:, None] * omega)]
    return torch.cat(out, 1)


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p).to(x.dtype)
        return x * keep / (1 - self.p)


class Block(nn.Module):
    """Pre-norm transformer block with SDPA (flash attention when available)."""

    def __init__(self, dim, heads, mlp_ratio, drop_path):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.dp = DropPath(drop_path)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        x = x + self.dp(self.proj(a))
        return x + self.dp(self.mlp(self.n2(x)))


def _conv(ci, co, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1), nn.GroupNorm(min(32, co // 2), co), nn.GELU())


class Stem(nn.Module):
    """RGB stem (v2b): s1 512 -> 32 ch, s2 256 -> 64, s4 128 -> 96, s8 64 -> 128."""

    def __init__(self):
        super().__init__()
        self.s1, self.s2, self.s4, self.s8 = _conv(3, 32), _conv(32, 64, 2), _conv(64, 96, 2), _conv(96, 128, 2)

    def forward(self, x):
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s4 = self.s4(s2)
        return s1, s2, s4, self.s8(s4)


class AmodalMaskNetV2(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = DinoTaps(cfg.taps)
        D, W = self.backbone.dim, cfg.width
        nt = len(cfg.taps)
        self.tap_norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(nt)])
        self.tap_proj = nn.ModuleList([nn.Linear(D, 256) for _ in range(nt)])
        self.fuse = nn.Sequential(nn.Linear(256 * nt, W), nn.GELU())
        if cfg.pos_emb == "learned":
            self.pos = nn.Parameter(torch.zeros(1, GRID * GRID, W))
            nn.init.trunc_normal_(self.pos, std=0.02)
        elif cfg.pos_emb == "sinusoidal":
            self.register_buffer("pos", sincos_2d(GRID, W)[None], persistent=False)
        else:
            self.pos = None
        dps = torch.linspace(0, cfg.drop_path, cfg.depth).tolist()
        self.blocks = nn.ModuleList([Block(W, cfg.heads, cfg.mlp_ratio, dp) for dp in dps])
        self.norm = nn.LayerNorm(W)
        self.logits32 = nn.Linear(W, 4)
        self.joint_head = nn.Linear(W, 21) if cfg.joints else None
        if cfg.decoder in ("lite", "full"):
            self.stem = Stem()
            self.d64 = _conv(W + 128, 128)
            self.d128 = _conv(128 + 96, 96)
            if cfg.decoder == "lite":
                self.out128 = nn.Conv2d(96, 4, 1)
            else:
                assert cfg.T == 512, "v2b-full predicts at 512 directly"
                self.d256 = _conv(96 + 64, 64)
                self.d512 = _conv(64 + 32, 32)
                self.out512 = nn.Conv2d(32, 4, 1)
        self.naf = load_naf() if cfg.decoder in ("naf", "lite") else None
        self.naf_ok = True  # set False by the trainer if NAF backward is unavailable (then bilinear for training)

    def trainable(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith(("backbone.", "naf."))]

    def upsample(self, logits, image, size):
        """NAF (frozen, differentiable) or bilinear when NAF backward is unavailable during training."""
        if self.naf is None or (self.training and not self.naf_ok):
            return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        B, C, h, w = logits.shape
        hd = self.cfg.naf_head_dim
        v = logits.new_zeros(B, 4 * hd, h, w)
        v[:, ::hd] = logits  # logit c -> channel 0 of head c
        out = self.naf(image.to(v.dtype), v, (size, size))
        return out[:, ::hd]

    def forward(self, image):
        """-> dict: logits [B,4,T,T], logits32 [B,4,32,32], joints [B,21,32,32] (when enabled)."""
        B = image.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=image.is_cuda):
            taps = self.backbone(image)
            x = torch.cat([p(n(t.float())) for t, n, p in zip(taps, self.tap_norm, self.tap_proj)], -1)
            x = self.fuse(x)
            if self.pos is not None:
                x = x + self.pos
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            out = {"logits32": self.logits32(x).transpose(1, 2).reshape(B, 4, GRID, GRID)}
            if self.joint_head is not None:
                out["joints"] = self.joint_head(x).transpose(1, 2).reshape(B, 21, GRID, GRID)
            T = self.cfg.T
            if self.cfg.decoder == "naf":
                out["logits"] = self.upsample(out["logits32"].float(), image, T)
            else:
                f = x.transpose(1, 2).reshape(B, -1, GRID, GRID)
                s1, s2, s4, s8 = self.stem(image)
                y = self.d64(torch.cat([F.interpolate(f, scale_factor=2, mode="bilinear"), s8], 1))
                y = self.d128(torch.cat([F.interpolate(y, scale_factor=2, mode="bilinear"), s4], 1))
                if self.cfg.decoder == "lite":
                    out["logits"] = self.upsample(self.out128(y).float(), image, T)
                else:
                    y = self.d256(torch.cat([F.interpolate(y, scale_factor=2, mode="bilinear"), s2], 1))
                    y = self.d512(torch.cat([F.interpolate(y, scale_factor=2, mode="bilinear"), s1], 1))
                    out["logits"] = self.out512(y)
        return {k: v.float() for k, v in out.items()}

    @torch.no_grad()
    def predict_masks(self, image):
        return torch.sigmoid(self(image)["logits"])


def head_state(model):
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith(("backbone.", "naf."))}
    return {"head": sd, "config": model.cfg.to_dict() | {"dino": DINO, "version": 2}}


def from_checkpoint(ck):
    cfg = dict(ck["config"])
    cfg.pop("dino", None)
    cfg.pop("version", None)
    cfg["taps"] = tuple(cfg["taps"])
    m = AmodalMaskNetV2(Config(**cfg))
    m.load_state_dict(ck["head"], strict=False)
    return m
