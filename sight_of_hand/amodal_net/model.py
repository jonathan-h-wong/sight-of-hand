"""Amodal mask predictor (docs/amodal_branch_spec.md, sections 1-2).

    predict_masks(image [B,3,512,512] in [0,1]) -> [B,4,T,T] sigmoid probabilities
    channels: 0 amodal hand, 1 amodal object, 2 modal hand, 3 modal object

Frozen DINOv3 ViT-L/16 (loaded and run exactly as Pixal3D's DinoV3ProjFeatureExtractor: transformers
DINOv3ViTModel, ImageNet normalisation, embeddings + rope + layers, then a parameter-free layer norm; 1 CLS + register
tokens dropped) -> patch tokens [B,1024,32,32] -> frozen NAF upsampling guided by the RGB in [0,1] to TxT ->
trainable head (about 0.43M parameters). No gradient reaches DINOv3 or NAF; they run under no_grad in bf16/fp16.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DINO = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class FrozenFeatures(nn.Module):
    def __init__(self, T=512, image_size=512, dino=DINO):
        super().__init__()
        from transformers import DINOv3ViTModel
        self.dino = DINOv3ViTModel.from_pretrained(dino).eval().requires_grad_(False)
        import os
        naf_dir = os.environ.get("NAF_DIR")  # a local clone of github.com/valeoai/NAF (avoids torch.hub's GitHub lookup)
        if naf_dir:
            self.naf = torch.hub.load(naf_dir, "naf", source="local", pretrained=True, device="cpu", trust_repo=True)
        else:
            self.naf = torch.hub.load("valeoai/NAF", "naf", pretrained=True, device="cpu", trust_repo=True)
        self.naf.eval().requires_grad_(False)
        self.T, self.image_size = T, image_size
        self.patch = self.dino.config.patch_size
        self.dim = self.dino.config.hidden_size
        self.n_reg = getattr(self.dino.config, "num_register_tokens", 4)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):  # always frozen
        return super().train(False)

    @torch.no_grad()
    def forward(self, image):
        x = (image - self.mean) / self.std
        x = x.to(self.dino.embeddings.patch_embeddings.weight.dtype)
        h = self.dino.embeddings(x, bool_masked_pos=None)
        pos = self.dino.rope_embeddings(x)
        for layer in self.dino.layer:
            h = layer(h, position_embeddings=pos)
        h = F.layer_norm(h, h.shape[-1:])
        n = self.image_size // self.patch
        tok = h[:, 1 + self.n_reg:].reshape(h.shape[0], n, n, -1).permute(0, 3, 1, 2)  # [B,1024,32,32]
        return self.naf(image.to(tok.dtype), tok, (self.T, self.T))  # [B,1024,T,T]


def _gn(c):
    return nn.GroupNorm(32, c)


class Head(nn.Module):
    """1x1 1024->128, GN, GELU; [concat RGB]; 3x3 ->128, GN, GELU; 3x3 dil 2, GN, GELU; 1x1 -> 4 logits."""

    def __init__(self, in_ch=1024, width=128, rgb_skip=True):
        super().__init__()
        self.rgb_skip = rgb_skip
        self.reduce = nn.Sequential(nn.Conv2d(in_ch, width, 1), _gn(width), nn.GELU())
        self.c1 = nn.Sequential(nn.Conv2d(width + (3 if rgb_skip else 0), width, 3, padding=1), _gn(width), nn.GELU())
        self.c2 = nn.Sequential(nn.Conv2d(width, width, 3, padding=2, dilation=2), _gn(width), nn.GELU())
        self.out = nn.Conv2d(width, 4, 1)

    def forward(self, feats, image):
        x = self.reduce(feats)
        if self.rgb_skip:
            x = torch.cat([x, F.interpolate(image, size=x.shape[-2:], mode="bilinear", align_corners=False)], 1)
        return self.out(self.c2(self.c1(x)))  # logits [B,4,T,T]


class AmodalMaskNet(nn.Module):
    def __init__(self, T=512, rgb_skip=True):
        super().__init__()
        self.features = FrozenFeatures(T=T)
        self.head = Head(self.features.dim, rgb_skip=rgb_skip)
        self.T = T

    def forward(self, image):
        """image [B,3,512,512] in [0,1] -> logits [B,4,T,T]."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=image.is_cuda):
            f = self.features(image)  # no_grad, frozen
            logits = self.head(f, image)  # bf16 compute keeps the 1024 x T x T map at half the memory
        return logits.float()

    @torch.no_grad()
    def predict_masks(self, image):
        return torch.sigmoid(self(image))


def head_state(model):
    return {"head": model.head.state_dict(), "config": {"T": model.T, "rgb_skip": model.head.rgb_skip, "dino": DINO}}
