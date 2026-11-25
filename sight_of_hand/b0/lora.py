"""LoRA for Pixal3D's dense and sparse DiTs (spec section 4).

Targets: every nn.Linear inside the transformer blocks (self-attention qkv/out, cross-attention q/kv/out,
`proj_linear`, the MLP) plus the model's `input_layer` and `out_layer`. Not targeted: the shared adaLN modulation and
the timestep embedder. `SparseLinear` (an nn.Linear subclass whose forward takes a VarLenTensor) is handled by the
same wrapper: it applies the base weight with F.linear on `.feats` and re-wraps with `.replace`.

    model = load_base(...)                 # pretrained weights first, while the key names are stock
    names = inject(model, rank=32, alpha=32)
    freeze_base(model)                     # only lora_A / lora_B train
    lora_state_dict(model)                 # what the trainer saves (requires_grad params) is exactly this
    merge(model)                           # fold B A into the base weights, restoring stock modules
"""
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_INCLUDE = r"(^|\.)blocks\.\d+\..*|^input_layer$|^out_layer$"
DEFAULT_EXCLUDE = r"adaLN_modulation|t_embedder"


def _autocast_dtype(x):
    dev = x.device.type
    try:
        if torch.is_autocast_enabled(dev):
            return torch.get_autocast_dtype(dev)
    except TypeError:  # older torch
        if dev == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return x.dtype


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.merged = False

    @property
    def weight(self):  # some code reads .weight (init, dtype checks)
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def _f(self, x):
        y = F.linear(x, self.base.weight, self.base.bias)
        if self.merged:
            return y
        dt = _autocast_dtype(x)
        h = F.linear(F.linear(x.to(dt), self.lora_A.to(dt)), self.lora_B.to(dt))
        return y + (h * self.scale).to(y.dtype)

    def forward(self, x):
        if hasattr(x, "feats") and hasattr(x, "replace"):
            return x.replace(self._f(x.feats))
        return self._f(x)


def _targets(model, include, exclude):
    inc, exc = re.compile(include), re.compile(exclude)
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and inc.search(n) and not exc.search(n)]


def _set(model, name, new):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, leaf, new)


def inject(model, rank=32, alpha=32, include=DEFAULT_INCLUDE, exclude=DEFAULT_EXCLUDE):
    names = _targets(model, include, exclude)
    if not names:
        raise ValueError("no LoRA targets matched")
    for n in names:
        _set(model, n, LoRALinear(model.get_submodule(n), rank, alpha))
    return names


def freeze_base(model):
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A" in n) or ("lora_B" in n)


def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def load_lora(model, sd, strict=True):
    own = {k for k in model.state_dict() if "lora_A" in k or "lora_B" in k}
    missing, extra = own - set(sd), set(sd) - own
    if strict and (missing or extra):
        raise KeyError(f"LoRA keys: missing {sorted(missing)[:5]}..., unexpected {sorted(extra)[:5]}...")
    model.load_state_dict(sd, strict=False)


@torch.no_grad()
def merge(model):
    """Fold every LoRA into its base Linear and put the base module back in place."""
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for n in names:
        m = model.get_submodule(n)
        delta = (m.lora_B.float() @ m.lora_A.float()) * m.scale
        m.base.weight.add_(delta.to(m.base.weight.dtype))
        _set(model, n, m.base)
    return names


def count(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, tr
