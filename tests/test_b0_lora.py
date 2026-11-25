"""LoRA wrapper (spec P5, CPU part): zero-init output equals the base, the sparse path works, merge is exact."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from sight_of_hand.b0 import lora  # noqa: E402


class VarLen:
    """Stand-in for pixal3d's VarLenTensor: .feats and .replace."""
    def __init__(self, feats):
        self.feats = feats

    def replace(self, f):
        return VarLen(f)


class SparseLinear(nn.Linear):
    def forward(self, x):
        return x.replace(super().forward(x.feats))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.to_qkv = nn.Linear(c, 3 * c)
        self.mlp = nn.Sequential(nn.Linear(3 * c, 4 * c), nn.GELU(), nn.Linear(4 * c, c))
        self.adaLN_modulation = nn.Linear(c, 6 * c)

    def forward(self, x):
        return x + self.mlp(self.to_qkv(x))


class Toy(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.input_layer = SparseLinear(8, c)
        self.blocks = nn.ModuleList([Block(c) for _ in range(2)])
        self.out_layer = SparseLinear(c, 8)
        self.t_embedder = nn.Linear(4, c)

    def forward(self, x):
        h = self.input_layer(x)
        f = h.feats
        for b in self.blocks:
            f = b(f)
        return self.out_layer(h.replace(f))


def test_targets_zero_init_and_merge():
    torch.manual_seed(0)
    m = Toy()
    x = VarLen(torch.randn(5, 8))
    y0 = m(x).feats
    names = lora.inject(m, rank=4, alpha=4)
    assert "input_layer" in names and "out_layer" in names
    assert not any("adaLN" in n or "t_embedder" in n for n in names)
    assert len(names) == 2 + 2 * 3
    assert torch.allclose(m(x).feats, y0)                      # B = 0 at init
    lora.freeze_base(m)
    sd = lora.lora_state_dict(m)
    assert set(sd) == {n for n, p in m.named_parameters() if p.requires_grad}
    for k, v in m.named_parameters():
        if "lora_B" in k:
            v.data.normal_()
    y1 = m(x).feats
    assert not torch.allclose(y1, y0)
    lora.merge(m)
    assert not any(isinstance(mm, lora.LoRALinear) for mm in m.modules())
    assert torch.allclose(m(x).feats, y1, atol=1e-5)


def test_load_roundtrip():
    torch.manual_seed(0)
    a, b = Toy(), Toy()
    b.load_state_dict(a.state_dict())
    lora.inject(a, 4, 8)
    lora.inject(b, 4, 8)
    for k, v in a.named_parameters():
        if "lora_B" in k:
            v.data.normal_()
    lora.load_lora(b, lora.lora_state_dict(a))
    x = VarLen(torch.randn(3, 8))
    assert torch.allclose(a(x).feats, b(x).feats)
