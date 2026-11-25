"""P5 on the released 1.3B models (GPU): with zero-init LoRA the output equals the base model, a perturbed LoRA changes
it, and merging reproduces the LoRA output; for the dense stage 1 DiT and the sparse stage 2 DiT (SparseLinear
input/output layers included). Also reports peak memory of one forward+backward at the training batch.

    python p5_lora_check.py --out /workspace/out/p5
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sight_of_hand.b0 import lora  # noqa: E402
from train_b0 import STAGES, load_released  # noqa: E402


def stage_model(stage, px):
    import pixal3d.models as models
    cfg = json.load(open(Path(px) / "configs" / "gen" / STAGES[stage]["config"]))
    m = cfg["models"]["denoiser"]
    model = getattr(models, m["name"])(**m["args"]).cuda().eval()
    load_released(model, STAGES[stage]["weights"])
    return model


def inputs(stage, model, B=2, n_tok=3000):
    g = torch.Generator(device="cuda").manual_seed(0)
    t = torch.rand(B, device="cuda", generator=g) * 1000
    glob = torch.randn(B, 5, 1024, device="cuda", generator=g)
    if stage == "ss64":
        R = model.resolution
        x = torch.randn(B, model.in_channels, R, R, R, device="cuda", generator=g)
        proj = torch.randn(B, R ** 3, 1024, device="cuda", generator=g)
        return x, t, (glob, proj)
    import pixal3d.modules.sparse as sp
    R = model.resolution
    coords = []
    for b in range(B):
        c = torch.unique(torch.randint(0, R, (n_tok, 3), device="cuda", generator=g), dim=0)
        coords.append(torch.cat([torch.full((len(c), 1), b, device="cuda"), c], 1))
    coords = torch.cat(coords).int()
    x = sp.SparseTensor(torch.randn(len(coords), model.in_channels, device="cuda", generator=g), coords)
    proj = x.replace(torch.randn(len(coords), 2048, device="cuda", generator=g))
    return x, t, (glob, proj)


def feats(y):
    return y.feats if hasattr(y, "feats") else y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="p5")
    ap.add_argument("--stages", default="ss64,shape512")
    a = ap.parse_args()
    import os
    px = os.environ.get("PX", ".")
    sys.path.insert(0, px)
    res = {}
    for stage in a.stages.split(","):
        model = stage_model(stage, px)
        x, t, cond = inputs(stage, model)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            y0 = feats(model(x, t, cond)).float()
            names = lora.inject(model, 32, 32)
            y1 = feats(model(x, t, cond)).float()
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.normal_(0, 1e-3)
            y2 = feats(model(x, t, cond)).float()
        lora.freeze_base(model)
        tot, trn = lora.count(model)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", torch.bfloat16):
            loss = feats(model(x, t, cond)).float().pow(2).mean()
        loss.backward()
        grads_ok = all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad)
        base_nograd = all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        model.eval()
        lora.merge(model)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            y3 = feats(model(x, t, cond)).float()
        rel = lambda a_, b_: float((a_ - b_).norm() / b_.norm())  # noqa: E731
        res[stage] = {"lora_layers": len(names), "trainable": trn, "total": tot,
                      "zero_init_rel_diff": rel(y1, y0), "perturbed_rel_diff": rel(y2, y0),
                      "merged_vs_lora_rel_diff": rel(y3, y2), "lora_grads": grads_ok, "base_frozen": base_nograd,
                      "peak_mem_fwd_bwd_GiB_B2": peak}
        res[stage]["pass"] = (res[stage]["zero_init_rel_diff"] < 1e-6 and res[stage]["perturbed_rel_diff"] > 1e-4
                              and res[stage]["merged_vs_lora_rel_diff"] < 2e-2 and grads_ok and base_nograd)
        print(stage, json.dumps(res[stage], indent=1), flush=True)
        del model
        torch.cuda.empty_cache()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "p5.json").write_text(json.dumps(res, indent=1))
    if not all(r["pass"] for r in res.values()):
        sys.exit("P5 failed")


if __name__ == "__main__":
    main()
