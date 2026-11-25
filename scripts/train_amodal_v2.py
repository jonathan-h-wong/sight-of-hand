"""Train the v2 amodal head (docs/amodal_branch_spec_v2.md) on the DexYCB s0 shards, with W&B logging.

    python scripts/train_amodal_v2.py --data /data/amodal --exp E2 --taps 5,11,17,23 --decoder naf
    python scripts/train_amodal_v2.py --data /data/amodal --exp E1 --taps 23
    python scripts/train_amodal_v2.py --data /data/amodal --exp E3lite --decoder lite
    python scripts/train_amodal_v2.py --data /data/amodal --exp E4 --joints     # needs <data>/joints_2d.parquet

At start: (1) NAF backward check (one real forward/backward; if gradients cannot flow through NAF, training falls back
to bilinear upsampling for the loss and NAF is used only at evaluation, as the spec allows; logged); (2) with
--diagnostics, the 32x32 resolution ceiling on all of s0 val (bilinear and NAF) and per-component cost (ms per image,
peak memory) at the run's T. Every --val-every steps: an evenly spread val sample and a train sample of the same size
(train/val gap), overlays to W&B, last.pt; best.pt on val hidden_iou_occ10/mean (frames >= 10% occluded, the spec's
stopping signal). At the end: all of s0 val (val_full/*). s0 test is never loaded.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from sight_of_hand.amodal_net import data as D  # noqa: E402
from sight_of_hand.amodal_net.metrics_v2 import EvaluatorV2, loss_v2, resolution_ceiling  # noqa: E402
from sight_of_hand.amodal_net.model_v2 import AmodalMaskNetV2, Config, head_state  # noqa: E402


def overlay(x, y, p):
    img = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    T = y.shape[-1]
    if img.shape[0] != T:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((T, T)))

    def paint(m):
        a = img.astype(np.float32) * 0.5
        a[..., 0] += 127 * m[0]
        a[..., 2] += 127 * m[1]
        a[..., 1] += 100 * (m[2] + m[3])
        return a.clip(0, 255).astype(np.uint8)
    return np.concatenate([img, paint((y.cpu().numpy() > 0.5).astype(np.float32)), paint(p.cpu().numpy())], 1)


def fit(y, T):
    return y if y.shape[-1] == T else torch.nn.functional.interpolate(y, size=(T, T), mode="nearest")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ev = EvaluatorV2()
    for x, y, meta in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        lg = model(x)["logits"]
        ev.add(lg, fit(y, lg.shape[-1]), {"hand": meta["hand_hidden_frac"].to(device), "obj": meta["obj_hidden_frac"].to(device)})
    model.train()
    return ev.result()


def naf_backward_check(model, x, y, meta):
    """One forward/backward through the NAF path; returns (ok, message)."""
    if model.naf is None:
        return True, "no NAF in this decoder"
    try:
        model.train()
        out = model(x)
        loss, _ = loss_v2(out, fit(y, out["logits"].shape[-1]), meta)
        loss.backward()
        g = model.logits32.weight.grad if model.cfg.decoder == "naf" else model.out128.weight.grad
        ok = g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0
        model.zero_grad(set_to_none=True)
        return ok, f"grad norm {0 if g is None else g.norm().item():.3e}"
    except Exception as e:  # noqa: BLE001
        model.zero_grad(set_to_none=True)
        return False, repr(e)[:300]


@torch.no_grad()
def cost(model, x):
    """ms per image and peak memory per component at batch len(x) (forward only, bf16)."""
    torch.cuda.synchronize()
    res = {}

    def timed(name, fn):
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            r = fn()
        torch.cuda.synchronize()
        res[f"cost/{name}_ms_per_img"] = (time.time() - t) / 5 / len(x) * 1000
        res[f"cost/{name}_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return r
    with torch.autocast("cuda", dtype=torch.bfloat16):
        taps = timed("dino_taps", lambda: model.backbone(x))
    model.eval()
    timed("full_forward", lambda: model(x))
    if model.naf is not None:
        lg = torch.randn(len(x), 4, 32 if model.cfg.decoder == "naf" else 128, 32 if model.cfg.decoder == "naf" else 128,
                         device=x.device)
        timed("naf", lambda: model.upsample(lg, x, model.cfg.T))
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--exp", required=True, help="experiment id (E1, E2, E3lite, E3full, E4, ...)")
    ap.add_argument("--taps", default="5,11,17,23")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pos-emb", default="learned", choices=["learned", "sinusoidal", "none"])
    ap.add_argument("--decoder", default="naf", choices=["naf", "lite", "full"])
    ap.add_argument("--joints", action="store_true")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--hidden-weight", type=float, default=5.0)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-images", type=int, default=1000)
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--no-full-val", action="store_true", help="skip the all-of-val evaluation at the end (smoke tests)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("runs"))
    ap.add_argument("--s3-prefix", default="work/amodal_v2/runs")
    ap.add_argument("--wandb-project", default="sight-of-hand-amodal")
    ap.add_argument("--run-name")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    joints = D.load_joints(a.data / "joints_2d.parquet") if a.joints else None
    items = D.index_shards(sorted(p for p in (a.data / "shards").glob("*.tar") if p.name.startswith(("train_", "val_"))))
    train = [it for it in items if it["meta"]["split"] == "train"]
    val = sorted([it for it in items if it["meta"]["split"] == "val"], key=lambda it: it["key"])
    tr_sub = [train[i] for i in np.linspace(0, len(train) - 1, min(len(train), a.val_images)).astype(int)]
    va_sub = [val[i] for i in np.linspace(0, len(val) - 1, min(len(val), a.val_images)).astype(int)]
    print(f"train {len(train)} samples, val {len(val)}; joints: {len(joints) if joints else 0}", flush=True)
    kw = dict(collate_fn=D.collate, pin_memory=True, num_workers=a.workers)
    dl = DataLoader(D.AmodalShards(train, True, joints=joints), batch_size=a.batch, shuffle=True, drop_last=True,
                    persistent_workers=a.workers > 0, **kw)
    mk = lambda its: DataLoader(D.AmodalShards(its, False), batch_size=a.batch, shuffle=False, **kw)  # noqa: E731
    vdl_full, vdl_sub, tdl_sub = mk(val), mk(va_sub), mk(tr_sub)
    vis = D.collate([D.AmodalShards(val, False)[i] for i in np.linspace(0, len(val) - 1, 8).astype(int)])

    cfg = Config(taps=tuple(int(t) for t in a.taps.split(",")), width=a.width, depth=a.depth, heads=a.heads,
                 pos_emb=a.pos_emb, decoder=a.decoder, joints=a.joints, T=a.T)
    model = AmodalMaskNetV2(cfg).to(device)
    params = model.trainable()
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))

    x0, y0, m0 = next(iter(dl))
    x0, y0 = x0.to(device), y0.to(device)
    naf_ok, naf_msg = naf_backward_check(model, x0[:4], y0[:4], {k: v[:4] for k, v in m0.items()})
    model.naf_ok = naf_ok
    print(f"NAF backward: {'ok' if naf_ok else 'UNAVAILABLE -> bilinear for training'} ({naf_msg})", flush=True)

    run = a.run_name or f"v2_{a.exp}_{time.strftime('%m%d_%H%M')}"
    meta_cfg = vars(a) | {"run": run, "head_params": n_params, "model": cfg.to_dict(), "naf_backward": naf_ok,
                          "naf_backward_msg": naf_msg, "train_samples": len(train), "val_samples": len(val)}
    import wandb
    wandb.init(project=a.wandb_project, name=run, group="v2", config={k: str(v) if isinstance(v, Path) else v for k, v in meta_cfg.items()})
    print(f"run {run}: {n_params / 1e6:.2f}M trainable params; wandb {wandb.run.url if wandb.run else None}", flush=True)
    ck = a.ckpt_dir / run
    ck.mkdir(parents=True, exist_ok=True)

    if a.diagnostics and device == "cuda":
        d = cost(model, x0)
        model.eval()
        evc = {}
        with torch.no_grad():
            for x, y, _ in vdl_full:
                x, y = x.to(device), y.to(device)
                for k, v in resolution_ceiling(fit(y, a.T), x, model.upsample if model.naf is not None else None).items():
                    evc.setdefault(k, []).append(v.cpu())
        model.train()
        d |= {k: float(torch.nanmean(torch.cat(v))) for k, v in evc.items()}
        wandb.log({f"diag/{k}": v for k, v in d.items()}, step=0)
        (ck / "diagnostics.json").write_text(json.dumps(d, indent=1))
        print("diagnostics", json.dumps(d), flush=True)

    def save(name, extra):
        torch.save(head_state(model) | {"step": step, "metrics": extra, "args": {k: str(v) for k, v in vars(a).items()}},
                   ck / name)
        if a.s3_prefix:
            import boto3
            boto3.client("s3").upload_file(str(ck / name), "xray-hands", f"{a.s3_prefix}/{run}/{name}")

    best, step, t0, seen = -1.0, 0, time.time(), 0
    model.train()
    it = iter(dl)
    res = {}
    while step < a.steps:
        try:
            x, y, meta = next(it)
        except StopIteration:
            it = iter(dl)
            x, y, meta = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss, parts = loss_v2(out, fit(y, out["logits"].shape[-1]), meta, a.hidden_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        seen += x.shape[0]
        if step % 20 == 0:
            wandb.log({"train/loss": loss.item(), **{f"train/{k}": v for k, v in parts.items()},
                       "lr": sched.get_last_lr()[0], "throughput/img_per_s": seen / (time.time() - t0),
                       "mem/max_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0}, step=step)
        if step % a.val_every == 0 or step == a.steps:
            full = step == a.steps and not a.no_full_val
            res = evaluate(model, vdl_full if full else vdl_sub, device)
            tr = evaluate(model, tdl_sub, device)
            with torch.no_grad():
                model.eval()
                pv = torch.sigmoid(model(vis[0].to(device))["logits"])
                model.train()
            yv = fit(vis[1].to(device), pv.shape[-1])
            panels = [wandb.Image(overlay(vis[0][i], yv[i], pv[i]), caption=f"val {i}: image | GT | pred") for i in range(len(pv))]
            gap = {f"gap/{k}": tr[k] - res[k] for k in ("hidden_iou/mean", "modal_iou/mean", "hidden_iou_occ10/mean")}
            wandb.log({("val_full/" if full else "val/") + k: v for k, v in res.items()}
                      | {f"train_eval/{k}": v for k, v in tr.items()} | gap | {"val/overlays": panels}, step=step)
            print(f"step {step}: loss {loss.item():.3f} | val modal {res['modal_iou/mean']:.3f} hidden {res['hidden_iou/hand']:.3f}/"
                  f"{res['hidden_iou/obj']:.3f} occ10 {res['hidden_iou_occ10/mean']:.3f} 10-60 {res['hidden_iou_10to60/mean']:.3f} | "
                  f"train-val gap hidden {gap['gap/hidden_iou/mean']:+.3f}  {time.time() - t0:.0f}s", flush=True)
            save("last.pt", res)
            if not full and res["hidden_iou_occ10/mean"] > best:
                best = res["hidden_iou_occ10/mean"]
                save("best.pt", res)
                wandb.run.summary["best/hidden_iou_occ10_mean"] = best
                wandb.run.summary["best/step"] = step
    (ck / "final_metrics.json").write_text(json.dumps(res, indent=1))
    if a.s3_prefix:
        import boto3
        boto3.client("s3").upload_file(str(ck / "final_metrics.json"), "xray-hands", f"{a.s3_prefix}/{run}/final_metrics.json")
    print(f"RUN_NAME={run}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
