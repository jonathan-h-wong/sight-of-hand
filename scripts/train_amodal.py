"""Train the amodal mask predictor head (docs/amodal_branch_spec.md) on the DexYCB s0 shards, with W&B logging.

    python scripts/train_amodal.py --data /data/amodal --steps 20000 --batch 8 --T 512
    python scripts/train_amodal.py --data /data/amodal --fold A   # head for fold-A predictions: trains on fold B

--fold k trains on the train sequences NOT in fold k (cross-fitting: fold k's cached predictions come from a head that
never saw it). Model selection uses s0 val; s0 test is never loaded here.
Checkpoints (head weights + config only; DINOv3 and NAF are not saved) go to --ckpt-dir and, with --s3-prefix, to
s3://xray-hands/<prefix>/<run>/: last.pt every validation, best.pt on the best val hidden_iou/mean.
W&B: project --wandb-project (default sight-of-hand-amodal); needs WANDB_API_KEY in the environment.
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
from sight_of_hand.amodal_net.metrics import Evaluator, loss_fn  # noqa: E402
from sight_of_hand.amodal_net.model import AmodalMaskNet, head_state  # noqa: E402


def overlay(x, y, p):
    """One panel row per sample: image | GT (amodal outline over modal fill) | prediction, for hand and object."""
    img = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    T = y.shape[-1]
    if img.shape[0] != T:
        from PIL import Image
        img = np.asarray(Image.fromarray(img).resize((T, T)))

    def paint(m):
        a = img.astype(np.float32) * 0.5
        a[..., 0] += 127 * m[0]  # amodal hand -> red
        a[..., 2] += 127 * m[1]  # amodal object -> blue
        a[..., 1] += 100 * (m[2] + m[3])  # modal (either part) -> green tint
        return a.clip(0, 255).astype(np.uint8)
    yy, pp = y.cpu().numpy() > 0.5, p.cpu().numpy()
    return np.concatenate([img, paint(yy.astype(np.float32)), paint(pp)], 1)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0):
    model.eval()
    ev = Evaluator()
    for b, (x, y, meta) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        lg = model(x)
        if lg.shape[-1] != y.shape[-1]:
            y = torch.nn.functional.interpolate(y, size=lg.shape[-2:], mode="nearest")
        ev.add(lg, y, {"hand": meta["hand_hidden_frac"].to(device), "obj": meta["obj_hidden_frac"].to(device)})
    model.head.train()
    return ev.result()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="dir with the downloaded shards/ and index.parquet")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--no-rgb-skip", action="store_true")
    ap.add_argument("--fold", choices=["A", "B"], help="train the cross-fitting head for this fold (uses the other)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--hidden-weight", type=float, default=5.0)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-batches", type=int, default=125, help="batches per periodic val (0 = all of s0 val)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ckpt-dir", type=Path, default=Path("runs"))
    ap.add_argument("--s3-prefix", default="work/amodal_net/runs")
    ap.add_argument("--wandb-project", default="sight-of-hand-amodal")
    ap.add_argument("--run-name")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    shards = sorted((a.data / "shards").glob("*.tar"))
    items = D.index_shards([p for p in shards if p.name.startswith(("train_", "val_"))])
    train = [it for it in items if it["meta"]["split"] == "train" and (a.fold is None or it["meta"]["fold"] != a.fold)]
    val = [it for it in items if it["meta"]["split"] == "val"]
    val.sort(key=lambda it: it["key"])
    print(f"train {len(train)} samples ({len({it['meta']['seq'] for it in train})} seqs), val {len(val)}", flush=True)
    dl = DataLoader(D.AmodalShards(train, True), batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    collate_fn=D.collate, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0)
    vdl = DataLoader(D.AmodalShards(val, False), batch_size=a.batch, shuffle=False, num_workers=a.workers,
                     collate_fn=D.collate, pin_memory=True)
    vis = D.collate([D.AmodalShards(val, False)[i] for i in np.linspace(0, len(val) - 1, 8).astype(int)])

    model = AmodalMaskNet(T=a.T, rgb_skip=not a.no_rgb_skip).to(device)
    n_params = sum(p.numel() for p in model.head.parameters())
    opt = torch.optim.AdamW(model.head.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))

    run = a.run_name or f"T{a.T}{'' if not a.fold else '_fold' + a.fold}{'_norgb' if a.no_rgb_skip else ''}_{time.strftime('%m%d_%H%M')}"
    cfg = vars(a) | {"run": run, "head_params": n_params, "train_samples": len(train), "val_samples": len(val),
                     "data_index": str(a.data)}
    import wandb
    wandb.init(project=a.wandb_project, name=run, config={k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()})
    print(f"run {run}: head {n_params / 1e6:.2f}M params; wandb {wandb.run.url}", flush=True)
    ck = a.ckpt_dir / run
    ck.mkdir(parents=True, exist_ok=True)

    def save(name, extra):
        torch.save(head_state(model) | {"step": step, "metrics": extra, "args": cfg}, ck / name)
        if a.s3_prefix:
            import boto3
            boto3.client("s3").upload_file(str(ck / name), "xray-hands", f"{a.s3_prefix}/{run}/{name}")

    best, step, t0, seen = -1.0, 0, time.time(), 0
    model.head.train()
    it = iter(dl)
    while step < a.steps:
        try:
            x, y, meta = next(it)
        except StopIteration:
            it = iter(dl)
            x, y, meta = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        lg = model(x)
        if lg.shape[-1] != y.shape[-1]:
            y = torch.nn.functional.interpolate(y, size=lg.shape[-2:], mode="nearest")
        loss, parts = loss_fn(lg, y, a.hidden_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        seen += x.shape[0]
        if step % 20 == 0:
            dt = time.time() - t0
            wandb.log({"train/loss": loss.item(), **{f"train/{k}": v for k, v in parts.items()},
                       "lr": sched.get_last_lr()[0], "throughput/img_per_s": seen / dt,
                       "mem/max_gb": torch.cuda.max_memory_allocated() / 1e9}, step=step)
        if step % a.val_every == 0 or step == a.steps:
            full = step == a.steps
            res = evaluate(model, vdl, device, 0 if full else a.val_batches)
            with torch.no_grad():
                model.eval()
                pv = torch.sigmoid(model(vis[0].to(device)))
                model.head.train()
            yv = vis[1].to(device)
            if yv.shape[-1] != pv.shape[-1]:
                yv = torch.nn.functional.interpolate(yv, size=pv.shape[-2:], mode="nearest")
            panels = [wandb.Image(overlay(vis[0][i], yv[i], pv[i]), caption=f"val {i}: image | GT | pred (red hand, blue object, green modal)") for i in range(len(pv))]
            wandb.log({("val_full/" if full else "val/") + k: v for k, v in res.items()} | {"val/overlays": panels}, step=step)
            print(f"step {step}: loss {loss.item():.3f} val hidden_iou hand {res['hidden_iou/hand']:.3f} obj {res['hidden_iou/obj']:.3f} "
                  f"(own-modal baseline {res['baseline_own_modal/hidden_iou/hand']:.3f}/{res['baseline_own_modal/hidden_iou/obj']:.3f}) "
                  f"iou amodal {res['iou/amodal_hand']:.3f}/{res['iou/amodal_obj']:.3f}  {time.time() - t0:.0f}s", flush=True)
            save("last.pt", res)
            if res["hidden_iou/mean"] > best:
                best = res["hidden_iou/mean"]
                save("best.pt", res)
                wandb.run.summary["best/hidden_iou_mean"] = best
                wandb.run.summary["best/step"] = step
    (ck / "final_metrics.json").write_text(json.dumps(res, indent=1))
    wandb.finish()


if __name__ == "__main__":
    main()
