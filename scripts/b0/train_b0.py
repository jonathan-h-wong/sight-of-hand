"""B0 LoRA training of one Pixal3D stage on the encoded DexYCB set (spec section 4). Run from a Pixal3D checkout.

Stages (stock config in Pixal3D/configs/gen, released weights in TencentARC/Pixal3D):
  ss64      ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json          <- ckpts/ss_flow_img_dit_1_3B_64_bf16
  shape512  slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune_ft512.json <- ckpts/slat_flow_img2shape_dit_1_3B_512_bf16
  shape1024 slat_flow_img2shape_dit_1_3B_512_bf16_proj_finetune_ft1024.json <- ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16

Changes from the stock config, and only these: num_views 1; no aesthetic filter (our metadata has no score column);
no finetune_ckpt (the released weights are loaded here, before LoRA wraps the layers); LoRA (rank, alpha) on every
block linear + input/output layers, base frozen; LR, EMA rate, batch, steps, save/log intervals from the flags; stock
snapshots off (i_sample -1; eval_b0.py samples instead). Checkpoints hold the LoRA tensors only.

    python train_b0.py --stage ss64 --data /workspace/b0 --out /workspace/out/ss64 --steps 20000 --wandb
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sight_of_hand.b0 import lora  # noqa: E402

STAGES = {
    "ss64": dict(config="ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json",
                 weights="TencentARC/Pixal3D/ckpts/ss_flow_img_dit_1_3B_64_bf16",
                 roots={"ss_latent": "ss_latents/ss_enc_conv3d_16l8_fp16_64_view"}, lr=1e-3, batch=8),
    "shape512": dict(config="slat_flow_img2shape_dit_1_3B_256_bf16_proj_finetune_ft512.json",
                     weights="TencentARC/Pixal3D/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16",
                     roots={"shape_latent": "shape_latents/shape_enc_next_dc_f16c32_fp16_512_view"}, lr=2e-4, batch=4),
    "shape1024": dict(config="slat_flow_img2shape_dit_1_3B_512_bf16_proj_finetune_ft1024.json",
                      weights="TencentARC/Pixal3D/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16",
                      roots={"shape_latent": "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"}, lr=2e-4, batch=2),
}


def build_cfg(a, px):
    st = STAGES[a.stage]
    cfg = json.load(open(Path(px) / "configs" / "gen" / st["config"]))
    ds = cfg["dataset"]["args"]
    ds["num_views"] = 1
    ds.pop("min_aesthetic_score", None)
    ds["load_camera_info"] = True
    if "image_size" in ds and a.image_size:
        ds["image_size"] = a.image_size
    tr = cfg["trainer"]["args"]
    tr.pop("finetune_ckpt", None)
    tr["optimizer"]["args"]["lr"] = a.lr or st["lr"]
    tr["ema_rate"] = [a.ema]
    tr["max_steps"] = a.steps
    tr["batch_size_per_gpu"] = a.batch or st["batch"]
    if a.batch_split:
        tr["batch_split"] = a.batch_split
    tr["i_sample"] = -1
    tr["i_save"] = a.save_every
    tr["i_log"] = a.log_every
    tr["i_print"] = a.log_every * 10
    tr["num_workers"] = a.workers
    if a.dino_size:
        tr["image_cond_model"]["args"]["image_size"] = a.dino_size
    cfg["models"]["denoiser"]["args"]["use_checkpoint"] = a.grad_ckpt
    cfg["b0"] = {"stage": a.stage, "weights": st["weights"], "lora_rank": a.rank, "lora_alpha": a.alpha,
                 "D_n": 1.0, "L_n_rule": "2 D_n tan(camera_angle_x/2), camera_angle_x = 2 atan(S/(2 fx))",
                 "scale_mode": "wrist-depth normalized, mesh_scale = 1/L_n", "data": a.data}
    return cfg


def load_released(model, path):
    """Load the released bf16 weights into the fp32 training model; every model key must be covered."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    p = path.split("/")
    sd = load_file(hf_hub_download(f"{p[0]}/{p[1]}", "/".join(p[2:]) + ".safetensors"))
    own = model.state_dict()
    missing = [k for k in own if k not in sd and "rope_phases" not in k]
    unexpected = [k for k in sd if k not in own]
    bad = [k for k in sd if k in own and sd[k].shape != own[k].shape]
    if missing or unexpected or bad:
        raise RuntimeError(f"weights mismatch: missing {missing[:5]} unexpected {unexpected[:5]} shape {bad[:5]}")
    model.load_state_dict(sd, strict=False)


def lora_only_checkpoints(trainer_cls):
    """Trainer saves/loads only the LoRA tensors (the trainable params); the base stays the released weights."""
    orig = trainer_cls._master_params_to_state_dicts

    def to_sd(self, master_params):
        sds = orig(self, master_params)
        return {n: {k: v for k, v in sd.items() if "lora_A" in k or "lora_B" in k} for n, sd in sds.items()}

    trainer_cls._master_params_to_state_dicts = to_sd


def patch_partial_load(model):
    full = model.load_state_dict

    def load(sd, strict=True):
        if sd and all(("lora_A" in k or "lora_B" in k) for k in sd):
            lora.load_lora(model, sd)
            return
        return full(sd, strict=strict)

    model.load_state_dict = load


def mixed_coords(pred_dir, frac, extra, swap_frac=0.0, swap_prob=0.0):
    """Stage 2 mixed training: with probability `frac` (when the frame has one) replace the GT 32^3 coordinates by the
    stage 1 prediction (gen_pred_coords.py). Voxels in both keep their GT latent. Extra voxels (predicted, not in GT)
    get the mean latent (0 after normalization); with extra='mask' they are left out of the loss, with extra='mean'
    they are trained toward it. Missed voxels (GT, not predicted) are dropped. A 33rd feature channel carries the
    per-token loss weight; masked_losses() strips it.
    Samples that do not use a prediction (no file, or the frac draw) get, with probability swap_prob, a synthetic
    boundary perturbation of the GT set: swap_frac of the GT voxels dropped and as many 6-neighbour non-GT voxels
    added (mean latent), the error pattern stage 1 makes."""
    import numpy as np
    from pixal3d.datasets import structured_latent_shape as sls
    orig = sls.SLatShapeView.get_instance
    pred_dir = Path(pred_dir)

    def get_instance(self, root, instance):
        out = orig(self, root, instance)
        c, f = out["coords"], out["feats"]
        w = torch.ones(len(c), 1)
        p = pred_dir / f"{instance}.npz"
        if np.random.rand() < frac and p.exists():
            P = torch.from_numpy(np.load(p)["coords"].astype(np.int32))
            if len(P):
                code = lambda x: (x[:, 0].long() * 32 + x[:, 1].long()) * 32 + x[:, 2].long()  # noqa: E731
                lut = torch.full((32 ** 3,), -1, dtype=torch.long)
                lut[code(c)] = torch.arange(len(c))
                j = lut[code(P)]
                hit = j >= 0
                f = torch.where(hit[:, None], f[j.clamp(min=0)], torch.zeros(len(P), f.shape[1]))
                w = (hit | torch.tensor(extra == "mean")).float()[:, None]
                c = P
        elif swap_frac > 0 and np.random.rand() < swap_prob and len(c) > 4:
            keep = torch.from_numpy(np.random.rand(len(c)) >= swap_frac)
            G_ = {tuple(x) for x in c.tolist()}
            offs = np.r_[np.eye(3, dtype=int), -np.eye(3, dtype=int)]
            nb = {tuple(x) for x in (c.numpy()[:, None, :] + offs[None]).reshape(-1, 3).tolist()}
            nb = np.array([x for x in nb if x not in G_ and min(x) >= 0 and max(x) < 32], np.int32).reshape(-1, 3)
            k = min(len(nb), int((~keep).sum()))
            add = torch.from_numpy(nb[np.random.choice(len(nb), k, replace=False)]) if k else torch.zeros(0, 3, dtype=torch.int32)
            f = torch.cat([f[keep], torch.zeros(k, f.shape[1])])
            w = torch.cat([torch.ones(int(keep.sum()), 1), torch.full((k, 1), float(extra == "mean"))])
            c = torch.cat([c[keep], add.to(c.dtype)])
        out["coords"], out["feats"] = c, torch.cat([f, w], 1)
        return out

    sls.SLatShapeView.get_instance = get_instance


def masked_losses(trainer_cls):
    """training_losses of the proj sparse trainer with a per-token loss weight in the last feature channel."""
    import numpy as np
    import torch.nn.functional as F
    from easydict import EasyDict as edict

    def training_losses(self, x_0, cond=None, **kwargs):
        w = x_0.feats[:, -1]
        x0 = x_0.replace(x_0.feats[:, :-1].contiguous())  # keeps the batch shape and the layout cache
        noise = x0.replace(torch.randn_like(x0.feats))
        t = self.sample_t(x0.shape[0]).to(x0.device).float()
        x_t = self.diffuse(x0, t, noise=noise)
        kwargs["coords"] = x0.coords
        cond = self.get_cond(cond, **kwargs)
        dk = {"concat_cond": kwargs["concat_cond"]} if "concat_cond" in kwargs else {}
        pred = self.training_models["denoiser"](x_t, t * 1000, cond, **dk)
        target = self.get_v(x0, noise, t)
        se = ((pred.feats - target.feats) ** 2).mean(1)
        terms = edict()
        terms["mse"] = (se * w).sum() / w.sum().clamp(min=1)
        terms["loss"] = terms["mse"]
        terms["frac_weighted"] = w.mean()
        per = np.array([((se[l] * w[l]).sum() / w[l].sum().clamp(min=1)).item() for l in x0.layout])
        tb = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (tb == i).sum():
                terms[f"bin_{i}"] = {"mse": per[tb == i].mean()}
        return terms, {}

    trainer_cls.training_losses = training_losses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--data", required=True, help="encode_b0.py output root holding <split>/")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--batch-split", type=int)
    ap.add_argument("--image-size", type=int)
    ap.add_argument("--dino-size", type=int)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pred-coords", help="dir of stage 1 predicted coords (<key>.npz) for stage 2 mixed training")
    ap.add_argument("--pred-frac", type=float, default=0.5, help="share of samples that use predicted coords")
    ap.add_argument("--extra-target", choices=["mask", "mean"], default="mask")
    ap.add_argument("--swap-frac", type=float, default=0.0, help="synthetic boundary swap share for non-pred samples")
    ap.add_argument("--swap-prob", type=float, default=0.0, help="probability of the synthetic swap for non-pred samples")
    ap.add_argument("--subset", help="text file of sample keys to keep (P4 overfit)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="sight-of-hand-b0")
    ap.add_argument("--run-name")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    px = os.environ.get("PX", ".")
    sys.path.insert(0, px)
    from easydict import EasyDict as edict
    from pixal3d import datasets, models, trainers

    cfg = build_cfg(a, px)
    os.makedirs(a.out, exist_ok=True)
    root = Path(a.data) / a.split
    roots = {"dexycb": {"render_cond": str(root / "renders_cond"),
                        **{k: str(root / v) for k, v in STAGES[a.stage]["roots"].items()}}}
    json.dump(cfg, open(Path(a.out) / "config.json", "w"), indent=1)

    wandb_run = None
    if a.wandb:
        import wandb
        wandb_run = wandb.init(project=a.wandb_project, name=a.run_name or Path(a.out).name, config=cfg,
                               dir=a.out, resume="allow", id=(a.run_name or Path(a.out).name).replace("/", "_"))

    if a.pred_coords:
        assert a.stage.startswith("shape"), "--pred-coords is for stage 2"
        mixed_coords(a.pred_coords, a.pred_frac, a.extra_target, a.swap_frac, a.swap_prob)
        cfg["b0"]["mixed"] = {"pred_coords": a.pred_coords, "pred_frac": a.pred_frac, "extra_target": a.extra_target,
                              "swap_frac": a.swap_frac, "swap_prob": a.swap_prob}
        json.dump(cfg, open(Path(a.out) / "config.json", "w"), indent=1)
    ds = getattr(datasets, cfg["dataset"]["name"])(json.dumps(roots), **cfg["dataset"]["args"])
    if a.subset:
        keep = set(Path(a.subset).read_text().split())
        ds.instances = [x for x in ds.instances if x[1] in keep]
        if hasattr(ds, "loads"):
            ds.loads = [ds.loads[0]] * len(ds.instances)
    print(f"dataset {cfg['dataset']['name']}: {len(ds)} samples", flush=True)

    mcfg = cfg["models"]["denoiser"]
    model = getattr(models, mcfg["name"])(**mcfg["args"]).cuda()
    load_released(model, STAGES[a.stage]["weights"])
    names = lora.inject(model, a.rank, a.alpha)
    lora.freeze_base(model)
    patch_partial_load(model)
    tot, trn = lora.count(model)
    print(f"LoRA r={a.rank} a={a.alpha}: {len(names)} layers, {trn:,} trainable of {tot:,}", flush=True)
    json.dump({"layers": names, "trainable": trn, "total": tot}, open(Path(a.out) / "lora_layers.json", "w"), indent=1)

    tcls = getattr(trainers, cfg["trainer"]["name"])
    lora_only_checkpoints(tcls)
    if a.pred_coords:
        masked_losses(tcls)
    step = None
    if a.resume:
        import glob
        f = glob.glob(os.path.join(a.out, "ckpts", "misc_step*.pt"))
        step = max(int(Path(x).stem.split("step")[-1]) for x in f) if f else None
    trainer = tcls({"denoiser": model}, ds, **copy.deepcopy(cfg["trainer"]["args"]), output_dir=a.out,
                   load_dir=a.out, step=step, wandb_run=wandb_run)
    if trainer.is_master and trainer.writer is None:
        class _W:  # stock run() closes a TensorBoard writer it never creates
            def close(self):
                pass

            def add_scalar(self, *a, **k):
                pass
        trainer.writer = _W()
    trainer.run()
    trainer.save(non_blocking=False)
    torch.save(lora.lora_state_dict(model), Path(a.out) / "lora_final_raw.pt")
    if trainer.is_master:
        ema = trainer._master_params_to_state_dicts(trainer.ema_params[0])["denoiser"]
        torch.save({k: v.detach().cpu() for k, v in ema.items()}, Path(a.out) / f"lora_final_ema{a.ema}.pt")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
