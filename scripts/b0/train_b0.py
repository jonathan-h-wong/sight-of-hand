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


def mixed_coords(pred_dir, frac, extra, swap_frac=0.0, swap_prob=0.0, dilate_prob=0.0, hand_dir=None):
    """Stage 2 coordinate augmentation. Each sample, in this order of precedence:
      prediction  with probability `frac` (when the frame has one) the stage 1 predicted 32^3 set (gen_pred_coords.py)
      swap        else with probability swap_prob a synthetic boundary swap of the GT set (swap_frac dropped, as many
                  6-neighbour non-GT voxels added)
      dilate      else with probability dilate_prob the GT set dilated by one voxel (all 6-neighbours added)
      clean       else the GT set.
    Voxels in GT keep their GT latent; added voxels get the mean latent (0 after normalization) and are trained toward
    it with extra='mean' or left out of the loss with extra='mask'; missed voxels are dropped. Two feature channels are
    appended: the per-token loss weight and a hand flag (the 32^3 cell contains a GT hand cell from
    gen_hand_vox.py; 0 without hand_dir). masked_losses() strips them."""
    import numpy as np
    from pixal3d.datasets import structured_latent_shape as sls
    orig = sls.SLatShapeView.get_instance
    pred_dir = Path(pred_dir) if pred_dir else None
    hand_dir = Path(hand_dir) if hand_dir else None
    offs = np.r_[np.eye(3, dtype=int), -np.eye(3, dtype=int)]

    def neighbours(c):
        G_ = {tuple(x) for x in c.tolist()}
        nb = {tuple(x) for x in (c.numpy()[:, None, :] + offs[None]).reshape(-1, 3).tolist()}
        return np.array([x for x in nb if x not in G_ and min(x) >= 0 and max(x) < 32], np.int32).reshape(-1, 3)

    def get_instance(self, root, instance):
        out = orig(self, root, instance)
        c, f = out["coords"], out["feats"]
        w = torch.ones(len(c), 1)
        ew = float(extra == "mean")
        p = pred_dir / f"{instance}.npz" if pred_dir else None
        if p is not None and np.random.rand() < frac and p.exists() and len(np.load(p)["coords"]):
            P = torch.from_numpy(np.load(p)["coords"].astype(np.int32))
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
            nb = neighbours(c)
            k = min(len(nb), int((~keep).sum()))
            add = torch.from_numpy(nb[np.random.choice(len(nb), k, replace=False)]) if k else torch.zeros(0, 3, dtype=torch.int32)
            f = torch.cat([f[keep], torch.zeros(k, f.shape[1])])
            w = torch.cat([torch.ones(int(keep.sum()), 1), torch.full((k, 1), ew)])
            c = torch.cat([c[keep], add.to(c.dtype)])
        elif dilate_prob > 0 and np.random.rand() < dilate_prob:
            add = torch.from_numpy(neighbours(c))
            f = torch.cat([f, torch.zeros(len(add), f.shape[1])])
            w = torch.cat([w, torch.full((len(add), 1), ew)])
            c = torch.cat([c, add.to(c.dtype)])
        hand = torch.zeros(len(c), 1)
        if hand_dir is not None and (hand_dir / f"{instance}.npz").exists():
            h = np.load(hand_dir / f"{instance}.npz")["coords"].astype(int) // 2
            hs = np.zeros(32 ** 3, bool)
            hs[(h[:, 0] * 32 + h[:, 1]) * 32 + h[:, 2]] = True
            cc = c.long().numpy()
            hand = torch.from_numpy(hs[(cc[:, 0] * 32 + cc[:, 1]) * 32 + cc[:, 2]].astype(np.float32))[:, None]
        out["coords"], out["feats"] = c, torch.cat([f, w, hand], 1)
        return out

    sls.SLatShapeView.get_instance = get_instance


def _ramp(self, occ):
    return occ["lam"] * min(1.0, self.step / max(occ["ramp"], 1))


def _dice(p, y, eps=1.0):
    return 1 - (2 * (p * y).sum() + eps) / (p.sum() + y.sum() + eps)


def masked_losses(trainer_cls, occ=None):
    """training_losses of the proj sparse trainer (stage 2) with a per-token loss weight and a hand flag in the last
    two feature channels. With occ (decoded-occupancy loss): x0_hat = x_t - t v_hat (exact for sigma_min -> 0) goes
    through the frozen shape decoder's first block level, whose subdivision logits (which of the 8 64^3 children of
    each 32^3 cell are occupied) are trained by BCE + Dice toward those of the GT latent, per sample weighted by
    (1 - t)^2, hand tokens up-weighted by occ['hand_w'], coefficient ramped from 0 to occ['lam'] over occ['ramp'] steps."""
    import numpy as np
    import torch.nn.functional as F
    from easydict import EasyDict as edict

    def level0(dec, x):
        h = dec.from_latent(x)
        h = h.type(dec.dtype)
        res = dec.blocks[0]
        for j, block in enumerate(res):
            if j == len(res) - 1:
                h, sub = block(h)
            else:
                h = block(h)
        return sub.feats.float()

    def training_losses(self, x_0, cond=None, **kwargs):
        w, hand = x_0.feats[:, -2], x_0.feats[:, -1]
        x0 = x_0.replace(x_0.feats[:, :-2].contiguous())  # keeps the batch shape and the layout cache
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
        if occ is not None and occ["lam"] > 0:
            dec, mean, std = occ["decoder"], occ["mean"], occ["std"]
            tt = torch.cat([t[i].expand(l.stop - l.start) for i, l in enumerate(x0.layout)])
            x0h = x_t.feats - tt[:, None] * pred.feats
            with torch.no_grad():
                tgt = (level0(dec, x0.replace(x0.feats * std + mean)) > 0).float()
            lg = level0(dec, x0.replace(x0h * std + mean))
            tw = (1 - tt) ** 2 * (1 + (occ["hand_w"] - 1) * hand)
            bce = F.binary_cross_entropy_with_logits(lg, tgt, reduction="none").mean(1)
            pr = torch.sigmoid(lg)
            dl = torch.stack([_dice(pr[l], tgt[l]) for l in x0.layout])
            occ_l = (bce * tw).sum() / tw.sum().clamp(min=1e-6) + (dl * (1 - t) ** 2).mean()
            lam = _ramp(self, occ)
            terms["occ"] = occ_l
            terms["occ_lam"] = torch.tensor(lam)
            terms["loss"] = terms["mse"] + lam * occ_l
        per = np.array([((se[l] * w[l]).sum() / w[l].sum().clamp(min=1)).item() for l in x0.layout])
        tb = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (tb == i).sum():
                terms[f"bin_{i}"] = {"mse": per[tb == i].mean()}
        return terms, {}

    trainer_cls.training_losses = training_losses


def ss_occ_data(hand_dir):
    """Stage 1 dataset: add the GT 64^3 occupancy (the 1024 shape latent's coordinates) and the GT hand cells."""
    import numpy as np
    from pixal3d.datasets import sparse_structure_latent as ssl
    orig = ssl.SparseStructureLatentView.get_instance
    hand_dir = Path(hand_dir) if hand_dir else None

    def get_instance(self, root, instance):
        out = orig(self, root, instance)
        occ = np.zeros((64,) * 3, np.uint8)
        c = np.load(Path(root["ss_latent"]).parents[1] / "shape_latents/shape_enc_next_dc_f16c32_fp16_1024_view"
                    / instance / "view00.npz")["coords"].astype(int)
        occ[tuple(c.T)] = 1
        hand = np.zeros((64,) * 3, np.uint8)
        if hand_dir is not None and (hand_dir / f"{instance}.npz").exists():
            h = np.load(hand_dir / f"{instance}.npz")["coords"].astype(int)
            if len(h):
                hand[tuple(h.T)] = 1
        out["occ64"], out["hand64"] = torch.from_numpy(occ)[None], torch.from_numpy(hand & occ)[None]
        return out

    ssl.SparseStructureLatentView.get_instance = get_instance


def ss_occ_losses(trainer_cls, occ):
    """training_losses of the proj dense trainer (stage 1) + the decoded-occupancy loss: x0_hat = x_t - t v_hat through
    the frozen conv3d decoder -> 64^3 logits; BCE (hand cells up-weighted by occ['hand_w']) + Dice against the GT
    occupancy, per sample weighted by (1 - t)^2, coefficient ramped to occ['lam'] over occ['ramp'] steps."""
    import numpy as np
    import torch.nn.functional as F
    from easydict import EasyDict as edict

    def training_losses(self, x_0, cond=None, **kwargs):
        occ64 = kwargs.pop("occ64").float()
        hand64 = kwargs.pop("hand64").float()
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)
        cond = self.get_cond(cond, **kwargs)
        pred = self.training_models["denoiser"](x_t, t * 1000, cond)
        target = self.get_v(x_0, noise, t)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        tv = t.view(-1, *[1] * (x_0.dim() - 1))
        lg = occ["decoder"]((x_t - tv * pred).float()).float()
        wcell = 1 + (occ["hand_w"] - 1) * hand64
        bce = (F.binary_cross_entropy_with_logits(lg, occ64, reduction="none") * wcell).flatten(1).sum(1) \
            / wcell.flatten(1).sum(1)
        pr = torch.sigmoid(lg)
        dl = torch.stack([_dice(pr[i], occ64[i]) for i in range(len(pr))])
        tw = (1 - t) ** 2
        occ_l = ((bce + dl) * tw).mean()
        lam = _ramp(self, occ)
        terms["occ"] = occ_l
        terms["occ_lam"] = torch.tensor(lam)
        with torch.no_grad():
            hit = ((lg > 0).float() * occ64).flatten(1).sum(1)
            terms["occ_iou"] = (hit / (((lg > 0).float() + occ64).clamp(max=1).flatten(1).sum(1) + 1)).mean()
            terms["hand_rec"] = (((lg > 0).float() * hand64).flatten(1).sum(1) / (hand64.flatten(1).sum(1) + 1)).mean()
        terms["loss"] = terms["mse"] + lam * occ_l
        mse_i = np.array([F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])])
        tb = np.digitize(t.cpu().numpy(), np.linspace(0, 1, 11)) - 1
        for i in range(10):
            if (tb == i).sum():
                terms[f"bin_{i}"] = {"mse": mse_i[tb == i].mean()}
        return terms, {}

    trainer_cls.training_losses = training_losses


def load_frozen(path):
    from pixal3d import models
    m = models.from_pretrained(path).cuda().eval()
    if hasattr(m, "convert_to_fp32"):
        m.convert_to_fp32()
    m.float()
    m.dtype = torch.float32
    for q in m.parameters():
        q.requires_grad_(False)
    return m


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
    ap.add_argument("--dilate-prob", type=float, default=0.0, help="stage 2: probability of the one-voxel GT dilation")
    ap.add_argument("--hand-vox", help="dir of gen_hand_vox.py outputs (hand cells, for hand up-weighting)")
    ap.add_argument("--occ-loss", type=float, default=0.0, help="decoded-occupancy loss coefficient (after ramp)")
    ap.add_argument("--occ-ramp", type=int, default=2000, help="steps over which the coefficient ramps from 0")
    ap.add_argument("--occ-hand-w", type=float, default=5.0, help="hand cell / token weight in the occupancy BCE")
    ap.add_argument("--init-lora", help="LoRA state dict to start from (fresh optimizer and EMA)")
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

    stage2_aug = a.stage.startswith("shape") and (a.pred_coords or a.dilate_prob > 0 or a.occ_loss > 0)
    if stage2_aug:
        mixed_coords(a.pred_coords, a.pred_frac, a.extra_target, a.swap_frac, a.swap_prob, a.dilate_prob, a.hand_vox)
        cfg["b0"]["mixed"] = {"pred_coords": a.pred_coords, "pred_frac": a.pred_frac, "extra_target": a.extra_target,
                              "swap_frac": a.swap_frac, "swap_prob": a.swap_prob, "dilate_prob": a.dilate_prob}
    if a.stage == "ss64" and a.occ_loss > 0:
        ss_occ_data(a.hand_vox)
    if a.occ_loss > 0:
        cfg["b0"]["occ_loss"] = {"lam": a.occ_loss, "ramp": a.occ_ramp, "hand_w": a.occ_hand_w, "hand_vox": a.hand_vox}
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
    if a.init_lora:
        lora.load_lora(model, torch.load(a.init_lora, map_location="cuda"))
        print(f"LoRA initialised from {a.init_lora}", flush=True)
    lora.freeze_base(model)
    patch_partial_load(model)
    tot, trn = lora.count(model)
    print(f"LoRA r={a.rank} a={a.alpha}: {len(names)} layers, {trn:,} trainable of {tot:,}", flush=True)
    json.dump({"layers": names, "trainable": trn, "total": tot}, open(Path(a.out) / "lora_layers.json", "w"), indent=1)

    tcls = getattr(trainers, cfg["trainer"]["name"])
    lora_only_checkpoints(tcls)
    occ = None
    if a.occ_loss > 0:
        dec = "TencentARC/Pixal3D/ckpts/" + ("ss_dec_conv3d_16l8_fp16" if a.stage == "ss64" else "shape_dec_next_dc_f16c32_fp16")
        occ = {"lam": a.occ_loss, "ramp": a.occ_ramp, "hand_w": a.occ_hand_w, "decoder": load_frozen(dec)}
        if a.stage != "ss64":
            nz = cfg["dataset"]["args"]["normalization"]
            occ["mean"] = torch.tensor(nz["mean"]).float().cuda()
            occ["std"] = torch.tensor(nz["std"]).float().cuda()
    if stage2_aug:
        masked_losses(tcls, occ)
    elif a.stage == "ss64" and occ is not None:
        ss_occ_losses(tcls, occ)
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
