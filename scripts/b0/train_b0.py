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
    step = None
    if a.resume:
        import glob
        f = glob.glob(os.path.join(a.out, "ckpts", "misc_step*.pt"))
        step = max(int(Path(x).stem.split("step")[-1]) for x in f) if f else None
    trainer = tcls({"denoiser": model}, ds, **copy.deepcopy(cfg["trainer"]["args"]), output_dir=a.out,
                   load_dir=a.out, step=step, wandb_run=wandb_run)
    if trainer.is_master and trainer.writer is None:
        class _W:
            def close(self):
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
