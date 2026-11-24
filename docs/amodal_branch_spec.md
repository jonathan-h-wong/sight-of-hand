# Amodal mask branch: implementation spec (v1, DexYCB first)

Status: design only. Nothing here has been run. Items marked (verify) are assumptions to check against the code or data before relying on them.

## 1. Purpose and interface

A separate network that predicts soft amodal and modal masks for the hand and the grasped object from one RGB crop. It is trained alone, then frozen. Later, Pixal3D's stage 1 and stage 2 DiTs consume its output through `ProjGrid` (see the planning doc, Architecture section). This spec covers only the mask network.

    predict_masks(image: [B,3,512,512] float in [0,1]) -> [B,4,T,T] sigmoid probabilities
    channels: 0 amodal hand, 1 amodal object, 2 modal hand, 3 modal object

T is the output resolution. Default T = 512, the same as the NAF target Pixal3D uses for its stage 2 (verified in the 512 and 1024 configs: `naf_target_size: 512`; the 256 config uses 256). Sweep T in {128, 256, 512} as an ablation. The mask map is sampled later with bilinear `grid_sample`, so T does not have to equal the DINO high-resolution map.

## 2. Architecture

    RGB 512x512
      -> DINOv3 ViT-L/16, frozen   (camenduru/dinov3-vitl16-pretrain-lvd1689m, same ImageNet normalization
                                    and patch/register-token handling as DinoV3ProjFeatureExtractor in
                                    pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py)
      -> patch tokens -> [B,1024,32,32]
      -> NAF upsample, frozen      (torch.hub.load("valeoai/NAF","naf",pretrained=True); guide = RGB in [0,1];
                                    target size T)  -> [B,1024,T,T]
      -> head (trainable, about 0.5M parameters):
           1x1 conv 1024->128, GroupNorm, GELU
           concat RGB resized to TxT (3 ch)            [skip for boundary detail; ablate]
           3x3 conv 131->128, GN, GELU
           3x3 conv 128->128 (dilation 2), GN, GELU
           1x1 conv 128->4
           sigmoid

Notes:
- NAF is a 0.66M-parameter, image-guided upsampler. Its weights depend on the RGB image only, so it works on any channel count. It was exercised in the repo on 1024-channel DINO features only.
- Memory: the NAF output is 1024 x T x T. In fp16 that is about 0.54 GB per image at T=512, 0.13 GB at T=256, 0.03 GB at T=128. Run DINO and NAF under `torch.no_grad()` and in fp16 or bf16.
- Ablation B (lean): 1x1 reduction 1024->128 at 32x32 first, then NAF on the 128-channel map. Needs gradients through NAF (verify it supports backward, and that autocast is safe).
- Ablation C: head at 32x32, NAF applied to the 4-channel logits afterwards.
- No gradient flows into DINO or NAF in the main design.

## 3. Data (DexYCB first)

Use DexYCB s0 train sequences, as in the planned M1 model, with val and test kept separate. Treat the 8 camera views as separate images. Subsample frames with a fixed stride (about every 5th of 30 fps) to cut redundancy. Do NOT filter for penetration: a mask is the projection of a mesh and barely changes with a few millimeters of hand-object overlap.

Per frame:
- Hand mesh from the MANO parameters, grasped-object mesh and 6D pose from the label file, and the camera intrinsics (verify the field names in the DexYCB toolkit: believe `pose_m`, `pose_y`, `ycb_grasp_ind`, and the `seg` map with 0 background, 1 to 21 objects, 255 hand).
- Modal masks: from the DexYCB segmentation for the hand and for the grasped object only (other table objects are distractors; do not include them).
- Amodal masks: rasterize the hand mesh alone and the object mesh alone with the same intrinsics (nvdiffrast, pyrender or PyTorch3D). Do not composite with the table or other objects. Self-occlusion of the hand is included naturally.
- Enforce modal subset of amodal: amodal = amodal OR modal. Report how often the raw rendering violated this (pose error), per sequence.
- Hidden region = amodal AND NOT modal. Store the per-frame occlusion ratio = hidden area / amodal area, per part.

Crop: a square crop around the union bounding box of the projected hand and object, with margin (start at 1.2x), resized to 512x512. Apply the identical crop to the image and the masks, and store the crop parameters. Development crops come from ground truth. OPEN: at test time the crop has to come from a detector or a first-pass prediction, and it must also match Pixal3D's fixed canonical front camera (verify; unresolved, see the planning doc).

Splits (by sequence, never by frame or view): split the train sequences into 2 folds, A and B. Fold k's cached predictions must come from a head trained on the other fold (this stops the later DiT training from over-trusting masks). Use s0 val for model selection. s0 test stays untouched until the final report.

## 4. Training

- Loss: per-channel BCE + soft Dice. For channels 0 and 1, up-weight BCE on hidden-region pixels (start at 5x; tune). Optional hinge penalty if amodal < modal.
- Augmentation: color jitter, blur, JPEG; crop jitter (scale +/-15 percent, shift +/-10 percent of the crop, applied to image and masks together, so the head sees test-time crop noise). No horizontal flip (it changes handedness; the MANO side would need mirroring).
- Optimizer: AdamW, lr 1e-3 for the head with cosine decay, weight decay 1e-2, bf16 autocast. Batch 8 at T=512 on a large GPU, batch 2 at T=256 on 8 GB. Steps: tune; stop on val hidden-region IoU.
- Save: head weights plus the config (T, channel counts, crop margin). DINO and NAF are not saved.
- Own optimizer and own checkpoint. Nothing couples to the DiT trainer.

## 5. Evaluation (report on s0 val; s0 test only at the end)

Per channel, threshold 0.5:
1. IoU (amodal hand, amodal object, modal hand, modal object).
2. Hidden-region IoU: IoU between (predicted amodal AND NOT GT modal) and (GT amodal AND NOT GT modal). This is the number that matters.
3. Boundary IoU on the visible boundary.
4. Calibration on the hidden region: Brier score and expected calibration error (the maps are soft on purpose).
5. All of the above binned by occlusion ratio (0 to 10, 10 to 30, 30 to 60, over 60 percent) and by part.

Baselines, to establish a floor:
- (i) predict amodal = GT modal (no completion).
- (ii) predict amodal = the network's own modal channel.

Ablations: main design vs B vs C; with and without the RGB skip; T in {128, 256, 512}; hidden-region loss weight.

Cost: milliseconds per image and peak memory for DINO, NAF and the head, at each T.

Also dump overlay images for about 50 frames per occlusion bin for visual review.

Gate: the head should beat baselines (i) and (ii) on hidden-region IoU on val. No numeric target is set; the first run establishes the baseline.

## 6. Outputs

- Trained head checkpoints (full data, plus the two fold heads).
- Cached predictions `[4,T,T]` in fp16 per frame, with crop parameters, for the training frames (from the fold heads) and for val and test (from the full head).
- A results table (section 5) and the overlay dumps.
- A short note listing every item above marked (verify) and what was found.

## 7. Open items (not for this agent to solve silently)

- Test-time crop source, and mapping a crop onto Pixal3D's fixed front camera (distance, FOV, mesh scale).
- Whether NAF supports backward and autocast (needed for ablation B only).
- DexYCB label field names and whether the segmentation convention above holds.
