# Amodal mask branch: spec v2 (transformer head), DexYCB first

Status: design only. Nothing here has been run. Items marked (verify) are assumptions to check against code or data. This spec replaces section 2 (Architecture) and parts of sections 4 and 5 of v1. Data (section 3), outputs (section 6) and the 2-fold scheme stay as in v1.

## 1. Why v2

v1 results on DexYCB val (step 20,000): hidden-region IoU 0.299 (hand) / 0.359 (object); amodal IoU 0.756 / 0.767; modal IoU 0.753 / 0.751; calibration error about 0.03. Findings:
- Modal IoU is only about 0.75, so base segmentation quality is weak, not just hidden-extent ambiguity.
- Amodal IoU is about equal to modal IoU, so completion adds little.
- Overlays: occluded hands come out as blobs. The wrist pose and the visible fingers carry enough information to infer the hidden fingers, but the v1 head (3x3 convs on NAF-upsampled features, 0.5M parameters) has no global context.

v2 changes: (a) read several DINO layers instead of only the last, (b) add global self-attention at token resolution, (c) upsample the logits with NAF, (d) optionally add an RGB-stem decoder for pixel-level boundaries, (e) optionally add a joint-heatmap auxiliary loss.

DINO stays intact. No LoRA, no gradient into DINO. Intermediate layers are read from the same forward pass that produces the final output Pixal3D uses; that output is unchanged.

## 2. Interface (unchanged)

    predict_masks(image: [B,3,512,512] float in [0,1]) -> [B,4,T,T] sigmoid probabilities
    channels: 0 amodal hand, 1 amodal object, 2 modal hand, 3 modal object

Default T = 512. Sweep T in {128, 256, 512} as before.

## 3. Architecture (v2a, the main design)

    RGB 512x512
      -> DINOv3 ViT-L/16, frozen, one forward pass, collect block outputs at taps L = {5, 11, 17, 23}
         (0-indexed; DPT convention for ViT-L; verify against DINOv3's own dense-head config)
         drop CLS and register tokens; reshape patch tokens to [B,1024,32,32] per tap
      -> per-tap LayerNorm (learnable affine) -> Linear 1024->256   (4 taps)
      -> concat (1024) -> Linear 1024->384 -> GELU
      -> + learned 2D position embedding [1024,384]                   (ablate: none / sinusoidal)
      -> Transformer encoder: 6 layers, width 384, 6 heads, MLP ratio 4, pre-norm,
         full self-attention over the 1024 tokens (use SDPA / flash attention),
         stochastic depth 0.1, no dropout
      -> LayerNorm -> Linear 384->4  : logits [B,4,32,32]
      -> NAF upsample the 4-channel logits to TxT, guide = RGB in [0,1]
      -> sigmoid

Parameter count: about 12M (tap projections about 1.4M, transformer about 10.6M).

Notes:
- Why NAF on logits: NAF output is a weighted average of its input. The last step before the sigmoid is linear, so NAF on 4 logits equals NAF on the 384 features followed by that linear layer. Same result, much cheaper (4 channels instead of 384 or 1024).
- NAF ratio: 32 -> 512 is 16x, the ratio Pixal3D already uses on DINO features. Other ratios (verify).
- Why several taps: the last DINO layer is semantic but smooth; early layers keep edges and local geometry, middle layers keep part structure. The head needs both: edges for the boundary, semantics for what is hand vs object, and part structure for completion. The tap set is a convention, not tuned; ablate it.
- Per-tap LayerNorm: intermediate ViT features have different scales and a few large-norm channels; normalizing each tap before projection keeps one tap from dominating.
- Backward through NAF (verify): the loss at TxT needs gradients through NAF's weighted average (NAF weights are frozen but the operation must be differentiable w.r.t. its input). If NAF does not support backward or autocast, fallback: compute the main loss at 128x128 on logits upsampled with a fixed bilinear op, plus the 32x32 deep-supervision loss, and apply NAF only at inference; then check the train/inference mismatch on val.
- Memory: DINO and the taps run under torch.no_grad() in bf16. Transformer activations at 1024 tokens are small. Batch 16 at T=512 should fit on a 24 GB GPU (verify); on 8 GB use batch 4 and T=256.

## 4. Optional components (separate experiments)

### 4a. RGB-stem decoder (v2b): boundary detail

Motivation: DINO patches are 16x16 pixels, so detail finer than a patch (finger gaps, thin edges) is absent from the features, and NAF can only re-weight them. A stem gives learned pixel-level edge and colour features that the decoder merges with the upsampled semantics (a U-Net skip).

Stem (on the 512x512 RGB, 0.5M parameters):
    s1: conv3x3 3->32,   GN, GELU          (512x512)
    s2: conv3x3 s2 32->64, GN, GELU        (256x256)
    s4: conv3x3 s2 64->96, GN, GELU        (128x128)
    s8: conv3x3 s2 96->128, GN, GELU       (64x64)

Decoder (from the transformer output, 384 ch at 32x32):
    up x2 (bilinear) -> concat s8 -> conv3x3 -> 128   (64x64)
    up x2            -> concat s4 -> conv3x3 -> 96    (128x128)
    two variants:
      v2b-lite: 1x1 conv -> 4 logits at 128x128, then NAF to T (cheap)
      v2b-full: continue up x2 -> concat s2 -> conv -> 64 (256), up x2 -> concat s1 -> conv -> 32 (512), 1x1 -> 4 logits; no NAF
    sigmoid

Memory for v2b-full at 512: the s1 map is about 16 MB per image in fp16 before activations for backward (verify batch size).

### 4b. Joint-heatmap auxiliary loss (v2c): structured target for inferring hidden fingers

Training only; discard at inference.
- Head: Linear 384->21 on the transformer output (32x32), predicting a Gaussian heatmap per hand joint (sigma = 1 token).
- Labels: DexYCB 2D joints for all 21 joints, including occluded ones (verify field name, believed `joint_2d`). Apply the same crop and jitter as the image.
- Loss: MSE on heatmaps, weight 0.5 (tune).
- Intent: force the transformer to place hidden joints from the wrist and visible fingers. Also useful as an extra cached output later (verify whether Pixal3D conditioning needs it; probably not).

## 5. Training

- Data, crops, augmentation, 2-fold scheme, splits: as in v1 (section 3 and 4).
- Loss: per-channel BCE + soft Dice, 5x BCE weight on hidden-region pixels (tune), optional hinge if amodal < modal. Add deep supervision: BCE on the 32x32 logits against area-downsampled GT soft masks, weight 0.3.
- Optimizer: AdamW, lr 3e-4 (transformer and projections), weight decay 0.05, 1k warmup steps, cosine decay, grad clip 1.0, bf16 autocast.
- Steps: v1 used 20k. Expect 40k to 60k; stop on val hidden-region IoU in the 10 to 60 percent bins and on modal IoU.
- Overfitting watch: DexYCB has 10 subjects and 20 objects, and the head now has about 12M parameters. Track the train/val gap every epoch; if it opens, add stochastic depth, stronger crop and colour augmentation, or reduce width or depth.
- Own optimizer and checkpoint; nothing couples to the DiT trainer.

## 6. Evaluation (s0 val; test only at the end)

Everything in v1 section 5, plus:
1. Modal IoU is the primary diagnostic for segmentation quality; report it first.
2. Boundary-tolerant hidden-region IoU: count a pixel as correct within 3 px of the GT hidden region (report 1, 2 and 3 px).
3. Hidden-region IoU restricted to frames with at least 10 percent occlusion.
4. Resolution ceiling: downsample GT modal and amodal masks to 32x32, upsample to T with NAF (and with bilinear), and score IoU against the original masks. This is the upper bound for any head that sees only 32x32 information; report it next to the model's numbers.
5. Overlays (about 50 frames per occlusion bin) with the new emphasis: hands with wrist and fingers partly hidden; mark blobs vs finger-shaped completions. Optionally dump the last transformer layer's attention from a few hidden-finger tokens to check that it attends to the wrist and the visible fingers.

Experiment matrix (change one thing at a time):
- E0: v1 head (baseline numbers above).
- E1: last tap only (23) + transformer + NAF on logits.
- E2: four taps + transformer + NAF on logits (v2a).
- E3: E2 + RGB-stem decoder (v2b-lite, then v2b-full).
- E4: E2 or E3 + joint-heatmap loss (v2c).
- E5: transformer depth {4, 6, 8}, width {256, 384, 512}; position embedding {none, learned, sinusoidal}; tap sets {last, {11,23}, {5,11,17,23}}.
- E6: T in {128, 256, 512}.

Gate: beat v1 on modal IoU and on hidden-region IoU in the 10 to 60 percent bins. No numeric target; E2 establishes the baseline. Report cost (ms per image, peak memory) for DINO, taps, transformer and NAF at each T.

## 7. Outputs

- Head checkpoints (full data plus the two fold heads), config (T, taps, widths, variant).
- Cached predictions [4,T,T] fp16 with crop parameters, as in v1.
- Results table (section 6) and overlay dumps.
- A note listing every (verify) item and what was found.

## 8. Open items

- Whether NAF supports backward and autocast (now needed for the main design, not only an ablation).
- NAF behaviour at ratios other than 16x (needed for v2b-lite).
- DINOv3 tap indices and whether intermediate outputs need the final norm layer.
- DexYCB field names for 2D joints and for segmentation.
- Test-time crop source and the mapping onto Pixal3D's fixed front camera (unresolved, as in v1).
