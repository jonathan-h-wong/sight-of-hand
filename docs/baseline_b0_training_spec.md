# B0 training spec v3: LoRA fine-tune of Pixal3D on DexYCB, wrist-depth-normalized cube, full frames

Status: design only, nothing run. Supersedes baseline_b0_lora_lean.md. Items marked (verify) are unchecked assumptions. Stock values below were read from configs/gen/ss_flow_img_dit_1_3B_32_bf16_proj_finetune_ft64.json and slat_flow_img2shape_dit_1_3B_512_bf16_proj_finetune_ft1024.json.

## 0. What this revision fixes

- Full frame, no crops, no warps. The only image operation is padding to a square.
- The scene is normalized by the wrist depth. The cube is the padded frame's extent at the wrist plane (the stock rule), and it is constant in normalized units.
- Pixal3D's own cascade and configs are used as they are; the only trainable parameters are LoRA weights.
- Metric scale at inference comes from MoGe-2 depth. The GT depth is used for oracle rows.
- "Native resolution" is read as Pixal3D's native stage resolutions (stage 1 at 32^3, stage 2 at 512, then 1024) with the stock image path per stage. If you meant feeding the camera's 640x480 straight to DINO, see the optional knob in section 4.

## 1. Normalization and cube

Definitions (camera frame is OpenCV: x right, y down, z forward):
- d = camera-frame depth z of the MANO wrist joint (joint 0), in meters. Normalized points p_n = p_c / d.
- Toolkit world frame (camera at (0, -D_n, 0) looking along +y, z up): x_w = x_n, y_w = z_n - D_n, z_w = -y_n, with D_n = 1. This is a proper rotation (determinant +1). The wrist then sits at y_w = 0. Verify signs with the alignment overlay (P2).
- camera_angle_x = 2 * atan(S / (2 * fx)), where S is the side of the padded square and the principal point is at its center. Pad each frame to the smallest square centered on its principal point. Assumes fx is close to fy (verify for DexYCB).
- Cube side L_n = 2 * D_n * tan(camera_angle_x / 2). This is exactly the stock rule (the cube's mid-plane width equals the image width), so the stock inference code's `distance_from_fov` returns D_n = 1 when mesh_scale = 1 / L_n (verify numerically).
- mesh_scale = total_scale = 1 / L_n (ProjGrid cube side is 1 / mesh_scale; verify). camera_distance = D_n = 1.
- Containment: the cube covers normalized depth [D_n - L_n/2, D_n + L_n/2] and lateral extent set by L_n. Assert that every soup vertex lies inside; drop and count frames that do not.

Illustrative numbers, to be replaced by P0: with S = 656 and fx = 615, L_n is about 1.07. At d = 0.5 m the cube side is 0.54 m, and:
- stage 1 latent (8^3): 6.7 cm per token; occupancy (32^3): 1.7 cm
- stage 2 at 512 (latent R = 32): 1.7 cm per token, 1.0 mm voxel
- stage 2 at 1024 (latent R = 64): 0.84 cm per token, 0.52 mm voxel
A 0.19 m hand spans about 11 tokens at R = 32 and about 23 at R = 64. Everything scales with d: nearer means finer in meters.

## 2. Sample construction (DexYCB s0 train, all 8 cameras)

Stride so that about 20 to 30K frames remain. Per (sequence, frame, camera):
1. Soup: MANO hand mesh plus the grasped object mesh at GT pose, in the camera frame, concatenated as one mesh, no boolean union.
2. d from the GT wrist joint (verify the field name, believed `joint_3d`). Record the distribution; reject d outside [d_min, d_max] set from P0.
3. Normalize, convert to the world frame, run the containment assert.
4. Filters: penetration filter (same threshold as A1's training set); visibility (at least a fraction, default 50 percent, of both the hand's and the object's amodal projection lies inside the frame); keep non-contact approach frames up to about 30 percent.
5. Image: padded square, RGBA, alpha = GT modal mask of the hand plus the grasped object (removes table and distractors). Resize per stage: the stage 1 and stage 2 at 512 configs use image_size 512; the 1024 config uses 1024 (verify; a 640 px source is upsampled, which adds no information).
6. Files in the toolkit layout: `transforms.json` with one frame (camera_angle_x; transform_matrix = canonical front pose, camera at (0, -1, 0), +z up), the RGBA image, the mesh dump, `view00_scale.json` with total_scale = 1 / L_n, and the metadata CSV row.
7. Encode: stage 2 shape latents at 512 and 1024 (`encode_shape_latent_view`), then SS latents (`encode_ss_latent_view`) at 32^3 (and 64^3 if training ft64).

Toolkit and dataset changes:
- Replace the adaptive per-mesh normalization with the constant scale; skip `sphere_normalize_torch` (its bbox-center assertion) and the `transform_mesh` rotation (identity here); keep `mesh_to_flexible_dual_grid` and the encoders unchanged.
- Dataset args: set `num_views` to 1 (stock is 2), remove the aesthetic-score filters, keep `load_camera_info: true`; check `max_tokens` (stock 32768 at 1024) is not binding (hand-object tokens at R = 64 are expected to be in the low thousands; measure in P0).

## 3. Pre-flight checks

- P0, statistics: wrist depth d, lateral offsets in normalized units, fraction of frames that fit the cube, fraction passing each filter, hand size in pixels, token counts per stage.
- P1, round trip: encode GT soups at 512 and 1024, decode, report Chamfer and F@1/2/5 mm using GT d. This is the ceiling; if 512 loses fingers, stage 2 at 1024 is required.
- P2, alignment: render a decoded GT latent with the stock fixed front camera over the padded image; the silhouette must match.
- P3, zero-shot: stock weights with this cube and camera on 200 stratified val frames (50 per occlusion bin), oracle matte and raw frame.
- P4, overfit 32 frames per stage.
- P5, LoRA tests: zero-init output equals the base model; sparse-linear wrapper works in stage 2 (unverified); merge works.
- P6, throughput: 100 steps per stage on the target GPU, then project cost.

## 4. Training

Order: stage 1 (32^3), then stage 2 at 512, then stage 2 at 1024 if P1 and P3 call for it. Each stage is its own LoRA. Stage 2 trains on GT coordinates.

Start from the stock settings and change only what LoRA needs:
- AdamW (betas 0.9 and 0.95, weight decay 0.01), bf16 amp, AdaptiveGradClipper (max_norm 1.0, 95th percentile), p_uncond 0.1, sigma_min 1e-5. Stage 1 uses the logitNormal(1, 1) timestep schedule; stage 2 uses uniform.
- Batch: stock is 8 per GPU for stage 1 and 2 per GPU for stage 2, with batch_split 2; adjust to the GPU in P6.
- LoRA: all linear layers in the 30 blocks (self-attention, cross-attention, `proj_linear`, MLP) plus the input and output projections; alpha 32; rank 32 to start (sweep {16, 64} on stage 1). Stock full-fine-tune LRs are 1e-4 (stage 1) and 2e-5 (stage 2 at 1024); the doc's recipe says LoRA LR about 10x, so start near 1e-3 and 2e-4 and tune.
- EMA: stock is 0.9999 (time constant about 10K steps), too slow for a short LoRA run. Use 0.999, or evaluate the raw weights. Decide before launching.
- Frozen: DINOv3, NAF, all VAEs and decoders.
- Augmentation: color, blur, JPEG on the condition image only. No geometric augmentation (cached targets are tied to the cube).
- Optional knob: DINO input resolution. Hands are small in a full frame (about 11 patches wide at 0.5 m with a 512 input). A larger DINO input gives a larger patch map; ProjGrid samples bilinearly, so no architecture change is expected, but check the NAF target size and `proj_in_channels` (verify).
- Snapshots: stock samples with CFG at fixed intervals; decode and overlay on the padded frame for visual checks.
- Save LoRA weights plus the config (frame list, filters, D_n, L_n rule, scale mode).

## 5. Inference and lifting

- Inputs: padded frame, matte (oracle in evaluation, BiRefNet in deployment), camera_angle_x from the real intrinsics or MoGe-2, D_n = 1, mesh_scale = 1 / L_n.
- Output: the fused soup in normalized units (scale-free).
- Metric: multiply by d. d = MoGe-2 metric depth at the generated wrist (or the median over the generated hand's pixels); oracle rows use the GT d. A robust 1-D scale fit is only a fallback if the MoGe-2 depth is unreliable on hands.

## 6. Evaluation (s0 val; test at the end)

- Modes: stage 1 occupancy; stage 2 with GT coordinates; full cascade.
- Scale-free metrics in normalized units: Chamfer and F-scores, relative hand-object placement, contact and penetration (these do not depend on d).
- Metric rows: the same metrics after multiplying by the GT d (the model's error) and by the MoGe-2 d (adds the depth source's error).
- Strata: occlusion bins as in the mask evaluation; image position (soup-center offset from the principal point); depth bins; hand size in pixels; matte condition (oracle vs raw frame).
- Paired occlusion gap: least vs most occluded of the 8 cameras per grasp.
- Stress frames: hand fully hidden or outside the frame. Report whether the model hallucinates a hand, drops it, or places it. Training targets always contain a hand, so it may produce one; this is a behavior to observe, not a bug.
- Rows: zero-shot, B0 LoRA; A1 later under the same protocol.

## 7. Open items

- Which Pixal3D checkpoints are released (stage 1 at 32^3, ft64, stage 2 at 512 and 1024); if there is no 64^3 stage 1 weight, the released cascade (stage 1 at 32^3 then 512 then 1024) is the path.
- Whether the stock inference rule and the constants agree numerically (P2 and a unit test).
- Sparse-linear LoRA wrapper (unverified).
- Whether `max_tokens` or elastic memory settings matter at these token counts.
- The matte at deployment: BiRefNet or another segmenter; B0 and A1 must use the same.
- GPU-hours per run: measure in P6; no estimate here.
