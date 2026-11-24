# sight-of-hand

Feedforward amodal hand-object reconstruction. This repository holds the training-data pipeline, starting with the
**amodal hand and object masks** for the first two training sets, DexYCB and GraspXL. Data, audits and physically
grounded subsets live in `s3://xray-hands`; the penetration audit and the subset tooling are in
[sight-of-hand-eval](https://github.com/jonathan-h-wong/sight-of-hand-eval) (`hoiaudit`), which this repo imports.

## Amodal masks

For every image and camera, four masks:

| mask | definition |
|---|---|
| `hand_amodal` | the hand mesh rendered alone: its full silhouette, whatever occludes it |
| `obj_amodal` | the object mesh rendered alone |
| `hand_modal` | pixels where the hand is the first surface hit with every mesh present (what is visible) |
| `obj_modal` | the same for the object |

They are rendered by ray casting the labelled meshes (Embree through trimesh + embreex, CPU, no GL): one ray per pixel
centre, OpenCV pinhole cameras, rays cast only inside the projected box of each mesh
(`sight_of_hand/amodal/raycast.py`).

| dataset | images | cameras | hand | object | checks |
|---|---|---|---|---|---|
| DexYCB | every frame of every camera (8 x ~72 x 1000), 640x480 | released calibration; labels already in each camera's frame | MANO (PCA-45, subject betas) | the grasped YCB object; the other scene objects occlude the modal masks | modal masks vs DexYCB's own `seg`: IoU 0.94 hand, 0.98 object (median, one sequence, all cameras) |
| GraspXL | 4 frames per motion (25, 55, 80, 100%) x 2 cameras, 512x512 | sampled per frame, aimed at that frame's hand + object (see `sight_of_hand/amodal/graspxl.py`) | MANO via `hoiaudit` (the audited loader) | the GraspXL object | no images exist; cameras are stored so an RGB pass reuses them |

GraspXL uses the audited v2 sample (3,000 objects per archive, seed 0), so mask rows join the audit and the
grounded subsets on (`seq`, `frame`).

## Store

```
s3://xray-hands/masks/amodal_v1/<dataset>/
  shards/<shard>.tar        <seq>/<cam>.npz: frames, the four masks bit-packed (N, H, ceil(W/8)), K, [R, t], meta
  shards/<shard>.parquet    per-image rows of that shard
  index.parquet             one row per image: seq, cam, frame, row, shard, member, offset, size, fx, fy, cx, cy,
                            mask areas, visible and occluded fractions, truncation flags (+ DexYCB: IoU vs seg)
  summary.json
```

```python
from sight_of_hand.amodal import store
idx = store.read_index("dexycb")
m = store.masks_for("dexycb", idx.iloc[0])     # dict of four (H, W) bool masks, one ranged S3 read
```

`obj_occluded_by_hand_frac` (share of the object's amodal mask covered by the visible hand) is the column to
stratify training by occlusion.

## Run

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PWD/../sight-of-hand-eval
bash scripts/prepare_data.sh data mano
bash scripts/prepare_data.sh data dexycb
python scripts/render_amodal.py --dataset dexycb --root data/dexycb --mano-dir data/mano/mano_v1_2/models
```

On RunPod: `scripts/runpod/amodal_bootstrap.sh` (CPU pod; prepares, renders both datasets, publishes, removes itself).
