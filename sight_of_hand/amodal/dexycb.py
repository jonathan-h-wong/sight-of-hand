"""DexYCB amodal masks: every frame of every camera, hand + grasped object, on the real 640x480 images.

Local layout (scripts/prepare_dexycb.sh: labels only, no images):
    ROOT/<date>-subject-NN/<seq>/meta.yml
    ROOT/<date>-subject-NN/<seq>/<serial>/labels_<frame:06d>.npz   seg (480,640), pose_y (K,3,4), pose_m (1,51)
    ROOT/calibration/intrinsics/<serial>_640x480.yml                color fx, fy, ppx, ppy
    ROOT/calibration/mano_<calib>/mano.yml                          subject betas
    ROOT/models/<ycb>/textured_simple.obj                           metres

Per-camera labels are already in that camera's frame (dex-ycb-toolkit): object k is v @ R_k.T + t_k with
pose_y[k] = [R_k | t_k]; the hand is manopth ManoLayer(use_pca=True, ncomps=45, flat_hand_mean=False) on pose_m
(3 global rot + 45 PCA + 3 trans) with the subject betas. pose_m == 0 means no hand; an all-zero pose_y row means
the object is not tracked in that frame.

Targets: "hand" and "obj" = the grasped object (meta ycb_grasp_ind). Every other scene object is an occluder for the
modal masks, so modal masks are comparable with DexYCB's own `seg` (255 = hand, YCB id = object); the agreement is
reported per frame (iou_hand_seg, iou_obj_seg).
"""
from pathlib import Path

import numpy as np
import yaml

from . import raycast

H, W = 480, 640
YCB = {1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box", 4: "005_tomato_soup_can",
       5: "006_mustard_bottle", 6: "007_tuna_fish_can", 7: "008_pudding_box", 8: "009_gelatin_box",
       9: "010_potted_meat_can", 10: "011_banana", 11: "019_pitcher_base", 12: "021_bleach_cleanser",
       13: "024_bowl", 14: "025_mug", 15: "035_power_drill", 16: "036_wood_block", 17: "037_scissors",
       18: "040_large_marker", 19: "051_large_clamp", 20: "052_extra_large_clamp", 21: "061_foam_brick"}
_MESH, _K, _BETAS = {}, {}, {}


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads the `!!python/tuple` tag DexYCB's intrinsics files use (as a list)."""


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, node: ld.construct_sequence(node))


def _yaml(p):
    with open(p) as f:
        return yaml.load(f, Loader=_Loader)


def list_tasks(root):
    return sorted(str(p.parent) for p in Path(root).glob("*-subject-*/*/meta.yml"))


def _mesh(root, yid):
    if yid not in _MESH:
        from hoiaudit.sources.common import load_mesh
        _MESH[yid] = load_mesh(Path(root) / "models" / YCB[yid] / "textured_simple.obj")
    return _MESH[yid]


def _intr(root, serial):
    if serial not in _K:
        c = _yaml(Path(root) / "calibration" / "intrinsics" / f"{serial}_640x480.yml")["color"]
        _K[serial] = np.array([[c["fx"], 0, c["ppx"]], [0, c["fy"], c["ppy"]], [0, 0, 1]], np.float64)
    return _K[serial]


def _betas(root, calib):
    if calib not in _BETAS:
        _BETAS[calib] = np.array(_yaml(Path(root) / "calibration" / f"mano_{calib}" / "mano.yml")["betas"])
    return _BETAS[calib]


def _iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def render_sequence(task):
    """Yields (serial, frames, masks dict name -> (N,H,W) bool, K, rows list[dict]) for every camera."""
    from hoiaudit.sources.common import mano
    task = Path(task)
    root = task.parent.parent
    meta = _yaml(task / "meta.yml")
    ids = list(meta["ycb_ids"])
    g = int(meta["ycb_grasp_ind"])
    side = meta["mano_sides"][0]
    m = mano(side, False)
    betas = _betas(root, meta["mano_calib"][0])
    meshes0 = [_mesh(root, i) for i in ids]
    for serial in meta["serials"]:
        K = _intr(root, str(serial))
        files = sorted((task / str(serial)).glob("labels_*.npz"))
        frames, rows, acc = [], [], {}
        for f in files:
            fr = int(f.stem.split("_")[1])
            z = np.load(f)
            pm = z["pose_m"][0]
            hand = (None, None)
            if np.any(pm != 0):
                V = m(betas, pm[None, :3], m.pca_to_pose(pm[None, 3:48]), pm[None, 48:51])[0]
                hand = (V, m.faces)
            objs = []
            for k, (V0, F0) in enumerate(meshes0):
                P = z["pose_y"][k]
                objs.append((None, None) if not np.any(P[:, :3]) else (V0 @ P[:, :3].T + P[:, 3], F0))
            meshes = [hand] + objs  # 0 = hand, 1 + k = object k
            mk = raycast.render(meshes, K, H, W, {"hand": 0, "obj": 1 + g})
            seg = z["seg"]
            r = {"frame": fr, "has_hand": hand[0] is not None, "obj_tracked": objs[g][0] is not None,
                 **raycast.stats(mk),
                 "iou_hand_seg": _iou(mk["hand_modal"], seg == 255),
                 "iou_obj_seg": _iou(mk["obj_modal"], seg == ids[g])}
            rows.append(r)
            frames.append(fr)
            for k, v in mk.items():
                acc.setdefault(k, []).append(v)
        masks = {k: np.stack(v) for k, v in acc.items()}
        yield str(serial), np.array(frames), masks, K, rows
