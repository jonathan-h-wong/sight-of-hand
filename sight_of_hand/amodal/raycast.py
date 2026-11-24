"""Amodal and modal hand/object masks by ray casting (Embree via trimesh + embreex; CPU, no GL).

For a pinhole camera (OpenCV axes: x right, y down, z forward) and meshes already in that camera's frame:
  amodal mask of mesh i  = pixels whose ray hits mesh i, with every other mesh removed
  modal mask of mesh i   = pixels whose FIRST hit among all meshes (targets and occluders) is mesh i
A ray goes through each pixel centre. Rays are only cast inside the projected bounding box of the relevant meshes, so
cost scales with the size of the hand/object on screen, not the image.
"""
import numpy as np

NEAR = 1e-3  # metres; vertices closer than this (or behind the camera) make the box fall back to the full image


def _bbox(V, K, H, W):
    """Pixel box [x0, x1) x [y0, y1) covering the projection of V (camera frame, metres)."""
    if (V[:, 2] <= NEAR).any():
        return 0, W, 0, H
    p = (K @ V.T).T
    p = p[:, :2] / p[:, 2:]
    x0, y0 = np.floor(p.min(0)).astype(int) - 1
    x1, y1 = np.ceil(p.max(0)).astype(int) + 2
    x0, y0 = min(max(0, x0), W), min(max(0, y0), H)  # clamp both ends: a mesh fully off one side gives an empty
    return x0, max(x0, min(W, x1)), y0, max(y0, min(H, y1))  # box, never a negative (wrap-around) slice end


def _rays(K, x0, x1, y0, y1):
    u, v = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    return np.zeros_like(d), d / np.linalg.norm(d, axis=1, keepdims=True)


def _first_hit(meshes, K, H, W, box):
    """Index of the first mesh hit per pixel inside box (-1 = none), as an (y1-y0, x1-x0) array."""
    x0, x1, y0, y1 = box
    out = np.full((max(0, y1 - y0), max(0, x1 - x0)), -1, np.int32)
    if out.size == 0:
        return out
    Vs, Fs, owner, off = [], [], [], 0
    for i, (V, F) in enumerate(meshes):
        Vs.append(V)
        Fs.append(F + off)
        owner.append(np.full(len(F), i, np.int32))
        off += len(V)
    import trimesh  # imported here so pack/unpack users need neither trimesh nor embreex
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    m = trimesh.Trimesh(np.concatenate(Vs), np.concatenate(Fs), process=False)
    o, d = _rays(K, x0, x1, y0, y1)
    tri = RayMeshIntersector(m).intersects_first(o, d)
    own = np.concatenate(owner)
    hit = tri >= 0
    flat = out.reshape(-1)
    flat[hit] = own[tri[hit]]
    return out


def render(meshes, K, H, W, targets):
    """meshes: list of (V (n,3) metres in camera frame, F (m,3)); targets: {name: mesh index}.
    Returns {"<name>_amodal": bool (H,W), "<name>_modal": bool (H,W)} for every target."""
    out = {}
    live = [(V, F) for V, F in meshes if V is not None and len(F)]
    idx = {i: j for j, i in enumerate(i for i, (V, F) in enumerate(meshes) if V is not None and len(F))}
    # modal: one pass over every mesh, restricted to the union of the targets' boxes
    tboxes = [_bbox(meshes[i][0], K, H, W) for i in targets.values() if i in idx]
    if tboxes:
        box = (min(b[0] for b in tboxes), max(b[1] for b in tboxes), min(b[2] for b in tboxes), max(b[3] for b in tboxes))
        first = _first_hit(live, K, H, W, box)
    for name, i in targets.items():
        a = np.zeros((H, W), bool)
        mo = np.zeros((H, W), bool)
        if i in idx:
            V, F = meshes[i]
            b = _bbox(V, K, H, W)
            a[b[2]:b[3], b[0]:b[1]] = _first_hit([(V, F)], K, H, W, b) >= 0
            mo[box[2]:box[3], box[0]:box[1]] = first == idx[i]
        out[f"{name}_amodal"] = a
        out[f"{name}_modal"] = mo
    return out


def stats(m, prefix=""):
    """Per-frame mask statistics for hand/object masks from render()."""
    ha, oa, hm, om = (m[k] for k in ("hand_amodal", "obj_amodal", "hand_modal", "obj_modal"))
    na, no = int(ha.sum()), int(oa.sum())

    def border(x):
        return bool(x[0].any() or x[-1].any() or x[:, 0].any() or x[:, -1].any())
    return {
        "hand_amodal_px": na, "obj_amodal_px": no, "hand_modal_px": int(hm.sum()), "obj_modal_px": int(om.sum()),
        "hand_visible_frac": hm.sum() / na if na else np.nan,
        "obj_visible_frac": om.sum() / no if no else np.nan,
        # share of the object's full extent hidden behind the hand, and vice versa (the cases amodal masks are for)
        "obj_occluded_by_hand_frac": (oa & hm).sum() / no if no else np.nan,
        "hand_occluded_by_obj_frac": (ha & om).sum() / na if na else np.nan,
        "hand_truncated": border(ha), "obj_truncated": border(oa),
    }


def pack(masks):
    """dict name -> (N,H,W) bool  ->  dict name -> (N,H,ceil(W/8)) uint8 (np.packbits along W)."""
    return {k: np.packbits(v, axis=-1) for k, v in masks.items()}


def unpack(packed, W):
    return np.unpackbits(packed, axis=-1, count=W).astype(bool)
