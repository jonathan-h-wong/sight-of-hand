"""GraspXL amodal masks on sampled virtual cameras (GraspXL ships motions, no images).

Geometry comes from hoiaudit.sources.graspxl (the audited loader: MANO right, betas 0, flat_hand_mean per archive; the
object posed by its stored rotation/translation), so masks, audit and subsets all agree on the same hand and object.

Sampling (deterministic per sequence, seeded by its name), chosen so a later RGB pass can reuse the same cameras:
  frames   FRAC = 0.25, 0.55, 0.8, 1.0 of the motion (approach -> final grasp)
  cameras  CAMS per sampled frame, each aimed at that frame's hand+object points, at 0.9-1.5x the distance that fits
           their bounding sphere in the view. Cameras are per frame, not per sequence: GraspXL moves the object (it is
           lifted after the grasp), so no fixed camera frames both the approach and the final grasp.
           mano_tabletop: elevation 10-80 deg above the table (world z up); mano_dataset_N (floating objects):
           uniform directions. Roll jitter +-15 deg. 512x512, horizontal FOV 50-80 deg, principal point +-8 px.
  Masks touching the border are kept and flagged (hand_truncated, obj_truncated).
Camera convention: OpenCV (x right, y down, z forward); X_cam = R @ X_world + t, metres.
"""
import hashlib

import numpy as np

from . import raycast

H = W = 512
FRAC = (0.25, 0.55, 0.8, 1.0)
CAMS = 2


def _rng(name):
    return np.random.default_rng(int(hashlib.sha1(name.encode()).hexdigest()[:12], 16))


def _look_at(eye, target, up, roll):
    z = target - eye
    z /= np.linalg.norm(z)
    if abs(z @ up) > 0.98:
        up = np.array([1.0, 0, 0])
    x = np.cross(z, up)  # OpenCV: x right, y down, z forward
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # rows: camera axes in world
    c, s = np.cos(roll), np.sin(roll)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R
    return R, -R @ eye


def sample_cameras(name, pts, tabletop, n=CAMS):
    rng = _rng(name)
    ctr = 0.5 * (pts.min(0) + pts.max(0))
    rad = max(np.linalg.norm(pts - ctr, axis=1).max(), 0.03)
    cams = []
    for _ in range(n):
        fov = np.deg2rad(rng.uniform(50, 80))
        f = W / 2 / np.tan(fov / 2)
        K = np.array([[f, 0, W / 2 + rng.uniform(-8, 8)], [0, f, H / 2 + rng.uniform(-8, 8)], [0, 0, 1]])
        if tabletop:
            el, az = np.deg2rad(rng.uniform(10, 80)), rng.uniform(0, 2 * np.pi)
            d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        else:
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
        dist = rad / np.sin(fov / 2) * rng.uniform(0.9, 1.5)
        R, t = _look_at(ctr + d * dist, ctr, np.array([0, 0, 1.0]), np.deg2rad(rng.uniform(-15, 15)))
        cams.append((K, R, t))
    return cams


def render_sequence(task):
    """Yields (cam index, frames, masks dict name -> (N,H,W) bool, K (N,3,3), rows, extra {R (N,3,3), t (N,3)}):
    for camera slot c, one image per sampled frame, each with its own camera."""
    from hoiaudit.metrics import apply_T
    from hoiaudit.sources import graspxl as src
    from hoiaudit.sources.common import mano
    S = src.load(task)
    Vh, valid = S.hands["right"]
    n = len(S.frame_idx)
    sel = sorted({int(round(f * (n - 1))) for f in FRAC})
    sel = [i for i in sel if valid[i] and np.isfinite(S.T_wo[i]).all()]
    if not sel:
        return
    part = S.parts[0]
    Fh = mano("right", False).faces
    tab = S.name.startswith("mano_tabletop")
    per = {}
    for i in sel:
        h = Vh[i] / 1000.0
        o = apply_T(S.T_wo[i:i + 1], part.V[None])[0] / 1000.0
        per[i] = (h, o, sample_cameras(f"{S.name}#{int(S.frame_idx[i])}", np.concatenate([h, o]), tab))
    for c in range(CAMS):
        rows, acc, Ks, Rs, ts = [], {}, [], [], []
        for i in sel:
            h, o, cams = per[i]
            K, R, t = cams[c]
            mk = raycast.render([(h @ R.T + t, Fh), (o @ R.T + t, part.F)], K, H, W, {"hand": 0, "obj": 1})
            rows.append({"frame": int(S.frame_idx[i]), "t_frac": i / max(1, n - 1), **raycast.stats(mk)})
            for k, v in mk.items():
                acc.setdefault(k, []).append(v)
            Ks.append(K)
            Rs.append(R)
            ts.append(t)
        yield c, np.array([S.frame_idx[i] for i in sel]), {k: np.stack(v) for k, v in acc.items()}, np.stack(Ks), \
            rows, {"R": np.stack(Rs), "t": np.stack(ts)}
