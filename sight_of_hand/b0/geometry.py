"""B0 geometry (docs/baseline_b0_training_spec.md, section 1): wrist-depth normalization, padded square, the cube.

Frames
  camera   OpenCV: x right, y down, z forward, metres.
  norm     p_n = p_c / d, d = camera-frame depth of the MANO wrist (joint 0).
  world    Pixal3D / Blender world of the fixed front camera: camera at (0, -D_n, 0) looking along +y, z up.
           x_w = x_n, y_w = z_n - D_n, z_w = -y_n  (a proper rotation plus a shift).
  grid     the voxel frame the toolkit encodes in, aabb [-0.5, 0.5]^3. ProjGrid maps grid -> world by
           w = R g / mesh_scale with R = [[1,0,0],[0,0,-1],[0,1,0]], so g = R^T w * mesh_scale
           = (x_w, z_w, -y_w) / L_n.
  The cube side in world (normalized) units is L_n = 1 / mesh_scale.

Camera rule: the image is padded to the smallest square centred on the principal point; camera_angle_x =
2 atan(S / (2 fx)); L_n = 2 D_n tan(camera_angle_x / 2) = S / fx (for D_n = 1), so the cube's mid-plane exactly
spans the padded image (the stock inference rule; see tests/test_b0_geometry.py).
"""
import math

import numpy as np

D_N = 1.0
R_PROJ = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # ProjGrid's grid -> world rotation


def pad_params(K, H, W):
    """Smallest square centred on the principal point that contains the HxW frame.

    Returns (S, ox, oy): side in pixels (integer), and the offset of the original frame inside the square
    (original pixel (u, v) -> padded (u + ox, v + oy)). The principal point lands at (S/2, S/2) up to rounding,
    which is absorbed by using the exact padded principal point in camera_angle_x (fx only; see note below)."""
    cx, cy = K[0, 2], K[1, 2]
    half = max(cx, W - cx, cy, H - cy)
    S = int(math.ceil(2 * half))
    ox = int(round(S / 2 - cx))
    oy = int(round(S / 2 - cy))
    return S, ox, oy


def pad_image(img, S, ox, oy):
    """Place img (H, W[, C]) into a black S x S canvas at (ox, oy)."""
    H, W = img.shape[:2]
    out = np.zeros((S, S) + img.shape[2:], img.dtype)
    out[oy:oy + H, ox:ox + W] = img
    return out


def camera_angle_x(S, fx):
    return 2.0 * math.atan(S / (2.0 * fx))


def cube_side(cam_angle_x, D_n=D_N):
    """L_n: the cube side in normalized units, equal to the image width at the wrist plane."""
    return 2.0 * D_n * math.tan(cam_angle_x / 2.0)


def cam_to_world(p_c, d, D_n=D_N):
    p = np.asarray(p_c, np.float64) / d
    return np.stack([p[..., 0], p[..., 2] - D_n, -p[..., 1]], -1)


def world_to_cam(w, d, D_n=D_N):
    w = np.asarray(w, np.float64)
    return np.stack([w[..., 0], -w[..., 2], w[..., 1] + D_n], -1) * d


def world_to_grid(w, L_n):
    return (np.asarray(w, np.float64) @ R_PROJ) / L_n          # R^T w, row vectors


def grid_to_world(g, L_n):
    return (np.asarray(g, np.float64) @ R_PROJ.T) * L_n


def cam_to_grid(p_c, d, L_n, D_n=D_N):
    return world_to_grid(cam_to_world(p_c, d, D_n), L_n)


def grid_to_cam(g, d, L_n, D_n=D_N):
    return world_to_cam(grid_to_world(g, L_n), d, D_n)


def contained(g, margin=0.0):
    """True when every grid point lies inside [-0.5 + margin, 0.5 - margin]^3."""
    g = np.asarray(g)
    return bool(np.all(np.abs(g) <= 0.5 - margin))


def front_transform_matrix(D_n=D_N):
    """Camera-to-world matrix of the canonical front camera (Blender convention), for transforms.json."""
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -D_n], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def frame_geometry(K, H, W, d):
    """Everything a sample needs about its camera and cube."""
    S, ox, oy = pad_params(K, H, W)
    fx = float(K[0, 0])
    cax = camera_angle_x(S, fx)
    L_n = cube_side(cax)
    return {"S": S, "pad_x": ox, "pad_y": oy, "fx": fx, "fy": float(K[1, 1]), "camera_angle_x": cax,
            "L_n": L_n, "mesh_scale": 1.0 / L_n, "d": float(d), "D_n": D_N,
            "pp_offset_px": [float(K[0, 2] + ox - S / 2), float(K[1, 2] + oy - S / 2)]}
