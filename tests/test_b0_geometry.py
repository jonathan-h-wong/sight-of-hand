"""B0 geometry against Pixal3D's own projection code (spec section 1, the "(verify)" items).

Needs a Pixal3D checkout: PIXAL3D_DIR (default ../Pixal3D). The functions are pulled out of the source with ast so the
heavy package imports are not needed.

    PIXAL3D_DIR=/path/to/Pixal3D python -m pytest tests/test_b0_geometry.py -q
"""
import ast
import math
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sight_of_hand.b0 import geometry as G  # noqa: E402

PX = Path(os.environ.get("PIXAL3D_DIR", Path(__file__).resolve().parents[2] / "Pixal3D"))


def _load(path, names):
    src = (PX / path).read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    import typing
    ns = {"torch": torch, "math": math, **{k: getattr(typing, k) for k in ("Tuple", "Optional", "List", "Dict")}}
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    ns.update(nn=nn, F=F, Image=Image, np=np)
    exec(compile(ast.Module(keep, []), str(path), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def px():
    if not PX.exists():
        pytest.skip(f"no Pixal3D checkout at {PX}")
    ns = _load("pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py",
               {"project_points_to_image_batch", "sample_features", "ProjGrid"})
    ns.update(_load("inference.py", {"compute_f_pixels", "distance_from_fov"}))
    return ns


K = np.array([[621.796997, 0, 302.413696], [0, 621.402893, 236.884338], [0, 0, 1]])  # DexYCB 840412060917
H, W = 480, 640


def test_frames_roundtrip():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(100, 3)) * 0.1 + [0, 0, 0.7]
    L = 1.1
    assert np.allclose(G.grid_to_cam(G.cam_to_grid(p, 0.7, L), 0.7, L), p)
    R = np.stack([G.cam_to_world(e, 1.0) - G.cam_to_world(np.zeros(3), 1.0) for e in np.eye(3)])
    assert abs(np.linalg.det(R) - 1) < 1e-9          # proper rotation
    assert np.allclose(G.cam_to_world([0, 0, 1.0], 1.0), 0)  # wrist on the axis lands at the world origin


def test_distance_from_fov_returns_one(px):
    """The stock inference rule gives D_n = 1 for mesh_scale = 1 / L_n (spec 1, 'verify numerically')."""
    S, _, _ = G.pad_params(K, H, W)
    cax = G.camera_angle_x(S, K[0, 0])
    L = G.cube_side(cax)
    for res in (512, 1024):
        d = px["distance_from_fov"](cax, torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0, res - 1]), 1.0 / L, res)
        # the stock rule maps the cube's -x face to pixel 0 (left edge centre convention): within half a pixel of 1
        assert abs(d["distance_from_x"] - 1.0) < 1.5 / res, d


def test_projgrid_matches_pinhole(px):
    """Points of the soup, sent through our grid frame and ProjGrid's front camera, land on the same padded-image
    pixel as the real pinhole projection (up to the <=0.5 px principal-point rounding)."""
    d = 0.62
    S, ox, oy = G.pad_params(K, H, W)
    geo = G.frame_geometry(K, H, W, d)
    rng = np.random.default_rng(1)
    p = np.c_[rng.uniform(-0.15, 0.15, 200), rng.uniform(-0.1, 0.1, 200), d + rng.uniform(-0.2, 0.2, 200)]
    g = G.cam_to_grid(p, d, geo["L_n"])
    assert G.contained(g)
    # ProjGrid: world = R (2 g) / mesh_scale / 2
    w = torch.tensor(2 * g @ G.R_PROJ.T, dtype=torch.float32) / geo["mesh_scale"] / 2
    T = torch.tensor(G.front_transform_matrix(1.0))[None]
    res = 512
    uv, depth, valid = px["project_points_to_image_batch"](w[None], T, torch.tensor([geo["camera_angle_x"]]), res)
    uv = uv[0].numpy()
    pin = (p @ K.T)[:, :2] / p[:, 2:3] + [ox, oy]       # padded-image pixels at size S
    pin = (pin + 0.5) * res / S - 0.5                    # to the res x res image (pixel-centre convention)
    err = np.abs(uv - pin).max()
    assert err < 1.0 * res / S + 1e-3, err


def test_pinhole_principal_point_rounding():
    S, ox, oy = G.pad_params(K, H, W)
    assert S >= W and S >= H
    assert abs(K[0, 2] + ox - S / 2) <= 0.5 + 1e-9 and abs(K[1, 2] + oy - S / 2) <= 0.5 + 1e-9
