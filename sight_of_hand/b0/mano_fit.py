"""Minimal differentiable MANO (torch, CPU or GPU) and a point-cloud fitter.

The layer reads the official MANO_{LEFT,RIGHT}.pkl without chumpy (a stub unpickler). Pose is axis-angle global
orientation + 45 PCA coefficients added to hands_mean (flat_hand_mean=False, as DexYCB / manopth), shape is 10 betas,
output in metres. The fitter minimises a truncated two-way surface distance between MANO and a target point cloud,
from several random global orientations, then refines pose and shape.
"""
import pickle
from pathlib import Path

import numpy as np
import torch

TIPS = [745, 317, 444, 556, 673]  # thumb, index, middle, ring, little fingertip vertices (right and left)


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, st):
        self.__dict__.update(st if isinstance(st, dict) else {"x": st})


def _load_pkl(path):
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith("chumpy"):
                return _Stub
            if mod == "scipy.sparse.csc":
                mod = "scipy.sparse"
            return super().find_class(mod, name)

    with open(path, "rb") as f:
        d = U(f, encoding="latin1").load()

    def arr(v):
        if isinstance(v, _Stub):
            st = v.__dict__
            if "x" in st:
                v = st["x"]
            elif "idxs" in st:  # chumpy Select: a.ravel()[idxs] in preferred_shape
                v = arr(st["a"]).ravel()[np.asarray(st["idxs"])].reshape(st["preferred_shape"])
            else:
                v = arr(st["a"])
        if hasattr(v, "toarray"):
            v = v.toarray()
        return np.asarray(v, np.float64)

    return {k: (arr(v) if k != "kintree_table" and k != "f" else np.asarray(v)) for k, v in d.items()
            if k in ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "kintree_table", "f",
                     "hands_components", "hands_mean")}


def _rodrigues(r):
    th = r.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = r / th
    K = torch.zeros(*r.shape[:-1], 3, 3, dtype=r.dtype, device=r.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand_as(K)
    return eye + s * K + (1 - c) * K @ K


class Mano(torch.nn.Module):
    def __init__(self, model_dir, side="right", dtype=torch.float32):
        super().__init__()
        d = _load_pkl(Path(model_dir) / f"MANO_{side.upper()}.pkl")
        t = lambda x: torch.tensor(x, dtype=dtype)  # noqa: E731
        self.register_buffer("v_template", t(d["v_template"]))
        self.register_buffer("shapedirs", t(d["shapedirs"][:, :, :10]))
        self.register_buffer("posedirs", t(d["posedirs"].reshape(778 * 3, 135).T))
        self.register_buffer("J_regressor", t(d["J_regressor"]))
        self.register_buffer("weights", t(d["weights"]))
        self.register_buffer("comps", t(d["hands_components"]))
        self.register_buffer("mean", t(d["hands_mean"]))
        self.parents = [int(p) for p in d["kintree_table"][0]]
        self.parents[0] = -1
        self.faces = np.asarray(d["f"], np.int64)

    def forward(self, rot, pca, betas, trans):
        """rot (B,3), pca (B,45), betas (B,10), trans (B,3) -> verts (B,778,3), joints (B,21,3), metres"""
        B = rot.shape[0]
        v = self.v_template + torch.einsum("vcs,bs->bvc", self.shapedirs, betas)
        J = torch.einsum("jv,bvc->bjc", self.J_regressor, v)
        pose = torch.cat([rot, self.mean + pca @ self.comps], 1).view(B, 16, 3)
        R = _rodrigues(pose)
        eye = torch.eye(3, dtype=R.dtype, device=R.device)
        v = v + ((R[:, 1:] - eye).reshape(B, 135) @ self.posedirs).view(B, 778, 3)
        T = [None] * 16
        for i in range(16):
            A = torch.zeros(B, 4, 4, dtype=R.dtype, device=R.device)
            A[:, :3, :3], A[:, 3, 3] = R[:, i], 1
            p = self.parents[i]
            A[:, :3, 3] = J[:, i] - (J[:, p] if p >= 0 else 0)
            T[i] = A if p < 0 else T[p] @ A
        T = torch.stack(T, 1)
        Rg, Jp = T[:, :, :3, :3], T[:, :, :3, 3]
        tg = Jp - torch.einsum("bjcd,bjd->bjc", Rg, J)  # skinning transform = world transform after the rest joint
        Wr = torch.einsum("vj,bjcd->bvcd", self.weights, Rg)
        Wt = torch.einsum("vj,bjc->bvc", self.weights, tg)
        v = torch.einsum("bvcd,bvd->bvc", Wr, v) + Wt
        joints = torch.cat([Jp, v[:, TIPS]], 1)
        return v + trans[:, None], joints + trans[:, None]

    def joints_of(self, verts):
        """21 joints (16 regressed + 5 tips) of given MANO-topology vertices (V,3) numpy"""
        Jr = self.J_regressor.cpu().numpy() @ verts
        return np.r_[Jr, verts[TIPS]]


def _rand_rots(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g)
    q = q / q.norm(dim=1, keepdim=True)
    ang = 2 * torch.acos(q[:, :1].clamp(-1, 1))
    ax = q[:, 1:] / q[:, 1:].norm(dim=1, keepdim=True).clamp(min=1e-8)
    return ax * ang


def _surface(V, F, bary, fidx):
    return (V[:, F[fidx]] * bary[None, :, :, None]).sum(2)


def _tdist(a, b, trunc):
    """mean truncated distance a->b for batched a (B,N,3), b (B,M,3)"""
    d = torch.cdist(a, b).min(2).values
    return d.clamp(max=trunc).mean(1)


def fit(mano, target, n_starts=64, iters_a=100, iters_b=400, trunc=0.02, w_pose=1e-6, w_beta=1e-5, seed=0, keep=8,
        rigid=40, n_sub=300):
    """Fit MANO to target points (N,3) metres. Returns dict of numpy params, verts, joints and the final loss."""
    tgt = torch.tensor(target, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    F = torch.from_numpy(mano.faces)
    nf = len(F)
    fidx = torch.from_numpy(rng.integers(0, nf, 2000))
    b = rng.dirichlet([1, 1, 1], 2000)
    bary = torch.tensor(b, dtype=torch.float32)
    sub = tgt[torch.from_numpy(rng.choice(len(tgt), min(n_sub, len(tgt)), replace=False))]
    # stage A: orientation + translation from many starts, mean pose and shape
    rot = _rand_rots(n_starts, seed).requires_grad_()
    with torch.no_grad():
        v0, _ = mano(rot, torch.zeros(n_starts, 45), torch.zeros(n_starts, 10), torch.zeros(n_starts, 3))
    trans = (tgt.mean(0) - v0.mean(1)).clone().requires_grad_()
    pca = torch.zeros(n_starts, 45, requires_grad=True)
    opt = torch.optim.Adam([rot, trans, pca], lr=0.02)
    for it in range(iters_a):
        if it < rigid:  # orientation and position only, at the mean pose
            pca.grad = None
        v, _ = mano(rot, pca if it >= rigid else pca.detach(), torch.zeros(n_starts, 10), trans)
        loss = _tdist(sub[None].expand(n_starts, -1, -1), v, trunc) + _tdist(v, sub[None].expand(n_starts, -1, -1), trunc) \
            + w_pose * (pca ** 2).sum(1)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    k = min(keep, n_starts)
    best = torch.argsort(loss.detach())[:k]
    # stage B: refine the best starts with pose and shape against all points and a MANO surface sample
    rot = rot.detach()[best].clone().requires_grad_()
    trans = trans.detach()[best].clone().requires_grad_()
    pca = pca.detach()[best].clone().requires_grad_()
    betas = torch.zeros(k, 10, requires_grad=True)
    opt = torch.optim.Adam([rot, trans, pca, betas], lr=0.01)
    gen = torch.Generator().manual_seed(seed)
    for it in range(iters_b):
        T = tgt[torch.randint(len(tgt), (2000,), generator=gen)][None].expand(k, -1, -1)
        if it == iters_b // 2:
            for pg in opt.param_groups:
                pg["lr"] = 0.003
        v, _ = mano(rot, pca, betas, trans)
        s = _surface(v, F, bary, fidx)
        loss = _tdist(T, s, trunc) + _tdist(s, T, trunc) + w_pose * (pca ** 2).sum(1) + w_beta * (betas ** 2).sum(1)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    i = int(torch.argmin(loss.detach()))
    with torch.no_grad():
        v, j = mano(rot[i:i + 1], pca[i:i + 1], betas[i:i + 1], trans[i:i + 1])
    return {"rot": rot[i].detach().numpy(), "pca": pca[i].detach().numpy(), "betas": betas[i].detach().numpy(),
            "trans": trans[i].detach().numpy(), "verts": v[0].numpy(), "joints": j[0].numpy(),
            "loss": float(loss[i].detach())}
