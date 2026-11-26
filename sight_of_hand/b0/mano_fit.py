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


def fit_batch(mano, targets, n_starts=64, iters_a=100, iters_b=400, trunc=0.02, w_pose=1e-6, w_beta=1e-5, seed=0,
              keep=8, rigid=40, n_sub=300, n_b=2000):
    """Fit MANO (one side) to several target clouds at once, all frames x all starts in one batch (GPU if `mano` is
    on one). targets: list of (N_i,3) metres. Stage A: n_starts random orientations, rigid for `rigid` iterations,
    then pose, on n_sub points; stage B: the best `keep` starts per frame refine pose and shape against n_b points
    resampled each iteration and a fixed MANO surface sample. Returns one dict per target (see fit)."""
    dev = mano.v_template.device
    rng = np.random.default_rng(seed)
    nF, S = len(targets), n_starts
    full = [torch.tensor(t, dtype=torch.float32, device=dev) for t in targets]
    lens = torch.tensor([len(t) for t in full], device=dev)
    pad = torch.zeros(nF, int(lens.max()), 3, device=dev)
    for i, t in enumerate(full):
        pad[i, :len(t)] = t
    F = torch.from_numpy(mano.faces).to(dev)
    fidx = torch.from_numpy(rng.integers(0, len(F), 2000)).to(dev)
    bary = torch.tensor(rng.dirichlet([1, 1, 1], 2000), dtype=torch.float32, device=dev)
    sub = torch.stack([t[torch.from_numpy(rng.choice(len(t), n_sub, replace=len(t) < n_sub)).to(dev)] for t in full])
    subS = sub.repeat_interleave(S, 0)
    rot = _rand_rots(S, seed).to(dev).repeat(nF, 1).requires_grad_()
    z45, z10 = torch.zeros(nF * S, 45, device=dev), torch.zeros(nF * S, 10, device=dev)
    with torch.no_grad():
        v0, _ = mano(rot, z45, z10, torch.zeros(nF * S, 3, device=dev))
    trans = (sub.mean(1).repeat_interleave(S, 0) - v0.mean(1)).clone().requires_grad_()
    pca = z45.clone().requires_grad_()
    opt = torch.optim.Adam([rot, trans, pca], lr=0.02)
    for it in range(iters_a):
        v, _ = mano(rot, pca if it >= rigid else pca.detach(), z10, trans)
        loss = _tdist(subS, v, trunc) + _tdist(v, subS, trunc) + w_pose * (pca ** 2).sum(1)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    k = min(keep, S)
    best = (torch.argsort(loss.detach().view(nF, S), 1)[:, :k] + torch.arange(nF, device=dev)[:, None] * S).reshape(-1)
    rot = rot.detach()[best].clone().requires_grad_()
    trans = trans.detach()[best].clone().requires_grad_()
    pca = pca.detach()[best].clone().requires_grad_()
    betas = torch.zeros(nF * k, 10, device=dev, requires_grad=True)
    opt = torch.optim.Adam([rot, trans, pca, betas], lr=0.01)
    gen = torch.Generator(device=dev).manual_seed(seed)
    fr = torch.arange(nF, device=dev)[:, None]
    for it in range(iters_b):
        j = (torch.rand(nF, n_b, generator=gen, device=dev) * lens[:, None]).long()
        T = pad[fr, j].repeat_interleave(k, 0)
        if it == iters_b // 2:
            for pg in opt.param_groups:
                pg["lr"] = 0.003
        v, _ = mano(rot, pca, betas, trans)
        s_ = _surface(v, F, bary, fidx)
        loss = _tdist(T, s_, trunc) + _tdist(s_, T, trunc) + w_pose * (pca ** 2).sum(1) + w_beta * (betas ** 2).sum(1)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
    L = loss.detach().view(nF, k)
    pick = (L.argmin(1) + torch.arange(nF, device=dev) * k)
    with torch.no_grad():
        v, jn = mano(rot[pick], pca[pick], betas[pick], trans[pick])
    c = lambda x: x.detach().cpu().numpy()  # noqa: E731
    return [{"rot": c(rot[p]), "pca": c(pca[p]), "betas": c(betas[p]), "trans": c(trans[p]), "verts": c(v[i]),
             "joints": c(jn[i]), "loss": float(L[i].min())} for i, p in enumerate(pick.tolist())]


def fit(mano, target, **kw):
    """Fit MANO to target points (N,3) metres. Returns dict of numpy params, verts, joints and the final loss."""
    return fit_batch(mano, [target], **kw)[0]
