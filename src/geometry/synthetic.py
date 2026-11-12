# Cours : Session 4 (Stereo) : modèle sténopé x = K [R | t] X (EP4 slides 10-14) ; données synthétiques pour les tests
"""Synthetic scenes with known ground truth, for testing the geometry core.

Camera 1 is [I | 0]; camera 2 maps a 3D point X (camera-1 frame) to K (R X + t).
Then x'^T F x = 0 with F = K^-T [t]_x R K^-1 (EP4 slides 85-91).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .homogeneous import from_h, skew, to_h

W, H = 1280, 720


def make_K(f: float = 1000.0, w: int = W, h: int = H) -> np.ndarray:
    return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


def random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.radians(rng.uniform(0.3 * max_deg, max_deg))
    return Rotation.from_rotvec(axis * angle).as_matrix()


def project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project (N, 3) points with K [R | t]."""
    return from_h((X @ R.T + t) @ K.T)


def _inside(x: np.ndarray, w: int, h: int) -> np.ndarray:
    return (x[:, 0] >= 0) & (x[:, 0] < w) & (x[:, 1] >= 0) & (x[:, 1] < h)


def _add_outliers(xp: np.ndarray, ratio: float, rng: np.random.Generator,
                  w: int, h: int) -> np.ndarray:
    """Replace a share of xp with uniform random points. Returns the inlier mask."""
    n = len(xp)
    n_out = int(round(ratio * n))
    out_idx = rng.choice(n, size=n_out, replace=False)
    xp[out_idx] = rng.uniform([0, 0], [w, h], size=(n_out, 2))
    inlier = np.ones(n, dtype=bool)
    inlier[out_idx] = False
    return inlier


def two_view_scene(rng: np.random.Generator, n: int = 200, noise_px: float = 0.5,
                   outlier_ratio: float = 0.0, w: int = W, h: int = H) -> dict:
    """General 3D scene seen by two cameras with a real translation."""
    K = make_K(rng.uniform(800, 1200), w, h)
    R = random_rotation(rng, 8.0)
    t = np.array([rng.uniform(0.5, 1.0) * rng.choice([-1, 1]), rng.uniform(-0.3, 0.3),
                  rng.uniform(-0.3, 0.3)])
    X_list, x_list, xp_list = [], [], []
    while sum(len(a) for a in x_list) < n:
        x = rng.uniform([0, 0], [w, h], size=(4 * n, 2))
        Z = rng.uniform(4.0, 12.0, size=(4 * n, 1))
        X = (to_h(x) @ np.linalg.inv(K).T) * Z
        Xc2 = X @ R.T + t
        xp = project(K, R, t, X)
        ok = (Xc2[:, 2] > 0.1) & _inside(xp, w, h)
        X_list.append(X[ok]); x_list.append(x[ok]); xp_list.append(xp[ok])
    X = np.concatenate(X_list)[:n]
    x_clean = np.concatenate(x_list)[:n]
    xp_clean = np.concatenate(xp_list)[:n]
    F = np.linalg.inv(K).T @ skew(t) @ R @ np.linalg.inv(K)
    F /= np.linalg.norm(F)
    x = x_clean + rng.normal(scale=noise_px, size=x_clean.shape)
    xp = xp_clean + rng.normal(scale=noise_px, size=xp_clean.shape)
    inlier = _add_outliers(xp, outlier_ratio, rng, w, h)
    return dict(x=x, xp=xp, x_clean=x_clean, xp_clean=xp_clean, X=X, F=F, K=K, R=R, t=t,
                inlier=inlier, shape=(h, w))


def homography_scene(rng: np.random.Generator, n: int = 200, noise_px: float = 0.5,
                     outlier_ratio: float = 0.0, planar: bool = True,
                     w: int = W, h: int = H) -> dict:
    """Matches related by a homography.

    planar=True: a 3D plane n^T X = d seen with translation, H = K (R + t n^T / d) K^-1.
    planar=False: pure rotation, H = K R K^-1.
    """
    K = make_K(rng.uniform(800, 1200), w, h)
    R = random_rotation(rng, 8.0)
    if planar:
        t = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)])
        nrm = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 1.0])
        nrm /= np.linalg.norm(nrm)
        d = rng.uniform(5.0, 10.0)
        Hgt = K @ (R + np.outer(t, nrm) / d) @ np.linalg.inv(K)
    else:
        Hgt = K @ R @ np.linalg.inv(K)
    Hgt /= Hgt[2, 2]
    x_all = rng.uniform([0, 0], [w, h], size=(6 * n, 2))
    xp_all = from_h(to_h(x_all) @ Hgt.T)
    ok = _inside(xp_all, w, h)
    x_clean, xp_clean = x_all[ok][:n], xp_all[ok][:n]
    x = x_clean + rng.normal(scale=noise_px, size=x_clean.shape)
    xp = xp_clean + rng.normal(scale=noise_px, size=xp_clean.shape)
    inlier = _add_outliers(xp, outlier_ratio, rng, w, h)
    return dict(x=x, xp=xp, x_clean=x_clean, xp_clean=xp_clean, H=Hgt, K=K, inlier=inlier,
                shape=(h, w))


def manhattan_segments(rng: np.random.Generator, R: np.ndarray | None = None,
                       n_per_axis: int = 40, n_outliers: int = 30, noise_px: float = 0.5,
                       min_len_px: float = 40.0, w: int = W, h: int = H) -> dict:
    """Segments along the 3 world axes (a "Manhattan" scene) plus random clutter.

    Ground-truth VP of axis i: K R e_i. Labels: 0, 1, 2 for the axes, -1 for clutter.
    """
    K = make_K(1000.0, w, h)
    if R is None:
        R = Rotation.from_euler("yx", [35.0, 15.0], degrees=True).as_matrix()
    segs, labels = [], []
    Kinv = np.linalg.inv(K)
    for axis in range(3):
        e = np.eye(3)[axis]
        count = 0
        while count < n_per_axis:
            x0 = rng.uniform([0.1 * w, 0.1 * h], [0.9 * w, 0.9 * h])
            P0 = (Kinv @ np.append(x0, 1.0)) * rng.uniform(5.0, 15.0)   # camera frame
            P1 = P0 + R @ e * rng.uniform(0.5, 2.5)                      # along the axis
            if P1[2] <= 0.5:
                continue
            p0 = from_h(K @ P0)
            p1 = from_h(K @ P1)
            if not (_inside(p1[None], w, h)[0]) or np.linalg.norm(p1 - p0) < min_len_px:
                continue
            segs.append(np.concatenate([p0, p1]))
            labels.append(axis)
            count += 1
    segs = np.array(segs) + rng.normal(scale=noise_px, size=(len(segs), 4))
    a = rng.uniform([0, 0], [w, h], size=(n_outliers, 2))
    ang = rng.uniform(0, np.pi, size=n_outliers)
    ln = rng.uniform(min_len_px, 150.0, size=n_outliers)
    b = a + ln[:, None] * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    segs = np.concatenate([segs, np.concatenate([a, b], axis=1)])
    labels = np.concatenate([labels, -np.ones(n_outliers, dtype=int)])
    vps_gt = (K @ R).T          # row i = K R e_i
    return dict(segments=segs, labels=labels, vps=vps_gt, K=K, R=R, shape=(h, w))


def vp_direction_error_deg(vp: np.ndarray, vp_gt: np.ndarray, K: np.ndarray) -> float:
    """Angle between the 3D directions K^-1 vp and K^-1 vp_gt (sign-free), in degrees."""
    d1 = np.linalg.solve(K, vp)
    d2 = np.linalg.solve(K, vp_gt)
    c = abs(d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))
