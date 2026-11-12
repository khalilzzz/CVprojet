# Cours : Session 3 (Warping, RANSAC, Panoramas) : homographie par DLT (EP3 slides 93-94, EP4 slide 66) + RANSAC
"""Homography estimation with the Direct Linear Transform.

Course notation (EP3 slide 94): x' ~ H x with H = [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]].
Each match gives two equations, linear in h:
    x'_i (h20 x_i + h21 y_i + h22) = h00 x_i + h01 y_i + h02
    y'_i (h20 x_i + h21 y_i + h22) = h10 x_i + h11 y_i + h12
so 4 matches are needed. We minimize ||A h|| with ||h|| = 1 (smallest singular vector).
"""
from __future__ import annotations

import numpy as np

from .errors import symmetric_transfer_error
from .homogeneous import from_h, to_h
from .normalization import T_hartley, apply_T
from .ransac import ransac


def apply_H(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Map (N, 2) points with H."""
    return from_h(to_h(x) @ H.T)


def _dlt(x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    xs, ys = x[:, 0], x[:, 1]
    xps, yps = xp[:, 0], xp[:, 1]
    z, o = np.zeros_like(xs), np.ones_like(xs)
    rows_x = np.stack([-xs, -ys, -o, z, z, z, xps * xs, xps * ys, xps], axis=1)
    rows_y = np.stack([z, z, z, -xs, -ys, -o, yps * xs, yps * ys, yps], axis=1)
    A = np.concatenate([rows_x, rows_y], axis=0)
    _, _, Vt = np.linalg.svd(A, full_matrices=A.shape[0] < 9)   # only V is needed (A is 2N x 9)
    return Vt[-1].reshape(3, 3)


def dlt_homography(x: np.ndarray, xp: np.ndarray, normalize: bool = True) -> np.ndarray:
    """H from N >= 4 matches, scaled so that h22 = 1 when possible."""
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    if len(x) < 4:
        raise ValueError("the DLT needs at least 4 matches")
    if normalize:
        T, Tp = T_hartley(x), T_hartley(xp)
        H = np.linalg.inv(Tp) @ _dlt(apply_T(T, x), apply_T(Tp, xp)) @ T
    else:
        H = _dlt(x, xp)
    if abs(H[2, 2]) > 1e-12:
        return H / H[2, 2]
    return H / np.linalg.norm(H)


def estimate_H_ransac(
    x: np.ndarray,
    xp: np.ndarray,
    rng: np.random.Generator,
    threshold_px: float = 3.0,
    p: float = 0.99,
    max_iters: int = 5000,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """Robust H: RANSAC with s = 4 and the symmetric transfer error. Returns (H, inlier_mask, n_iters)."""
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)

    def fit(idx: np.ndarray):
        H = dlt_homography(x[idx], xp[idx])
        if not np.all(np.isfinite(H)):
            return None
        # Reject near-singular H (degenerate samples, e.g. 3 collinear points).
        sv = np.linalg.svd(H, compute_uv=False)
        return None if sv[-1] < 1e-8 * sv[0] else H

    def residuals(H: np.ndarray) -> np.ndarray:
        return symmetric_transfer_error(H, x, xp)

    return ransac(len(x), fit, residuals, s=4, threshold=threshold_px,
                  rng=rng, p=p, max_iters=max_iters)
