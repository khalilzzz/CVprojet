# Cours : Session 4 (Stereo) : estimation de F, algorithme des 8 points normalisé (EP4 slides 98-104) + RANSAC (Session 3)
"""Fundamental matrix estimation, with the course notation.

x = (u, v, 1)^T in image 1, x' = (u', v', 1)^T in image 2, x'^T F x = 0.
Each match gives one row of A:
    [u u', v u', u', u v', v v', v', u, v, 1] . (f11, f12, f13, f21, f22, f23, f31, f32, f33) = 0
"""
from __future__ import annotations

import numpy as np

from .errors import sampson_distance
from .normalization import T_hartley, T_image_bounds, apply_T
from .ransac import ransac


def enforce_rank2(F: np.ndarray) -> np.ndarray:
    """Closest rank-2 matrix in Frobenius norm: F' = U diag(s1, s2, 0) V^T (EP4 slide 100)."""
    U, D, Vt = np.linalg.svd(F)
    return U @ np.diag([D[0], D[1], 0.0]) @ Vt


def eight_point(x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """Unnormalized 8-point algorithm on N >= 8 matches (EP4 slide 101)."""
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    if len(x) < 8:
        raise ValueError("the 8-point algorithm needs at least 8 matches")
    u, v = x[:, 0], x[:, 1]
    up, vp = xp[:, 0], xp[:, 1]
    # 1- build the constraint matrix A
    A = np.stack([u * up, v * up, up, u * vp, v * vp, vp, u, v, np.ones_like(u)], axis=1)
    # 2- SVD, 3- singular vector of the smallest singular value, 4- reshape to 3x3
    _, _, Vt = np.linalg.svd(A, full_matrices=A.shape[0] < 9)   # only V is needed (A is N x 9)
    F = Vt[-1].reshape(3, 3)
    # 5- enforce rank 2
    return enforce_rank2(F)


def normalized_eight_point(x: np.ndarray, xp: np.ndarray,
                           Tx: np.ndarray, Txp: np.ndarray) -> np.ndarray:
    """Normalized 8-point (EP4 slide 104): x^ = T x, x'^ = T' x', F^ = 8pt(x^, x'^), F = T'^T F^ T.

    The result is scaled to unit Frobenius norm.
    """
    F_hat = eight_point(apply_T(Tx, x), apply_T(Txp, xp))
    F = Txp.T @ F_hat @ Tx
    return F / np.linalg.norm(F)


def normalization_transforms(x: np.ndarray, xp: np.ndarray, mode: str = "course",
                             img_shapes: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, T') for the given mode.

    mode "course" needs img_shapes = ((h1, w1), (h2, w2)); mode "hartley" uses the points.
    """
    if mode == "course":
        if img_shapes is None:
            raise ValueError("course normalization needs img_shapes=((h1, w1), (h2, w2))")
        (h1, w1), (h2, w2) = img_shapes
        return T_image_bounds(w1, h1), T_image_bounds(w2, h2)
    if mode == "hartley":
        return T_hartley(x), T_hartley(xp)
    raise ValueError(f"unknown normalization mode: {mode}")


def estimate_F_ransac(
    x: np.ndarray,
    xp: np.ndarray,
    rng: np.random.Generator,
    img_shapes: tuple | None = None,
    normalization: str = "course",
    threshold_px: float = 1.0,
    p: float = 0.99,
    max_iters: int = 5000,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """Robust F: RANSAC with minimal samples of s = 8 and the Sampson distance as residual.

    Returns (F, inlier_mask, n_iters).
    """
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    Tx, Txp = normalization_transforms(x, xp, normalization, img_shapes)

    def fit(idx: np.ndarray):
        F = normalized_eight_point(x[idx], xp[idx], Tx, Txp)
        return F if np.all(np.isfinite(F)) else None

    def residuals(F: np.ndarray) -> np.ndarray:
        return sampson_distance(F, x, xp)

    return ransac(len(x), fit, residuals, s=8, threshold=threshold_px,
                  rng=rng, p=p, max_iters=max_iters)
