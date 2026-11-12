# Cours : Session 7 (Structure from Motion) : triangulation, calibration d'une caméra par DLT, erreur de reprojection
"""Three-view consistency with the tools of the SfM lecture (course7.pdf).

Given matches followed over three frames:
1. two projective cameras from F (Hartley & Zisserman, result 9.14):
       P1 = [I | 0],  P2 = [[e']x F | e'],  with F^T e' = 0,
2. linear triangulation of each match of views 1 and 2 (DLT, smallest singular vector),
3. camera of view 3 from the 3D-2D matches (course: "Camera calibration", the system
   A vec(M) = 0 with 2 equations per point, at least 6 points), inside RANSAC,
4. reprojection error of the points in view 3.

No calibration K is needed: the reconstruction is projective. A pairwise test with F
only constrains a point to a line; this test constrains it to a point, so it also sees
points that slide along their epipolar line (for example a depth that changes in time).
It stays valid when the three camera centres are aligned (walking straight ahead), where
the transfer by intersection of epipolar lines is degenerate.
"""
from __future__ import annotations

import numpy as np

from .homogeneous import from_h, skew, to_h
from .normalization import T_hartley, apply_T
from .ransac import ransac


def cameras_from_F(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projective cameras (P1, P2) compatible with F (x'^T F x = 0, x in image 1)."""
    U, _, _ = np.linalg.svd(F)
    e2 = U[:, -1]                                   # epipole of image 2: F^T e' = 0
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([skew(e2) @ F, e2[:, None]])
    return P1, P2


def triangulate(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Linear triangulation. x1, x2: (N, 2). Returns (N, 4) homogeneous points, unit norm.

    For each view, w p = P X gives (P_z X) x - P_x X = 0 and (P_z X) y - P_y X = 0.
    """
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    A = np.stack([x1[:, 0:1] * P1[2] - P1[0], x1[:, 1:2] * P1[2] - P1[1],
                  x2[:, 0:1] * P2[2] - P2[0], x2[:, 1:2] * P2[2] - P2[1]], axis=1)   # (N, 4, 4)
    _, _, Vt = np.linalg.svd(A)
    return Vt[:, -1, :]


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project (N, 4) homogeneous points with the 3x4 camera P. Returns (N, 2)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return from_h(X @ P.T)


def reprojection_error(P: np.ndarray, X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Distance (px) between the projections of X and the observed points x. inf if not finite."""
    e = np.linalg.norm(project(P, X) - x, axis=1)
    return np.where(np.isfinite(e), e, np.inf)


def camera_dlt(X: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Camera matrix M (3x4) from N >= 6 matches between 3D points X (N, 4) and image points x (N, 2).

    Linear system A vec(M) = 0 with the rows (course7, "Camera calibration and triangulation"):
        (M_z X) x - M_x X = 0   and   (M_z X) y - M_y X = 0.
    The image points are normalized (Hartley) and the 3D points scaled to unit norm.
    """
    X, x = np.asarray(X, float), np.asarray(x, float)
    if len(X) < 6:
        raise ValueError("the camera DLT needs at least 6 points")
    T = T_hartley(x)
    xn = apply_T(T, x)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    z = np.zeros_like(Xn)
    rows_x = np.hstack([-Xn, z, xn[:, 0:1] * Xn])
    rows_y = np.hstack([z, -Xn, xn[:, 1:2] * Xn])
    A = np.vstack([rows_x, rows_y])
    _, _, Vt = np.linalg.svd(A, full_matrices=A.shape[0] < 12)
    M = np.linalg.inv(T) @ Vt[-1].reshape(3, 4)
    return M / np.linalg.norm(M)


def estimate_camera_ransac(X: np.ndarray, x: np.ndarray, rng: np.random.Generator,
                           threshold_px: float = 2.0, p: float = 0.99, max_iters: int = 1000):
    """Robust camera of one view: RANSAC with samples of 6 points and the reprojection error.

    Returns (M, inlier_mask, n_iters).
    """
    X, x = np.asarray(X, float), np.asarray(x, float)

    def fit(idx):
        M = camera_dlt(X[idx], x[idx])
        return M if np.all(np.isfinite(M)) else None

    def residuals(M):
        return reprojection_error(M, X, x)

    return ransac(len(X), fit, residuals, s=6, threshold=threshold_px, rng=rng, p=p,
                  max_iters=max_iters)


def three_view_error(F12: np.ndarray, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray,
                     rng: np.random.Generator, threshold_px: float = 2.0,
                     max_iters: int = 1000) -> dict | None:
    """Reprojection error in view 3 of points triangulated from views 1 and 2.

    x1, x2, x3: (N, 2) positions of the same N points in the three frames; (x1, x2) should
    be inliers of F12. Returns None when the third camera cannot be estimated.
    """
    if len(x1) < 12:
        return None
    P1, P2 = cameras_from_F(F12)
    X = triangulate(P1, P2, x1, x2)
    M, mask, _ = estimate_camera_ransac(X, x3, rng, threshold_px=threshold_px, max_iters=max_iters)
    if M is None or mask.sum() < 6:
        return None
    e3 = reprojection_error(M, X, x3)
    e12 = 0.5 * (reprojection_error(P1, X, x1) + reprojection_error(P2, X, x2))
    return dict(n_tracks=int(len(x1)), reproj3_med_px=float(np.median(e3)),
                reproj3_q90_px=float(np.quantile(e3, 0.9)),
                reproj3_inlier_ratio=float(mask.mean()),
                reproj12_med_px=float(np.median(e12)))
