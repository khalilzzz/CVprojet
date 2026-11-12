# Cours : Session 4 (Stereo / Single view modeling) : dualité point-droite (EP4 slide 75) ; Session 3 : RANSAC
"""Vanishing point estimation from line segments.

Parallel 3D lines project to image lines that meet in one vanishing point (VP).
- A segment (x1, y1, x2, y2) gives the homogeneous line l = p1 x p2.
- Two lines give a VP hypothesis v = l1 x l2 (minimal sample s = 2).
- Inlier test (angular): the angle between the segment direction and the line from the
  segment midpoint to v. This also works for a VP at infinity (w = 0).
- Refit: v minimizing sum (l_i^T v)^2 with ||v|| = 1 (SVD), in normalized coordinates.
- Several VPs: sequential RANSAC (remove the inliers of each VP, then repeat).

The vanishing-point RANSAC is not detailed in the slides. Reference for the use as an
AI-image cue: Sarkar et al., "Shadows Don't Lie and Lines Can't Bend!", CVPR 2024.
"""
from __future__ import annotations

import numpy as np

from .homogeneous import line_through
from .normalization import T_hartley, apply_T
from .ransac import ransac


def segments_to_lines(segments: np.ndarray) -> np.ndarray:
    """(N, 4) segments [x1, y1, x2, y2] -> (N, 3) homogeneous lines."""
    segments = np.asarray(segments, dtype=float)
    return line_through(segments[:, :2], segments[:, 2:])


def vp_residual_angle(vp: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Angle (degrees) between each segment and the line joining its midpoint to vp.

    vp is homogeneous (3,). The direction from midpoint m to vp is vp[:2] - vp[2] * m,
    up to scale, so a VP at infinity (vp[2] = 0) is handled.
    """
    segments = np.asarray(segments, dtype=float)
    vp = np.asarray(vp, dtype=float)
    p1, p2 = segments[:, :2], segments[:, 2:]
    m = 0.5 * (p1 + p2)
    d = p2 - p1
    g = vp[:2][None, :] - vp[2] * m
    nd = np.linalg.norm(d, axis=1)
    ng = np.linalg.norm(g, axis=1)
    cos = np.abs(np.einsum("ij,ij->i", d, g)) / np.maximum(nd * ng, 1e-300)
    ang = np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))
    ang[ng < 1e-12] = 0.0  # vp on the midpoint: the segment line passes through it
    return ang


def refine_vp(segments: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares VP from N >= 2 segments: min sum w_i (l_i^T v)^2, ||v|| = 1.

    Computed in Hartley-normalized coordinates for conditioning. Returns a unit-norm vp.
    """
    segments = np.asarray(segments, dtype=float)
    T = T_hartley(segments.reshape(-1, 2))
    s_n = np.concatenate([apply_T(T, segments[:, :2]), apply_T(T, segments[:, 2:])], axis=1)
    L = segments_to_lines(s_n)
    L = L / np.linalg.norm(L[:, :2], axis=1, keepdims=True)
    if weights is not None:
        L = L * np.sqrt(np.asarray(weights, dtype=float))[:, None]
    _, _, Vt = np.linalg.svd(L, full_matrices=L.shape[0] < 3)   # only V is needed (L is N x 3)
    v = np.linalg.inv(T) @ Vt[-1]
    return v / np.linalg.norm(v)


def estimate_vps(
    segments: np.ndarray,
    rng: np.random.Generator,
    n_vps: int = 3,
    angle_threshold_deg: float = 2.0,
    min_inliers: int = 5,
    p: float = 0.99,
    max_iters: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential RANSAC for up to n_vps vanishing points.

    Returns:
        vps: (K, 3) unit-norm homogeneous VPs, K <= n_vps, strongest first.
        labels: (N,) index of the VP each segment supports, -1 for unexplained segments.
    """
    segments = np.asarray(segments, dtype=float)
    labels = -np.ones(len(segments), dtype=int)
    vps: list[np.ndarray] = []
    remaining = np.arange(len(segments))
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)

    for k in range(n_vps):
        if len(remaining) < max(2, min_inliers):
            break
        seg_k = segments[remaining]
        lines_k = segments_to_lines(seg_k)

        def fit(idx: np.ndarray):
            if len(idx) == 2:
                v = np.cross(lines_k[idx[0]], lines_k[idx[1]])
                n = np.linalg.norm(v)
                return None if n < 1e-12 else v / n
            return refine_vp(seg_k[idx], weights=lengths[remaining][idx])

        def residuals(v: np.ndarray) -> np.ndarray:
            return vp_residual_angle(v, seg_k)

        v, mask, _ = ransac(len(seg_k), fit, residuals, s=2, threshold=angle_threshold_deg,
                            rng=rng, p=p, max_iters=max_iters)
        if v is None or mask.sum() < min_inliers:
            break
        vps.append(v)
        labels[remaining[mask]] = k
        remaining = remaining[~mask]

    return np.array(vps).reshape(-1, 3), labels
