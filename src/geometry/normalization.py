# Cours : Session 4 (Stereo) : normalisation des coordonnées pour l'algorithme des 8 points (EP4 slides 102-104)
"""Similarity transforms T used to condition linear least-squares problems.

- ``T_image_bounds``: the course version (EP4 slide 103), maps the image
  [0, w] x [0, h] to [-1, 1] x [-1, 1].
- ``T_hartley``: Hartley's version, centroid at the origin and mean distance sqrt(2).
"""
from __future__ import annotations

import numpy as np


def T_image_bounds(w: float, h: float) -> np.ndarray:
    """Course normalization: T = [[2/w, 0, -1], [0, 2/h, -1], [0, 0, 1]]."""
    return np.array([[2.0 / w, 0.0, -1.0],
                     [0.0, 2.0 / h, -1.0],
                     [0.0, 0.0, 1.0]])


def T_hartley(pts: np.ndarray) -> np.ndarray:
    """Hartley normalization of (N, 2) points: zero centroid, mean distance sqrt(2)."""
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    mean_dist = np.linalg.norm(pts - c, axis=1).mean()
    s = np.sqrt(2.0) / max(mean_dist, 1e-12)
    return np.array([[s, 0.0, -s * c[0]],
                     [0.0, s, -s * c[1]],
                     [0.0, 0.0, 1.0]])


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine normalization T to (N, 2) Cartesian points."""
    pts = np.asarray(pts, dtype=float)
    return pts @ T[:2, :2].T + T[:2, 2]
