# Cours : Session 4 (Stereo) : coordonnées homogènes, dualité point-droite, produit vectoriel (EP4 slides 6, 73-75)
"""Homogeneous coordinates helpers.

Conventions (course): an image point (u, v) is x = (u, v, 1)^T. A line l = (a, b, c)
contains x iff l^T x = 0. The line through two points is p x q and the intersection of
two lines is l1 x l2 (point-line duality, EP4 slide 75).
"""
from __future__ import annotations

import numpy as np


def to_h(pts: np.ndarray) -> np.ndarray:
    """(N, 2) Cartesian points -> (N, 3) homogeneous points with w = 1."""
    pts = np.asarray(pts, dtype=float)
    return np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1)


def from_h(pts_h: np.ndarray) -> np.ndarray:
    """(N, 3) homogeneous points -> (N, 2) Cartesian points (divide by w)."""
    pts_h = np.asarray(pts_h, dtype=float)
    return pts_h[..., :-1] / pts_h[..., -1:]


def skew(t: np.ndarray) -> np.ndarray:
    """[t]_x, the 3x3 skew-symmetric matrix such that [t]_x v = t x v (EP4 slide 73)."""
    t = np.asarray(t, dtype=float).ravel()
    return np.array([[0.0, -t[2], t[1]],
                     [t[2], 0.0, -t[0]],
                     [-t[1], t[0], 0.0]])


def _as_h(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return to_h(p) if p.shape[-1] == 2 else p


def line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Homogeneous line through points p and q (Cartesian or homogeneous, batched)."""
    return np.cross(_as_h(p), _as_h(q))


def intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Homogeneous intersection point of lines l1 and l2 (w = 0 for parallel lines)."""
    return np.cross(np.asarray(l1, dtype=float), np.asarray(l2, dtype=float))


def normalize_lines(lines: np.ndarray) -> np.ndarray:
    """Scale lines so that a^2 + b^2 = 1; then |l^T x| is the point-line distance in px."""
    lines = np.asarray(lines, dtype=float)
    return lines / np.linalg.norm(lines[..., :2], axis=-1, keepdims=True)
