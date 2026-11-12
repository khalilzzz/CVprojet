# Cours : Session 4 (Stereo) : erreurs épipolaires ; Session 3 : erreur de transfert d'une homographie
"""Residuals for two-view models. All distances are in pixels.

Course convention: x'^T F x = 0 with x in image 1 and x' in image 2 (EP4 slide 98).
The Sampson distance is not in the slides; reference: Hartley & Zisserman,
Multiple View Geometry, 2nd ed., eq. 11.9 (first-order geometric error).
"""
from __future__ import annotations

import numpy as np

from .homogeneous import from_h, to_h


def algebraic_error(F: np.ndarray, x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """|x'^T F x| for each correspondence."""
    return np.abs(np.einsum("ij,jk,ik->i", to_h(xp), F, to_h(x)))


def sampson_distance(F: np.ndarray, x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """Sampson distance sqrt( (x'^T F x)^2 / ((Fx)_1^2 + (Fx)_2^2 + (F^T x')_1^2 + (F^T x')_2^2) )."""
    xh, xph = to_h(x), to_h(xp)
    Fx = xh @ F.T            # epipolar lines in image 2
    Ftxp = xph @ F           # epipolar lines in image 1
    num = np.einsum("ij,ij->i", xph, Fx)
    den = Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftxp[:, 0] ** 2 + Ftxp[:, 1] ** 2
    return np.abs(num) / np.sqrt(np.maximum(den, 1e-300))


def symmetric_epipolar_distance(F: np.ndarray, x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """Mean of the two point-to-epipolar-line distances: d(x', Fx) and d(x, F^T x')."""
    xh, xph = to_h(x), to_h(xp)
    Fx = xh @ F.T
    Ftxp = xph @ F
    num = np.abs(np.einsum("ij,ij->i", xph, Fx))
    d2 = num / np.maximum(np.hypot(Fx[:, 0], Fx[:, 1]), 1e-300)
    d1 = num / np.maximum(np.hypot(Ftxp[:, 0], Ftxp[:, 1]), 1e-300)
    return 0.5 * (d1 + d2)


def transfer_error(H: np.ndarray, x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """One-way transfer error ||x' - H x||."""
    return np.linalg.norm(from_h(to_h(x) @ H.T) - xp, axis=1)


def symmetric_transfer_error(H: np.ndarray, x: np.ndarray, xp: np.ndarray) -> np.ndarray:
    """Mean of ||x' - H x|| and ||x - H^-1 x'||. Returns +inf if H is singular."""
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(x), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = 0.5 * (transfer_error(H, x, xp) + transfer_error(Hinv, xp, x))
    return np.nan_to_num(e, nan=np.inf)
