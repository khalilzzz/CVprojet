# Cours : Session 3 (Features, Warping, RANSAC, Panoramas) : RANSAC (EP3 slides 100-112)
"""Generic RANSAC (hypothesize and test), following the course:

1. draw a minimal sample of size s and fit a model,
2. count inliers (residual < threshold),
3. keep the model with the most inliers,
4. adapt the number of rounds N >= log(1 - p) / log(1 - (1 - e)^s) (EP3 slide 109),
5. final least-squares fit on all inliers (EP3 slide 112).
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


def ransac_num_iters(p: float, e: float, s: int) -> int:
    """Number of rounds so that, with probability p, one sample of size s is outlier-free.

    e is the outlier ratio. Returns a very large number when (1 - e)^s underflows.
    """
    if e <= 0.0:
        return 1
    if e >= 1.0:
        return 10**9
    good = (1.0 - e) ** s
    denom = math.log1p(-good) if good < 1.0 else -math.inf
    if denom == 0.0:
        return 10**9
    return int(math.ceil(math.log(1.0 - p) / denom))


def ransac(
    n_data: int,
    fit: Callable[[np.ndarray], Any],
    residuals: Callable[[Any], np.ndarray],
    s: int,
    threshold: float,
    rng: np.random.Generator,
    p: float = 0.99,
    max_iters: int = 5000,
    refit: bool = True,
    refit_rounds: int = 3,
) -> tuple[Any, np.ndarray, int]:
    """Run RANSAC.

    Args:
        n_data: number of data items (indices 0..n_data-1).
        fit: maps an index array to a model, or None if the sample is degenerate.
            It must accept s indices (minimal sample) and more (refit on inliers).
        residuals: maps a model to an (n_data,) array of residuals.
        s: minimal sample size.
        threshold: inlier if residual < threshold.

    Returns:
        (model, inlier_mask, n_iters). model is None if no valid model was found.
    """
    if n_data < s:
        return None, np.zeros(n_data, dtype=bool), 0

    best_model, best_mask, best_count, best_cost = None, np.zeros(n_data, dtype=bool), -1, np.inf
    n_needed = max_iters
    it = 0
    while it < min(n_needed, max_iters):
        it += 1
        idx = rng.choice(n_data, size=s, replace=False)
        model = fit(idx)
        if model is None:
            continue
        r = residuals(model)
        mask = r < threshold
        count = int(mask.sum())
        cost = float(r[mask].sum())
        if count > best_count or (count == best_count and cost < best_cost):
            best_model, best_mask, best_count, best_cost = model, mask, count, cost
            e = 1.0 - count / n_data
            n_needed = ransac_num_iters(p, e, s)

    if best_model is None:
        return None, best_mask, it

    # Final least-squares fit on all inliers (EP3 slide 112). The inlier set is
    # recomputed with the refit model; repeat a few times until it is stable.
    if refit:
        for _ in range(refit_rounds):
            if best_mask.sum() < s:
                break
            model = fit(np.flatnonzero(best_mask))
            if model is None:
                break
            mask = residuals(model) < threshold
            if mask.sum() < s:
                break
            stable = np.array_equal(mask, best_mask)
            best_model, best_mask = model, mask
            if stable:
                break
    return best_model, best_mask, it
