# Cours : Sessions 3-4 : figure de contrôle du cœur géométrique sur données synthétiques
"""Sanity figure for the geometry core (synthetic data only).

Run from the repo root:  python -m src.geometry.demo
Writes results/figures/geometry_synthetic.png:
  left   - image 2 of a synthetic pair: RANSAC inliers with their epipolar lines, outliers;
  right  - synthetic Manhattan segments coloured by the vanishing point they support.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.geometry.errors import symmetric_epipolar_distance
from src.geometry.fundamental import estimate_F_ransac
from src.geometry.homogeneous import to_h
from src.geometry.synthetic import manhattan_segments, two_view_scene, vp_direction_error_deg
from src.geometry.vanishing import estimate_vps
from src.utils.config import REPO_ROOT, load_config, set_seed


def _draw_line(ax, l, w, h, **kw):
    a, b, c = l
    if abs(b) > abs(a):
        xs = np.array([0.0, w])
        ax.plot(xs, -(a * xs + c) / b, **kw)
    else:
        ys = np.array([0.0, h])
        ax.plot(-(b * ys + c) / a, ys, **kw)


def main() -> None:
    cfg = load_config("configs/geometry.yaml")
    rng = set_seed(cfg["seed"])

    sc = two_view_scene(rng, n=200, noise_px=0.5, outlier_ratio=0.3)
    F, mask, n_iters = estimate_F_ransac(
        sc["x"], sc["xp"], rng, img_shapes=(sc["shape"],) * 2,
        normalization=cfg["fundamental"]["normalization"],
        threshold_px=cfg["fundamental"]["threshold_px"],
        p=cfg["ransac"]["p"], max_iters=cfg["ransac"]["max_iters"])
    err = np.median(symmetric_epipolar_distance(F, sc["x_clean"], sc["xp_clean"]))
    truth = sc["inlier"]
    print(f"F: {mask.sum()} inliers / {len(mask)}, {n_iters} iterations, "
          f"recall {(mask & truth).sum() / truth.sum():.3f}, "
          f"median epipolar distance to clean data {err:.3f} px")

    ms = manhattan_segments(rng)
    vps, labels = estimate_vps(ms["segments"], rng, n_vps=cfg["vanishing"]["n_vps"],
                               angle_threshold_deg=cfg["vanishing"]["angle_threshold_deg"],
                               min_inliers=cfg["vanishing"]["min_inliers"])
    errs = [min(vp_direction_error_deg(v, g, ms["K"]) for g in ms["vps"]) for v in vps]
    print(f"VPs: {len(vps)} found, direction errors (deg): {np.round(errs, 3).tolist()}, "
          f"unexplained segments: {(labels < 0).sum()}")

    h, w = sc["shape"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axes[0]
    lines2 = to_h(sc["x"]) @ F.T
    for i in np.flatnonzero(mask)[:15]:
        _draw_line(ax, lines2[i], w, h, color="tab:green", lw=0.6, alpha=0.7)
    ax.scatter(*sc["xp"][mask].T, s=6, c="tab:green", label="inliers")
    ax.scatter(*sc["xp"][~mask].T, s=10, c="tab:red", marker="x", label="outliers")
    ax.set(xlim=(0, w), ylim=(h, 0), aspect="equal",
           title=f"Epipolar lines in image 2 (median error {err:.2f} px)")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    colors = ["tab:blue", "tab:orange", "tab:purple"]
    for seg, lab in zip(ms["segments"], labels):
        c = colors[lab] if lab >= 0 else "0.6"
        ax.plot(seg[[0, 2]], seg[[1, 3]], color=c, lw=1.2 if lab >= 0 else 0.8,
                ls="-" if lab >= 0 else ":")
    ax.set(xlim=(0, w), ylim=(h, 0), aspect="equal",
           title="Segments by vanishing point (grey: unexplained)")

    out = REPO_ROOT / "results" / "figures" / "geometry_synthetic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
