"""Session 7 tools on synthetic data: triangulation, camera DLT, three-view error."""
import numpy as np
import pytest

from src.geometry.errors import sampson_distance
from src.geometry.fundamental import normalized_eight_point
from src.geometry.multiview import (camera_dlt, cameras_from_F, estimate_camera_ransac,
                                    reprojection_error, three_view_error, triangulate)
from src.geometry.normalization import T_image_bounds
from src.geometry.synthetic import make_K, project, random_rotation

W, H = 1280, 720


def _scene(rng, n=300, noise=0.0, collinear=True):
    """Three cameras looking at random 3D points. collinear: centres on one line (forward walk)."""
    K = make_K(1000.0, W, H)
    X = np.column_stack([rng.uniform(-4, 4, n), rng.uniform(-2.5, 2.5, n), rng.uniform(6, 20, n)])
    poses = [(np.eye(3), np.zeros(3))]
    for k in (1, 2):
        R = random_rotation(rng, 2.0)
        c = np.array([0.02 * k, 0.0, 0.5 * k]) if collinear else rng.uniform(-0.6, 0.6, 3)
        poses.append((R, -R @ c))
    xs = [project(K, R, t, X) + noise * rng.normal(size=(n, 2)) for R, t in poses]
    cams = [K @ np.hstack([R, t[:, None]]) for R, t in poses]
    return X, xs, cams


def _F(x1, x2):
    T = T_image_bounds(W, H)
    return normalized_eight_point(x1, x2, T, T)


def test_cameras_from_F_are_compatible_with_F():
    rng = np.random.default_rng(0)
    _, xs, _ = _scene(rng)
    F = _F(xs[0], xs[1])
    P1, P2 = cameras_from_F(F)
    X = triangulate(P1, P2, xs[0], xs[1])
    assert np.median(reprojection_error(P1, X, xs[0])) < 1e-6
    assert np.median(reprojection_error(P2, X, xs[1])) < 1e-6


def test_triangulate_recovers_points_with_true_cameras():
    rng = np.random.default_rng(1)
    Xgt, xs, cams = _scene(rng)
    X = triangulate(cams[0], cams[1], xs[0], xs[1])
    X = X[:, :3] / X[:, 3:]
    assert np.max(np.abs(X - Xgt)) < 1e-6


def test_camera_dlt_recovers_projection():
    rng = np.random.default_rng(2)
    Xgt, xs, cams = _scene(rng, n=40)
    Xh = np.column_stack([Xgt, np.ones(len(Xgt))])
    M = camera_dlt(Xh, xs[2])
    assert np.max(reprojection_error(M, Xh, xs[2])) < 1e-6
    with pytest.raises(ValueError):
        camera_dlt(Xh[:5], xs[2][:5])


def test_camera_ransac_rejects_outliers():
    rng = np.random.default_rng(3)
    Xgt, xs, _ = _scene(rng, n=200, noise=0.3)
    Xh = np.column_stack([Xgt, np.ones(len(Xgt))])
    x3 = xs[2].copy()
    bad = rng.choice(200, 50, replace=False)
    x3[bad] += rng.uniform(20, 80, size=(50, 2))
    M, mask, _ = estimate_camera_ransac(Xh, x3, rng, threshold_px=2.0)
    good = np.setdiff1d(np.arange(200), bad)
    assert mask[good].mean() > 0.95 and mask[bad].mean() < 0.05


@pytest.mark.parametrize("collinear", [True, False])
def test_three_view_error_small_on_rigid_scene(collinear):
    rng = np.random.default_rng(4)
    _, xs, _ = _scene(rng, noise=0.3, collinear=collinear)
    out = three_view_error(_F(xs[0], xs[1]), xs[0], xs[1], xs[2], rng)
    assert out is not None
    assert out["reproj3_med_px"] < 1.5 and out["reproj3_inlier_ratio"] > 0.8


def test_three_view_error_sees_motion_along_epipolar_lines():
    """Points that slide along their epipolar line in view 3 still satisfy F between
    views 2 and 3, but they cannot be reprojected: only the three-view test sees them."""
    rng = np.random.default_rng(5)
    _, xs, _ = _scene(rng, noise=0.2)
    F12, F23 = _F(xs[0], xs[1]), _F(xs[1], xs[2])
    lines = np.column_stack([xs[1], np.ones(len(xs[1]))]) @ F23.T        # epipolar lines in view 3
    d = np.column_stack([-lines[:, 1], lines[:, 0]])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    x3_slid = xs[2] + rng.uniform(-10, 10, size=(len(d), 1)) * d          # slide by up to 10 px
    assert np.median(sampson_distance(F23, xs[1], x3_slid)) < 0.5         # F does not see it
    rigid = three_view_error(F12, xs[0], xs[1], xs[2], rng)
    slid = three_view_error(F12, xs[0], xs[1], x3_slid, rng)
    assert slid is None or slid["reproj3_med_px"] > 4 * rigid["reproj3_med_px"]
