import cv2
import numpy as np
import pytest

from src.geometry.errors import algebraic_error, sampson_distance, symmetric_epipolar_distance
from src.geometry.fundamental import (eight_point, estimate_F_ransac, normalization_transforms,
                                      normalized_eight_point)
from src.geometry.normalization import apply_T
from src.geometry.synthetic import two_view_scene


def _same_up_to_scale(A, B):
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    return min(np.linalg.norm(A - B), np.linalg.norm(A + B))


@pytest.mark.parametrize("mode", ["course", "hartley"])
def test_noise_free_recovers_ground_truth(mode):
    rng = np.random.default_rng(1)
    sc = two_view_scene(rng, n=50, noise_px=0.0)
    Tx, Txp = normalization_transforms(sc["x"], sc["xp"], mode, (sc["shape"], sc["shape"]))
    F = normalized_eight_point(sc["x"], sc["xp"], Tx, Txp)
    assert sampson_distance(F, sc["x"], sc["xp"]).max() < 1e-6
    sv = np.linalg.svd(F, compute_uv=False)
    assert sv[2] / sv[0] < 1e-10                         # rank 2
    assert _same_up_to_scale(F, sc["F"]) < 1e-6


def test_ground_truth_F_satisfies_epipolar_constraint():
    sc = two_view_scene(np.random.default_rng(2), n=50, noise_px=0.0)
    assert sampson_distance(sc["F"], sc["x"], sc["xp"]).max() < 1e-8


def _constraint_matrix(x, xp):
    u, v = x.T
    up, vp = xp.T
    return np.stack([u * up, v * up, up, u * vp, v * vp, vp, u, v, np.ones_like(u)], axis=1)


def test_normalization_improves_conditioning():
    """EP4 slide 102: columns of A differ by orders of magnitude without normalization."""
    sc = two_view_scene(np.random.default_rng(3), n=100, noise_px=1.0)
    Tx, Txp = normalization_transforms(sc["x"], sc["xp"], "course", (sc["shape"], sc["shape"]))
    cond_raw = np.linalg.cond(_constraint_matrix(sc["x"], sc["xp"]))
    cond_norm = np.linalg.cond(_constraint_matrix(apply_T(Tx, sc["x"]), apply_T(Txp, sc["xp"])))
    assert cond_norm < 1e-3 * cond_raw


def test_normalization_reduces_error_under_noise():
    """Over many random scenes, the normalized version is usually more accurate.

    Error is measured on the noise-free correspondences (distance to the true geometry).
    A single scene can go either way, so the test is statistical.
    """
    e_norm, e_raw = [], []
    for seed in range(30):
        sc = two_view_scene(np.random.default_rng(100 + seed), n=100, noise_px=1.0)
        Tx, Txp = normalization_transforms(sc["x"], sc["xp"], "course", (sc["shape"],) * 2)
        F_norm = normalized_eight_point(sc["x"], sc["xp"], Tx, Txp)
        F_raw = eight_point(sc["x"], sc["xp"])
        e_norm.append(np.median(symmetric_epipolar_distance(F_norm, sc["x_clean"], sc["xp_clean"])))
        e_raw.append(np.median(symmetric_epipolar_distance(F_raw, sc["x_clean"], sc["xp_clean"])))
    e_norm, e_raw = np.array(e_norm), np.array(e_raw)
    assert (e_norm < e_raw).mean() >= 0.8
    assert np.median(e_norm) < 0.5 * np.median(e_raw)
    assert np.median(e_norm) < 1.0


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_ransac_with_outliers(seed):
    rng = np.random.default_rng(seed)
    sc = two_view_scene(rng, n=200, noise_px=0.5, outlier_ratio=0.3)
    F, mask, n_iters = estimate_F_ransac(sc["x"], sc["xp"], rng, img_shapes=(sc["shape"],) * 2,
                                         threshold_px=1.5)
    truth = sc["inlier"]
    recall = (mask & truth).sum() / truth.sum()
    precision = (mask & truth).sum() / mask.sum()
    assert recall > 0.95 and precision > 0.95, (recall, precision)
    assert np.median(symmetric_epipolar_distance(F, sc["x_clean"], sc["xp_clean"])) < 1.0
    assert n_iters < 5000


def test_matches_opencv_8point():
    rng = np.random.default_rng(7)
    sc = two_view_scene(rng, n=150, noise_px=0.5)
    Tx, Txp = normalization_transforms(sc["x"], sc["xp"], "hartley")
    F_ours = normalized_eight_point(sc["x"], sc["xp"], Tx, Txp)
    F_cv, _ = cv2.findFundamentalMat(sc["x"], sc["xp"], cv2.FM_8POINT)
    # Same algorithm (Hartley normalization + rank 2): nearly the same matrix.
    assert _same_up_to_scale(F_ours, F_cv) < 1e-3
    e_ours = np.median(sampson_distance(F_ours, sc["x"], sc["xp"]))
    e_cv = np.median(sampson_distance(F_cv, sc["x"], sc["xp"]))
    assert abs(e_ours - e_cv) < 0.05


@pytest.mark.parametrize("seed", [4, 5, 8])
def test_ransac_at_least_as_good_as_opencv(seed):
    """OpenCV thresholds a different distance, so the masks differ near the threshold.
    Compare both methods to the ground truth instead of comparing the masks."""
    rng = np.random.default_rng(seed)
    sc = two_view_scene(rng, n=200, noise_px=0.5, outlier_ratio=0.3)
    F, mask, _ = estimate_F_ransac(sc["x"], sc["xp"], rng, img_shapes=(sc["shape"],) * 2,
                                   threshold_px=1.5)
    F_cv, mask_cv = cv2.findFundamentalMat(sc["x"], sc["xp"], cv2.FM_RANSAC, 1.5, 0.99)
    mask_cv = mask_cv.ravel().astype(bool)
    truth = sc["inlier"]
    recall = (mask & truth).sum() / truth.sum()
    recall_cv = (mask_cv & truth).sum() / truth.sum()
    assert recall >= recall_cv - 0.02
    assert (mask & truth).sum() / mask.sum() > 0.95
    e = np.median(symmetric_epipolar_distance(F, sc["x_clean"], sc["xp_clean"]))
    e_cv = np.median(symmetric_epipolar_distance(F_cv, sc["x_clean"], sc["xp_clean"]))
    assert e < max(1.5 * e_cv, 0.3), (e, e_cv)


def test_algebraic_error_zero_on_clean_data():
    sc = two_view_scene(np.random.default_rng(9), n=30, noise_px=0.0)
    assert algebraic_error(sc["F"], sc["x"], sc["xp"]).max() < 1e-8
