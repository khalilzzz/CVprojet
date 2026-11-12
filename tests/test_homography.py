import cv2
import numpy as np
import pytest

from src.geometry.errors import transfer_error
from src.geometry.homography import apply_H, dlt_homography, estimate_H_ransac
from src.geometry.synthetic import homography_scene


@pytest.mark.parametrize("planar", [True, False])
def test_exact_from_four_points(planar):
    sc = homography_scene(np.random.default_rng(10), n=50, noise_px=0.0, planar=planar)
    for normalize in (True, False):
        H = dlt_homography(sc["x"][:4], sc["xp"][:4], normalize=normalize)
        assert np.allclose(H, sc["H"], atol=1e-6 * np.abs(sc["H"]).max())
        assert transfer_error(H, sc["x"], sc["xp"]).max() < 1e-5


def test_apply_H_identity():
    pts = np.array([[1.0, 2.0], [100.0, 50.0]])
    assert np.allclose(apply_H(np.eye(3), pts), pts)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_ransac_with_outliers(seed):
    rng = np.random.default_rng(seed)
    sc = homography_scene(rng, n=200, noise_px=0.5, outlier_ratio=0.3)
    H, mask, _ = estimate_H_ransac(sc["x"], sc["xp"], rng, threshold_px=3.0)
    truth = sc["inlier"]
    recall = (mask & truth).sum() / truth.sum()
    precision = (mask & truth).sum() / mask.sum()
    assert recall > 0.95 and precision > 0.95, (recall, precision)
    assert np.median(transfer_error(H, sc["x_clean"], sc["xp_clean"])) < 1.0


def test_matches_opencv():
    rng = np.random.default_rng(14)
    sc = homography_scene(rng, n=150, noise_px=0.5)
    H_ours = dlt_homography(sc["x"], sc["xp"])
    H_cv, _ = cv2.findHomography(sc["x"], sc["xp"], 0)
    # OpenCV refines with Levenberg-Marquardt, so compare the mapped points, not matrices.
    d = np.linalg.norm(apply_H(H_ours, sc["x_clean"]) - apply_H(H_cv, sc["x_clean"]), axis=1)
    assert np.median(d) < 0.2
