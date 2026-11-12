import numpy as np
import pytest

from src.geometry.synthetic import manhattan_segments, vp_direction_error_deg
from src.geometry.vanishing import estimate_vps, refine_vp, vp_residual_angle


def _match_vps(vps, vps_gt, K):
    """Best ground-truth match for each estimated VP (greedy)."""
    errs = []
    used = set()
    for v in vps:
        cands = [(vp_direction_error_deg(v, g, K), i) for i, g in enumerate(vps_gt) if i not in used]
        e, i = min(cands)
        used.add(i)
        errs.append(e)
    return errs


def test_residual_zero_for_segments_through_vp():
    vp = np.array([500.0, 300.0, 1.0])
    segs = np.array([[0.0, 0.0, 250.0, 150.0], [1000.0, 300.0, 800.0, 300.0]])
    assert np.allclose(vp_residual_angle(vp, segs), 0.0, atol=1e-9)


def test_residual_vp_at_infinity():
    vp = np.array([1.0, 0.0, 0.0])                    # horizontal direction
    segs = np.array([[0.0, 5.0, 10.0, 5.0], [0.0, 0.0, 0.0, 10.0]])
    ang = vp_residual_angle(vp, segs)
    assert np.isclose(ang[0], 0.0) and np.isclose(ang[1], 90.0)


def test_refine_vp_noise_free():
    sc = manhattan_segments(np.random.default_rng(20), noise_px=0.0, n_outliers=0)
    segs = sc["segments"][sc["labels"] == 0]
    v = refine_vp(segs)
    assert vp_direction_error_deg(v, sc["vps"][0], sc["K"]) < 1e-6


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_three_vps_with_clutter(seed):
    rng = np.random.default_rng(seed)
    sc = manhattan_segments(rng)
    vps, labels = estimate_vps(sc["segments"], rng, n_vps=3, angle_threshold_deg=2.0)
    assert len(vps) == 3
    errs = _match_vps(vps, sc["vps"], sc["K"])
    assert max(errs) < 1.0, errs
    structured = sc["labels"] >= 0
    assert (labels[structured] >= 0).mean() > 0.9


def test_vps_at_infinity():
    """Camera looking straight at a box: x and y edges are fronto-parallel (VP at infinity)."""
    rng = np.random.default_rng(24)
    sc = manhattan_segments(rng, R=np.eye(3))
    vps, _ = estimate_vps(sc["segments"], rng, n_vps=3)
    assert len(vps) == 3
    errs = _match_vps(vps, sc["vps"], sc["K"])
    assert max(errs) < 1.0, errs
