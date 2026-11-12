"""Tests of the video pipeline pieces: frame reading, keypoint selection, matching, pair statistics."""
import cv2
import numpy as np
import pytest

from src.data.frames import read_frames
from src.features.video_epipolar import (match_descriptors, pair_stats, select_on_grid,
                                         video_rng)
from src.geometry.synthetic import homography_scene, two_view_scene

V1_CFG = dict(min_matches=16,
              fundamental=dict(normalization="course", threshold_px=1.5, p=0.99, max_iters=2000),
              homography=dict(threshold_px=3.0, p=0.99, max_iters=1000))


def _write_video(path, n, fps, size=(96, 64)):
    """Frame k is uniformly gray with value 2 * k, so a frame identifies its index."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert w.isOpened()
    for k in range(n):
        w.write(np.full((size[1], size[0], 3), 2 * k, np.uint8))
    w.release()


def _index(frame):
    return int(round(float(frame.mean()) / 2))


def test_read_frames_skip_and_duration(tmp_path):
    p = tmp_path / "v.avi"
    _write_video(p, n=90, fps=30)
    vf = read_frames(p, skip_s=0.5, duration_s=1.0)
    assert vf.indices[0] == 15 and vf.indices[-1] == 45
    assert [_index(f) for f in vf.frames] == vf.indices          # frames match their index
    assert vf.times[0] == pytest.approx(0.5, abs=1e-3)


def test_read_frames_resamples_30_to_24_fps(tmp_path):
    p = tmp_path / "v.avi"
    _write_video(p, n=90, fps=30)
    vf = read_frames(p, skip_s=0.0, target_fps=24, duration_s=2.0)
    assert len(vf.frames) == 49                                   # 2 s at 24 fps, both ends
    assert set(np.diff(vf.indices)) <= {1, 2}                     # every 5th source frame dropped
    t = np.array(vf.indices) / 30.0
    assert np.max(np.abs(t - np.arange(49) / 24.0)) <= 0.5 / 30 + 1e-6   # nearest frame in time


def test_read_frames_crop_and_short_side(tmp_path):
    p = tmp_path / "v.avi"
    _write_video(p, n=10, fps=24, size=(200, 100))
    vf = read_frames(p, skip_s=0.0, crop_bottom=0.1, short_side=50)
    assert vf.shape == (45, 100)                                  # 90 rows kept, then scale 0.5
    assert vf.original_shape == (100, 200)


def test_select_on_grid_spreads_keypoints():
    rng = np.random.default_rng(0)
    # 5000 strong keypoints packed in one corner, 500 weak ones everywhere
    corner = rng.uniform(0, 40, size=(5000, 2))
    spread = rng.uniform(0, 400, size=(500, 2))
    pts = np.vstack([corner, spread])
    resp = np.r_[np.full(5000, 10.0), np.full(500, 1.0)]
    keep = select_on_grid(pts, resp, (400, 400), n=640, grid=(8, 8))
    assert len(keep) == 640 == len(set(keep.tolist()))
    assert (keep >= 5000).sum() > 300        # most weak-but-spread keypoints survive
    by_response = np.argsort(-resp)[:640]
    assert (by_response >= 5000).sum() == 0  # a pure top-response choice keeps none of them


def test_match_descriptors_mutual_and_ratio():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(60, 128)).astype(np.float32)
    perm = rng.permutation(60)
    b = a[perm] + 0.01 * rng.normal(size=(60, 128)).astype(np.float32)
    m = match_descriptors(a, b, ratio=0.75, mutual=True)
    assert len(m) == 60
    assert np.all(perm[m[:, 1]] == m[:, 0])                       # every match is the true one
    # an ambiguous descriptor (two equal candidates) is rejected by the ratio test
    b2 = np.vstack([b, b[:1]])
    m2 = match_descriptors(a, b2, ratio=0.75, mutual=False)
    assert perm[0] not in m2[:, 0]


def test_match_descriptors_mutual_rejects_one_way_matches():
    a = np.zeros((3, 4), np.float32)
    a[0, 0], a[1, 0], a[2, 1] = 1.0, 1.2, 5.0
    b = np.zeros((2, 4), np.float32)
    b[0, 0], b[1, 2] = 1.0, 9.0
    one_way = match_descriptors(a, b, ratio=0.99, mutual=False)
    both = match_descriptors(a, b, ratio=0.99, mutual=True)
    assert {0, 1} <= set(one_way[:, 0].tolist())                  # a0 and a1 both pick b0
    assert both[:, 0].tolist() == [0]                             # only a0 is b0's nearest


def test_pair_stats_general_scene_has_parallax():
    rng = np.random.default_rng(2)
    sc = two_view_scene(rng, n=300, noise_px=0.3, outlier_ratio=0.1)
    row, F, mask = pair_stats(sc["x"], sc["xp"], sc["shape"], video_rng(0, "a"), V1_CFG)
    assert row["valid"] and F is not None
    assert row["inlier_ratio_F"] > 0.8
    assert 0.0 <= row["h_cover"] < 0.7                            # a homography cannot explain it
    assert row["parallax_px"] > 3.0


def test_pair_stats_planar_scene_is_flagged_degenerate():
    rng = np.random.default_rng(3)
    sc = homography_scene(rng, n=300, noise_px=0.3, outlier_ratio=0.1, planar=True)
    row, _, _ = pair_stats(sc["x"], sc["xp"], sc["shape"], video_rng(0, "a"), V1_CFG)
    assert row["valid"]
    assert row["h_cover"] > 0.95 and row["h_cover"] <= 1.0
    assert row["parallax_px"] < 1.0


def test_pair_stats_too_few_matches_is_invalid():
    x = np.random.default_rng(4).uniform(0, 100, size=(10, 2))
    row, F, mask = pair_stats(x, x + 1.0, (100, 100), video_rng(0, "a"), V1_CFG)
    assert not row["valid"] and F is None and not mask.any()


def test_video_rng_is_per_video_and_reproducible():
    a1, a2 = video_rng(0, "vid1").random(4), video_rng(0, "vid1").random(4)
    b = video_rng(0, "vid2").random(4)
    assert np.array_equal(a1, a2) and not np.array_equal(a1, b)
