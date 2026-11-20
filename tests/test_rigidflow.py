"""V5 (src/features/video_rigidflow.py): the rigid flow model, its robust fit, and the video
features, on synthetic data. The depth network is replaced by a synthetic depth so that no
download is needed."""
import numpy as np
import pytest

from src.features import video_rigidflow as v5
from src.geometry.synth_video import render_video

H, W = 120, 160
K = v5.assumed_K((H, W), 0.8)


def _grid():
    ys, xs = np.mgrid[0:H, 0:W]
    return xs.astype(float), ys.astype(float)


def _synthetic_flow(params, rho):
    xs, ys = _grid()
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    x, y = ((xs - cx) / f).ravel(), ((ys - cy) / f).ravel()
    fl = v5.rigid_flow(np.asarray(params, float), x, y, rho.ravel(), f)
    return fl.reshape(H, W, 2)


def test_exact_rigid_flow_is_recovered():
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.5 * rng.random((H, W))                       # inverse depth
    params = np.r_[0.02, -0.01, 0.03, 0.0, 0.0, 0.0, 0.004, -0.003, 0.002]  # a t, b t, w
    flow = _synthetic_flow(params, rho)
    mask = np.ones((H, W), bool)
    _, res = v5.fit_rigid(flow, rho, mask, K, True, iters=3)
    assert np.nanmedian(res) < 1e-6
    # the same flow has translation: a rotation-only camera cannot explain it
    _, res_rot = v5.fit_rigid(flow, rho, mask, K, False, iters=3)
    assert np.nanmedian(res_rot) > 0.1


def test_depth_scale_and_shift_do_not_matter():
    rng = np.random.default_rng(1)
    rho = 1.0 + 0.5 * rng.random((H, W))
    params = np.r_[0.02, 0.01, -0.02, 0.0, 0.0, 0.0, 0.0, 0.002, 0.0]
    flow = _synthetic_flow(params, rho)
    d = 3.0 * rho - 2.0                                           # what a relative-depth network gives
    _, res = v5.fit_rigid(flow, d, np.ones((H, W), bool), K, True, iters=3)
    assert np.nanmedian(res) < 1e-6


def test_irls_ignores_moving_object():
    rng = np.random.default_rng(2)
    rho = 1.0 + 0.5 * rng.random((H, W))
    params = np.r_[0.02, -0.01, 0.0, 0.0, 0.0, 0.0, 0.003, 0.0, -0.002]
    flow = _synthetic_flow(params, rho)
    flow[40:80, 50:110] += np.array([3.0, -2.0])                  # an object moving on its own
    _, res = v5.fit_rigid(flow, rho, np.ones((H, W), bool), K, True, iters=6)
    obj = np.zeros((H, W), bool)
    obj[40:80, 50:110] = True
    assert np.nanmedian(res[~obj]) < 0.05                          # background still explained
    assert np.nanmedian(res[obj]) > 2.0                            # the object stands out


def test_video_features_on_synthetic_clip(monkeypatch):
    frames = render_video(seed=0, n_frames=30, shape=(180, 320))["frames"]
    # synthetic depth: smooth ramp, no network
    monkeypatch.setattr(v5, "inverse_depth", lambda g, model=None: np.tile(
        np.linspace(1, 2, g.shape[1], dtype=np.float32), (g.shape[0], 1)))
    feats = v5.frames_features(frames, dict(gap=4, n_pairs=3, max_px=5000))
    assert feats["v5_n_pairs"] == 3
    for k in v5.FEATURES:
        assert np.isfinite(feats[k]), k
    assert feats["v5_res_px"] < feats["v5_flow_px"]


def test_duplicated_frames_are_skipped(monkeypatch):
    frames = render_video(seed=0, n_frames=12, shape=(120, 200))["frames"]
    frozen = [frames[0]] * 12
    monkeypatch.setattr(v5, "inverse_depth", lambda g, model=None: np.ones(g.shape, np.float32))
    feats = v5.frames_features(frozen, dict(gap=2, n_pairs=3))
    assert feats["v5_n_pairs"] == 0 and np.isnan(feats["v5_res_px"])


def test_pair_indices():
    assert v5.pair_indices(10, 20, 3) == []
    idx = v5.pair_indices(100, 8, 6)
    assert idx[0] == 0 and idx[-1] == 91 and len(idx) == 6
