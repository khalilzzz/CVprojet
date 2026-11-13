"""I1 (frequency), V3 (temporal) and the classifier, on synthetic data with a known answer."""
import cv2
import numpy as np
import pandas as pd

from src.features import image_frequency as i1
from src.features import video_temporal as v3
from src.models import classify

I1_CFG = dict(crop=128, n_bins=16, median_ksize=3)


def _natural(rng, n=160):
    """Image with a 1/f amplitude spectrum, like a natural image."""
    f = i1.radial_frequency(n)
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / np.maximum(f, 1.0 / n)
    img = np.real(np.fft.ifft2(np.fft.ifftshift(spec)))
    img = (img - img.min()) / (img.max() - img.min())
    return (40 + 170 * img + rng.normal(0, 2.0, img.shape)).clip(0, 255).astype(np.uint8)


def test_i1_image_slope_of_natural_image():
    g = _natural(np.random.default_rng(0))
    out = i1.frames_features([g], I1_CFG)
    assert -2.6 < out["i1_img_slope"] < -1.4          # power ~ 1/f^2


def test_i1_detects_periodic_upsampling_trace():
    rng = np.random.default_rng(1)
    clean = [_natural(rng) for _ in range(4)]
    grid = np.zeros((160, 160))
    grid[::4, ::4] = 6.0                                 # period-4 pattern, like a 4x upsampler
    traced = [(c + grid).clip(0, 255).astype(np.uint8) for c in clean]
    a, b = i1.frames_features(clean, I1_CFG), i1.frames_features(traced, I1_CFG)
    assert b["i1_peak_max"] > a["i1_peak_max"] + 0.3
    assert b["i1_peak_frac"] > a["i1_peak_frac"]


def _shifted_clip(rng, n=6, boil=0.0):
    """A textured image translated by 1 px per frame; `boil` adds fresh fine texture each frame."""
    base = cv2.GaussianBlur(rng.normal(128, 40, (140, 180)).astype(np.float32), (0, 0), 1.2)
    frames = []
    for k in range(n):
        f = np.roll(base, k, axis=1)[:, 10:170]
        f = f + boil * rng.normal(0, 1, f.shape)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return frames


def test_v3_boiling_texture_raises_warp_residual():
    rng = np.random.default_rng(2)
    stable = v3.frames_features(_shifted_clip(rng), stride=1, max_pairs=10)
    boiling = v3.frames_features(_shifted_clip(rng, boil=6.0), stride=1, max_pairs=10)
    assert 0.5 < stable["v3_flow_med"] < 1.5
    assert boiling["v3_hf_res_med"] > 2 * stable["v3_hf_res_med"]


def test_classifier_groups_scenes_and_finds_signal():
    rng = np.random.default_rng(3)
    rows = []
    for s in range(40):
        scene_level = rng.normal(0, 3)                   # scene effect, shared by real and fake
        rows.append(dict(video=f"r{s}", label="real", scene_id=f"s{s}", generator="",
                         i1_a=scene_level + rng.normal(0, 0.3), v1_noise=rng.normal()))
        for g in ("g1", "g2"):
            rows.append(dict(video=f"{g}{s}", label="fake", scene_id=f"s{s}", generator=g,
                             i1_a=scene_level + 1.0 + rng.normal(0, 0.3), v1_noise=rng.normal()))
    res = classify.run(pd.DataFrame(rows), seed=0, n_splits=5, unseen=True)
    summary = res["summary"].set_index("family")
    assert summary.loc["V1 geometry", "AUC"] < 0.7               # pure noise: no signal
    assert set(res["per_generator"]["generator"]) == {"g1", "g2"}
    assert "unseen_generator" in res
    # out-of-fold scores exist for every video
    assert np.isfinite(res["scores"]["I1 frequency"]).all()
