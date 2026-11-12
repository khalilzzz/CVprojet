"""What do the video measures react to? Checked on synthetic clips with known geometry.

Three renderings of the same rigid scene and camera path (src/geometry/synth_video.py):
clean, blurred (rendering change only), and slowly deformed (3D drift).
- The raw Sampson distance goes up with blur: it is confounded with rendering.
- The growth of the three-view reprojection error with the frame gap reacts to the drift
  and not to the blur: it separates geometry from rendering.
"""
import copy

import pytest

from src.data.frames import VideoFrames
from src.features import video_epipolar as v1
from src.geometry.synth_video import render_video
from src.utils.config import load_config

N = 64


def _measures(**kw):
    cfg = copy.deepcopy(load_config("configs/pipeline.yaml"))
    cfg["v1"]["deltas"] = [1, 20]
    cfg["v1"]["three_view"]["gaps"] = [5, 24]
    frames = render_video(seed=0, n_frames=N, **kw)["frames"]
    vf = VideoFrames(frames, 24.0, 0, frames[0].shape, frames[0].shape, list(range(N)), 24.0,
                     [k / 24 for k in range(N)])
    pairs, trip, _ = v1.analyse_frames(vf, "synthetic", cfg, 0)
    s = pairs[pairs["valid"]].groupby("delta")["sampson_med_all"].median()
    r = trip[trip["valid"]].groupby("delta")["reproj3_med_px"].median()
    return dict(sampson20=s[20], reproj_growth=r[24] / r[5])


@pytest.fixture(scope="module")
def measures():
    return dict(clean=_measures(), blur=_measures(blur_sigma=1.5), drift=_measures(drift_px=4.0))


def test_raw_sampson_is_confounded_with_rendering(measures):
    assert measures["blur"]["sampson20"] > 1.15 * measures["clean"]["sampson20"]


def test_three_view_growth_ignores_blur(measures):
    ratio = measures["blur"]["reproj_growth"] / measures["clean"]["reproj_growth"]
    assert 0.7 < ratio < 1.35


def test_three_view_growth_sees_drift(measures):
    assert measures["drift"]["reproj_growth"] > 1.6 * measures["clean"]["reproj_growth"]
    assert measures["drift"]["reproj_growth"] > 1.6 * measures["blur"]["reproj_growth"]
