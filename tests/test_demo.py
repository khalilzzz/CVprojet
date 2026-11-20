"""Demo (demo.py): per-frame measurements on a synthetic clip and drawing of the frames."""
import copy

import cv2
import numpy as np

from src.demo import measure, render
from src.features import video_epipolar as v1
from src.geometry.synth_video import render_video
from src.utils.config import load_config


def _cfg():
    cfg = copy.deepcopy(load_config("configs/pipeline.yaml"))
    cfg["demo"] = dict(load_config("configs/demo.yaml")["demo"], v1_gap=4, three_view_gap=3)
    cfg["v1"]["matching"]["n_features"] = 1500
    return cfg


def test_three_view_points_equals_pipeline_triplet():
    cfg = _cfg()
    frames = render_video(seed=3, n_frames=10)["frames"]
    feats = v1.sift_features(frames, 1500)
    ref = v1.triplet_stats(feats, 0, 4, frames[0].shape, np.random.default_rng(5), cfg["v1"])
    ours = measure.three_view_points(feats[0], feats[4], feats[8], frames[0].shape,
                                     np.random.default_rng(5), cfg["v1"])
    assert ref is not None and ours is not None
    assert np.isclose(ours["err_med"], ref["reproj3_med_px"])
    assert len(ours["a3"]) == len(ours["proj"]) == ref["n_tracks"]


def test_measure_and_render_on_synthetic_clip():
    cfg = _cfg()
    grays = render_video(seed=1, n_frames=14)["frames"]
    res = measure.measure_chunk(("synth", 0, len(grays), 0, grays, cfg))
    full = dict(frames=res, i1=measure.i1_series(grays, cfg["i1"]),
                i2=measure.i2_series([m["i2"] for m in res], cfg["i2"]))
    m = res[10]
    assert m["v1"]["row"]["inlier_ratio_F"] > 0.8                      # rigid synthetic scene
    h, w = grays[0].shape
    xp = m["v1"]["xp"][m["v1"]["mask"]]
    assert (xp[:, 0] >= 0).all() and (xp[:, 0] < w).all() and (xp[:, 1] >= 0).all() and (xp[:, 1] < h).all()
    assert m["tv"] is not None and np.isfinite(m["tv"]["err_med"])
    assert "v3" not in res[0] and np.isfinite(m["v3"]["stats"]["v3_hf_res_norm"])
    ser = measure.indicator_series(full)
    assert np.isfinite(ser["v1_inlier"][1][-1]) and np.isfinite(ser["i1_hf"][1][-1])

    colour = [cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) for g in grays]
    keys = ["title", "v1", "tv", "i2", "v3", "i1", "summary"]
    rend = render.Renderer(("synth", "synth"), (colour, colour), (full, full), (ser, ser), 24.0,
                           cfg["demo"], keys)
    for k in keys:
        img = rend.frame(12, k)
        assert img.shape == (render.H, render.W, 3) and img.dtype == np.uint8


def test_portrait_clip_is_letterboxed():
    panel, (s, ox, oy), (nw, nh) = render.fit_frame(np.zeros((1280, 720, 3), np.uint8))
    assert panel.shape == (render.PH, render.PW, 3)
    assert nh == render.PH and nw < render.PW and ox > 0 and oy == 0


def test_timeline_auto_speed_covers_the_clip():
    import demo
    chapters = [dict(key="title", seconds=1.0), dict(key="v1", seconds=2.0)]
    tl = demo.timeline(30, 24.0, chapters, 24.0, "auto")
    assert len(tl) == 72 and tl[0] == (0, "title") and tl[-1][0] == 29 and tl[-1][1] == "v1"
