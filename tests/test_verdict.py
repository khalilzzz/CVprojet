"""Comparative verdict of the demo (src/demo/verdict.py) on synthetic indicator series."""
import numpy as np

from src.demo import verdict


def _series(motion, factor, rng, n=120, slope=0.5, noise=0.08):
    """Per-frame series where error = factor * motion^slope * noise, like the demo's output."""
    m = motion * np.exp(rng.normal(0, 0.3, n))
    err = factor * m ** slope * np.exp(rng.normal(0, noise, n))
    s = {"v1_motion": (m, None), "tv_err": (err, None)}
    s["v1_sampson"] = (0.1 * err, None)
    s["v3_hf"] = (np.full(n, 0.2) * np.exp(rng.normal(0, noise, n)), None)
    return s


def test_generated_clip_with_larger_error_at_equal_motion():
    rng = np.random.default_rng(0)
    v = verdict.compare(_series(10, 1.0, rng), _series(10, 1.5, rng))
    assert v["code"] == "fake_less_coherent"
    assert 1.3 < v["main"]["ratio"] < 1.7 and v["main"]["lo"] > 1
    assert v["support"][0]["call"] == "fake_higher" and v["n_agree"] >= 1


def test_more_motion_alone_is_not_a_violation():
    # the generated clip moves 1.6x more: raw error higher, same law -> no difference
    rng = np.random.default_rng(1)
    v = verdict.compare(_series(10, 1.0, rng), _series(16, 1.0, rng))
    assert v["main"]["raw_ratio"] > 1.15
    assert v["code"] == "no_difference"
    assert v["main"]["lo"] < 1 < v["main"]["hi"]


def test_motions_that_do_not_overlap_give_no_verdict():
    rng = np.random.default_rng(2)
    v = verdict.compare(_series(2, 1.0, rng), _series(200, 1.5, rng))
    assert v["code"] == "not_comparable"


def test_too_few_frames_give_no_verdict():
    rng = np.random.default_rng(3)
    v = verdict.compare(_series(10, 1.0, rng, n=10), _series(10, 1.5, rng, n=10))
    assert v["code"] == "no_data" and not np.isfinite(v["main"]["ratio"])
