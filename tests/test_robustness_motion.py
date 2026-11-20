"""Robustness evaluation and motion-vs-coherence analysis on synthetic feature tables."""
import numpy as np
import pandas as pd

from src.eval import indicators, motion_vs_coherence, robustness


def _table(rng, n_scenes=60, noise=0.0, shift=1.0):
    """One real and one generated clip per scene. v1_inlier_ratio_F_d8 is lower in generated
    clips; noise blurs it (a degradation)."""
    rows = []
    for s in range(n_scenes):
        motion = rng.uniform(1, 10)
        for lab in ("real", "fake"):
            coh = 0.9 - 0.02 * motion - (shift * 0.05 if lab == "fake" else 0) + rng.normal(0, 0.02 + noise)
            rows.append(dict(video=f"{lab}{s}", label=lab, scene_id=f"s{s}",
                             generator="" if lab == "real" else "gen",
                             v1_inlier_ratio_F_d8=coh, v1_median_flow_px_d8=motion + rng.normal(0, 0.3),
                             i1_peak_max=rng.normal()))
    return pd.DataFrame(rows)


def test_evaluate_auc_drops_with_degradation():
    rng = np.random.default_rng(0)
    tables = {"clean": _table(rng), "bad": _table(np.random.default_rng(0), noise=0.3)}
    table, single = robustness.evaluate(tables, seed=0, n_boot=20)
    v1 = table[(table["family"] == "V1 geometry") & (table["mode"] == "trained on clean")].set_index("level")
    assert v1.loc["clean", "AUC"] > 0.7
    assert v1.loc["bad", "AUC"] < v1.loc["clean", "AUC"] - 0.1
    assert "physics + CLIP" not in set(table["family"])          # no CLIP column: skipped
    f = single[single["column"] == "v1_inlier_ratio_F_d8"].set_index("level")
    assert f.loc["clean", "AUC"] < 0.4                           # lower in generated clips


def test_motion_groups_and_correction():
    df = _table(np.random.default_rng(1), n_scenes=80)
    g = motion_vs_coherence.v1_groups(df)
    assert g["motion"] == ["v1_median_flow_px_d8"] and g["coherence"] == ["v1_inlier_ratio_F_d8"]
    table, single = motion_vs_coherence.analyse(df, n_boot=20)
    t = table.set_index("variant")
    # the coherence signal is hidden by the motion trend; the correction recovers it
    assert t.loc["3D coherence corrected for motion", "AUC"] > t.loc["3D coherence (raw)", "AUC"]
    assert t.loc["camera motion", "AUC"] < 0.65


def test_indicators_resolve_middle_gap():
    cols = ["v1_inlier_ratio_F_d2", "v1_inlier_ratio_F_d8", "v1_inlier_ratio_F_d32",
            "v1_reproj3_med_px_g3", "v1_reproj3_med_px_g16", "i2_rot_resid_deg"]
    ind = indicators.resolve(cols)
    assert "v1_inlier_ratio_F_d8" in set(ind["column"])
    assert "v1_reproj3_med_px_g16" in set(ind["column"]) and "v1_reproj_growth" in set(ind["column"])
    assert "i2_rot_resid_deg" in set(ind["column"])
