# Cours : toutes sessions (synthèse) : liste des indicateurs lisibles, un par mesure physique
"""The readable indicators: one number per physical measure, with what a real camera gives.

Used by the one-video inspection (src/eval/inspect_video.py) and by the robustness test
(src/eval/robustness.py). Each indicator is a column of a features table.

kind = "cue": a physical constraint that a real camera respects (the expected direction is
given); kind = "context": describes the clip (motion, amount of lines, noise level) and is
needed to read the cues, but is not a cue itself.
"""
from __future__ import annotations

import re

import pandas as pd

# (column template, family, course session, name, kind, expected for a real camera)
# {d} = middle frame gap of V1, {g} = largest gap of the three-view test
TEMPLATES = [
    ("v1_inlier_ratio_F_d{d}", "V1", "Sessions 3-4", "share of matches that fit one F", "cue", "high"),
    ("v1_sampson_med_all_d{d}", "V1", "Sessions 3-4", "Sampson distance to F, median (px)", "cue", "low"),
    ("v1_reproj3_med_px_g{g}", "V1", "Session 7", "three-view reprojection error, largest gap (px)", "cue", "low"),
    ("v1_reproj_growth", "V1", "Session 7", "growth of the three-view error with the gap", "cue", "low"),
    ("v1_median_flow_px_d{d}", "V1", "Session 3", "image motion (px)", "context", ""),
    ("v1_parallax_px_d{d}", "V1", "Sessions 3-4", "parallax: motion a homography cannot explain (px)", "context", ""),
    ("v1_h_cover_d{d}", "V1", "Sessions 3-4", "share of F inliers a homography explains", "context", ""),
    ("i1_peak_max", "I1", "Sessions 1-2", "periodic peaks in the noise spectrum (log10)", "cue", "low"),
    ("i1_hf_ratio", "I1", "Sessions 1-2", "share of noise energy at high frequency", "cue", "high"),
    ("i1_res_slope", "I1", "Sessions 1-2", "slope of the noise spectrum", "cue", ""),
    ("i1_noise_std", "I1", "Sessions 1-2", "noise level (grey levels)", "context", ""),
    ("v3_hf_res_norm", "V3", "Session 6", "fine texture change after motion compensation (/ noise)", "cue", "low"),
    ("v3_static_res_mean", "V3", "Session 6", "change of pixels that do not move, mean (grey levels)", "cue", "low"),
    ("v3_fb_bad_frac", "V3", "Session 6", "share of pixels failing the forward-backward flow check", "cue", "low"),
    ("v3_flow_med", "V3", "Session 6", "dense flow, median (px)", "context", ""),
    ("i2_resid_med_deg", "I2", "Sessions 2-4", "misfit of lines to their vanishing point (deg)", "cue", "low"),
    ("i2_rot_resid_deg", "I2", "Sessions 3-4", "directions not explained by one camera rotation (deg)", "cue", "low"),
    ("i2_pair_angle_std_deg", "I2", "Sessions 3-4", "change of the angle between two 3D directions (deg)", "cue", "low"),
    ("i2_jitter_med_deg", "I2", "Sessions 3-4", "direction jitter between frames, includes camera shake (deg)", "context", ""),
    ("i2_dominant_kept_frac", "I2", "Sessions 3-4", "share of frames where the main vanishing point is found again", "cue", "high"),
    ("i2_frac_frames_vp", "I2", "Sessions 2-4", "share of frames with a vanishing point", "context", ""),
    ("v5_res_px", "V5", "outside the course", "flow not explained by a camera motion given the learned depth (px)", "cue", "low"),
    ("v5_bad_frac", "V5", "outside the course", "share of pixels off the rigid-camera model by more than 1 px", "cue", "low"),
    ("v5_gain", "V5", "outside the course", "rotation-only residual / with-depth residual (what the depth explains)", "context", ""),
    ("v5_flow_px", "V5", "outside the course", "dense flow over the V5 gap, median (px)", "context", ""),
]


def _gaps(columns, prefix: str) -> list[int]:
    return sorted({int(m.group(1)) for c in columns if (m := re.fullmatch(prefix + r"(\d+)", c))})


def resolve(columns) -> pd.DataFrame:
    """Indicators available in a features table (the gap placeholders filled in)."""
    columns = list(columns)
    deltas = _gaps(columns, r"v1_inlier_ratio_F_d")
    gaps = _gaps(columns, r"v1_reproj3_med_px_g")
    d = deltas[len(deltas) // 2] if deltas else None
    g = gaps[-1] if gaps else None
    rows = []
    for tpl, fam, session, name, kind, expected in TEMPLATES:
        if ("{d}" in tpl and d is None) or ("{g}" in tpl and g is None):
            continue
        col = tpl.format(d=d, g=g)
        if col == "v1_reproj_growth" and len(gaps) < 2:
            continue
        if col in columns or col == "v1_reproj_growth":
            label = name + (f", gap {d}" if "{d}" in tpl else "")
            rows.append(dict(column=col, family=fam, session=session, indicator=label, kind=kind,
                             real_camera=expected))
    return pd.DataFrame(rows)
