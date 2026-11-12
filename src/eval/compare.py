# Cours : Sessions 3-4 : comparaison réel / généré des mesures V1, scène par scène
"""Real vs generated comparison of the V1 pair measures, one scene at a time.

Two confounders are removed here:
- motion: every measure degrades with the amount of image motion, and generated clips
  often move less. Videos of a scene are compared inside shared **motion** bins. The motion
  of a frame pair is its median flow (px): on the real clips it predicts the Sampson
  distance better than the parallax, which saturates near the homography threshold.
- conditioning: when a generated clip was given the first ``cond_s`` seconds of the real
  clip, that part is a re-rendered copy. The comparison uses only what comes after it,
  for every video of the scene. The copied part is reported separately as a control:
  same geometry as the real clip, rendered by the generator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ["sampson_med_all", "inlier_ratio_F", "h_cover"]


def attach_meta(pairs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    m = meta[["id", "label", "scene_id", "cond_s"]].copy()
    m["cond_s"] = pd.to_numeric(m["cond_s"], errors="coerce").fillna(0.0)
    out = pairs.merge(m, left_on="video", right_on="id", how="left").drop(columns="id")
    return out[out["valid"].astype(bool)]


def paired_scenes(pairs: pd.DataFrame) -> list[str]:
    """Scenes that have at least one real and one generated video."""
    n = pairs.groupby("scene_id")["label"].nunique()
    return sorted(n[n >= 2].index)


def split_windows(pairs: pd.DataFrame, margin_s: float = 0.1):
    """(generated part, copied part). The window of a scene is the largest cond_s of its videos."""
    start = pairs.groupby("scene_id")["cond_s"].transform("max")
    generated = pairs[(start <= 0) | (pairs["t0_s"] >= start + margin_s)]
    copied = pairs[(start > 0) & (pairs["t1_s"] <= start)]
    return generated, copied


MOTION = "median_flow_px"      # covariate used to compare videos at equal motion


def bin_by_motion(pairs: pd.DataFrame, n_bins: int = 5, min_pairs: int = 3,
                  col: str = MOTION) -> pd.DataFrame:
    """Median of each measure per (scene, video, motion bin).

    The bins of a scene cover the motion range that all its videos share (between their
    10th and 90th percentiles), in log scale. Bins with fewer than ``min_pairs`` pairs of
    a video are dropped.
    """
    rows = []
    for scene, g in pairs[pairs[col] > 0].groupby("scene_id"):
        q = g.groupby("video")[col].quantile([0.1, 0.9]).unstack()
        lo, hi = q[0.1].max(), q[0.9].min()
        if not np.isfinite(lo) or hi <= lo * 1.2:
            continue                                    # no shared motion range
        edges = np.geomspace(lo, hi, n_bins + 1)
        b = np.digitize(g[col], edges) - 1
        g = g.assign(bin=b)[(b >= 0) & (b < n_bins)]
        for (video, k), gb in g.groupby(["video", "bin"]):
            if len(gb) < min_pairs:
                continue
            rows.append(dict(scene_id=scene, video=video, label=gb["label"].iloc[0], bin=int(k),
                             motion_mid=float(np.sqrt(edges[k] * edges[k + 1])), n_pairs=len(gb))
                        | {m: float(gb[m].median()) for m in METRICS})
    return pd.DataFrame(rows)


def matched_summary(binned: pd.DataFrame) -> pd.DataFrame:
    """One row per video: measures averaged over the motion bins shared by the whole scene.

    ``sampson_matched`` is a geometric mean (the measure spans a wide range);
    ``ratio_to_real`` divides it by the value of the real video of the scene.
    """
    rows = []
    if binned.empty:
        return pd.DataFrame(rows)
    for scene, g in binned.groupby("scene_id"):
        videos = g["video"].unique()
        count = g.groupby("bin")["video"].nunique()
        common = count[count == len(videos)].index
        if len(videos) < 2 or len(common) == 0:
            continue
        gc = g[g["bin"].isin(common)]
        for video, gv in gc.groupby("video"):
            rows.append(dict(scene_id=scene, video=video, label=gv["label"].iloc[0],
                             n_bins=len(common), n_pairs=int(gv["n_pairs"].sum()),
                             motion_range=f"{gv['motion_mid'].min():.0f}-{gv['motion_mid'].max():.0f}",
                             sampson_matched=float(np.exp(np.log(gv["sampson_med_all"]).mean())),
                             inlier_ratio_F=float(gv["inlier_ratio_F"].mean()),
                             h_cover=float(gv["h_cover"].mean())))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    real = out[out["label"] == "real"].groupby("scene_id")["sampson_matched"].mean()
    out["ratio_to_real"] = out["sampson_matched"] / out["scene_id"].map(real)
    return out.sort_values(["scene_id", "label", "video"], ascending=[True, False, True])


def video_table(pairs: pd.DataFrame, meta: pd.DataFrame, deltas=(2, 20, 96)) -> pd.DataFrame:
    """One row per video with a few medians, for the report."""
    rows = []
    for video, g in pairs.groupby("video"):
        m = meta[meta["id"] == video].iloc[0]
        row = dict(video=video, label=m["label"], scene=m["scene_id"],
                   generator=m["generator"] or "-", valid_pairs=len(g))
        for d in deltas:
            gd = g[g["delta"] == d]
            row[f"Sampson px, gap {d}"] = gd["sampson_med_all"].median() if len(gd) else np.nan
        g20 = g[g["delta"] == 20]
        row["F inliers, gap 20"] = g20["inlier_ratio_F"].median() if len(g20) else np.nan
        row["parallax px, gap 20"] = g20["parallax_px"].median() if len(g20) else np.nan
        row["H cover, gap 20"] = g20["h_cover"].median() if len(g20) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scene", "label", "video"], ascending=[True, False, True])


def three_view_table(trip: pd.DataFrame, min_triplets: int = 3) -> pd.DataFrame:
    """One row per video: median three-view reprojection error per gap, and its growth.

    ``growth`` divides the error at the largest gap by the error at the smallest gap of the
    same video, over the gaps that all the videos of the scene have. A change of rendering (blur, re-encoding) scales the error at every gap, so
    it leaves the growth unchanged; a scene that deforms in time makes it larger (checked on
    synthetic clips, tests/test_synth_measures.py). ``growth_to_real`` compares it with the
    real clip of the scene.
    """
    rows = []
    for scene, gs in trip.groupby("scene_id"):
        # gaps with enough triplets in every video of the scene: the growth of two videos
        # is only comparable over the same gaps
        n = gs.groupby(["video", "delta"]).size().unstack("delta")
        common = [d for d in n.columns if (n[d].fillna(0) >= min_triplets).all()]
        if len(common) < 2:
            continue
        lo, hi = min(common), max(common)
        for video, g in gs.groupby("video"):
            med = g.groupby("delta")["reproj3_med_px"].median()
            flow = g.groupby("delta")["median_flow_px"].median()
            row = dict(scene_id=scene, video=video, label=g["label"].iloc[0])
            row.update({f"error px, gap {d}": float(med[d]) for d in common})
            row.update(gaps=f"{lo} to {hi}", growth=float(med[hi] / med[lo]),
                       motion_growth=float(flow[hi] / flow[lo]))
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    real = out[out["label"] == "real"].groupby("scene_id")["growth"].mean()
    out["growth_to_real"] = out["growth"] / out["scene_id"].map(real)
    return out.sort_values(["scene_id", "label", "video"], ascending=[True, False, True])


def benchmark_table(matched: pd.DataFrame, three_view: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per generator, over all the scenes it shares with a real clip.

    For each measure: number of real/generated pairs, median of generated / real, share of
    pairs where the generated clip is worse (> 1), and the two-sided sign-test p-value of
    that share against 0.5.
    """
    from scipy.stats import binomtest

    gen = meta.set_index("id")["generator"]
    rows = {}
    for name, df, col in [("Sampson at equal motion", matched, "ratio_to_real"),
                          ("three-view growth", three_view, "growth_to_real")]:
        if df is None or df.empty:
            continue
        f = df[df["label"] == "fake"].assign(generator=lambda d: d["video"].map(gen).replace("", "unknown"))
        f = f[np.isfinite(f[col])]
        for g, gg in f.groupby("generator"):
            k, n = int((gg[col] > 1).sum()), len(gg)
            rows.setdefault(g, dict(generator=g)).update({
                f"{name}: pairs": n, f"{name}: median gen/real": float(gg[col].median()),
                f"{name}: share gen worse": k / n,
                f"{name}: sign-test p": float(binomtest(k, n, 0.5).pvalue) if n else np.nan})
    return pd.DataFrame(rows.values()).sort_values("generator") if rows else pd.DataFrame()
