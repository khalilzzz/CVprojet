# Cours : Sessions 4 et 7 (géométrie épipolaire, SfM) et Session 5 (évaluation) : le classifieur V1 détecte-t-il un mouvement de caméra différent, ou une scène 3D incohérente ?
"""Does V1 detect "the camera moves differently" or "the 3D scene is inconsistent"?

The V1 features mix three kinds of numbers:
- camera motion: image motion, parallax, share explained by a homography (``median_flow_px``,
  ``parallax_px``, ``parallax_q90_px``, ``h_cover``). Generators move the camera differently
  (often less), so these can separate real from generated without any 3D error;
- matching: how many points are found and followed (``n_matches``, ``n_tracks``, shares of
  valid pairs and triplets). This depends on texture and rendering;
- 3D coherence: how well one rigid scene explains the matches (F inliers, Sampson distances,
  three-view reprojection errors and their growth with the gap). This is the physical cue.

Coherence numbers also depend on motion (every error grows with the motion). So the
coherence features are also tested **corrected for motion**: inside each training fold, a
ridge regression learns, on the REAL clips only, the coherence expected for a real camera
at a given motion; a clip is then described by its deviation from that expectation.

Classifiers (gradient boosting, 5 folds grouped by scene, as in src/models/classify.py) on
each group, and the paired bootstrap difference "all V1 - (motion + matching)": what the
3D coherence adds. Also the AUC of each single coherence feature, raw and corrected.

    python -m src.eval.motion_vs_coherence results/latest_vifbench
Writes <run>/classification/motion_vs_coherence*.csv and motion_vs_coherence.png.
"""
from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.eval import plots
from src.models import classify

MOTION = ("median_flow_px", "parallax_px", "parallax_q90_px", "h_cover")
MATCHING = ("n_matches", "n_tracks", "frac_valid", "frac_valid_triplets", "frac_valid_pairs")
COHERENCE = ("inlier_ratio_F", "sampson_med_all", "sampson_q90_all", "sampson_med_in",
             "reproj3_med_px", "reproj3_q90_px", "reproj3_inlier_ratio", "reproj12_med_px", "reproj_growth")


def metric_of(col: str) -> str:
    """'v1_sampson_med_all_d8' -> 'sampson_med_all'."""
    return re.sub(r"_(d|g)\d+$", "", col[3:])


def v1_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = classify.family_columns(df, ("v1_",))
    g = {"motion": [], "matching": [], "coherence": []}
    for c in cols:
        m = metric_of(c)
        key = "motion" if m in MOTION else "matching" if m in MATCHING else "coherence" if m in COHERENCE else None
        if key:
            g[key].append(c)
    return g


def _impute(M: np.ndarray, ref: np.ndarray) -> np.ndarray:
    med = np.nanmedian(ref, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(M), M, med)


def residualize(C: np.ndarray, M: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Coherence C minus what a ridge regression on motion M predicts; the regression is
    fitted on fit_rows only (real clips of the training fold). NaN stays NaN."""
    Mi = _impute(M, M[fit_rows])
    mu, sd = Mi[fit_rows].mean(0), Mi[fit_rows].std(0) + 1e-9
    Z = (Mi - mu) / sd
    out = np.full_like(C, np.nan)
    for j in range(C.shape[1]):
        ok = fit_rows & np.isfinite(C[:, j])
        if ok.sum() < 10:
            continue
        r = Ridge(alpha=1.0).fit(Z[ok], C[ok, j])
        out[:, j] = C[:, j] - r.predict(Z)
    return out


def oof_corrected(C, M, y, folds, seed) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold classifier scores on motion-corrected coherence, and the corrected
    features themselves (each test row corrected by the regression of its training fold)."""
    s = np.full(len(y), np.nan)
    R_all = np.full_like(C, np.nan)
    for tr, te in folds:
        fit_rows = np.zeros(len(y), bool)
        fit_rows[tr[y[tr] == 0]] = True
        R = residualize(C, M, fit_rows)
        R_all[te] = R[te]
        m = classify.make_model("physics", seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(R[tr], y[tr])
        s[te] = m.predict_proba(R[te])[:, 1]
    return s, R_all


def analyse(df: pd.DataFrame, seed: int = 0, n_boot: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = classify.add_derived(df.reset_index(drop=True))
    y = (df["label"] == "fake").to_numpy(int)
    groups = df["scene_id"].to_numpy()
    folds = classify._folds(y, groups, 5, seed)
    g = v1_groups(df)
    variants = {"camera motion": g["motion"], "matching (texture)": g["matching"],
                "motion + matching": g["motion"] + g["matching"],
                "3D coherence (raw)": g["coherence"],
                "all V1": g["motion"] + g["matching"] + g["coherence"]}
    scores = {}
    for name, cols in variants.items():
        if cols:
            print(f"[motion-vs-coherence] {name}: {len(cols)} features", flush=True)
            scores[name] = classify.oof_scores(df[cols].to_numpy(float), y, folds, "physics", seed)
    C, M = df[g["coherence"]].to_numpy(float), df[g["motion"]].to_numpy(float)
    R = None
    if g["coherence"] and g["motion"]:
        print("[motion-vs-coherence] 3D coherence corrected for motion", flush=True)
        scores["3D coherence corrected for motion"], R = oof_corrected(C, M, y, folds, seed)
    rows = []
    for name, s in scores.items():
        lo, hi = classify.bootstrap_auc(y, s, groups, n=n_boot, seed=seed)
        n = len(variants.get(name, g["coherence"]))
        rows.append(dict(variant=name, n_features=n, AUC=classify._auc(y, s), AUC_low=lo, AUC_high=hi))
    # what coherence adds on top of motion + matching, paired bootstrap over scenes
    if "all V1" in scores and "motion + matching" in scores:
        rng = np.random.default_rng(seed)
        idx_by = pd.Series(groups).groupby(groups).indices
        sc = np.array(list(idx_by))
        d = []
        for _ in range(n_boot):
            idx = np.concatenate([idx_by[k] for k in rng.choice(sc, len(sc), replace=True)])
            d.append(classify._auc(y[idx], scores["all V1"][idx]) - classify._auc(y[idx], scores["motion + matching"][idx]))
        d = np.array(d)
        rows.append(dict(variant="all V1 - (motion + matching)", n_features=len(g["coherence"]),
                         AUC=rows[[r["variant"] for r in rows].index("all V1")]["AUC"]
                         - rows[[r["variant"] for r in rows].index("motion + matching")]["AUC"],
                         AUC_low=float(np.nanquantile(d, 0.025)), AUC_high=float(np.nanquantile(d, 0.975))))
    # single features: AUC > 0.5 = higher in generated clips
    single = []
    for name, cols, X in [("motion", g["motion"], None), ("matching", g["matching"], None),
                          ("coherence", g["coherence"], None), ("coherence corrected", g["coherence"], R)]:
        if X is None and name == "coherence corrected":
            continue
        for j, c in enumerate(cols):
            v = df[c].to_numpy(float) if X is None else X[:, j]
            single.append(dict(group=name, feature=c, AUC=classify._auc(y, v)))
    single = pd.DataFrame(single)
    if not single.empty:
        single["distance_to_chance"] = (single["AUC"] - 0.5).abs()
        single = single.sort_values(["group", "distance_to_chance"], ascending=[True, False])
    return pd.DataFrame(rows), single


def figure(table: pd.DataFrame, out: Path) -> Path:
    plots.apply_style()
    t = table[~table["variant"].str.contains(" - ")]
    sets = list(dict.fromkeys(t["subset"]))
    order = list(dict.fromkeys(t["variant"]))
    fig, ax = plt.subplots(figsize=(9.5, 1.5 + 0.55 * len(order) * max(1, len(sets)) ** 0.5))
    for k, sname in enumerate(sets):
        for i, v in enumerate(order):
            r = t[(t["subset"] == sname) & (t["variant"] == v)]
            if r.empty:
                continue
            r = r.iloc[0]
            yv = i + (k - (len(sets) - 1) / 2) * 0.25
            c = ["#1baf7a", "#7a5cd6", "#eda100"][k % 3]     # not blue/orange: those mean real/generated
            ax.plot([r["AUC_low"], r["AUC_high"]], [yv, yv], color=c, lw=2.2, alpha=0.6)
            ax.scatter(r["AUC"], yv, s=55, color=c, zorder=3, edgecolor=plots.SURFACE, linewidth=1.2,
                       label=sname if i == 0 else None)
            ax.annotate(f"{r['AUC']:.2f}", (r["AUC_high"], yv), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=plots.INK2)
    ax.axvline(0.5, color="#b9b8b2", ls="--", lw=1.2)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(0.4, 1.0)
    ax.grid(True, axis="x", color=plots.GRID)
    ax.grid(False, axis="y")
    ax.set_xlabel("AUC, real against generated (dot = estimate, bar = 95 % interval over scenes)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("V1: camera motion or 3D coherence? Classifiers on each group of V1 features",
                 loc="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder with features.csv, e.g. results/latest_vifbench")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "classification"
    out.mkdir(exist_ok=True)
    feats = pd.read_csv(run / "features.csv", low_memory=False)
    feats["generator"] = feats["generator"].fillna("")
    subsets = {"all generators": feats}
    if feats["video"].astype(str).str.startswith("vif__").any():
        from src.data.vifbench import _cond_type
        i2v = feats[(feats["label"] == "real") | (feats["generator"].map(_cond_type) == "image")]
        if (i2v["label"] == "fake").sum() >= 20:
            subsets["image-to-video only"] = i2v
    tables, singles = [], []
    for name, df in subsets.items():
        print(f"== {name}: {len(df)} videos")
        t, s = analyse(df, args.seed, args.n_boot)
        tables.append(t.assign(subset=name))
        singles.append(s.assign(subset=name))
    table = pd.concat(tables, ignore_index=True)
    single = pd.concat(singles, ignore_index=True)
    table.to_csv(out / "motion_vs_coherence.csv", index=False)
    single.to_csv(out / "motion_vs_coherence_single_features.csv", index=False)
    fig = figure(table, out / "motion_vs_coherence.png")
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print("\nstrongest single features per group (AUC > 0.5 = higher in generated clips):")
    print(single.groupby(["subset", "group"]).head(4).round(3).to_string(index=False))
    print(f"\nfigure: {fig}")


if __name__ == "__main__":
    main()
