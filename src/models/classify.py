# Cours : Session 5 (classification) : classifieurs sur nos features, ablations, test inter-générateurs
"""Classify videos as real or generated from the per-video features, family by family.

Families: V1 geometry (``v1_*``), I1 frequency (``i1_*``), V3 temporal (``v3_*``), their
union ("physics"), the CLIP baseline (``clip_*``) and physics + CLIP.

Protocol (CLAUDE.md sections 5 and 7):
- 5-fold cross-validation **grouped by scene**: a real clip and its generated versions are
  always in the same fold, so the classifier cannot match a fake to its real twin;
- physics families: gradient boosting (handles missing values); families with CLIP: logistic
  regression on standardized features (linear probe);
- metrics on out-of-fold scores: AUC (with a bootstrap interval over scenes), balanced
  accuracy, F1 of the "generated" class; AUC per generator (all real clips against the clips
  of that generator);
- unseen generator: the generator is removed from the training folds, then tested.

Standalone use:  python -m src.models.classify results/latest_vifbench/features.csv
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAMILIES = {
    "V1 geometry": ("v1_",),
    "I1 frequency": ("i1_",),
    "V3 temporal": ("v3_",),
    "physics (V1+I1+V3)": ("v1_", "i1_", "v3_"),
    "CLIP baseline": ("clip_",),
    "physics + CLIP": ("v1_", "i1_", "v3_", "clip_"),
}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Growth of the three-view error between the smallest and largest gap (V1, Session 7)."""
    cols = sorted([c for c in df.columns if c.startswith("v1_reproj3_med_px_g")],
                  key=lambda c: int(c.rsplit("g", 1)[1]))
    if len(cols) >= 2:
        lo, hi = df[cols[0]], df[cols[-1]]
        df = df.assign(v1_reproj_growth=np.where((lo > 0) & np.isfinite(lo), hi / lo, np.nan))
    return df


def family_columns(df: pd.DataFrame, prefixes, max_missing: float = 0.5) -> list[str]:
    cols = [c for c in df.columns if c.startswith(tuple(prefixes))]
    return [c for c in cols if df[c].isna().mean() <= max_missing and df[c].nunique() > 1]


def make_model(name: str, seed: int):
    if "CLIP" in name:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=seed)


def _folds(y, groups, n_splits, seed):
    n = min(n_splits, pd.Series(groups).nunique())
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups))


def oof_scores(X, y, folds, name, seed, train_mask=None, test_mask=None) -> np.ndarray:
    """Out-of-fold probability of "generated". Rows outside test_mask stay NaN."""
    s = np.full(len(y), np.nan)
    for tr, te in folds:
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        if test_mask is not None:
            te = te[test_mask[te]]
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        m = make_model(name, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return s


def _auc(y, s):
    ok = np.isfinite(s)
    return roc_auc_score(y[ok], s[ok]) if ok.any() and len(np.unique(y[ok])) == 2 else np.nan


def bootstrap_auc(y, s, groups, n=200, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    g = pd.Series(groups)
    scenes = g.unique()
    idx_by = g.groupby(g).indices
    vals = []
    for _ in range(n):
        pick = rng.choice(scenes, len(scenes), replace=True)
        idx = np.concatenate([idx_by[p] for p in pick])
        a = _auc(y[idx], s[idx])
        if np.isfinite(a):
            vals.append(a)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else (np.nan, np.nan)


def run(df: pd.DataFrame, seed: int = 0, n_splits: int = 5, unseen: bool = True) -> dict:
    """df: one row per video with columns label, scene_id, generator and feature columns."""
    df = add_derived(df.reset_index(drop=True))
    y = (df["label"] == "fake").to_numpy(int)
    groups = df["scene_id"].to_numpy()
    gen = df["generator"].fillna("").replace("", "unknown").to_numpy()
    folds = _folds(y, groups, n_splits, seed)
    summary, per_gen, unseen_rows, rocs, scores = [], {}, {}, {}, pd.DataFrame({"video": df["video"], "label": df["label"], "generator": gen})
    gens = sorted(set(gen[y == 1]))
    for name, prefixes in FAMILIES.items():
        cols = family_columns(df, prefixes)
        if not cols:
            continue
        X = df[cols].to_numpy(float)
        t0 = time.time()
        print(f"[classify] {name}: {len(cols)} features, {len(y)} videos", flush=True)
        s = oof_scores(X, y, folds, name, seed)
        scores[name] = s
        auc = _auc(y, s)
        lo, hi = bootstrap_auc(y, s, groups, seed=seed)
        pred = (s >= 0.5).astype(int)
        summary.append(dict(family=name, model="logistic regression" if "CLIP" in name else "gradient boosting",
                            n_features=len(cols), AUC=auc, AUC_low=lo, AUC_high=hi,
                            balanced_accuracy=balanced_accuracy_score(y, pred),
                            F1_generated=f1_score(y, pred)))
        fpr, tpr, _ = roc_curve(y, s)
        rocs[name] = (fpr, tpr, auc)
        per_gen[name] = {g: _auc(y[(y == 0) | (gen == g)], s[(y == 0) | (gen == g)]) for g in gens}
        if unseen:
            col = {}
            for g in gens:
                keep_train = gen != g                              # never train on generator g
                test = (y == 0) | (gen == g)
                su = oof_scores(X, y, folds, name, seed, train_mask=keep_train, test_mask=test)
                col[g] = _auc(y[test], su[test])
            unseen_rows[name] = col
        print(f"[classify] {name}: AUC {auc:.3f} ({time.time() - t0:.0f} s)", flush=True)
    out = dict(summary=pd.DataFrame(summary), per_generator=pd.DataFrame(per_gen).rename_axis("generator").reset_index(),
               rocs=rocs, scores=scores, importance=importance(df, seed))
    if unseen_rows:
        out["unseen_generator"] = pd.DataFrame(unseen_rows).rename_axis("generator").reset_index()
    return out


def importance(df: pd.DataFrame, seed: int, top: int = 15) -> pd.DataFrame:
    """Standardized logistic-regression coefficients of the physics features (fit on all rows).
    Positive = pushes towards "generated"."""
    cols = family_columns(df, ("v1_", "i1_", "v3_"))
    if not cols:
        return pd.DataFrame()
    y = (df["label"] == "fake").to_numpy(int)
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df[cols].to_numpy(float), y)
    coef = pd.Series(m[-1].coef_[0], index=cols)
    return coef.reindex(coef.abs().sort_values(ascending=False).index[:top]).rename("coefficient") \
               .rename_axis("feature").reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features", help="features.csv of a run (one row per video)")
    ap.add_argument("--out", help="output folder (default: next to the csv)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from src.eval import plots
    df = pd.read_csv(args.features)
    out = Path(args.out or Path(args.features).parent / "classification")
    out.mkdir(parents=True, exist_ok=True)
    res = run(df, args.seed)
    for k in ("summary", "per_generator", "unseen_generator", "importance"):
        if k in res:
            res[k].to_csv(out / f"classify_{k}.csv", index=False)
    plots.classification_figures(res, out)
    print(res["summary"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
