# Cours : Session 5 (évaluation) : détection selon ce que le générateur a reçu (image ou texte)
"""Does a feature family detect "generated", or only "different content"?

A text-to-video clip shows other content than its real counterpart; an image-to-video clip
starts from a frame of the real clip, so its content is nearly the same. A family whose AUC
falls on image-to-video clips was partly detecting content.

Two analyses on a finished run (no new feature computation):
1. out-of-fold scores of the main classification (trained on every generator), AUC of the
   real clips against each group of generators, with a bootstrap interval over scenes, and
   the paired difference CLIP - physics on the same bootstrap samples;
2. classifiers retrained on the real clips and the image-to-video clips only.

    python -m src.eval.by_conditioning results/latest_vifbench
Writes <run>/classification/by_conditioning*.csv and by_conditioning.png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.vifbench import _cond_type
from src.eval import plots
from src.models import classify

GROUPS = {"image": "image-to-video", "text": "text-to-video", "": "not stated (closed models)"}
def _physics(names) -> str | None:
    """Name of the physics family in a score table (it changed when I2 was added)."""
    return next((n for n in names if n.startswith("physics (")), None)


def _shown(names) -> list[str]:
    return [n for n in ["V1 geometry", _physics(names), "CLIP baseline", "physics + CLIP"] if n in names]


def _auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def bootstrap(df: pd.DataFrame, fams: list[str], n: int = 1000, seed: int = 0) -> pd.DataFrame:
    """AUC per family and its bootstrap over scenes; also CLIP - physics on the same samples."""
    phys = _physics(fams)
    rng = np.random.default_rng(seed)
    y = (df["label"] == "fake").to_numpy(int)
    idx_by = df.groupby("scene_id").indices
    scenes = np.array(list(idx_by))
    point = {f: _auc(y, df[f].to_numpy()) for f in fams}
    draws = {f: [] for f in fams}
    diff = []
    for _ in range(n):
        idx = np.concatenate([idx_by[s] for s in rng.choice(scenes, len(scenes), replace=True)])
        a = {f: _auc(y[idx], df[f].to_numpy()[idx]) for f in fams}
        for f in fams:
            draws[f].append(a[f])
        if "CLIP baseline" in a and phys in a:
            diff.append(a["CLIP baseline"] - a[phys])
    rows = [dict(family=f, AUC=point[f], low=np.nanquantile(draws[f], 0.025),
                 high=np.nanquantile(draws[f], 0.975)) for f in fams]
    if diff:
        d = np.array(diff)
        rows.append(dict(family="CLIP - physics (difference)",
                         AUC=point["CLIP baseline"] - point[phys],
                         low=np.nanquantile(d, 0.025), high=np.nanquantile(d, 0.975),
                         share_CLIP_better=float((d > 0).mean())))
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, out: Path) -> Path:
    plots.apply_style()
    groups = [g for g in GROUPS.values() if g in set(table["group"])]
    fams = _shown(set(table["family"]))
    fig, ax = plt.subplots(figsize=(9.5, 1.4 + 1.1 * len(groups)))
    for i, g in enumerate(groups):
        for k, f in enumerate(fams):
            r = table[(table["group"] == g) & (table["family"] == f)]
            if r.empty:
                continue
            r = r.iloc[0]
            yv = i + (k - (len(fams) - 1) / 2) * 0.18
            c = plots.FAMILY_COLORS[k]
            ax.plot([r["low"], r["high"]], [yv, yv], color=c, lw=2.2, alpha=0.6)
            ax.scatter(r["AUC"], yv, s=60, color=c, zorder=3, edgecolor=plots.SURFACE, linewidth=1.2,
                       label=f if i == 0 else None)
            ax.annotate(f"{r['AUC']:.2f}", (r["high"], yv), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=plots.INK2)
    ax.axvline(0.5, color="#b9b8b2", ls="--", lw=1.2)
    ax.set_yticks(range(len(groups)), groups)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC, real clips against the clips of this group (dot = estimate, bar = 95 % interval)")
    ax.grid(True, axis="x", color=plots.GRID)
    ax.grid(False, axis="y")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Is CLIP detecting AI, or different content? Detection by kind of generator",
                 loc="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder, e.g. results/latest_vifbench")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run, cdir = Path(args.run), Path(args.run) / "classification"
    feats = pd.read_csv(run / "features.csv", low_memory=False)
    feats["generator"] = feats["generator"].fillna("")
    scores = pd.read_csv(cdir / "classify_oof_scores.csv", low_memory=False)
    df = scores.drop(columns=["label", "generator"]).merge(
        feats[["video", "label", "scene_id", "generator"]], on="video")
    df["cond"] = df["generator"].map(lambda g: _cond_type(g) if g else "real")
    fams = [c for c in scores.columns if c not in ("video", "label", "generator")]

    # 1. scores of the main classification, per group of generators
    rows = []
    for cond, name in GROUPS.items():
        sub = df[(df["cond"] == "real") | (df["cond"] == cond)]
        if (sub["label"] == "fake").sum() == 0:
            continue
        t = bootstrap(sub, fams, args.n_boot, args.seed)
        t.insert(0, "group", name)
        t.insert(1, "n_generated", int((sub["label"] == "fake").sum()))
        rows.append(t)
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(cdir / "by_conditioning.csv", index=False)
    fig = figure(table[~table["family"].str.startswith("CLIP - ")], cdir / "by_conditioning.png")

    # 2. classifiers retrained on real + image-to-video clips only
    sub = feats[(feats["label"] == "real") | (feats["generator"].map(_cond_type) == "image")]
    retrained = classify.run(sub, seed=args.seed, unseen=False)["summary"]
    retrained.to_csv(cdir / "by_conditioning_retrained_image_to_video.csv", index=False)

    pd.set_option("display.width", 200)
    print("== 1. main classifiers, AUC per group of generators (95 % bootstrap over scenes)")
    print(table.round(3).to_string(index=False))
    print("\n== 2. classifiers retrained on real + image-to-video clips only")
    print(retrained[["family", "n_features", "AUC", "AUC_low", "AUC_high"]].round(3).to_string(index=False))
    print(f"\nfigure: {fig}")


if __name__ == "__main__":
    main()
