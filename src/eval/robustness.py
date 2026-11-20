# Cours : Sessions 1-2 (rééchantillonnage, compression) et Session 5 (évaluation) : expérience 3, robustesse aux dégradations
"""Experiment 3 of CLAUDE.md: do the cues survive recompression and downscaling?

Every selected video (already normalized, H.264 CRF 18) is degraded once more, as a
re-upload on a social network would do: H.264 at a higher CRF, and/or downscaled. Each
level is a separate set of files in data/processed/degraded/<level>/ and goes through the
same feature extraction (cached like the main run).

Two questions, answered on the same scene-grouped folds as src/models/classify.py:
1. trained on clean clips, tested on degraded clips (the detector meets a recompressed
   video it was not trained on): AUC per family and level;
2. trained and tested at the same level (is the information still there at all?).
And for each readable indicator (src/eval/indicators.py), its own AUC at each level.

    python -m src.eval.robustness --config configs/vifbench.yaml --dataset vifbench \
        --max-scenes 40 --features v1,i1,v3,i2,clip --workers 16
Writes results/robustness_<dataset>/ (tables, figures, report.html).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import run as pipeline
from src.data.ingest import ingest
from src.data.normalize import _ffmpeg
from src.eval import indicators, plots
from src.eval.report import Report
from src.models import classify
from src.utils.config import REPO_ROOT, load_config

DEGRADED = REPO_ROOT / "data" / "processed" / "degraded"
DEFAULT_LEVELS = [dict(name="clean"), dict(name="crf28", crf=28), dict(name="crf35", crf=35),
                  dict(name="half", scale=0.5), dict(name="half_crf35", scale=0.5, crf=35)]


def degrade(src: Path, level: dict, out_dir: Path = DEGRADED) -> Path:
    """Re-encode src with the level's CRF (default 18) and scale (default 1). Cached."""
    if level["name"] == "clean":
        return src
    dst = out_dir / level["name"] / f"{src.stem}__{level['name']}.mp4"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-an"]
    s = level.get("scale", 1.0)
    if s != 1.0:
        cmd += ["-vf", f"scale=trunc(iw*{s}/2)*2:trunc(ih*{s}/2)*2:flags=area"]
    cmd += ["-c:v", "libx264", "-crf", str(level.get("crf", 18)), "-preset", "medium",
            "-pix_fmt", "yuv420p", str(dst)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed on {src.name}: {done.stderr.strip()[-400:]}")
    return dst


def _align(tables: dict[str, pd.DataFrame], order: pd.Series) -> dict[str, pd.DataFrame]:
    """Same rows (video order of the clean table) and same columns in every level."""
    cols = sorted(set().union(*[t.columns for t in tables.values()]))
    return {k: classify.add_derived(t.set_index("video").reindex(order).reset_index().reindex(columns=cols))
            for k, t in tables.items()}


def evaluate(tables: dict[str, pd.DataFrame], seed: int = 0, n_splits: int = 5,
             n_boot: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """AUC per family and level, trained on clean or retrained at the level; AUC of each
    readable indicator per level. tables[level]: one row per video (same videos)."""
    tables = _align(tables, tables["clean"]["video"])
    base = tables["clean"]
    y = (base["label"] == "fake").to_numpy(int)
    groups = base["scene_id"].to_numpy()
    folds = classify._folds(y, groups, n_splits, seed)
    rows = []
    for fam, prefixes in classify.FAMILIES.items():
        cols = classify.family_columns(base, prefixes)
        if not cols:
            continue
        print(f"[robustness] {fam}: {len(cols)} features", flush=True)
        Xc = base[cols].to_numpy(float)
        clean_scores = {lv: np.full(len(y), np.nan) for lv in tables}
        for tr, te in folds:
            m = classify.make_model(fam, seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(Xc[tr], y[tr])
                for lv, t in tables.items():
                    clean_scores[lv][te] = m.predict_proba(t[cols].to_numpy(float)[te])[:, 1]
        for lv, t in tables.items():
            for mode, s in [("trained on clean", clean_scores[lv]),
                            ("retrained at this level",
                             clean_scores[lv] if lv == "clean" else
                             classify.oof_scores(t[cols].to_numpy(float), y, folds, fam, seed))]:
                lo, hi = classify.bootstrap_auc(y, s, groups, n=n_boot, seed=seed)
                rows.append(dict(family=fam, level=lv, mode=mode, AUC=classify._auc(y, s),
                                 AUC_low=lo, AUC_high=hi))
    single = []
    ind = indicators.resolve(base.columns)
    for _, r in ind.iterrows():
        for lv, t in tables.items():
            single.append(dict(indicator=r["indicator"], column=r["column"], family=r["family"],
                               kind=r["kind"], level=lv,
                               AUC=classify._auc(y, t[r["column"]].to_numpy(float))))
    return pd.DataFrame(rows), pd.DataFrame(single)


def figure_auc(table: pd.DataFrame, levels: list[str], out: Path) -> Path:
    plots.apply_style()
    fams = list(dict.fromkeys(table["family"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, mode in zip(axes, ["trained on clean", "retrained at this level"]):
        for k, fam in enumerate(fams):
            t = table[(table["family"] == fam) & (table["mode"] == mode)].set_index("level").reindex(levels)
            x = np.arange(len(levels))
            c = plots.FAMILY_COLORS[k % len(plots.FAMILY_COLORS)]
            ls = "--" if "CLIP" in fam else "-"
            ax.fill_between(x, t["AUC_low"], t["AUC_high"], color=c, alpha=0.08, linewidth=0)
            ax.plot(x, t["AUC"], marker="o", ls=ls, color=c, label=fam)
        ax.axhline(0.5, color="#b9b8b2", ls="--", lw=1.2)
        ax.set_xticks(range(len(levels)), levels)
        ax.set_xlabel("degradation (crfN = H.264 at CRF N; half = width and height / 2)")
        ax.set_title(mode, loc="left", fontsize=11.5)
    axes[0].set_ylabel("AUC, real against generated")
    axes[1].legend(loc="upper right", fontsize=8.5, ncol=2)
    fig.suptitle("Robustness: does each family still tell real from generated after recompression?\n"
                 "Left: the classifier never saw degraded clips. Right: trained at each level. "
                 "Dashed = deep learning baseline.", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return plots._save(fig, out)


def figure_indicators(single: pd.DataFrame, levels: list[str], out: Path) -> Path:
    plots.apply_style()
    t = single.pivot_table(index=["family", "indicator"], columns="level", values="AUC", sort=False)
    t = t.reindex(columns=levels)
    fig, ax = plt.subplots(figsize=(2.6 + 1.25 * len(levels), 1.2 + 0.36 * len(t)))
    v = t.to_numpy(float)
    im = ax.imshow(np.abs(v - 0.5) + 0.5, cmap="Greens", vmin=0.5, vmax=0.75, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, "-" if not np.isfinite(v[i, j]) else f"{v[i, j]:.2f}", ha="center",
                    va="center", fontsize=8.5,
                    color="white" if np.isfinite(v[i, j]) and abs(v[i, j] - 0.5) > 0.17 else plots.INK)
    ax.set_xticks(range(len(levels)), levels, fontsize=9)
    ax.set_yticks(range(len(t)), [f"{f}: {n}" for f, n in t.index], fontsize=8.5)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("distance to chance, shown as AUC (0.5 = none)")
    ax.set_title("Each indicator alone: AUC per degradation level\n"
                 "above 0.5 = higher in generated clips, below 0.5 = lower; colour = strength",
                 loc="left", fontsize=11.5)
    fig.tight_layout()
    return plots._save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    pipeline.add_selection_args(ap)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    levels = cfg.get("robustness", {}).get("levels", DEFAULT_LEVELS)
    if levels[0]["name"] != "clean":
        levels = [dict(name="clean")] + levels
    meta = ingest()
    videos = pipeline.select_videos(meta, args)
    processed = pipeline.normalize_all(videos, cfg)
    families = [f.strip() for f in args.features.split(",") if f.strip()]
    workers = args.workers or cfg["run"]["workers"]
    out = REPO_ROOT / "results" / f"robustness_{args.dataset}"
    run_dir = REPO_ROOT / "results" / "runs" / f"robustness-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    tables = {}
    for lv in levels:
        name = lv["name"]
        print(f"== level {name}", flush=True)
        if not args.report_only:
            src = {vid: degrade(p, lv) for vid, p in processed.items()}
        else:
            src = {vid: (p if name == "clean" else DEGRADED / name / f"{p.stem}__{name}.mp4")
                   for vid, p in processed.items()}
            src = {k: v for k, v in src.items() if v.exists()}
        sub = videos[videos["id"].isin(src)]
        dirs = pipeline.compute_families(families, sub, src, cfg, args.force, workers, args.report_only)
        tables[name] = pipeline.collect_features(meta, sub, dirs)
        tables[name].to_csv(run_dir / f"features_{name}.csv", index=False)
    names = [lv["name"] for lv in levels if lv["name"] in tables]
    table, single = evaluate(tables, cfg["seed"], cfg.get("classify", {}).get("n_splits", 5), args.n_boot)
    table.to_csv(run_dir / "robustness_auc.csv", index=False)
    single.to_csv(run_dir / "robustness_indicators.csv", index=False)

    figs = run_dir / "figures"
    rep = Report("Robustness to recompression and downscaling",
                 f"{datetime.now():%Y-%m-%d %H:%M} | {len(videos)} videos "
                 f"({(videos['label'] == 'real').sum()} real) | levels: "
                 + ", ".join(f"{lv['name']} (CRF {lv.get('crf', 18)}, scale {lv.get('scale', 1.0)})"
                             for lv in levels))
    rep.p("Each clip, already normalized (H.264 CRF 18), is encoded once more at each level. "
          "Same scene-grouped folds as the main classification. Physical cues are expected to be "
          "less sensitive to compression than a deep model, except the frequency fingerprint (I1), "
          "which lives in the fine noise that compression removes.")
    rep.figure(figure_auc(table, names, figs / "robustness_auc.png"), "AUC per family and level")
    wide = table.pivot_table(index=["family", "mode"], columns="level", values="AUC", sort=False)
    rep.table(wide.reindex(columns=names).reset_index())
    rep.figure(figure_indicators(single, names, figs / "robustness_indicators.png"),
               "AUC of each indicator per level")
    rep.write(run_dir / "report.html")
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(run_dir, out)
    pd.set_option("display.width", 200)
    print(wide.reindex(columns=names).round(3).to_string())
    print(f"[done] {time.time() - t0:.0f} s | report: {(out / 'report.html').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
