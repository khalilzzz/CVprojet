# Cours : hors cours (visualisation) : style commun et figures de chaque run
"""Figures of the pipeline, with one shared style.

Conventions (same in every figure):
- real = blue, generated = orange, control (degraded real) = green; a second generated
  video of the same scene keeps the orange and changes marker and line style;
- one y axis per panel (never two scales in one panel), units in the axis label;
- the title of a figure says what to look at.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.geometry.homogeneous import to_h

COLORS = {"real": "#2a78d6", "fake": "#eb6834", "control": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
MARKERS = ["o", "s", "D", "^", "v"]
STYLES = ["-", "--", ":", "-."]
LABEL_NAME = {"real": "real", "fake": "generated", "control": "control"}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": "#b9b8b2", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "lines.linewidth": 2.0, "lines.markersize": 6.5,
        "legend.frameon": False, "savefig.dpi": 140, "figure.dpi": 100,
    })


def _series_style(videos: pd.DataFrame) -> dict:
    """video id -> dict(color, marker, ls, label). `videos` has columns video, label."""
    out, seen = {}, {}
    for _, r in videos.drop_duplicates("video").sort_values(["label", "video"],
                                                            ascending=[False, True]).iterrows():
        k = seen.get(r["label"], 0)
        seen[r["label"]] = k + 1
        out[r["video"]] = dict(color=COLORS.get(r["label"], INK2), marker=MARKERS[k % len(MARKERS)],
                               ls=STYLES[k % len(STYLES)],
                               label=f"{r['video']} ({LABEL_NAME.get(r['label'], r['label'])})")
    return out


def _plain_log(ax, axis: str = "y") -> None:
    """Log scale with plain tick labels (0.1, 0.2, 0.5). Call it after plotting: the ticks
    are 1-2-5 per decade on a short range and one per decade on a wide one."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    a = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    decades = np.log10(hi / lo) if lo > 0 else 9
    subs = (1.0, 2.0, 5.0) if decades <= 1.6 else (1.0, 3.0) if decades <= 2.6 else (1.0,)
    a.set_major_locator(LogLocator(base=10, subs=subs, numticks=12))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- one video
def _draw_epilines(ax, img, F, x, xp, mask, n_lines=18):
    h, w = img.shape[:2]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    idx = np.flatnonzero(mask)
    if len(idx) > n_lines:
        idx = idx[np.linspace(0, len(idx) - 1, n_lines).astype(int)]
    xs = np.array([0.0, w])
    for l in to_h(x[idx]) @ F.T:
        if abs(l[1]) > 1e-12:
            ax.plot(xs, -(l[0] * xs + l[2]) / l[1], color="#7ee081", lw=0.8, alpha=0.9)
    ax.scatter(*xp[mask].T, s=4, c="#7ee081", linewidths=0)
    ax.scatter(*xp[~mask].T, s=14, c="#ff5a52", marker="x", linewidths=1.0)
    ax.set(xlim=(0, w), ylim=(h, 0))
    ax.axis("off")


def video_sheet(df: pd.DataFrame, info: dict, out: Path, label: str = "") -> Path:
    """Per-video sheet: epipolar lines on two pairs, then the measures against the gap."""
    apply_style()
    ex, vf = info["examples"], info["frames"]
    video = df["video"].iloc[0]
    color = COLORS.get(label, INK2)
    h, w = vf.shape
    landscape = w >= h
    fig = plt.figure(figsize=(15.5, 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.1,
                             left=0.01, right=0.985, top=0.86, bottom=0.09)
    left = outer[0].subgridspec(2, 1, hspace=0.16) if landscape else outer[0].subgridspec(1, 2, wspace=0.04)
    right = outer[1].subgridspec(2, 2, hspace=0.5, wspace=0.3)
    deltas = sorted(ex)
    show = [d for d in (2, 20) if d in ex] or deltas[:1]
    if len(show) == 1 and len(deltas) > 1:
        show.append(deltas[-1])
    for k, delta in enumerate(show[:2]):
        ax = fig.add_subplot(left[k])
        t, x, xp, F, mask = ex[delta]
        _draw_epilines(ax, vf.frames[t + delta], F, x, xp, mask)
        ax.set_title(f"gap {delta} frames: {int(mask.sum())} of {len(mask)} fit F",
                     fontsize=10.5, color=INK2, loc="left")
    v = df[df["valid"]]
    g = v.groupby("delta")
    pos = {d: i for i, d in enumerate(sorted(df["delta"].unique()))}
    panels = [("sampson_med_all", "Sampson distance, median (px)", None),
              ("inlier_ratio_F", "share of matches that fit F", (0, 1.02)),
              ("median_flow_px", "image motion (px)", None),
              ("h_cover", "share of F inliers explained by H", (0, 1.02))]
    for k, (col, title, ylim) in enumerate(panels):
        ax = fig.add_subplot(right[k // 2, k % 2])
        med, lo, hi = g[col].median(), g[col].quantile(0.25), g[col].quantile(0.75)
        xs = [pos[d] for d in med.index]
        ax.fill_between(xs, lo.values, hi.values, color=color, alpha=0.18, linewidth=0)
        ax.plot(xs, med.values, "o-", color=color)
        ax.set_xticks(list(pos.values()), [str(d) for d in pos])
        ax.set_title(title, fontsize=10.5, loc="left")
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
        if k >= 2:
            ax.set_xlabel("frame gap (24 frames = 1 s)")
    name = f"{video} ({LABEL_NAME.get(label, label)})" if label else video
    fig.suptitle(f"{name}: epipolar consistency between frames\n"
                 f"Images: green = matches that fit F and some of their epipolar lines, red = other "
                 f"matches.   Curves: median over frame pairs, band = middle half.",
                 x=0.01, y=0.985, ha="left", va="top", fontsize=12)
    return _save(fig, out)


# ---------------------------------------------------------------- real vs generated
def scene_delta_grid(pairs: pd.DataFrame, out: Path) -> Path:
    """Small multiples: one row per scene, measures against the frame gap."""
    apply_style()
    cols = [("sampson_med_all", "Sampson distance, median (px)", True),
            ("inlier_ratio_F", "share of matches that fit F", False),
            ("parallax_px", "parallax (px)", True),
            ("h_cover", "share of F inliers explained by H", False)]
    scenes = sorted(pairs["scene_id"].unique())
    fig, axes = plt.subplots(len(scenes), len(cols), squeeze=False,
                             figsize=(4.1 * len(cols), 2.75 * len(scenes) + 0.9))
    pos = {d: i for i, d in enumerate(sorted(pairs["delta"].unique()))}
    for i, scene in enumerate(scenes):
        g = pairs[pairs["scene_id"] == scene]
        style = _series_style(g[["video", "label"]])
        for j, (col, title, log) in enumerate(cols):
            ax = axes[i, j]
            for video, gv in g.groupby("video"):
                q = gv.groupby("delta")[col]
                med = q.median().dropna()
                xs = [pos[d] for d in med.index]
                st = style[video]
                ax.fill_between(xs, q.quantile(0.25).reindex(med.index).values,
                                q.quantile(0.75).reindex(med.index).values,
                                color=st["color"], alpha=0.12, linewidth=0)
                ax.plot(xs, med.values, marker=st["marker"], ls=st["ls"], color=st["color"],
                        label=st["label"])
            if log:
                _plain_log(ax, "y")
            else:
                ax.set_ylim(0, 1.03)
            ax.set_xticks(list(pos.values()), [str(d) for d in pos])
            if i == 0:
                ax.set_title(title, fontsize=11)
            if i == len(scenes) - 1:
                ax.set_xlabel("frame gap (24 frames = 1 s)")
            if j == 0:
                ax.set_ylabel(scene, fontsize=10.5, color=INK)
                ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Same frame gap, not the same motion: a clip that moves less looks more consistent "
                 "(check the parallax column).\nLine = median over frame pairs, band = middle half.",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _save(fig, out)


def equal_motion(pairs: pd.DataFrame, binned: pd.DataFrame, out: Path,
                 col: str = "sampson_med_all", motion: str = "median_flow_px",
                 ylabel: str = "Sampson distance, median (px)") -> Path:
    """One panel per scene: measure against motion; dots = frame pairs, line = binned median."""
    apply_style()
    scenes = sorted(pairs["scene_id"].unique())
    ncol = min(3, len(scenes))
    nrow = int(np.ceil(len(scenes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(max(5.2 * ncol, 9.5), 3.9 * nrow + 0.9))
    for k, scene in enumerate(scenes):
        ax = axes[k // ncol, k % ncol]
        g = pairs[(pairs["scene_id"] == scene) & (pairs[motion] > 0)]
        style = _series_style(g[["video", "label"]])
        for video, gv in g.groupby("video"):
            st = style[video]
            ax.scatter(gv[motion], gv[col], s=9, color=st["color"], alpha=0.22, linewidths=0)
        b = binned[binned["scene_id"] == scene]
        for video, bv in b.groupby("video"):
            st = style[video]
            ax.plot(bv["motion_mid"], bv[col], marker=st["marker"], ls=st["ls"],
                    color=st["color"], label=st["label"], markeredgecolor=SURFACE,
                    markeredgewidth=1.2, markersize=7.5)
        if len(g):
            ylo, yhi = np.nanquantile(g[col], [0.01, 0.99])
            ax.set_ylim(ylo / 1.3, yhi * 1.3)
            xlo, xhi = np.nanquantile(g[motion], [0.01, 0.99])
            ax.set_xlim(xlo / 1.3, xhi * 1.3)
        _plain_log(ax, "x")
        _plain_log(ax, "y")
        ax.grid(True, axis="both", color=GRID)
        ax.set_title(scene, loc="left", fontsize=11.5)
        ax.set_xlabel("image motion of the frame pair (px)")
        if k % ncol == 0:
            ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", fontsize=8.5)
    for k in range(len(scenes), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("Same image motion: real against generated\n"
                 "Dot = one frame pair. Line = median in motion bins shared by the videos of the scene.",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _save(fig, out)


def scene_summary(matched: pd.DataFrame, out: Path, col: str = "sampson_matched",
                  xlabel: str = "Sampson distance at equal motion (px), lower = more consistent",
                  title: str = "Per scene, at equal image motion: real against generated",
                  fmt: str = "{:.3f}") -> Path:
    """One row per scene: real and generated values on one axis, joined by a line."""
    apply_style()
    scenes = list(dict.fromkeys(matched["scene_id"]))
    fig, ax = plt.subplots(figsize=(9.5, 1.2 + 0.85 * len(scenes)))
    used = set()
    for i, scene in enumerate(scenes):
        g = matched[matched["scene_id"] == scene].sort_values(col)
        ax.plot([g[col].min(), g[col].max()], [i, i], color="#b9b8b2", lw=1.5, zorder=1)
        n_fake = 0
        for _, r in g.iterrows():
            lab = LABEL_NAME.get(r["label"], r["label"])
            ax.scatter(r[col], i, s=95, color=COLORS.get(r["label"], INK2), zorder=3,
                       edgecolor=SURFACE, linewidth=1.5, label=None if lab in used else lab)
            used.add(lab)
            if r["label"] == "real":
                dy = 9
            else:                                   # stack the labels of several generated clips
                dy, n_fake = -15 - 11 * n_fake, n_fake + 1
            ax.annotate(fmt.format(r[col]), (r[col], i), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(scenes)), scenes)
    ax.set_ylim(len(scenes) - 0.3, -0.6)
    lo, hi = matched[col].min(), matched[col].max()
    ax.set_xlim(lo / 1.15, hi * 1.15)
    _plain_log(ax, "x")
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return _save(fig, out)


def three_view_grid(trip: pd.DataFrame, out: Path) -> Path:
    """One panel per scene: three-view reprojection error against the frame gap."""
    apply_style()
    scenes = sorted(trip["scene_id"].unique())
    ncol = min(3, len(scenes))
    nrow = int(np.ceil(len(scenes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(max(5.0 * ncol, 9.5), 3.7 * nrow + 0.9))
    pos = {d: i for i, d in enumerate(sorted(trip["delta"].unique()))}
    for k, scene in enumerate(scenes):
        ax = axes[k // ncol, k % ncol]
        g = trip[trip["scene_id"] == scene]
        style = _series_style(g[["video", "label"]])
        for video, gv in g.groupby("video"):
            q = gv.groupby("delta")["reproj3_med_px"]
            med = q.median().dropna()
            xs = [pos[d] for d in med.index]
            st = style[video]
            ax.fill_between(xs, q.quantile(0.25).reindex(med.index).values,
                            q.quantile(0.75).reindex(med.index).values,
                            color=st["color"], alpha=0.12, linewidth=0)
            ax.plot(xs, med.values, marker=st["marker"], ls=st["ls"], color=st["color"],
                    label=st["label"])
        _plain_log(ax, "y")
        ax.set_xticks(list(pos.values()), [str(d) for d in pos])
        ax.set_title(scene, loc="left", fontsize=11.5)
        ax.set_xlabel("frame gap k of the triplet (t, t+k, t+2k)")
        if k % ncol == 0:
            ax.set_ylabel("reprojection error in frame 3 (px)")
        ax.legend(loc="upper left", fontsize=8.5)
    for k in range(len(scenes), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    fig.suptitle("Three-view test: points triangulated from two frames, reprojected in a third\n"
                 "Look at the slope, not the level: rendering shifts a curve up or down, "
                 "a deforming scene makes it steeper.", x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _save(fig, out)


def ratio_by_generator(df: pd.DataFrame, meta: pd.DataFrame, col: str, out: Path, title: str,
                       xlabel: str) -> Path:
    """One row per generator: generated / real ratio of every scene (dots) and their median (bar)."""
    apply_style()
    gen = meta.set_index("id")["generator"]
    f = df[df["label"] == "fake"].assign(generator=lambda d: d["video"].map(gen).replace("", "unknown"))
    f = f[np.isfinite(f[col]) & (f[col] > 0)]
    order = f.groupby("generator")[col].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(9.5, 1.3 + 0.42 * len(order)))
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        v = f.loc[f["generator"] == g, col].values
        ax.scatter(v, i + rng.uniform(-0.16, 0.16, len(v)), s=9, color=COLORS["fake"], alpha=0.35,
                   linewidths=0)
        m = np.median(v)
        ax.plot([m, m], [i - 0.32, i + 0.32], color=INK, lw=2.2)
        ax.annotate(f"{m:.2f}", (m, i - 0.36), ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.axvline(1.0, color=COLORS["real"], lw=1.5)
    lo, hi = np.quantile(f[col], [0.02, 0.98])
    ax.set_xlim(min(lo, 0.8) / 1.1, max(hi, 1.25) * 1.1)
    _plain_log(ax, "x")
    ax.set_yticks(range(len(order)), [f"{g} (n={int((f['generator'] == g).sum())})" for g in order])
    ax.set_ylim(len(order) - 0.4, -0.7)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.set_xlabel(xlabel)
    ax.set_title(title + "\nDot = one scene, black bar = median, blue line = same as the real clip",
                 loc="left", fontsize=11.5)
    fig.tight_layout()
    return _save(fig, out)
