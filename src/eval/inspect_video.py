# Cours : toutes sessions (synthèse) : appliquer toutes nos méthodes à une vidéo et lire chaque indicateur
"""Every method of the project on one video (or a few side by side), and what each says.

    python -m src.eval.inspect_video path/to/clip.mp4
    python -m src.eval.inspect_video real.mp4 fake.mp4 --labels real fake
    python -m src.eval.inspect_video clip.mp4 --reference results/latest/features.csv

For each video: the same normalization and the same feature code as the main run
(src/run.py), then
- V1 (Sessions 3, 4, 7): epipolar lines, Sampson distance, three-view test,
- I1 (Sessions 1-2): noise spectrum and its periodic peaks,
- V3 (Session 6): warping residual after optical flow,
- I2 (Sessions 1-4): vanishing points and their stability over time,
and a table of the readable indicators (src/eval/indicators.py).

What a number means depends on the clip (motion, resolution, content). A reference makes
it readable: the features.csv of a run made with the SAME config (for example 20 real
stock clips run with python -m src.run). Each indicator is then placed among the real clips
of the reference (percentile), and a classifier trained on the reference gives a score.
Writes results/inspect/<name>/report.html.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.frames import read_frames
from src.data.normalize import normalize_video
from src.eval import indicators, plots
from src.eval.report import Report
from src.features import image_frequency as i1
from src.features import image_perspective as i2
from src.features import video_epipolar as v1
from src.features import video_temporal as v3
from src.models import classify
from src.utils.config import REPO_ROOT, load_config

INSPECT = REPO_ROOT / "data" / "processed" / "inspect"
VP_COLORS = ["#e8463a", "#2fb36d", "#3a7fe8"]


def all_features(path: Path, name: str, cfg: dict, label: str, figs: Path) -> tuple[dict, list]:
    """Normalize, compute every family, draw the figures. Returns (features, [(png, alt)])."""
    norm = normalize_video(path, cfg["normalize"], out_dir=INSPECT, name=name)
    print(f"[{name}] V1 (epipolar, three views)", flush=True)
    df, trip, info = v1.analyse_video(norm, name, cfg, cfg["seed"])
    feats = v1.video_features(df, trip)
    out = [(plots.video_sheet(df, info, figs / f"{name}_v1.png", label), f"V1 sheet of {name}")]
    print(f"[{name}] I1 (frequency), V3 (temporal), I2 (vanishing points)", flush=True)
    feats |= i1.video_features(norm, cfg)
    feats |= v3.video_features(norm, cfg)
    f2, det = i2.video_features(norm, cfg, return_frames=True)
    feats |= f2
    out.append((sheet_i1_v3_i2(norm, name, cfg, det, label, figs / f"{name}_i1_v3_i2.png"),
                f"I1, V3 and I2 sheet of {name}"))
    feats.pop("video", None)
    return classify.add_derived(pd.DataFrame([feats])).iloc[0].to_dict(), out


def sheet_i1_v3_i2(norm: Path, name: str, cfg: dict, det: dict, label: str, out: Path) -> Path:
    plots.apply_style()
    fr, c1 = cfg["frames"], cfg["i1"]
    vf = read_frames(norm, skip_s=fr["skip_s"], crop_bottom=fr["crop_bottom"], crop_top=fr["crop_top"],
                     duration_s=fr.get("duration_s"))
    idx = np.unique(np.linspace(0, len(vf.frames) - 1, c1["n_frames"]).round().astype(int))
    spec = np.mean([i1.log_spectrum(i1.noise_residual(vf.frames[i], c1.get("median_ksize", 3)), c1["crop"])
                    for i in idx], axis=0)
    centres, prof = i1.radial_profile(spec, c1["n_bins"])
    excess = spec - np.interp(i1.radial_frequency(spec.shape[0]), centres, prof)
    color = plots.COLORS.get(label, plots.INK2)
    fig = plt.figure(figsize=(16, 9.2))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.36, left=0.04, right=0.97, top=0.88, bottom=0.07)
    # I1
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(excess, cmap="magma", vmin=0, vmax=1.5, extent=(-0.5, 0.5, 0.5, -0.5))
    ax.set_title("I1: noise spectrum minus its ring average\nbright isolated dots = periodic peaks",
                 loc="left", fontsize=10.5)
    ax.set_xlabel("frequency (cycles / pixel)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("excess (log10 power)")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(centres, prof, "o-", color=color, ms=4)
    ax.set(xlabel="frequency (cycles / pixel), 0.5 = finest", ylabel="log10 power of the noise residual")
    ax.set_title("I1: radial profile of the noise spectrum", loc="left", fontsize=10.5)
    # V3
    t = len(vf.frames) // 2
    hres, ok, _ = v3.residual_map(vf.frames[t], vf.frames[min(t + 1, len(vf.frames) - 1)])
    show = np.where(ok, hres, np.nan)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(show, cmap="viridis", vmin=0, vmax=np.nanquantile(show, 0.99) if np.isfinite(show).any() else 1)
    ax.set_title("V3: fine texture change after motion compensation\n(middle frame pair; white = "
                 "flow check failed)", loc="left", fontsize=10.5)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("grey levels")
    # I2: one frame with its segments, coloured by vanishing point
    pf, frames, times = det["per_frame"], det["frames"], det["times"]
    best = int(np.argmax([len(p["dirs"]) * 1000 + p["n_segments"] for p in pf])) if pf else 0
    ax = fig.add_subplot(gs[1, 0])
    if pf:
        ax.imshow(frames[best], cmap="gray", vmin=0, vmax=255)
        p = pf[best]
        for s, lab in zip(p["segments"], p["labels"]):
            ax.plot([s[0], s[2]], [s[1], s[3]], lw=1.4 if lab >= 0 else 0.8,
                    color=VP_COLORS[lab % 3] if lab >= 0 else "#c9c9c9", alpha=0.95 if lab >= 0 else 0.6)
    when = f" (t = {times[best]:.1f} s)" if pf else ""
    ax.set_title(f"I2: segments of one frame{when}\ncolour = vanishing point, grey = none",
                 loc="left", fontsize=10.5)
    ax.axis("off")
    # I2 over time
    tracks, _ = i2.build_tracks(pf, cfg["i2"].get("max_jump_deg", 6.0))
    long = sorted([t for t in tracks if len(t) >= cfg["i2"].get("min_track_len", 5)], key=len, reverse=True)[:3]
    ax = fig.add_subplot(gs[1, 1])
    for k, tr in enumerate(long):
        d0 = tr[0][1]
        ax.plot([times[j] for j, _ in tr], [i2.angle_deg(d0, d) for _, d in tr], "o-", ms=3,
                color=VP_COLORS[k], label=f"direction {k + 1} ({len(tr)} frames)")
    ax.set(xlabel="time (s)", ylabel="angle to its first position (deg)")
    ax.set_title("I2: each 3D direction over time\nsmooth = camera turning; jagged = unstable scene",
                 loc="left", fontsize=10.5)
    if long:
        ax.legend(fontsize=8.5)
    ax = fig.add_subplot(gs[1, 2])
    drawn = 0
    min_len = cfg["i2"].get("min_track_len", 5)
    every = sorted([dict(t) for t in tracks if len(t) >= min_len], key=len, reverse=True)
    for a in range(len(every)):                  # same pairs as the feature i2_pair_angle_std_deg
        for b in range(a + 1, len(every)):
            A, B = every[a], every[b]
            common = sorted(set(A) & set(B))
            if len(common) >= min_len and drawn < 4:
                ax.plot([times[j] for j in common], [i2.angle_deg(A[j], B[j]) for j in common], "o-",
                        ms=3, label=f"pair {drawn + 1} ({len(common)} frames)")
                drawn += 1
    if drawn:
        ax.legend(fontsize=8.5)
    else:
        ax.text(0.5, 0.5, "no two directions followed together\n(too few straight lines)",
                ha="center", va="center", transform=ax.transAxes, color=plots.INK2)
    ax.set(xlabel="time (s)", ylabel="angle between two directions (deg)")
    ax.set_title("I2: angle between two 3D directions\nrigid scene = flat line (rotation keeps angles)",
                 loc="left", fontsize=10.5)
    fig.suptitle(f"{name}" + (f" ({plots.LABEL_NAME.get(label, label)})" if label else "")
                 + ": frequency (I1), temporal coherence (V3), vanishing points (I2)",
                 x=0.01, ha="left", fontsize=13)
    return plots._save(fig, out)


def read_reference(path: Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    ref = classify.add_derived(pd.read_csv(path, low_memory=False))
    print(f"[reference] {path}: {(ref['label'] == 'real').sum()} real, {(ref['label'] == 'fake').sum()} generated")
    return ref


def reading(col: str, value: float, expected: str, kind: str, ref: pd.DataFrame | None) -> tuple[float, str]:
    """(percentile among the real clips of the reference, sentence)."""
    if kind == "context":
        return np.nan, "context"
    if ref is None or col not in ref:
        return np.nan, "no reference"
    real = ref.loc[ref["label"] == "real", col].dropna().to_numpy(float)
    fake = ref.loc[ref["label"] == "fake", col].dropna().to_numpy(float)
    if len(real) < 8 or not np.isfinite(value):
        return np.nan, "too few real clips in the reference" if len(real) < 8 else "not measured"
    pct = 100.0 * ((real < value).mean() + 0.5 * (real == value).mean())
    if 5.0 <= pct <= 95.0:
        return pct, "typical of real clips"
    high = pct > 95.0
    if len(fake) >= 8:
        gen_high = np.median(fake) > np.median(real)
    elif expected:
        gen_high = expected == "low"
    else:
        return pct, "unusual (" + ("higher" if high else "lower") + " than 95 % of real clips)"
    side = "on the generated side" if high == gen_high else "on the other side"
    return pct, f"unusual, {side} ({'higher' if high else 'lower'} than 95 % of real clips)"


def classifier_scores(ref: pd.DataFrame | None, feats: dict[str, dict], seed: int) -> dict[str, float]:
    """P(generated) from a physics classifier trained on the whole reference."""
    if ref is None or (ref["label"] == "real").sum() < 10 or (ref["label"] == "fake").sum() < 10:
        return {}
    new = pd.DataFrame(feats).T
    cols = [c for c in classify.family_columns(ref, classify.PHYSICS_PREFIXES) if c in new.columns]
    if not cols:
        return {}
    m = classify.make_model("physics", seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(ref[cols].to_numpy(float), (ref["label"] == "fake").to_numpy(int))
    p = m.predict_proba(new[cols].to_numpy(float))[:, 1]
    return dict(zip(new.index, p.astype(float)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+", help="video files")
    ap.add_argument("--labels", nargs="*", help="real / fake for each video, if known (colours only)")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--reference", help="features.csv of a run made with the same config")
    ap.add_argument("--out", help="output folder (default results/inspect/<first video>)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = [Path(v) for v in args.videos]
    labels = (args.labels or []) + [""] * len(paths)
    names = [p.stem for p in paths]
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "inspect" / "_vs_".join(names)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    ref_path = args.reference or cfg.get("inspect", {}).get("reference") or None
    ref = read_reference(Path(ref_path) if ref_path else None)

    feats, sheets = {}, {}
    for p, n, lab in zip(paths, names, labels):
        feats[n], sheets[n] = all_features(p, n, cfg, lab, figs)
    (out / "features.json").write_text(json.dumps(feats, indent=1, default=float))

    ind = indicators.resolve(set().union(*[f.keys() for f in feats.values()]))
    rows = []
    for _, r in ind.iterrows():
        row = dict(family=r["family"], session=r["session"], indicator=r["indicator"],
                   real_camera=r["real_camera"] or ("context" if r["kind"] == "context" else "?"))
        if ref is not None and r["column"] in ref and r["kind"] == "cue":
            rv = ref.loc[ref["label"] == "real", r["column"]].dropna()
            fv = ref.loc[ref["label"] == "fake", r["column"]].dropna()
            row["real clips (5-95 %)"] = (f"{rv.quantile(0.05):.3g} to {rv.quantile(0.95):.3g}"
                                          if len(rv) >= 8 else "-")
            row["generated median"] = f"{fv.median():.3g}" if len(fv) else "-"
        for n in names:
            v = feats[n].get(r["column"], np.nan)
            pct, txt = reading(r["column"], v, r["real_camera"], r["kind"], ref)
            row[n] = v
            row[f"{n}: reading"] = txt if not np.isfinite(pct) else f"{txt} (percentile {pct:.0f})"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "indicators.csv", index=False)
    scores = classifier_scores(ref, feats, cfg["seed"])

    rep = Report("All our methods on " + ", ".join(names),
                 f"{datetime.now():%Y-%m-%d %H:%M} | config {args.config}"
                 + (f" | reference {ref_path}" if ref is not None else " | no reference"))
    rep.p("Each indicator applies one method of the course. <b>cue</b> rows: a physical constraint "
          "a real camera respects; the column <i>real camera</i> says whether a real clip should "
          "give a low or a high value. <b>context</b> rows describe the clip (motion, lines, noise) "
          "and are needed to read the cues: every error grows with motion, and V1 says little when "
          "a homography explains the motion (camera turning on the spot, flat or far scene).")
    if ref is None:
        rep.p("No reference: the values cannot be called typical or unusual. Give "
              "<code>--reference</code> the features.csv of a run made with the same config, with "
              "real clips (and generated ones, if possible) of the same kind.", muted=True)
    else:
        rep.p("The reference must come from the SAME config (same resolution, frame rate, gaps); "
              "otherwise the comparison is meaningless.", muted=True)
    rep.table(table)
    if scores:
        rep.p("Physics classifier (gradient boosting on V1+I1+V3+I2) trained on the whole reference, "
              "probability that the clip is generated: "
              + ", ".join(f"<b>{n}</b>: {s:.2f}" for n, s in scores.items())
              + ". On ViF-Bench, this family reaches an AUC of about 0.65: one score is weak evidence.")
    for n in names:
        rep.h2(n)
        for path, alt in sheets[n]:
            rep.figure(path, alt)
    rep.write(out / "report.html")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print(table.drop(columns=["session"]).to_string(index=False))
    if scores:
        print("P(generated), physics classifier on the reference:", {k: round(v, 3) for k, v in scores.items()})
    print(f"[done] report: {out / 'report.html'}")


if __name__ == "__main__":
    main()
