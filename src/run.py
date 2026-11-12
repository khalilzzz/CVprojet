# Cours : hors cours (automatisation) : commande unique du pipeline
"""One command for the whole pipeline:

    python -m src.run                     # every video listed in data/metadata.csv
    python -m src.run --videos vid4real vid4fake
    python -m src.run --force             # ignore the cache

Steps: (1) ingest new files of data/raw/ into data/metadata.csv, (2) normalize the videos
into data/processed/videos/, (3) extract the features (cached per video), (4) compare real
and generated, (5) write figures and a self-contained report to
results/runs/<date-time>/ and copy it to results/latest/ (results/latest_<dataset>/ for a
public dataset).

The cache key of a feature covers the normalized file, the config and the code version of
the feature, so only new or changed videos are computed.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")          # workers run in parallel: keep BLAS threads low

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.ingest import ingest
from src.data.normalize import normalize_video
from src.eval import compare, plots
from src.eval.report import Report
from src.features import video_epipolar as v1
from src.utils.config import REPO_ROOT, load_config

RESULTS = REPO_ROOT / "results"


def _key(path: Path, cfg: dict, version: str) -> str:
    st = path.stat()
    blob = json.dumps([path.name, st.st_size, int(st.st_mtime), cfg, version], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def v1_job(video_id: str, path: str, label: str, cfg: dict, out_dir: str) -> str:
    """Worker: V1 on one video. Writes pairs.csv, features.json and sheet.png in out_dir."""
    import cv2
    cv2.setNumThreads(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, trip, info = v1.analyse_video(Path(path), video_id, cfg, cfg["seed"])
    plots.video_sheet(df, info, out / "sheet.png", label)
    (out / "features.json").write_text(json.dumps(v1.video_features(df, trip)))
    trip.to_csv(out / "triplets.csv", index=False)
    df.to_csv(out / "pairs.csv", index=False)         # written last: marks the job as done
    return video_id


def run_v1(videos: pd.DataFrame, processed: dict, cfg: dict, force: bool, workers: int) -> dict:
    """Returns {video id: cache folder}. Computes only what is missing."""
    sub = dict(frames=cfg["frames"], v1=cfg["v1"], seed=cfg["seed"])
    dirs, jobs = {}, []
    for _, r in videos.iterrows():
        path = processed[r["id"]]
        d = RESULTS / "cache" / "v1" / f"{r['id']}__{_key(path, sub, v1.CODE_VERSION)}"
        dirs[r["id"]] = d
        if force or not (d / "pairs.csv").exists():
            jobs.append((r["id"], str(path), r["label"], sub, str(d)))
    print(f"[v1] {len(videos) - len(jobs)} videos from cache, {len(jobs)} to compute")
    if jobs:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
            futures = {pool.submit(v1_job, *j): j[0] for j in jobs}
            for k, f in enumerate(as_completed(futures), 1):
                f.result()
                print(f"[v1] {k}/{len(jobs)} done: {futures[f]} ({time.time() - t0:.0f} s)")
    return dirs


def build_report(run_dir: Path, meta: pd.DataFrame, videos: pd.DataFrame, dirs: dict,
                 cfg: dict) -> Path:
    pairs = pd.concat([pd.read_csv(dirs[i] / "pairs.csv") for i in videos["id"]], ignore_index=True)
    pairs.to_csv(run_dir / "v1_pairs.csv", index=False)
    feats = pd.DataFrame([json.loads((dirs[i] / "features.json").read_text()) for i in videos["id"]])
    feats.merge(meta[["id", "label", "scene_id", "generator"]], left_on="video", right_on="id") \
         .drop(columns="id").to_csv(run_dir / "features.csv", index=False)

    valid = compare.attach_meta(pairs, meta)
    scenes = compare.paired_scenes(valid)
    paired = valid[valid["scene_id"].isin(scenes)]
    generated, copied = compare.split_windows(paired)
    binned = compare.bin_by_motion(generated)
    matched = compare.matched_summary(binned)
    many = len(scenes) > 12            # benchmark-sized run: aggregate per generator
    if many:
        # a motion range shared by ALL the generated versions of a scene is rare:
        # compare each generator with the real clip on its own
        gen = generated["video"].map(meta.set_index("id")["generator"])
        parts = [compare.matched_summary(compare.bin_by_motion(
                     generated[(generated["label"] == "real") | (gen == g)]))
                 for g in sorted(set(gen[generated["label"] == "fake"]))]
        parts = [x for x in parts if not x.empty]
        matched = pd.concat(parts, ignore_index=True) if parts else matched
    binned_copy = compare.bin_by_motion(copied, n_bins=3)
    matched_copy = compare.matched_summary(binned_copy)
    table = compare.video_table(valid, meta)
    for name, df in [("v1_matched_motion", matched), ("v1_matched_motion_copied_part", matched_copy),
                     ("v1_video_table", table)]:
        df.to_csv(run_dir / f"{name}.csv", index=False)

    figs = run_dir / "figures"
    figs.mkdir(exist_ok=True)
    rep = Report("AI-video detection: run report",
                 f"{datetime.now():%Y-%m-%d %H:%M} | {len(videos)} videos "
                 f"({(videos['label'] == 'real').sum()} real, {(videos['label'] == 'fake').sum()} generated) "
                 f"| {len(scenes)} scenes with both a real and a generated clip | seed {cfg['seed']}")
    rep.h2("1. Real against generated, at equal image motion")
    rep.p("Every measure degrades when the camera moves more, and generated clips often move "
          "less. So videos of one scene are compared inside shared bins of image motion (median "
          "displacement of the matches of a frame pair, in px). "
          "When a generated clip was given the first seconds of the real clip, only what comes "
          "after those seconds is used here.")
    if many:
        bench = compare.benchmark_table(matched, None, meta)
        if not matched.empty:
            rep.figure(plots.ratio_by_generator(
                matched, meta, "ratio_to_real", figs / "v1_sampson_ratio_by_generator.png",
                "Sampson distance at equal image motion: generated / real",
                "generated / real (above 1 = the generated clip fits one F less well)"),
                "Sampson ratio generated / real, per generator")
            rep.table(bench)
    elif not matched.empty:
        rep.figure(plots.scene_summary(matched, figs / "v1_scene_summary.png"),
                   "Sampson distance at equal motion, real against generated, per scene")
        show = matched.rename(columns={"scene_id": "scene", "n_bins": "shared bins",
                                       "n_pairs": "frame pairs", "motion_range": "motion px",
                                       "sampson_matched": "Sampson px", "inlier_ratio_F": "F inliers",
                                       "h_cover": "H cover", "ratio_to_real": "Sampson / real"})
        rep.table(show)
        rep.figure(plots.equal_motion(generated, binned, figs / "v1_equal_motion.png"),
                   "Sampson distance against image motion, per scene")
    else:
        rep.p("No scene has a motion range shared by its real and generated clips.", muted=True)
    missing = [] if many else [s for s in scenes
                               if matched.empty or s not in set(matched["scene_id"])]
    if missing:
        names = (", ".join(f"<code>{m}</code>" for m in missing) if len(missing) <= 12
                 else f"{len(missing)} of {len(scenes)} scenes")
        rep.p("No shared motion range (clips move too differently), so no fair comparison for: "
              + names + ".", muted=True)

    if not matched_copy.empty:
        rep.h2("2. Control: the part copied from the real clip")
        rep.p("These generated clips start with a re-rendered copy of the real clip. In that part "
              "the 3D scene is the real one. A gap between real and generated here cannot come from "
              "3D inconsistency: it comes from how the generator renders the frames.")
        rep.table(matched_copy.rename(columns={"scene_id": "scene", "n_bins": "shared bins",
                                               "n_pairs": "frame pairs", "motion_range": "motion px",
                                               "sampson_matched": "Sampson px", "inlier_ratio_F": "F inliers",
                                               "h_cover": "H cover", "ratio_to_real": "Sampson / real"}))

    # three-view test (Session 7)
    trip = pd.concat([pd.read_csv(dirs[i] / "triplets.csv") for i in videos["id"]], ignore_index=True)
    trip.to_csv(run_dir / "v1_triplets.csv", index=False)
    tvalid = compare.attach_meta(trip, meta)
    tpaired = tvalid[tvalid["scene_id"].isin(scenes)]
    tgen, tcopy = compare.split_windows(tpaired)
    tv_table, tv_copy = compare.three_view_table(tgen), compare.three_view_table(tcopy)
    tv_table.to_csv(run_dir / "v1_three_view.csv", index=False)
    tv_copy.to_csv(run_dir / "v1_three_view_copied_part.csv", index=False)
    rep.h2("3. Three-view test: does the scene stay rigid over time?")
    rep.p("Points seen in three frames (t, t+k, t+2k) are triangulated from the first two and "
          "reprojected in the third (SfM lecture). The level of this error depends on how the "
          "frames are rendered, like the Sampson distance. Its <b>growth</b> with the gap k does "
          "not: on synthetic clips, blur leaves the growth unchanged and a slowly deforming scene "
          "doubles it. Growth = error at the largest gap / error at the smallest gap.")
    if many and not tv_table.empty:
        rep.figure(plots.ratio_by_generator(
            tv_table, meta, "growth_to_real", figs / "v1_growth_ratio_by_generator.png",
            "Growth of the three-view error: generated / real",
            "generated / real (above 1 = the generated scene is less rigid over time)"),
            "Three-view growth ratio generated / real, per generator")
        bench = compare.benchmark_table(matched, tv_table, meta)
        bench.to_csv(run_dir / "benchmark_by_generator.csv", index=False)
        rep.table(bench)
    elif not tv_table.empty:
        rep.figure(plots.scene_summary(
            tv_table, figs / "v1_three_view_growth.png", col="growth",
            xlabel="growth of the three-view error with the frame gap, lower = more rigid",
            title="Per scene: growth of the three-view error, real against generated", fmt="{:.2f}"),
            "Growth of the three-view error, real against generated, per scene")
        rep.table(tv_table.rename(columns={"scene_id": "scene", "motion_growth": "motion growth",
                                           "growth_to_real": "growth / real"}))
        rep.figure(plots.three_view_grid(tgen, figs / "v1_three_view.png"),
                   "Three-view reprojection error against the frame gap, per scene")
    if not tv_copy.empty:
        rep.h3("Control: the part copied from the real clip")
        rep.table(tv_copy.rename(columns={"scene_id": "scene", "motion_growth": "motion growth",
                                          "growth_to_real": "growth / real"}))

    rep.h2("4. Same frame gap (not a fair comparison, shown for reference)")
    if scenes and not many:
        rep.figure(plots.scene_delta_grid(paired, figs / "v1_by_gap.png"),
                   "V1 measures against the frame gap, one row per scene")
    rep.h2("5. All videos")
    rep.table(table)
    sheets = []
    for i in (videos["id"] if len(videos) <= 40 else []):      # keep the report small
        dst = figs / f"v1_sheet_{i}.png"
        shutil.copyfile(dirs[i] / "sheet.png", dst)
        sheets.append(f"<h3>{i}</h3>" + rep.img_html(dst, f"V1 sheet of {i}"))
    rep.details(f"Per-video sheets ({len(sheets)})", sheets)
    rep.h2("How to read the measures")
    rep.ul(["<b>Sampson distance</b>: distance (px) of a match to the epipolar constraint of the "
            "estimated F. Lower = the frame pair fits one rigid 3D scene better.",
            "<b>F inliers</b>: share of matches within 1.5 px of that constraint.",
            "<b>H cover</b>: share of the F inliers that a homography also explains. Near 1 = pure "
            "rotation or flat / far scene: F says little.",
            "<b>Parallax</b>: median error (px) of the F inliers under that homography.",
            "<b>Frame gap</b>: number of frames between the two frames of a pair, at 24 fps.",
            "<b>Three-view error</b>: distance (px) between a point seen in a third frame and the "
            "reprojection of its 3D position, triangulated from two other frames."])
    rep.h2("Configuration")
    rep.parts.append(f"<pre style='font-size:12px;overflow-x:auto'>{json.dumps(cfg, indent=1)}</pre>")
    return rep.write(run_dir / "report.html")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--videos", nargs="*", help="ids to process (default: all)")
    ap.add_argument("--dataset", default="own", help="rows of this dataset (own, vifbench)")
    ap.add_argument("--generators", nargs="*", help="keep the real clips and these generators")
    ap.add_argument("--max-scenes", type=int, help="keep the first N scenes (sorted by id)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()

    meta = ingest()
    videos = meta[(meta["media_type"] == "video") & (meta["path"] != "")]
    videos = videos[videos["dataset"] == args.dataset]
    if args.videos:
        videos = videos[videos["id"].isin(args.videos)]
    if args.generators:
        videos = videos[(videos["label"] == "real") | videos["generator"].isin(args.generators)]
    if args.max_scenes:
        keep = sorted(videos["scene_id"].unique())[: args.max_scenes]
        videos = videos[videos["scene_id"].isin(keep)]
    # data/raw is not in git: on another machine some listed files may be missing
    present = videos["path"].map(lambda p: (REPO_ROOT / p).exists())
    if not present.all():
        print(f"[warning] {int((~present).sum())} videos listed in data/metadata.csv are missing "
              f"from data/raw/ on this machine and are skipped (see gpuecole.md)")
        videos = videos[present]
    if videos.empty:
        raise SystemExit("no video to process")

    processed = {}
    for k, (_, r) in enumerate(videos.iterrows(), 1):
        processed[r["id"]] = normalize_video(REPO_ROOT / r["path"], cfg["normalize"], force=False,
                                             name=r["id"])
        if k % 50 == 0:
            print(f"[normalize] {k}/{len(videos)}")
    print(f"[normalize] {len(processed)} videos in data/processed/videos/")

    dirs = run_v1(videos, processed, cfg, args.force, args.workers or cfg["run"]["workers"])

    run_dir = RESULTS / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    report = build_report(run_dir, meta, videos, dirs, cfg)
    latest = RESULTS / ("latest" if args.dataset == "own" else f"latest_{args.dataset}")
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(run_dir, latest)
    print(f"[done] {time.time() - t0:.0f} s | report: {(latest / 'report.html').relative_to(REPO_ROOT)}"
          f" (copy of {run_dir.relative_to(REPO_ROOT)})")


if __name__ == "__main__":
    main()
