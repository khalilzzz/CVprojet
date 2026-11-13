# Cours : hors cours (automatisation) : commande unique du pipeline
"""One command for the whole pipeline:

    python -m src.run                     # every video listed in data/metadata.csv
    python -m src.run --videos vid4real vid4fake
    python -m src.run --force             # ignore the cache

    python -m src.run --features v1,i1,v3,clip   # add the CLIP baseline (GPU if available)

Steps: (1) ingest new files of data/raw/ into data/metadata.csv, (2) normalize the videos
into data/processed/videos/, (3) extract the feature families V1, I1, V3 (and CLIP), cached
per video, (4) compare real and generated, (5) classify real/generated per family,
(6) write figures and a self-contained report to
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingest import ingest
from src.data.normalize import normalize_video
from src.eval import compare, plots
from src.eval.report import Report
from src.features import image_frequency as i1
from src.features import video_epipolar as v1
from src.features import video_temporal as v3
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


def simple_job(family: str, video_id: str, path: str, label: str, cfg: dict, out_dir: str) -> str:
    """Worker: one per-video feature family (i1, v3). Writes features.json in out_dir."""
    import cv2
    cv2.setNumThreads(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = FAMILY_MODULES[family].video_features(Path(path), cfg)
    (out / "features.json").write_text(json.dumps(dict(video=video_id) | feats))
    return video_id


FAMILY_MODULES = {"i1": i1, "v3": v3}
DONE_FILE = {"v1": "pairs.csv", "i1": "features.json", "v3": "features.json"}


def _run_pool(fn, jobs: list, tag: str, workers: int) -> None:
    """Run fn(*job) for every job in worker processes. Memory safety: at most `workers` jobs
    in flight, a new job starts only when enough RAM is free, and each worker process is
    replaced after a few videos (no slow growth). A failing video is reported and skipped."""
    t0 = time.time()
    n_workers = max(1, min(workers, len(jobs)))
    pool = ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=MAX_TASKS_PER_CHILD)
    try:
        pending, todo, done = {}, list(jobs), 0
        while todo or pending:
            while todo and len(pending) < n_workers and _ram_ok(len(pending)):
                j = todo.pop(0)
                pending[pool.submit(fn, *j)] = j[0] if fn is v1_job else j[1]
            if not pending:                     # nothing running and RAM still low: wait
                time.sleep(5)
                continue
            finished, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for f in finished:
                vid = pending.pop(f)
                done += 1
                try:
                    f.result()
                    status = "done"
                except Exception as exc:            # one bad video must not stop the run
                    status = f"FAILED ({type(exc).__name__}: {exc})"
                if status != "done" or done % 25 == 0 or done == len(jobs):
                    print(f"[{tag}] {done}/{len(jobs)} {status}: {vid} "
                          f"({time.time() - t0:.0f} s, {_free_gb():.1f} GB RAM free)", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def run_family(family: str, videos: pd.DataFrame, processed: dict, cfg: dict, force: bool,
               workers: int, report_only: bool = False) -> dict:
    """Compute one feature family for every video (only what is missing from the cache).
    Returns {video id: cache folder}."""
    if family == "v1":
        sub, version = dict(frames=cfg["frames"], v1=cfg["v1"], seed=cfg["seed"]), v1.CODE_VERSION
    else:
        sub, version = dict(frames=cfg["frames"], **{family: cfg[family]}), FAMILY_MODULES[family].CODE_VERSION
    dirs, jobs = {}, []
    for _, r in videos.iterrows():
        path = processed[r["id"]]
        d = RESULTS / "cache" / family / f"{r['id']}__{_key(path, sub, version)}"
        dirs[r["id"]] = d
        if force or not (d / DONE_FILE[family]).exists():
            if family == "v1":
                jobs.append((r["id"], str(path), r["label"], sub, str(d)))
            else:
                jobs.append((family, r["id"], str(path), r["label"], sub, str(d)))
    print(f"[{family}] {len(videos) - len(jobs)} videos from cache, {len(jobs)} to compute"
          + (" (skipped: --report-only)" if report_only and jobs else ""), flush=True)
    if jobs and not report_only:
        _run_pool(v1_job if family == "v1" else simple_job, jobs, family, workers)
    return dirs


def run_clip(videos: pd.DataFrame, processed: dict, cfg: dict, force: bool,
             report_only: bool = False) -> dict:
    """CLIP embeddings, computed in this process (GPU if available). Returns {id: .npy path}."""
    from src.models import baseline_clip
    sub = dict(frames=cfg["frames"], clip=cfg["clip"])
    paths, todo = {}, []
    for _, r in videos.iterrows():
        path = processed[r["id"]]
        f = RESULTS / "cache" / "clip" / f"{r['id']}__{_key(path, sub, baseline_clip.CODE_VERSION)}.npy"
        paths[r["id"]] = f
        if force or not f.exists():
            todo.append((r["id"], path, f))
    print(f"[clip] {len(videos) - len(todo)} videos from cache, {len(todo)} to compute"
          + (" (skipped: --report-only)" if report_only and todo else ""), flush=True)
    if todo and not report_only:
        baseline_clip.embed_videos(todo, cfg)
    return paths


def collect_features(meta: pd.DataFrame, videos: pd.DataFrame, dirs: dict) -> pd.DataFrame:
    """One row per video: every computed family, plus label, scene and generator."""
    table = videos[["id", "label", "scene_id", "generator"]].rename(columns={"id": "video"})
    for family, d in dirs.items():
        rows = []
        for vid in videos["id"]:
            if family == "clip":
                f = d.get(vid)
                if f is not None and f.exists():
                    v = np.load(f)
                    rows.append(dict(video=vid, **{f"clip_{k:03d}": float(x) for k, x in enumerate(v)}))
            else:
                f = d[vid] / "features.json"
                if f.exists() and (d[vid] / DONE_FILE[family]).exists():
                    rows.append(json.loads(f.read_text()) | dict(video=vid))
        if rows:
            table = table.merge(pd.DataFrame(rows), on="video", how="left")
    return table


MAX_TASKS_PER_CHILD = 25        # videos processed by one worker process before it is replaced
MIN_FREE_GB = 2.0               # do not start a new video below this amount of free RAM


def _free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except ImportError:                             # psutil missing: no memory check
        return float("inf")


def _ram_ok(running: int) -> bool:
    """True if a new job may start. The first job always starts, so the run cannot stall."""
    return running == 0 or _free_gb() >= MIN_FREE_GB


def build_report_v1(run_dir: Path, meta: pd.DataFrame, videos: pd.DataFrame, dirs: dict,
                    cfg: dict) -> Report:
    """Sections 1 to 5 of the report (V1: geometry between frames)."""
    ok = [i for i in videos["id"] if (dirs[i] / "pairs.csv").exists()]
    if len(ok) < len(videos):
        print(f"[warning] {len(videos) - len(ok)} videos have no result (failed or not computed) "
              f"and are left out of the report")
    videos = videos[videos["id"].isin(ok)]
    pairs = pd.concat([pd.read_csv(dirs[i] / "pairs.csv") for i in videos["id"]], ignore_index=True)
    pairs.to_csv(run_dir / "v1_pairs.csv", index=False)

    valid = compare.attach_meta(pairs, meta)
    scenes = compare.paired_scenes(valid)
    paired = valid[valid["scene_id"].isin(scenes)]
    generated, copied = compare.split_windows(paired)
    binned = compare.bin_by_motion(generated)
    matched = compare.matched_summary(binned)
    # One panel per scene and one line per video is readable only for a few scenes with a few
    # videos each (our own clips). Otherwise aggregate: one row per generator.
    per_scene = valid[valid["scene_id"].isin(scenes)].groupby("scene_id")["video"].nunique()
    many = len(scenes) > 6 or (len(per_scene) and per_scene.max() > 4)
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
    return rep



def classification_section(rep: Report, feats: pd.DataFrame, run_dir: Path, cfg: dict) -> None:
    """Section 6: real/generated classifiers per feature family (src/models/classify.py)."""
    from src.models import classify
    n_scenes = int(feats.groupby("scene_id")["label"].nunique().eq(2).sum())
    rep.h2("6. Classification: which features tell real from generated?")
    if n_scenes < 10:
        rep.p(f"Only {n_scenes} scenes with both a real and a generated clip: too few to train "
              "and test a classifier (at least 10 needed).", muted=True)
        return
    ccfg = cfg.get("classify", {})
    res = classify.run(feats, seed=cfg["seed"], n_splits=ccfg.get("n_splits", 5),
                       unseen=ccfg.get("unseen_generator", True))
    out = run_dir / "classification"
    out.mkdir(exist_ok=True)
    for k in ("summary", "per_generator", "unseen_generator", "importance"):
        if k in res:
            res[k].to_csv(out / f"classify_{k}.csv", index=False)
    res["scores"].to_csv(out / "classify_oof_scores.csv", index=False)
    rep.p("Cross-validation in 5 folds grouped by scene: a real clip and its generated versions "
          "are always on the same side, so the classifier cannot recognise a scene. Physics "
          "families use gradient boosting; families with CLIP use a logistic regression (linear "
          "probe). AUC: 0.5 = chance, 1 = perfect; interval = 95 % bootstrap over scenes. "
          "Balanced accuracy and F1 use the threshold 0.5.")
    rep.table(res["summary"].rename(columns={"n_features": "features", "AUC_low": "AUC 2.5 %",
                                             "AUC_high": "AUC 97.5 %",
                                             "balanced_accuracy": "balanced accuracy",
                                             "F1_generated": "F1 (generated)"}))
    for path, alt in plots.classification_figures(res, out):
        rep.figure(path, alt)
    if "unseen_generator" in res:
        rep.p("Heat map 2: the generator is removed from training and only tested. A family whose "
              "AUC holds there captures something common to generators, not one model's signature.",
              muted=True)


def build_report(run_dir: Path, meta: pd.DataFrame, videos: pd.DataFrame, fam_dirs: dict,
                 cfg: dict, classify: bool = True) -> Path:
    print("[report] collecting features and writing the report (a few minutes on a big run)", flush=True)
    feats = collect_features(meta, videos, fam_dirs)
    feats.to_csv(run_dir / "features.csv", index=False)
    if "v1" in fam_dirs:
        rep = build_report_v1(run_dir, meta, videos, fam_dirs["v1"], cfg)
    else:
        rep = Report("AI-video detection: run report",
                     f"{datetime.now():%Y-%m-%d %H:%M} | {len(videos)} videos | seed {cfg['seed']}")
    if classify:
        classification_section(rep, feats, run_dir, cfg)
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
    ap.add_argument("--max-scenes", type=int,
                    help="keep the first N scenes that have a real and a generated clip (sorted by id)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--report-only", action="store_true",
                    help="compute nothing: build the report from the videos already in the cache")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--features", default="v1,i1,v3",
                    help="feature families, comma separated: v1, i1, v3, clip (default: v1,i1,v3)")
    ap.add_argument("--no-classify", action="store_true", help="skip the classification section")
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
        paired = videos.groupby("scene_id")["label"].nunique().eq(2)   # scenes with real AND generated
        keep = sorted(paired[paired].index)[: args.max_scenes]
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

    families = [f.strip() for f in args.features.split(",") if f.strip()]
    workers = args.workers or cfg["run"]["workers"]
    fam_dirs = {}
    for fam in families:
        if fam == "clip":
            fam_dirs["clip"] = run_clip(videos, processed, cfg, args.force, args.report_only)
        else:
            fam_dirs[fam] = run_family(fam, videos, processed, cfg, args.force, workers,
                                       args.report_only)

    run_dir = RESULTS / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    report = build_report(run_dir, meta, videos, fam_dirs, cfg, classify=not args.no_classify)
    latest = RESULTS / ("latest" if args.dataset == "own" else f"latest_{args.dataset}")
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(run_dir, latest)
    print(f"[done] {time.time() - t0:.0f} s | report: {(latest / 'report.html').relative_to(REPO_ROOT)}"
          f" (copy of {run_dir.relative_to(REPO_ROOT)})")


if __name__ == "__main__":
    main()
