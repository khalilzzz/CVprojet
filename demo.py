# Cours : Sessions 1-7 : démo côte à côte (vidéo réelle / vidéo générée), méthodes du cours appliquées image par image
"""Side-by-side demo: the real clip on the left, the generated clip on the right, the method
of each chapter drawn on every frame (points in red, epipolar lines, parallel lines and
vanishing points, flow residual, noise spectrum) and all values in a table below.

    python demo.py data/raw/videos/real/vid6.mp4 data/raw/videos/fake/vid6fake.mp4
    python demo.py REAL FAKE --no-window          # only write the MP4
    python demo.py REAL FAKE --speed 0.5          # fixed slow motion (the clip loops if needed)
    python demo.py REAL FAKE --short-side 1080    # keep 1080p (only if both clips are 1080p)
    python demo.py REAL FAKE --no-video           # only the verdict and the CSV (fast once cached)

Steps: the two clips are normalized like in the pipeline (24 fps, short side 720, cached in
data/processed/videos/), every frame is measured by our code (SIFT + RANSAC + 8 points,
three-view DLT, LSD + RANSAC vanishing points, Farnebäck flow, FFT; about 1 s per frame and
per clip on several processes, cached in results/demo/cache/), then the demo is written to
results/demo/<real>_vs_<fake>.mp4 and played in a window. The last chapter shows a comparative
verdict (src/demo/verdict.py: three-view error of the two clips at equal motion). The values
and the verdict are also written to results/demo/<real>_vs_<fake>.csv.

Keys in the window: space pause, left / right arrow (or a / d) one frame, 1-7 chapter,
f full screen, q or Esc quit. The video restarts at the end.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.normalize import PROCESSED, normalize_video  # noqa: E402
from src.demo import measure, render, verdict  # noqa: E402
from src.utils.config import load_config  # noqa: E402

OUT = ROOT / "results" / "demo"


def display_name(path: Path) -> str:
    """File id, plus the generator recorded in data/metadata.csv if any."""
    try:
        from src.data.ingest import read_metadata
        md = read_metadata()
        row = md[md["id"] == path.stem]
        gen = str(row["generator"].iloc[0]) if len(row) else ""
    except Exception:                                   # no metadata: the name alone
        gen = ""
    return f"{path.stem}  ({gen})" if gen and gen != "unknown" else path.stem


def timeline(n: int, fps_src: float, chapters: list[dict], fps_out: float, speed) -> list[tuple[int, str]]:
    """(source frame, chapter) of each output frame. speed "auto": the clip fills all chapters."""
    bounds = np.cumsum([0.0] + [float(c["seconds"]) for c in chapters])
    n_out = int(round(bounds[-1] * fps_out))
    out = []
    for k in range(n_out):
        tau = k / fps_out
        c = chapters[min(int(np.searchsorted(bounds, tau, side="right")) - 1, len(chapters) - 1)]["key"]
        src = min(n - 1, k * n // n_out) if speed == "auto" else int(tau * float(speed) * fps_src) % n
        out.append((src, c))
    return out


def write_mp4(path: Path, frames_iter, fps: float, n: int) -> None:
    import imageio_ffmpeg
    w = imageio_ffmpeg.write_frames(str(path), (render.W, render.H), fps=fps, codec="libx264",
                                    quality=None, output_params=["-crf", "18", "-preset", "medium"],
                                    macro_block_size=8, pix_fmt_in="rgb24")
    w.send(None)
    t0 = time.time()
    for k, img in enumerate(frames_iter):
        w.send(np.ascontiguousarray(img[..., ::-1]))
        if (k + 1) % 48 == 0 or k + 1 == n:
            print(f"\r[render] {k + 1}/{n} frames ({time.time() - t0:.0f} s)", end="", flush=True)
    w.close()
    print()


def play(path: Path, fps: float, chapter_starts: list[int]) -> None:
    """Play the MP4 in a window at its frame rate (keys: see the module docstring)."""
    name = "Réel vs IA - démo (espace: pause, flèches: image, 1-7: chapitre, f: plein écran, q: quitter)"
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1600, 900)
    idx, paused, full, seek = 0, False, False, True
    frame = None
    while True:
        t0 = time.time()
        if seek:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            seek = False
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:                                   # end: start again
                idx, seek, frame = 0, True, None
                continue
            idx += 1
        cv2.imshow(name, frame)
        wait = max(1, int(1000 / fps - (time.time() - t0) * 1000))
        key = cv2.waitKeyEx(wait)
        if key in (ord("q"), 27) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            paused = not paused
        elif key in (2424832, ord("a")):                 # left arrow
            idx, seek, paused, frame = max(0, idx - 2), True, True, None
        elif key in (2555904, ord("d")):                 # right arrow
            seek, paused, frame = True, True, None
        elif ord("1") <= key <= ord("9") and key - ord("1") < len(chapter_starts):
            idx, seek, frame = chapter_starts[key - ord("1")], True, None
        elif key == ord("f"):
            full = not full
            cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        idx = min(idx, n - 1)
    cap.release()
    cv2.destroyAllWindows()


def write_csv(path: Path, ids: list[str], series: list[dict], n: int, verd: dict) -> None:
    """Medians over the clip (one row per indicator) and the verdict rows, in one CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["section", "indicator", "real", "generated", "gap_pct", "ci_low", "ci_high",
                    "motion_overlap", "call", "real_id", "generated_id"])
        for key, label, *_ in measure.INDICATORS:
            vr, vf = series[0][key][1][n - 1], series[1][key][1][n - 1]
            ok = np.isfinite(vr) and np.isfinite(vf) and abs(vr) > 1e-9
            w.writerow(["median", key, vr, vf, (vf - vr) / abs(vr) * 100 if ok else np.nan,
                        "", "", "", "", ids[0], ids[1]])
        for r in verdict.rows(verd):
            w.writerow([f"verdict_{r['role']}", r["indicator"], "", r["ratio_at_equal_motion"], "",
                        r["ci_low"], r["ci_high"], r["motion_overlap"], r["call"], ids[0], ids[1]])
        w.writerow(["verdict", verd["code"], "", "", "", "", "", "", verd["text"], ids[0], ids[1]])


def main() -> None:
    ap = argparse.ArgumentParser(description="Side-by-side demo of the course methods on a real and a generated clip.")
    ap.add_argument("real", help="real video")
    ap.add_argument("fake", help="generated video")
    ap.add_argument("--config", default="configs/demo.yaml")
    ap.add_argument("--speed", help="playback speed of the clips (default: auto, fills the chapters)")
    ap.add_argument("--out", help="output MP4 (default results/demo/<real>_vs_<fake>.mp4)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--force", action="store_true", help="recompute the measurements")
    ap.add_argument("--no-window", action="store_true", help="only write the MP4")
    ap.add_argument("--no-video", action="store_true",
                    help="no MP4 and no window: print the verdict and write the CSV only")
    ap.add_argument("--short-side", type=int,
                    help="short side of the normalized clips (default: the pipeline config, 720). "
                         "Downscale only. Another value writes to data/processed/videos_<N>/ and "
                         "never touches the pipeline's clips")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):              # Windows consoles are not UTF-8 by default
        try:
            stream.reconfigure(errors="replace")
        except AttributeError:
            pass

    dcfg = load_config(args.config)
    cfg = load_config(dcfg["pipeline_config"])
    cfg["demo"] = dcfg["demo"]
    chapters = dcfg["chapters"]
    speed = args.speed or cfg["demo"].get("speed", "auto")
    workers = args.workers or int(cfg["demo"].get("workers", 6))

    paths = [Path(args.real), Path(args.fake)]
    ids = [p.stem for p in paths]
    if ids[0] == ids[1]:
        ids = [ids[0] + "_real", ids[1] + "_fake"]
    t0 = time.time()
    out_dir, tag = PROCESSED, ""
    if args.short_side and args.short_side != int(cfg["normalize"]["short_side"]):
        cfg["normalize"] = {**cfg["normalize"], "short_side": args.short_side}
        out_dir, tag = PROCESSED.parent / f"videos_{args.short_side}", f"_{args.short_side}p"
    norm = [normalize_video(p, cfg["normalize"], out_dir=out_dir, name=i) for p, i in zip(paths, ids)]
    colour, fps = zip(*[measure.read_clip(p, cfg["frames"]) for p in norm])
    n = min(len(c) for c in colour)
    colour = [list(c[:n]) for c in colour]
    print(f"[demo] {ids[0]} vs {ids[1]}: {n} frames each ({n / fps[0]:.1f} s at {fps[0]:.0f} fps), "
          f"{colour[0][0].shape[1]}x{colour[0][0].shape[0]} after the crop", flush=True)
    grays = [[cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in c] for c in colour]
    meas = measure.measure_clips([(i, p, g) for i, p, g in zip(ids, norm, grays)], cfg,
                                 workers=workers, force=args.force,
                                 log=lambda s: print(s, flush=True))
    meas = [meas[i] for i in ids]
    series = [measure.indicator_series(m) for m in meas]
    print(f"[demo] measurements ready ({time.time() - t0:.0f} s)", flush=True)
    verd = verdict.compare(series[0], series[1], dict(dcfg.get("verdict") or {}, seed=cfg.get("seed", 0)))
    out = Path(args.out) if args.out else OUT / f"{ids[0]}_vs_{ids[1]}{tag}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out.with_suffix(".csv"), ids, series, n, verd)

    fps_out = float(cfg["demo"].get("fps_out", 24))
    tl = timeline(n, fps[0], chapters, fps_out, speed)
    if not args.no_video:
        rend = render.Renderer((display_name(paths[0]), display_name(paths[1])), tuple(colour),
                               tuple(meas), tuple(series), fps[0], cfg["demo"], [c["key"] for c in chapters],
                               verdict=verd)
        write_mp4(out, (rend.frame(src, ch) for src, ch in tl), fps_out, len(tl))
        print(f"[demo] {len(tl) / fps_out:.1f} s of video -> {out}")

    print("\nmedian over the clip            real        generated")
    for key, label, *_ in measure.INDICATORS:
        vr, vf = series[0][key][1][n - 1], series[1][key][1][n - 1]
        print(f"  {label[:52]:52s} {vr:9.3f}   {vf:9.3f}")
    print(f"\nVERDICT: {verd['text']}")
    print(f"  {verdict.describe(verd['main'])}, motion overlap {verd['main']['overlap']:.0%}")
    for r in verd["support"]:
        print(f"  support: {verdict.describe(r)} ({r['call']})")
    print(f"[demo] values and verdict -> {out.with_suffix('.csv')}")
    if not args.no_window and not args.no_video:
        starts, k = [], 0
        for c in chapters:
            starts.append(int(round(k * fps_out)))
            k += float(c["seconds"])
        play(out, fps_out, starts)


if __name__ == "__main__":
    main()
