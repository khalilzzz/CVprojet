# Cours : Session 1 (images, lecture vidéo) : extraction de l'image de départ pour la génération image-to-video
"""Extract the first frame of real clips, to give it to an image-to-video generator.

    python extract.py data/raw/videos/real/vid5real.mp4
    python extract.py data/raw/videos/real/              # every video of the folder
    python extract.py clip.mp4 --time 0.5                # frame at 0.5 s (fade-in, blur at 0 s)

The frame is saved at full resolution, without crop, as a lossless PNG, in
data/processed/first_frames/<video>_t<time>.png (data/raw/ is never modified). The
generator resizes it itself; our pipeline later crops and downscales real and generated
clips the same way.

A warning is printed if the frame is very dark (fade-in) or blurry: then try --time 0.3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
OUT = Path(__file__).resolve().parent / "data" / "processed" / "first_frames"


def frame_at(path: Path, time_s: float):
    """First frame whose timestamp is >= time_s. Returns (BGR image, timestamp s, index)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    i, img, t = 0, None, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0 and i > 0:                      # container without timestamps
            t = i / fps
        img = frame
        if t >= time_s - 0.5 / fps:
            break
        i += 1
    cap.release()
    if img is None:
        raise ValueError(f"no frame read from {path}")
    return img, t, i


def check(img) -> list[str]:
    """Warnings: dark frame (fade-in) or blurry frame (motion blur, focus)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, None, fx=720 / min(gray.shape), fy=720 / min(gray.shape),
                       interpolation=cv2.INTER_AREA) if min(gray.shape) > 720 else gray
    sharp = cv2.Laplacian(small, cv2.CV_64F).var()
    out = []
    if gray.mean() < 25:
        out.append(f"very dark (mean grey {gray.mean():.0f}): fade-in?")
    if sharp < 30:
        out.append(f"blurry (Laplacian variance {sharp:.0f} at 720p)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract the first frame of real clips (image-to-video input).")
    ap.add_argument("inputs", nargs="+", help="video files or folders")
    ap.add_argument("--time", type=float, default=0.0, help="time of the frame, seconds (default 0)")
    ap.add_argument("--out", default=str(OUT), help=f"output folder (default {OUT})")
    ap.add_argument("--jpg", action="store_true", help="JPEG quality 95 instead of PNG (for upload limits)")
    args = ap.parse_args()

    videos = []
    for p in map(Path, args.inputs):
        videos += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT) if p.is_dir() else [p]
    if not videos:
        raise SystemExit("no video found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in videos:
        try:
            img, t, i = frame_at(v, args.time)
        except (IOError, ValueError) as exc:
            print(f"[skip] {v.name}: {exc}")
            continue
        ext = ".jpg" if args.jpg else ".png"
        dst = out / f"{v.stem}_t{args.time:.2f}{ext}"
        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if args.jpg else [cv2.IMWRITE_PNG_COMPRESSION, 3]
        if not cv2.imwrite(str(dst), img, params):
            print(f"[skip] {v.name}: cannot write {dst}")
            continue
        h, w = img.shape[:2]
        warn = check(img)
        print(f"{v.name}: frame {i} at {t:.2f} s, {w}x{h} -> {dst}"
              + ("".join(f"\n    warning: {m}" for m in warn)))


if __name__ == "__main__":
    main()
