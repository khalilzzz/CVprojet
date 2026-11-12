# Cours : Sessions 1-2 (rééchantillonnage) : protocole de normalisation des vidéos (CLAUDE.md section 6)
"""Normalize every video the same way, real and generated: H.264 with one CRF, one frame
rate, short side at most ``short_side`` pixels (downscale only), no audio.

The ffmpeg binary comes from the pip package ``imageio-ffmpeg`` (no system install).
Inputs in data/raw/ are only read. Outputs go to data/processed/videos/<id>.mp4, with a
``<id>.json`` file that records the source file and the parameters, so a video is
re-encoded only when its source or the parameters change.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.utils.config import REPO_ROOT

PROCESSED = REPO_ROOT / "data" / "processed" / "videos"


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _stamp(src: Path, cfg: dict) -> dict:
    st = src.stat()
    return dict(source=src.name, size=st.st_size, mtime=int(st.st_mtime), params=cfg)


def normalize_video(src: Path, cfg: dict, out_dir: Path = PROCESSED, force: bool = False,
                    name: str | None = None) -> Path:
    """Re-encode one video. Returns the path of the normalized file, named after ``name``
    (the metadata id; default: the stem of the source file)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or src.stem
    dst, side = out_dir / f"{name}.mp4", out_dir / f"{name}.json"
    stamp = _stamp(src, cfg)
    if not force and dst.exists() and side.exists() and json.loads(side.read_text()) == stamp:
        return dst
    s = int(cfg["short_side"])
    # scale the short side down to `s` (never up); -2 keeps the aspect ratio with an even size
    scale = (f"scale=w='if(gt(iw,ih),-2,min(iw,{s}))':h='if(gt(iw,ih),min(ih,{s}),-2)'"
             ":flags=area")
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src), "-an",
           "-vf", f"fps={cfg['fps']},{scale}",
           "-c:v", "libx264", "-crf", str(cfg["crf"]), "-preset", str(cfg["preset"]),
           "-pix_fmt", "yuv420p", str(dst)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed on {src.name}: {done.stderr.strip()[-400:]}")
    side.write_text(json.dumps(stamp))
    return dst
