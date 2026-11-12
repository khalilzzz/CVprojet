# Cours : Session 1 (Light, color and images) : lecture des frames, conversion en niveaux de gris
"""Frame extraction with OpenCV (ffmpeg is not required).

Normalization applied identically to real and generated videos (CLAUDE.md section 6):
- skip the first ``skip_s`` seconds (the first frame of an image-to-video clip is real),
- optional temporal resampling to ``target_fps`` (nearest frame in time, using the
  container timestamps, so variable frame rate phone videos are handled),
- keep at most ``duration_s`` seconds after the skip,
- crop a band at the bottom (visible watermarks, e.g. "Dreamina AI"),
- optional resize so that the longest side is <= ``max_side`` and the short side of the
  original frame is <= ``short_side`` (downscale only, never upscale).
Raw files in data/raw/ are only read, never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class VideoFrames:
    frames: list[np.ndarray]    # grayscale uint8 frames, after crop / resize
    fps: float                  # frame rate of the returned frames
    first_index: int            # index in the original video of frames[0]
    shape: tuple[int, int]      # (h, w) of the processed frames
    original_shape: tuple[int, int]
    indices: list[int] = field(default_factory=list)   # original index of each frame
    native_fps: float = 0.0
    times: list[float] = field(default_factory=list)   # timestamp (s) of each frame in the file


def _process(frame, crop_top, crop_bottom, max_side, short_side=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    gray = gray[int(round(crop_top * h)): h - int(round(crop_bottom * h))]
    s = 1.0
    if max_side is not None:
        s = min(s, max_side / max(gray.shape))
    if short_side is not None:
        # short side measured before the crop, so that cropped real and generated
        # frames end with the same pixel scale (1080p and 720p sources -> 720p)
        s = min(s, short_side / min(frame.shape[:2]))
    if s < 1.0:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return gray


def read_frames(path: str, skip_s: float = 0.5, crop_bottom: float = 0.0,
                crop_top: float = 0.0, max_side: int | None = None, short_side: int | None = None,
                max_frames: int | None = None, target_fps: float | None = None,
                duration_s: float | None = None) -> VideoFrames:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    out_fps = target_fps or native_fps
    t_end = skip_s + duration_s if duration_s is not None else np.inf
    # read all frames with their timestamps (seconds) in [skip_s - 1 frame, t_end + 1 frame]
    raw, ts, idx, original_shape, i = [], [], [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > t_end + 1.0 / native_fps:
            break
        if t >= skip_s - 1.0 / native_fps:
            original_shape = frame.shape[:2]
            raw.append(_process(frame, crop_top, crop_bottom, max_side, short_side))
            ts.append(t)
            idx.append(i)
        i += 1
    cap.release()
    if not raw:
        raise ValueError(f"no frames read from {path}")
    ts = np.asarray(ts)
    # Some containers give no usable timestamps (all zero, or not increasing):
    # fall back to index / fps so that the resampling below stays correct.
    if len(ts) > 1 and not np.all(np.diff(ts) > 0):
        ts = np.asarray(idx, dtype=float) / native_fps
    if target_fps is None:
        keep = [k for k in range(len(raw)) if ts[k] >= skip_s - 1e-6 and ts[k] <= t_end + 1e-6]
    else:
        # target times skip_s + k / target_fps, nearest available frame for each
        targets = np.arange(skip_s, min(t_end, ts[-1]) + 1e-9, 1.0 / target_fps)
        keep = [int(np.argmin(np.abs(ts - tt))) for tt in targets]
    if max_frames is not None:
        keep = keep[:max_frames]
    frames = [raw[k] for k in keep]
    indices = [idx[k] for k in keep]
    return VideoFrames(frames, out_fps, indices[0], frames[0].shape[:2], original_shape,
                       indices, native_fps, [float(ts[k]) for k in keep])
