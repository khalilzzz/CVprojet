# Cours : Session 6 (Dense Matching) et Session 3 (Warping) : V3, cohérence temporelle par flot optique
"""V3: temporal coherence. In a camera video, frame t+1 warped back with the optical flow
matches frame t up to noise, except at occlusions. Generated videos often "boil": fine
texture changes from frame to frame even where nothing moves.

For frame pairs (t, t+1):
1. dense flow forward and backward (Farnebäck, OpenCV),
2. forward-backward check: a pixel is kept if f(x) + b(x + f(x)) is small
   (|.| < 0.5 + 0.05 |f|), which removes occlusions and flow failures,
3. warp frame t+1 onto frame t and measure the residual on the kept pixels:
   on the image, on its high-pass part (fine texture), and on pixels that barely move.

Features (medians over pairs): ``v3_res_med``, ``v3_res_q90``, ``v3_hf_res_med``,
``v3_hf_res_norm`` (high-pass residual / noise level of the frame), ``v3_static_res_med``
(pixels moving less than 0.5 px), ``v3_fb_bad_frac`` (share of pixels failing the check),
``v3_flow_med`` (median motion, px, as a covariate).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.frames import read_frames

CODE_VERSION = "v3-2026-11-13a"
FB = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def _flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(a, b, None, **FB)


def _highpass(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    f = img.astype(np.float32)
    return f - cv2.GaussianBlur(f, (0, 0), sigma)


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Warping residual statistics of the grayscale pair (a, b)."""
    h, w = a.shape
    fwd, bwd = _flow(a, b), _flow(b, a)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + fwd[..., 0], gy + fwd[..., 1]
    inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    b_at = cv2.remap(bwd, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mag = np.hypot(fwd[..., 0], fwd[..., 1])
    ok = inside & (np.hypot(*(fwd + b_at).transpose(2, 0, 1)) < 0.5 + 0.05 * mag)
    if ok.sum() < 100:
        return {}
    warped = cv2.remap(b.astype(np.float32), mx, my, cv2.INTER_LINEAR)
    res = np.abs(a.astype(np.float32) - warped)[ok]
    ha = _highpass(a)
    hres = np.abs(ha - cv2.remap(_highpass(b), mx, my, cv2.INTER_LINEAR))[ok]
    noise = 1.4826 * np.median(np.abs(ha)) + 1e-6       # robust noise level of the frame
    static = ok & (mag < 0.5)
    return dict(v3_res_med=float(np.median(res)), v3_res_q90=float(np.quantile(res, 0.9)),
                v3_hf_res_med=float(np.median(hres)), v3_hf_res_norm=float(np.median(hres) / noise),
                v3_static_res_med=(float(np.median(np.abs(a.astype(np.float32) - warped)[static]))
                                   if static.sum() >= 100 else np.nan),
                v3_fb_bad_frac=float(1.0 - ok[inside].mean()) if inside.any() else np.nan,
                v3_flow_med=float(np.median(mag)))


FEATURES = ["v3_res_med", "v3_res_q90", "v3_hf_res_med", "v3_hf_res_norm", "v3_static_res_med",
            "v3_fb_bad_frac", "v3_flow_med"]


def frames_features(frames: list[np.ndarray], stride: int, max_pairs: int) -> dict:
    starts = list(range(0, len(frames) - 1, stride))
    if len(starts) > max_pairs:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_pairs).round().astype(int)]
    rows = [pair_stats(frames[t], frames[t + 1]) for t in starts]
    rows = [r for r in rows if r]
    out = {k: (float(np.nanmedian([r[k] for r in rows])) if rows and
               np.isfinite([r[k] for r in rows]).any() else np.nan) for k in FEATURES}
    out["v3_n_pairs"] = len(rows)
    return out


def video_features(path: Path, cfg: dict) -> dict:
    """cfg: keys ``frames`` and ``v3`` of the pipeline config."""
    fr, c = cfg["frames"], cfg["v3"]
    vf = read_frames(path, skip_s=fr["skip_s"], crop_bottom=fr["crop_bottom"],
                     crop_top=fr["crop_top"], duration_s=fr.get("duration_s"))
    return frames_features(vf.frames, c["stride"], c["max_pairs"])
