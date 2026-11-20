# Cours : Session 6 (flot dense) et Session 7 (mouvement de caméra) ; hors cours : modèle de
# mouvement rigide instantané (Longuet-Higgins & Prazdny 1980 ; Szeliski ch. 9, 11, 12) avec une
# profondeur monoculaire apprise (Depth Anything V2) : V5, le flot est-il celui d'une caméra ?
"""V5: does the optical flow agree with the depth a monocular network sees?

A camera moving with velocity (t, w) in a rigid scene produces, at the normalized image point
(x, y) = ((u - cx) / f, (v - cy) / f) with inverse depth rho = 1 / Z, the flow
(Longuet-Higgins & Prazdny 1980):

    u_dot = rho (-tx + x tz) + [ x y wx - (1 + x^2) wy + y wz ]
    v_dot = rho (-ty + y tz) + [ (1 + y^2) wx - x y wy - x wz ]

The depth comes from Depth Anything V2 (relative inverse depth d, up to a scale a and a shift
b: rho = a d + b). The model is then LINEAR in 9 unknowns (a t, b t, w), fitted by iteratively
re-weighted least squares on textured pixels that pass the forward-backward flow check. The
residual is the flow that no camera motion can explain given that depth: moving objects, and a
generated scene whose motion does not follow its own depth. The unknown focal length is
assumed (``focal_factor`` x the longest side, as in I2).

Compared with the three-view test (V1, Session 7): dense instead of sparse, differential
(suited to small motions), and robust to blur (checked on synthetic clips: blur leaves the
residual unchanged, a time-growing warp of the scene raises it). Like every geometric test
it also fires on real moving objects, so the per-pixel residual map matters as much as the
number: it shows WHERE the video breaks the rigid-camera model.

Features (medians over ``n_pairs`` frame pairs (t, t + gap)): ``v5_res_px`` (median residual,
px), ``v5_res_rel`` (residual / median flow), ``v5_res_rot_px`` (residual of the rotation-only
model, 3 unknowns, no depth), ``v5_gain`` (rotation-only / with-depth: how much the depth
explains), ``v5_bad_frac`` (share of pixels with residual > ``bad_px``), ``v5_flow_px``,
``v5_n_pairs``.

    python -m src.features.video_rigidflow data/processed/videos/vid6.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.data.frames import read_frames
from src.features.video_temporal import _flow

CODE_VERSION = "v5-2026-11-16a"
FEATURES = ["v5_res_px", "v5_res_rel", "v5_res_rot_px", "v5_gain", "v5_bad_frac", "v5_flow_px"]
DEFAULTS = dict(gap=8, n_pairs=6, short_side=360, focal_factor=0.8, texture_quantile=0.6,
                iters=5, max_px=30000, bad_px=1.0, duplicate_diff=0.3,
                model="depth-anything/Depth-Anything-V2-Small-hf")

_DEPTH_MODEL = {}


# ---------------------------------------------------------------- monocular depth (learned prior)
def inverse_depth(gray: np.ndarray, model: str = DEFAULTS["model"]) -> np.ndarray:
    """Relative inverse depth of a grayscale frame (float32, same shape, unit standard deviation).

    Depth Anything V2 is loaded once per process (transformers pipeline, CPU or GPU)."""
    from PIL import Image

    if model not in _DEPTH_MODEL:
        import torch
        from transformers import pipeline
        _DEPTH_MODEL[model] = pipeline("depth-estimation", model=model,
                                       device=0 if torch.cuda.is_available() else -1)
    out = _DEPTH_MODEL[model](Image.fromarray(np.ascontiguousarray(gray, np.uint8)).convert("RGB"))
    d = np.asarray(out["predicted_depth"], dtype=np.float32)
    if d.shape != gray.shape:
        d = cv2.resize(d, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return d / (d.std() + 1e-6)


# ---------------------------------------------------------------- rigid motion model
def assumed_K(shape: tuple[int, int], focal_factor: float) -> np.ndarray:
    h, w = shape
    f = focal_factor * max(h, w)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def rigid_flow_basis(x: np.ndarray, y: np.ndarray, rho: np.ndarray, with_depth: bool = True) -> np.ndarray:
    """Design matrix of the instantaneous rigid flow at normalized points (x, y).

    Rows: the u components of every point, then the v components. Columns: a*t (3), b*t (3),
    w (3) when ``with_depth``; w (3) only otherwise (rotation-only camera)."""
    zero, one = np.zeros_like(x), np.ones_like(x)
    rot_u = [x * y, -(1 + x ** 2), y]
    rot_v = [(1 + y ** 2), -x * y, -x]
    if not with_depth:
        return np.vstack([np.stack(rot_u, 1), np.stack(rot_v, 1)])
    tr_u = [-rho, zero, rho * x, -one, zero, x]
    tr_v = [zero, -rho, rho * y, zero, -one, y]
    return np.vstack([np.stack(tr_u + rot_u, 1), np.stack(tr_v + rot_v, 1)])


def rigid_flow(params: np.ndarray, x: np.ndarray, y: np.ndarray, rho: np.ndarray, f: float) -> np.ndarray:
    """Flow (N, 2) in pixels predicted by the 9 (or 3) parameters."""
    A = rigid_flow_basis(x, y, rho, with_depth=len(params) == 9)
    p = A @ params
    return f * np.stack([p[: len(x)], p[len(x):]], 1)


def fit_rigid(flow: np.ndarray, rho: np.ndarray, mask: np.ndarray, K: np.ndarray,
              with_depth: bool = True, iters: int = 5, max_px: int = 30000,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Robust fit of the rigid flow model on the pixels of ``mask``.

    flow: (h, w, 2) in pixels; rho: (h, w) inverse depth (any affine scale). The fit uses at
    most ``max_px`` pixels (IRLS, Cauchy-like weights); the returned residual map (h, w, px)
    is evaluated on every pixel of the mask, NaN elsewhere. Returns (params, residual_map)."""
    f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None, np.full(mask.shape, np.nan)
    fit_idx = np.arange(len(xs))
    if len(xs) > max_px:
        fit_idx = (rng or np.random.default_rng(0)).choice(len(xs), max_px, replace=False)
    x, y = (xs - cx) / f, (ys - cy) / f
    A = rigid_flow_basis(x[fit_idx], y[fit_idx], rho[ys[fit_idx], xs[fit_idx]], with_depth)
    b = np.concatenate([flow[ys[fit_idx], xs[fit_idx], 0], flow[ys[fit_idx], xs[fit_idx], 1]]) / f
    n = len(fit_idx)
    wgt = np.ones(2 * n)
    params = None
    for _ in range(iters):
        sw = np.sqrt(wgt)[:, None]
        params = np.linalg.lstsq(A * sw, b * sw[:, 0], rcond=None)[0]
        r = (A @ params - b) * f
        rr = np.hypot(r[:n], r[n:])
        s = 1.4826 * np.median(rr) + 1e-3
        wv = 1.0 / np.maximum(1.0, (rr / (2.5 * s)) ** 2)
        wgt = np.concatenate([wv, wv])
    pred = rigid_flow(params, x, y, rho[ys, xs], f)
    res = np.full(mask.shape, np.nan, dtype=np.float32)
    res[ys, xs] = np.hypot(pred[:, 0] - flow[ys, xs, 0], pred[:, 1] - flow[ys, xs, 1])
    return params, res


# ---------------------------------------------------------------- one frame pair
def _prepare(a: np.ndarray, b: np.ndarray, c: dict):
    """Flow a -> b, forward-backward mask, textured-pixel mask."""
    h, w = a.shape
    fwd, bwd = _flow(a, b), _flow(b, a)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + fwd[..., 0], gy + fwd[..., 1]
    inside = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    b_at = cv2.remap(bwd, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mag = np.hypot(fwd[..., 0], fwd[..., 1])
    ok = inside & (np.hypot(*(fwd + b_at).transpose(2, 0, 1)) < 0.5 + 0.05 * mag)
    gm = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    tex = gm > np.quantile(gm, c.get("texture_quantile", DEFAULTS["texture_quantile"]))
    return fwd, mag, ok & tex


def pair_analysis(a: np.ndarray, b: np.ndarray, c: dict, rho: np.ndarray | None = None) -> dict | None:
    """Rigid-flow residual of the pair (a, b). Returns the statistics and the maps
    (``res``: residual with depth, ``res_rot``: rotation only, ``rho``, ``mask``, ``flow``)."""
    c = DEFAULTS | c
    fwd, mag, mask = _prepare(a, b, c)
    if mask.sum() < 500:
        return None
    if rho is None:
        rho = inverse_depth(a, c["model"])
    K = assumed_K(a.shape, c["focal_factor"])
    rng = np.random.default_rng(0)
    _, res = fit_rigid(fwd, rho, mask, K, True, c["iters"], c["max_px"], rng)
    _, res_rot = fit_rigid(fwd, rho, mask, K, False, c["iters"], c["max_px"], rng)
    r, rr = res[mask], res_rot[mask]
    flow_med = float(np.median(mag[mask]))
    return dict(v5_res_px=float(np.median(r)), v5_res_rel=float(np.median(r) / (flow_med + 1e-3)),
                v5_res_rot_px=float(np.median(rr)), v5_gain=float(np.median(rr) / (np.median(r) + 1e-3)),
                v5_bad_frac=float((r > c["bad_px"]).mean()), v5_flow_px=flow_med,
                res=res, res_rot=res_rot, rho=rho, mask=mask, flow=fwd)


def pair_stats(a: np.ndarray, b: np.ndarray, c: dict, rho: np.ndarray | None = None) -> dict:
    out = pair_analysis(a, b, c, rho)
    return {k: out[k] for k in FEATURES} if out else {}


# ---------------------------------------------------------------- one video
def pair_indices(n_frames: int, gap: int, n_pairs: int) -> list[int]:
    if n_frames <= gap:
        return []
    return sorted(set(np.linspace(0, n_frames - gap - 1, n_pairs).round().astype(int).tolist()))


def frames_features(frames: list[np.ndarray], c: dict) -> dict:
    c = DEFAULTS | c
    rows = []
    for t in pair_indices(len(frames), c["gap"], c["n_pairs"]):
        a, b = frames[t], frames[t + c["gap"]]
        if np.abs(a.astype(np.float32) - b).mean() < c["duplicate_diff"]:
            continue                                    # duplicated / frozen frames
        r = pair_stats(a, b, c)
        if r:
            rows.append(r)
    out = {k: (float(np.median([r[k] for r in rows])) if rows else np.nan) for k in FEATURES}
    out["v5_n_pairs"] = len(rows)
    return out


def video_features(path: Path, cfg: dict) -> dict:
    """cfg: keys ``frames`` and ``v5`` of the pipeline config."""
    fr, c = cfg["frames"], DEFAULTS | cfg["v5"]
    vf = read_frames(path, skip_s=fr["skip_s"], crop_bottom=fr["crop_bottom"],
                     crop_top=fr["crop_top"], duration_s=fr.get("duration_s"),
                     short_side=c["short_side"])
    return frames_features(vf.frames, c)


def residual_figure(paths: list[Path], cfg: dict, out: Path, t: int | None = None) -> Path:
    """One row per video: frame, learned inverse depth, flow, residual map (px) of the pair
    (t, t + gap). Same colour scale for every row."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr, c = cfg["frames"], DEFAULTS | cfg["v5"]
    rows = []
    for p in paths:
        vf = read_frames(p, skip_s=fr["skip_s"], crop_bottom=fr["crop_bottom"], crop_top=fr["crop_top"],
                         duration_s=fr.get("duration_s"), short_side=c["short_side"])
        tt = t if t is not None else (len(vf.frames) - c["gap"]) // 2
        a = pair_analysis(vf.frames[tt], vf.frames[tt + c["gap"]], c)
        rows.append((Path(p).stem, vf.frames[tt], a))
    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3.3 * len(rows)), squeeze=False)
    vmax = 3.0
    for (name, frame, a), ax in zip(rows, axes):
        ax[0].imshow(frame, cmap="gray")
        ax[0].set_title(name, loc="left")
        if a is None:
            continue
        ax[1].imshow(a["rho"], cmap="magma")
        ax[1].set_title("learned inverse depth (Depth Anything V2)", fontsize=9)
        mag = np.hypot(a["flow"][..., 0], a["flow"][..., 1])
        ax[2].imshow(mag, cmap="viridis")
        ax[2].set_title(f"flow over {c['gap']} frames, median {a['v5_flow_px']:.1f} px", fontsize=9)
        im = ax[3].imshow(a["res"], cmap="inferno", vmin=0, vmax=vmax)
        ax[3].set_title(f"flow no camera motion explains\nmedian {a['v5_res_px']:.2f} px, "
                        f"{100 * a['v5_bad_frac']:.0f} % above 1 px", fontsize=9)
        for x in ax:
            x.set_axis_off()
    fig.colorbar(im, ax=axes[:, 3].tolist(), fraction=0.02, label="residual (px)")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    from src.utils.config import load_config

    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--figure", help="write the residual maps of the videos to this PNG")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for v in args.videos:
        feats = video_features(Path(v), cfg)
        print(Path(v).stem, {k: round(x, 3) if isinstance(x, float) else x for k, x in feats.items()})
    if args.figure:
        print(residual_figure([Path(v) for v in args.videos], cfg, Path(args.figure)))


if __name__ == "__main__":
    main()
