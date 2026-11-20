# Cours : Sessions 1-7 : mesures image par image pour la démo (V1 épipolaire, test à trois vues, I2 points de fuite, V3 flot, I1 spectre)
"""Per-frame measurements of one clip, for the side-by-side demo (demo.py).

Every value comes from the functions of the pipeline, applied to each frame t:

- V1 (Sessions 3-4): SIFT matches between frames t - gap and t, F by our RANSAC + normalized
  8-point algorithm, Sampson distance (``video_epipolar.pair_stats``);
- three views (Session 7): points followed over t - 2k, t - k, t, triangulated from the first
  two views, third camera by DLT in RANSAC, reprojection error (same steps and same random
  draws as ``video_epipolar.triplet_stats``, but the points are kept for the drawing);
- I2 (Sessions 1-3): LSD segments and vanishing points by RANSAC on frame t
  (``image_perspective.frame_vps``); the temporal values use the frames seen so far;
- V3 (Session 6): Farnebäck flow between t - 1 and t, warping residual (``video_temporal``);
- I1 (Sessions 1-2): noise residual and its power spectrum, averaged over the frames seen so far.

The heavy part (SIFT, RANSAC, flow) runs in parallel over chunks of frames and is cached in
results/demo/cache/.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from src.features import image_frequency as i1
from src.features import image_perspective as i2
from src.features import video_epipolar as v1
from src.features import video_temporal as v3
from src.geometry.fundamental import estimate_F_ransac
from src.geometry.multiview import (cameras_from_F, estimate_camera_ransac, project,
                                    reprojection_error, triangulate)
from src.utils.config import REPO_ROOT

CODE_VERSION = "demo-2026-11-15a"
CACHE = REPO_ROOT / "results" / "demo" / "cache"
NAN = float("nan")


# ---------------------------------------------------------------- reading
def read_clip(path: Path, frames_cfg: dict) -> tuple[list[np.ndarray], float]:
    """Colour frames (BGR) of a normalized clip: skip_s dropped, at most duration_s kept,
    same top / bottom crop as the pipeline (so the points of the methods fall on these frames)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    skip = int(round(frames_cfg.get("skip_s", 0.0) * fps))
    n_max = int(round((frames_cfg.get("duration_s") or 1e6) * fps))
    out, i = [], 0
    while len(out) < n_max:
        ok, f = cap.read()
        if not ok:
            break
        if i >= skip:
            h = f.shape[0]
            out.append(f[int(round(frames_cfg.get("crop_top", 0.0) * h)):
                         h - int(round(frames_cfg.get("crop_bottom", 0.0) * h))])
        i += 1
    cap.release()
    if not out:
        raise ValueError(f"no frame read from {path}")
    return out, fps


# ---------------------------------------------------------------- three views, with the points
def three_view_points(f1, f2, f3, shape, rng: np.random.Generator, c: dict) -> dict | None:
    """Session 7 test on three frames (same steps and random draws as triplet_stats).

    f1, f2, f3: (points, descriptors) of the three frames; c: the ``v1`` config.
    Returns the followed points a1, a2, a3, their reprojection in view 3, the errors.
    """
    m = c["matching"]
    (p1, d1), (p2, d2), (p3, d3) = f1, f2, f3
    m12 = v1.match_descriptors(d1, d2, m["ratio"], m.get("mutual", True))
    m23 = v1.match_descriptors(d2, d3, m["ratio"], m.get("mutual", True))
    if len(m12) < c["min_matches"] or len(m23) < c["min_matches"]:
        return None
    fc = c["fundamental"]
    x1, x2 = p1[m12[:, 0]], p2[m12[:, 1]]
    F, mask, _ = estimate_F_ransac(x1, x2, rng, img_shapes=(shape, shape),
                                   normalization=fc["normalization"], threshold_px=fc["threshold_px"],
                                   p=fc["p"], max_iters=fc["max_iters"])
    if F is None:
        return None
    to3 = dict(zip(m23[:, 0].tolist(), m23[:, 1].tolist()))
    keep = [j for j in np.flatnonzero(mask) if int(m12[j, 1]) in to3]
    if len(keep) < 12:
        return None
    a1, a2 = x1[keep], x2[keep]
    a3 = p3[[to3[int(m12[j, 1])] for j in keep]]
    P1, P2 = cameras_from_F(F)
    X = triangulate(P1, P2, a1, a2)
    tv = c["three_view"]
    M, inl, _ = estimate_camera_ransac(X, a3, rng, threshold_px=tv["threshold_px"],
                                       max_iters=tv["max_iters"])
    if M is None or inl.sum() < 6:
        return None
    err = reprojection_error(M, X, a3)
    return dict(a1=a1.astype(np.float32), a2=a2.astype(np.float32), a3=a3.astype(np.float32),
                proj=project(M, X).astype(np.float32), err=err.astype(np.float32), inliers=inl,
                err_med=float(np.median(err)))


# ---------------------------------------------------------------- one chunk of frames
def _frame_rng(seed: int, video_id: str, t: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(video_id.encode("utf-8")), int(t)])


def measure_chunk(args) -> list[dict]:
    """Measurements of frames start..end-1. grays holds frames offset..end-1."""
    video_id, start, end, offset, grays, cfg = args
    c1, c2, d = cfg["v1"], cfg["i2"], cfg["demo"]
    gap, k = int(d["v1_gap"]), int(d["three_view_gap"])
    mc = c1["matching"]
    feats = v1.sift_features(grays, mc["n_features"], mc.get("grid", (8, 8)))
    out = []
    for t in range(start, end):
        lt = t - offset
        g = grays[lt]
        rng = _frame_rng(cfg["seed"], video_id, t)
        m: dict = dict(t=t)
        if lt >= gap:                                                    # V1, two views
            (p1, d1), (p2, d2) = feats[lt - gap], feats[lt]
            mm = v1.match_descriptors(d1, d2, mc["ratio"], mc.get("mutual", True))
            x, xp = p1[mm[:, 0]], p2[mm[:, 1]]
            row, F, mask = v1.pair_stats(x, xp, g.shape, rng, c1)
            m["v1"] = dict(x=x.astype(np.float32), xp=xp.astype(np.float32), mask=mask, F=F, row=row)
        if lt >= 2 * k:                                                  # Session 7, three views
            m["tv"] = three_view_points(feats[lt - 2 * k], feats[lt - k], feats[lt], g.shape, rng, c1)
        seg = i2.detect_segments(g, c2.get("min_len_frac", 0.04), c2.get("max_segments", 300))
        m["i2"] = i2.frame_vps(seg, g.shape, rng, c2)                    # I2, this frame
        if lt >= 1:                                                      # V3, frames t-1 and t
            a = grays[lt - 1]
            hres, ok, _ = v3.residual_map(a, g)
            half = (g.shape[1] // 2, g.shape[0] // 2)
            m["v3"] = dict(stats=v3.pair_stats(a, g),
                           hres=cv2.resize(np.clip(hres * (255 / 25), 0, 255).astype(np.uint8), half,
                                           interpolation=cv2.INTER_AREA),
                           bad=cv2.resize((~ok).astype(np.uint8) * 255, half,
                                          interpolation=cv2.INTER_AREA))
        out.append(m)
    return out


# ---------------------------------------------------------------- running values over the clip
def i1_series(grays: list[np.ndarray], c: dict) -> list[dict]:
    """I1 on the frames seen so far: log spectra averaged over frames 0..t (as in the pipeline,
    which averages spread frames). Small inset image of the spectrum excess per frame."""
    res_sum = img_sum = None
    noise, out = [], []
    for n, g in enumerate(grays, start=1):
        r = i1.noise_residual(g, c.get("median_ksize", 3))
        rl, il = i1.log_spectrum(r, c["crop"]), i1.log_spectrum(g, c["crop"])
        res_sum = rl if res_sum is None else res_sum + rl
        img_sum = il if img_sum is None else img_sum + il
        noise.append(float(r.std()))
        avg = res_sum / n
        f = i1.spectrum_features(avg, img_sum / n, c["n_bins"], float(np.median(noise)))
        centres, prof = i1.radial_profile(avg, c["n_bins"])
        excess = avg - np.interp(i1.radial_frequency(avg.shape[0]), centres, prof)
        inset = cv2.resize((np.clip(excess / 1.5, 0, 1) * 255).astype(np.uint8), (200, 200),
                           interpolation=cv2.INTER_AREA)
        out.append(dict(hf=f["i1_hf_ratio"], peak=f["i1_peak_max"], prof=prof, centres=centres,
                        inset=inset))
    return out


def i2_series(per_frame: list[dict], c: dict) -> list[dict]:
    """I2 temporal values on the sampled frames seen so far (one in ``stride``, as in the pipeline)."""
    stride = max(1, int(c.get("stride", 2)))
    out = []
    for t in range(len(per_frame)):
        sel = [per_frame[j] for j in range(0, t + 1, stride)]
        f = i2.temporal_features(sel, c) if len(sel) >= 3 else {}
        out.append(dict(rot=f.get("i2_rot_resid_deg", NAN), pair=f.get("i2_pair_angle_std_deg", NAN)))
    return out


# ---------------------------------------------------------------- whole clips, cached
def _key(path: Path, video_id: str, n: int, cfg: dict) -> str:
    st = Path(path).stat()
    blob = json.dumps(dict(id=video_id, size=st.st_size, mtime=int(st.st_mtime), n=n,
                           cfg={k: cfg[k] for k in ("seed", "frames", "v1", "i2", "i1", "demo")},
                           code=[CODE_VERSION, v1.CODE_VERSION, i2.CODE_VERSION,
                                 v3.CODE_VERSION, i1.CODE_VERSION]), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def measure_clips(clips: list[tuple[str, Path, list[np.ndarray]]], cfg: dict, workers: int = 6,
                  force: bool = False, log=print) -> dict[str, dict]:
    """clips: (video id, normalized file, grayscale frames). Returns {id: dict(frames, i1, i2)},
    frames being the per-frame measurement dicts."""
    d = cfg["demo"]
    need = max(int(d["v1_gap"]), 2 * int(d["three_view_gap"]), 1)
    results, tasks, keys = {}, [], {}
    CACHE.mkdir(parents=True, exist_ok=True)
    for vid, path, grays in clips:
        keys[vid] = CACHE / f"{vid}_{_key(path, vid, len(grays), cfg)}.pkl"
        if keys[vid].exists() and not force:
            results[vid] = pickle.loads(keys[vid].read_bytes())
            log(f"[measure] {vid}: {len(grays)} frames from cache")
            continue
        n_chunks = max(1, workers // max(1, len(clips)))
        bounds = np.linspace(0, len(grays), n_chunks + 1).round().astype(int)
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e > s:
                off = max(0, s - need)
                tasks.append((vid, int(s), int(e), off, grays[off:e], cfg))
    if tasks:
        log(f"[measure] {len(tasks)} chunks of frames on {workers} processes "
            f"(SIFT, RANSAC, LSD, flow: about 1 s per frame and per clip)")
        per: dict[str, list] = {}
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for (vid, *_), res in zip(tasks, ex.map(measure_chunk, tasks)):
                    per.setdefault(vid, []).extend(res)
                    log(f"[measure] {vid}: frames up to {res[-1]['t']} done")
        else:
            for task in tasks:
                per.setdefault(task[0], []).extend(measure_chunk(task))
        for vid, path, grays in clips:
            if vid not in per:
                continue
            frames = sorted(per[vid], key=lambda m: m["t"])
            res = dict(frames=frames, i1=i1_series(grays, cfg["i1"]),
                       i2=i2_series([m["i2"] for m in frames], cfg["i2"]))
            keys[vid].write_bytes(pickle.dumps(res, protocol=pickle.HIGHEST_PROTOCOL))
            results[vid] = res
    return results


# ---------------------------------------------------------------- indicator series
def _running_median(v: np.ndarray) -> np.ndarray:
    out = np.full(len(v), NAN)
    for t in range(len(v)):
        w = v[: t + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[t] = float(np.median(w))
    return out


def _get(m: dict, *path):
    for p in path:
        if m is None:
            return NAN
        m = m.get(p) if isinstance(m, dict) else None
    return NAN if m is None else float(m)


# key, label, chapter, decimals, kind ("frame": per-frame value, shown as a running median;
# "cumul": the value is already computed on the frames seen so far)
INDICATORS = [
    ("v1_inlier", "V1  part des appariements compatibles avec F", "v1", 3, "frame"),
    ("v1_sampson", "V1  distance de Sampson médiane (px)", "v1", 3, "frame"),
    ("v1_motion", "V1  mouvement image entre t−10 et t (px)", "v1", 1, "frame"),
    ("tv_err", "V1  erreur de reprojection à trois vues (px)", "tv", 3, "frame"),
    ("i2_resid", "I2  écart des segments au point de fuite (°)", "i2", 2, "frame"),
    ("i2_rot", "I2  directions hors rotation de la caméra (°)", "i2", 2, "cumul"),
    ("i2_pair", "I2  variation de l'angle entre deux directions (°)", "i2", 2, "cumul"),
    ("v3_hf", "V3  changement de texture après compensation (/ bruit)", "v3", 3, "frame"),
    ("v3_static", "V3  changement des pixels immobiles (niveaux de gris)", "v3", 2, "frame"),
    ("i1_hf", "I1  part du bruit en haute fréquence", "i1", 3, "cumul"),
    ("i1_peak", "I1  pics périodiques du spectre (log10)", "i1", 2, "cumul"),
]


def indicator_series(res: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """{key: (curve, shown value)} over the frames: the curve is the per-frame value (or the
    running value for "cumul" indicators), the shown value its running median."""
    fr = res["frames"]
    raw = {
        "v1_inlier": [_get(m.get("v1"), "row", "inlier_ratio_F") for m in fr],
        "v1_sampson": [_get(m.get("v1"), "row", "sampson_med_all") for m in fr],
        "v1_motion": [_get(m.get("v1"), "row", "median_flow_px") for m in fr],
        "tv_err": [_get(m.get("tv"), "err_med") for m in fr],
        "i2_resid": [_get(m.get("i2"), "resid") for m in fr],
        "i2_rot": [s["rot"] for s in res["i2"]],
        "i2_pair": [s["pair"] for s in res["i2"]],
        "v3_hf": [_get(m.get("v3"), "stats", "v3_hf_res_norm") for m in fr],
        "v3_static": [_get(m.get("v3"), "stats", "v3_static_res_mean") for m in fr],
        "i1_hf": [s["hf"] for s in res["i1"]],
        "i1_peak": [s["peak"] for s in res["i1"]],
    }
    out = {}
    for key, _, _, _, kind in INDICATORS:
        curve = np.asarray(raw[key], float)
        out[key] = (curve, curve.copy() if kind == "cumul" else _running_median(curve))
    return out
