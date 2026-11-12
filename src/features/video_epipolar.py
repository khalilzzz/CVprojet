# Cours : Session 3 (SIFT, ratio test, RANSAC), Session 4 (Stereo) et Session 7 (SfM) : V1, géométrie épipolaire entre frames
"""V1: does the video respect the epipolar constraint x'^T F x = 0 between frames?

For each pair (t, t + delta):
1. SIFT keypoints spread over a grid, matched with the Lowe ratio test (EP3 slide 51)
   and a mutual nearest-neighbour check,
2. F by RANSAC + normalized 8-point algorithm (our implementation, src/geometry),
3. Sampson distances of all matches,
4. H by RANSAC on the same matches, to measure how planar / rotational the pair is:
   - ``h_cover``: share of the F inliers that H also explains (1 = degenerate pair,
     F is not informative),
   - ``parallax_px``: median transfer error of the F inliers under H, i.e. the image
     motion that a homography cannot explain. Real and generated clips are compared
     at equal parallax, because every measure degrades with the amount of 3D motion.

For each triplet (t, t + k, t + 2k): the three-view test of the SfM lecture (Session 7),
which also sees points that slide along their epipolar line.

Used by ``python -m src.run``. One video alone:
    python -m src.features.video_epipolar data/processed/videos/vid4real.mp4
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.frames import read_frames
from src.geometry.errors import sampson_distance, symmetric_transfer_error
from src.geometry.fundamental import estimate_F_ransac
from src.geometry.homography import estimate_H_ransac
from src.geometry.multiview import three_view_error
from src.utils.config import load_config

CODE_VERSION = "v1-2026-11-12a"     # change it to invalidate the cache of this feature
TRIPLET_METRICS = ["reproj3_med_px", "reproj3_inlier_ratio", "reproj12_med_px", "n_tracks"]
PAIR_COLUMNS = ["video", "delta", "t", "t0_s", "t1_s", "n_matches", "valid", "n_inliers_F",
                "inlier_ratio_F", "inlier_ratio_H", "h_cover", "parallax_px", "parallax_q90_px",
                "sampson_med_all", "sampson_q90_all", "sampson_med_in", "sampson_rmse_in",
                "median_flow_px"]
TRIPLET_COLUMNS = ["video", "delta", "t", "t0_s", "t1_s", "valid", "n_tracks", "reproj3_med_px",
                   "reproj3_q90_px", "reproj3_inlier_ratio", "reproj12_med_px", "median_flow_px"]
PAIR_METRICS = ["n_matches", "inlier_ratio_F", "sampson_med_all", "sampson_q90_all",
                "sampson_med_in", "h_cover", "parallax_px", "parallax_q90_px", "median_flow_px"]


def video_rng(seed: int, video_id: str) -> np.random.Generator:
    """One generator per video, so a result does not depend on the other videos of the run."""
    return np.random.default_rng([int(seed), zlib.crc32(video_id.encode("utf-8"))])


# ---------------------------------------------------------------- features and matching
def select_on_grid(pts: np.ndarray, response: np.ndarray, shape: tuple[int, int],
                   n: int, grid: tuple[int, int]) -> np.ndarray:
    """Indices of at most n keypoints, spread over a grid: best responses of each cell
    first (same quota per cell), then the best remaining ones."""
    if len(pts) <= n:
        return np.arange(len(pts))
    h, w = shape
    gy, gx = grid
    cell = (np.clip((pts[:, 1] * gy / h).astype(int), 0, gy - 1) * gx
            + np.clip((pts[:, 0] * gx / w).astype(int), 0, gx - 1))
    order = np.lexsort((-response, cell))               # by cell, best response first
    cell_sorted = cell[order]
    start = np.r_[0, np.flatnonzero(np.diff(cell_sorted)) + 1]
    rank = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    quota = int(np.ceil(n / (gx * gy)))
    keep = order[rank < quota]
    if len(keep) > n:
        keep = keep[np.argsort(-response[keep])[:n]]
    elif len(keep) < n:
        rest = np.setdiff1d(np.arange(len(pts)), keep)
        keep = np.r_[keep, rest[np.argsort(-response[rest])[: n - len(keep)]]]
    return np.sort(keep)


def sift_features(frames: list[np.ndarray], n_features: int, grid=(8, 8)):
    """SIFT keypoints and descriptors of each frame: list of ((N, 2) points, (N, 128) descriptors)."""
    sift = cv2.SIFT_create(nfeatures=0)
    out = []
    for f in frames:
        kp = sift.detect(f, None)
        if not kp:
            out.append((np.zeros((0, 2)), np.zeros((0, 128), np.float32)))
            continue
        pts = np.array([k.pt for k in kp], dtype=float)
        resp = np.array([k.response for k in kp])
        keep = select_on_grid(pts, resp, f.shape[:2], n_features, tuple(grid))
        kp, des = sift.compute(f, [kp[i] for i in keep])
        pts = np.array([k.pt for k in kp], dtype=float).reshape(-1, 2)
        out.append((pts, des if des is not None else np.zeros((0, 128), np.float32)))
    return out


def match_descriptors(des1: np.ndarray, des2: np.ndarray, ratio: float,
                      mutual: bool = True) -> np.ndarray:
    """Nearest-neighbour matching in descriptor space (EP3 slides 48-51).

    A match (i, j) is kept if j is the nearest neighbour of i, if d1 < ratio * d2
    (Lowe ratio test), and, when ``mutual``, if i is also the nearest neighbour of j.
    The nearest neighbours come from OpenCV's brute-force matcher (exact, L2).
    Returns (M, 2) index pairs.
    """
    if len(des1) < 2 or len(des2) < 2:
        return np.zeros((0, 2), dtype=int)
    a, b = np.ascontiguousarray(des1, np.float32), np.ascontiguousarray(des2, np.float32)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(a, b, k=2)
    fwd = np.array([(m.queryIdx, m.trainIdx, m.distance, n.distance) for m, n in knn])
    keep = fwd[:, 2] < ratio * fwd[:, 3]
    i, j = fwd[:, 0].astype(int), fwd[:, 1].astype(int)
    if mutual:
        back = np.full(len(b), -1)
        for m in bf.match(b, a):
            back[m.queryIdx] = m.trainIdx
        keep &= back[j] == i
    return np.stack([i[keep], j[keep]], axis=1)


# ---------------------------------------------------------------- one pair of frames
def pair_stats(x: np.ndarray, xp: np.ndarray, shape, rng, cfg: dict):
    """Epipolar statistics of one frame pair. Returns (row, F or None, F inlier mask)."""
    n = len(x)
    row = dict(n_matches=n, valid=False)
    if n < cfg["min_matches"]:
        return row, None, np.zeros(n, dtype=bool)
    fc, hc = cfg["fundamental"], cfg["homography"]
    F, mask, _ = estimate_F_ransac(x, xp, rng, img_shapes=(shape, shape),
                                   normalization=fc["normalization"],
                                   threshold_px=fc["threshold_px"], p=fc["p"],
                                   max_iters=fc["max_iters"])
    if F is None or mask.sum() < 8:
        return row, None, np.zeros(n, dtype=bool)
    H, mask_h, _ = estimate_H_ransac(x, xp, rng, threshold_px=hc["threshold_px"], p=hc["p"],
                                     max_iters=hc["max_iters"])
    if H is None:
        h_cover, parallax, parallax90, ratio_h = 0.0, np.nan, np.nan, 0.0
    else:
        h_cover = float((mask & mask_h).sum() / mask.sum())
        eh = symmetric_transfer_error(H, x[mask], xp[mask])
        parallax, parallax90 = float(np.median(eh)), float(np.quantile(eh, 0.9))
        ratio_h = float(mask_h.mean())
    d = sampson_distance(F, x, xp)
    row.update(
        valid=True,
        n_inliers_F=int(mask.sum()),
        inlier_ratio_F=float(mask.mean()),
        inlier_ratio_H=ratio_h,
        h_cover=h_cover,
        parallax_px=parallax,
        parallax_q90_px=parallax90,
        sampson_med_all=float(np.median(d)),
        sampson_q90_all=float(np.quantile(d, 0.9)),
        sampson_med_in=float(np.median(d[mask])),
        sampson_rmse_in=float(np.sqrt(np.mean(d[mask] ** 2))),
        median_flow_px=float(np.median(np.linalg.norm(xp - x, axis=1))),
    )
    return row, F, mask


# ---------------------------------------------------------------- three frames
def triplet_stats(feats, i: int, k: int, shape, rng, cfg: dict) -> dict | None:
    """Three-view test on frames (i, i + k, i + 2k), see src/geometry/multiview.py.

    Points are followed over the three frames through their keypoint in the middle frame.
    F of the first two frames comes from our RANSAC; the points that fit it are
    triangulated, the third camera is estimated, and the reprojection error is measured.
    """
    m = cfg["matching"]
    (p1, d1), (p2, d2), (p3, d3) = feats[i], feats[i + k], feats[i + 2 * k]
    m12 = match_descriptors(d1, d2, m["ratio"], m.get("mutual", True))
    m23 = match_descriptors(d2, d3, m["ratio"], m.get("mutual", True))
    if len(m12) < cfg["min_matches"] or len(m23) < cfg["min_matches"]:
        return None
    fc = cfg["fundamental"]
    x1, x2 = p1[m12[:, 0]], p2[m12[:, 1]]
    F, mask, _ = estimate_F_ransac(x1, x2, rng, img_shapes=(shape, shape),
                                   normalization=fc["normalization"],
                                   threshold_px=fc["threshold_px"], p=fc["p"],
                                   max_iters=fc["max_iters"])
    if F is None:
        return None
    to3 = dict(zip(m23[:, 0].tolist(), m23[:, 1].tolist()))      # keypoint of frame 2 -> frame 3
    keep = [j for j in np.flatnonzero(mask) if int(m12[j, 1]) in to3]
    if len(keep) < 12:
        return None
    a1, a2 = x1[keep], x2[keep]
    a3 = p3[[to3[int(m12[j, 1])] for j in keep]]
    tv = cfg["three_view"]
    out = three_view_error(F, a1, a2, a3, rng, threshold_px=tv["threshold_px"],
                           max_iters=tv["max_iters"])
    if out is None:
        return None
    out["median_flow_px"] = float(0.5 * (np.median(np.linalg.norm(a2 - a1, axis=1))
                                         + np.median(np.linalg.norm(a3 - a2, axis=1))))
    return out


# ---------------------------------------------------------------- one video
def analyse_frames(vf, video_id: str, cfg: dict, seed: int = 0):
    """All frame pairs and frame triplets of one clip (a VideoFrames object).

    cfg has the key ``v1`` of configs/pipeline.yaml.
    Returns (pairs table, triplets table, info); info holds the frames and one example pair
    per gap.
    """
    v1 = cfg["v1"]
    rng = video_rng(seed, video_id)
    m = v1["matching"]
    feats = sift_features(vf.frames, m["n_features"], m.get("grid", (8, 8)))
    rows, examples = [], {}
    for delta in v1["deltas"]:
        starts = list(range(0, len(vf.frames) - delta, v1["stride"]))
        for k, t in enumerate(starts):
            (p1, d1), (p2, d2) = feats[t], feats[t + delta]
            mm = match_descriptors(d1, d2, m["ratio"], m.get("mutual", True))
            x, xp = p1[mm[:, 0]], p2[mm[:, 1]]
            row, F, mask = pair_stats(x, xp, vf.shape, rng, v1)
            rows.append(dict(video=video_id, delta=delta, t=t, t0_s=vf.times[t],
                             t1_s=vf.times[t + delta]) | row)
            # keep the pair in the middle of the clip as the example of this gap
            if F is not None and (delta not in examples or k <= len(starts) // 2):
                examples[delta] = (t, x, xp, F, mask)
    trip = []
    tv = v1.get("three_view")
    if tv:
        for gap in tv["gaps"]:
            for t in range(0, len(vf.frames) - 2 * gap, tv["stride"]):
                out = triplet_stats(feats, t, gap, vf.shape, rng, v1)
                trip.append(dict(video=video_id, delta=gap, t=t, t0_s=vf.times[t],
                                 t1_s=vf.times[t + 2 * gap], valid=out is not None) | (out or {}))
    # a video may have no valid pair or triplet: keep every column so later steps still work
    pairs = pd.DataFrame(rows).reindex(columns=PAIR_COLUMNS)
    triplets = pd.DataFrame(trip).reindex(columns=TRIPLET_COLUMNS)
    return pairs, triplets, dict(frames=vf, examples=examples)


def analyse_video(path: Path, video_id: str, cfg: dict, seed: int = 0):
    """Read a normalized clip (key ``frames`` of the config) and analyse it."""
    fr = cfg["frames"]
    vf = read_frames(path, skip_s=fr["skip_s"], crop_bottom=fr["crop_bottom"],
                     crop_top=fr["crop_top"], duration_s=fr.get("duration_s"),
                     target_fps=fr.get("target_fps"), short_side=fr.get("short_side"))
    return analyse_frames(vf, video_id, cfg, seed)


def video_features(df: pd.DataFrame, trip: pd.DataFrame | None = None) -> dict:
    """One feature vector per video: medians over pairs (and triplets), per gap."""
    out = dict(video=df["video"].iloc[0], v1_frac_valid_pairs=float(df["valid"].mean()))
    for delta, g in df.groupby("delta"):
        out[f"v1_frac_valid_d{delta}"] = float(g["valid"].mean())
        v = g[g["valid"]]
        for col in PAIR_METRICS:
            out[f"v1_{col}_d{delta}"] = float(v[col].median()) if len(v) else np.nan
    if trip is not None and len(trip):
        for gap, g in trip.groupby("delta"):
            out[f"v1_frac_valid_triplets_g{gap}"] = float(g["valid"].mean())
            v = g[g["valid"]]
            for col in TRIPLET_METRICS:
                out[f"v1_{col}_g{gap}"] = float(v[col].median()) if len(v) else np.nan
    return out


def main() -> None:
    from src.eval import plots

    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--out", default="results/v1_single")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for v in args.videos:
        path = Path(v)
        df, trip, info = analyse_video(path, path.stem, cfg, cfg["seed"])
        df.to_csv(out / f"v1_pairs_{path.stem}.csv", index=False)
        trip.to_csv(out / f"v1_triplets_{path.stem}.csv", index=False)
        plots.video_sheet(df, info, out / f"v1_sheet_{path.stem}.png")
        print(df[df["valid"]].groupby("delta")[PAIR_METRICS].median().round(3).to_string())


if __name__ == "__main__":
    main()
