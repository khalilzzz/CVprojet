# Cours : Sessions 1-2 (contours), Session 3 (RANSAC) et Session 4 (single view modeling, points de fuite) : I2, perspective et stabilité des points de fuite dans le temps
"""I2: vanishing points, on one image and over the frames of a video.

One frame:
1. line segments (LSD of OpenCV; Canny + probabilistic Hough if LSD is missing), only the
   long ones (``min_len_frac`` of the image diagonal),
2. up to 3 vanishing points (VPs) by sequential RANSAC (our src/geometry/vanishing.py),
3. 3D direction of each VP: d = K^-1 v, with an ASSUMED camera K (focal = ``focal_factor``
   x the longest image side, principal point at the centre). The sign of d is free.

A video (the cue of Grab-3D, 2025): parallel 3D lines of a rigid scene keep the same 3D
direction. When the camera rotates, every VP direction turns with the same rotation, so:
- all directions turn with ONE rotation between two frames (``i2_rot_resid_deg``: the
  rotation is fitted on the tracked directions by orthogonal Procrustes, the residual is
  the angle left to each direction). Camera shake does not change it: this is the cue,
- each direction moves smoothly from frame to frame (``i2_jitter_med_deg``: second
  difference of a tracked direction). Insensitive to a smooth camera rotation, but a
  handheld walking camera shakes, so it mixes camera shake and scene instability,
- the angle between two directions stays constant (``i2_pair_angle_std_deg``): a rotation
  keeps angles. With the assumed K this holds only approximately (wrong focal length), but
  the error is smooth in time, while a drifting generated scene changes it frame to frame.
- ``i2_resid_med_deg``, the misfit of the segments to their VP in one frame, measures how
  straight and how precisely located the lines are (rendering), and serves as the noise
  floor: ``i2_jitter_norm`` = jitter / residual.

Features (one video): ``i2_frac_frames_vp`` (share of frames with a VP), ``i2_n_segments_med``,
``i2_n_vps_med``, ``i2_explained_frac_med`` (length-weighted share of segments that meet a
VP), ``i2_resid_med_deg``, ``i2_orth_err_med_deg`` (|angle between the two main directions -
90|), ``i2_jump_med_deg`` (direction change between sampled frames), ``i2_jitter_med_deg``,
``i2_jitter_norm``, ``i2_pair_angle_std_deg``, ``i2_rot_resid_deg``, ``i2_track_len_med`` (sampled frames a direction
survives), ``i2_dominant_kept_frac`` (share of steps where the strongest VP is found again).

Limits: needs straight man-made structures (buildings, corridors, interiors); a frame
without long lines gives no VP (NaN features). Lens distortion bends lines (fisheye, action
cameras). Reference: Sarkar et al., CVPR 2024 (single images); Grab-3D, 2025 (videos).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np

from src.data.frames import read_frames
from src.geometry.vanishing import estimate_vps, vp_residual_angle

CODE_VERSION = "i2-2026-11-13b"

FEATURES = ["i2_frac_frames_vp", "i2_n_segments_med", "i2_n_vps_med", "i2_explained_frac_med",
            "i2_resid_med_deg", "i2_orth_err_med_deg", "i2_jump_med_deg", "i2_jitter_med_deg",
            "i2_jitter_norm", "i2_pair_angle_std_deg", "i2_rot_resid_deg", "i2_track_len_med",
            "i2_dominant_kept_frac"]


# ---------------------------------------------------------------- one frame
def detect_segments(gray: np.ndarray, min_len_frac: float = 0.04, max_segments: int = 300) -> np.ndarray:
    """(N, 4) segments [x1, y1, x2, y2], longest first, at least min_len_frac x diagonal long."""
    g = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = g.shape
    min_len = min_len_frac * np.hypot(h, w)
    seg = None
    if hasattr(cv2, "createLineSegmentDetector"):
        try:
            lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
            seg = np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4).astype(float)
        except cv2.error:                       # some OpenCV builds ship LSD disabled
            seg = None
    if seg is None:
        edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 1.2), 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40, minLineLength=int(min_len),
                                maxLineGap=4)
        seg = np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4).astype(float)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    keep = np.argsort(-length)
    keep = keep[length[keep] >= min_len][:max_segments]
    return seg[keep]


def assumed_K(shape: tuple[int, int], focal_factor: float) -> np.ndarray:
    h, w = shape
    f = focal_factor * max(h, w)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle_deg(d1: np.ndarray, d2: np.ndarray) -> float:
    """Angle between two directions, sign-free (0 to 90 degrees)."""
    c = abs(float(np.dot(_unit(d1), _unit(d2))))
    return float(np.degrees(np.arccos(min(1.0, c))))


def frame_vps(segments: np.ndarray, shape: tuple[int, int], rng: np.random.Generator, c: dict) -> dict:
    """VPs of one frame from its segments. Returns a dict (empty VP list if too few lines)."""
    n = len(segments)
    out = dict(n_segments=n, dirs=np.zeros((0, 3)), vps=np.zeros((0, 3)), counts=[],
               labels=-np.ones(n, dtype=int), explained=np.nan, resid=np.nan, orth_err=np.nan,
               segments=segments)
    if n < c.get("min_segments", 12):
        return out
    vps, labels = estimate_vps(segments, rng, n_vps=c.get("n_vps", 3),
                               angle_threshold_deg=c.get("angle_deg", 2.0),
                               min_inliers=c.get("min_inliers", 6), max_iters=c.get("max_iters", 1000))
    if len(vps) == 0:
        return out
    K = assumed_K(shape, c.get("focal_factor", 0.8))
    dirs = _unit(np.linalg.solve(K, vps.T).T)
    length = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    res = np.concatenate([vp_residual_angle(v, segments[labels == k]) for k, v in enumerate(vps)])
    out.update(dirs=dirs, vps=vps, labels=labels, counts=[int((labels == k).sum()) for k in range(len(vps))],
               explained=float(length[labels >= 0].sum() / length.sum()),
               resid=float(np.median(res)) if len(res) else np.nan,
               orth_err=abs(90.0 - angle_deg(dirs[0], dirs[1])) if len(dirs) >= 2 else np.nan)
    return out


# ---------------------------------------------------------------- over time
def build_tracks(per_frame: list[dict], max_jump_deg: float):
    """Link the VP directions of consecutive sampled frames (greedy, smallest angle first,
    one to one, angle < max_jump_deg).

    Returns (tracks, assign): a track is a list of (frame index, unit direction), the sign
    of each direction aligned on the previous one; assign[j][k] = track of VP k of frame j.
    """
    tracks, assign = [], []
    prev: list[int] = []                          # tracks that reached the previous frame
    for j, fr in enumerate(per_frame):
        dirs = fr["dirs"]
        cand = sorted((angle_deg(tracks[t][-1][1], d), k, t) for t in prev for k, d in enumerate(dirs))
        here: dict[int, int] = {}                 # VP index in frame j -> track
        for ang, k, t in cand:
            if ang >= max_jump_deg or k in here or t in here.values():
                continue
            d = dirs[k] if np.dot(dirs[k], tracks[t][-1][1]) >= 0 else -dirs[k]
            tracks[t].append((j, d))
            here[k] = t
        for k, d in enumerate(dirs):
            if k not in here:
                tracks.append([(j, d)])
                here[k] = len(tracks) - 1
        assign.append(here)
        prev = list(here.values())
    return tracks, assign


def rotation_between(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rotation R minimizing sum |R a_k - b_k|^2 (orthogonal Procrustes, SVD), rows = directions."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    return U @ D @ Vt


def temporal_features(per_frame: list[dict], c: dict) -> dict:
    """I2 features of a sequence of frames analysed by frame_vps (equally spaced in time)."""
    nan = float("nan")
    med = lambda v: float(np.nanmedian(v)) if len(v) and np.isfinite(v).any() else nan
    has = [len(f["dirs"]) > 0 for f in per_frame]
    out = dict(i2_frac_frames_vp=float(np.mean(has)) if per_frame else nan,
               i2_n_segments_med=med([f["n_segments"] for f in per_frame]),
               i2_n_vps_med=med([len(f["dirs"]) for f in per_frame]),
               i2_explained_frac_med=med([f["explained"] for f in per_frame]),
               i2_resid_med_deg=med([f["resid"] for f in per_frame]),
               i2_orth_err_med_deg=med([f["orth_err"] for f in per_frame]))
    tracks, assign = build_tracks(per_frame, c.get("max_jump_deg", 6.0))
    jumps, jitter, lens = [], [], []
    for tr in tracks:
        lens.append(len(tr))
        for (j0, d0), (j1, d1) in zip(tr, tr[1:]):
            jumps.append(np.degrees(np.arccos(np.clip(d0 @ d1, -1, 1))))
        for (j0, d0), (j1, d1), (j2, d2) in zip(tr, tr[1:], tr[2:]):
            jitter.append(np.degrees(np.linalg.norm(d2 - 2 * d1 + d0)))   # small-angle arc
    # angle between two directions seen together in many frames
    min_len = c.get("min_track_len", 5)
    long = [dict(tr) for tr in tracks if len(tr) >= min_len]
    pair_std = []
    for a in range(len(long)):
        for b in range(a + 1, len(long)):
            common = sorted(set(long[a]) & set(long[b]))
            if len(common) >= min_len:
                pair_std.append(np.std([angle_deg(long[a][j], long[b][j]) for j in common]))
    # is the strongest VP of a frame found again in the next one?
    # one rotation for all directions: fit it on the directions seen in frames j and j+1
    # (orthogonal Procrustes), then measure how far each direction is from it
    rot_res = []
    pos = [dict(tr) for tr in tracks]
    for j in range(len(per_frame) - 1):
        pairs = [(p[j], p[j + 1]) for p in pos if j in p and j + 1 in p]
        if len(pairs) >= 2:
            R = rotation_between(np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))
            rot_res.append(np.median([angle_deg(R @ a, b) for a, b in pairs]))
    kept = []
    for j in range(len(per_frame) - 1):
        f = per_frame[j]
        if len(f["dirs"]) == 0 or len(per_frame[j + 1]["dirs"]) == 0:
            continue
        t = assign[j][int(np.argmax(f["counts"]))]
        kept.append(j + 1 in dict(tracks[t]))
    jit, res = med(jitter), out["i2_resid_med_deg"]
    out.update(i2_jump_med_deg=med(jumps), i2_jitter_med_deg=jit,
               i2_jitter_norm=jit / res if np.isfinite(res) and res > 0 else nan,
               i2_pair_angle_std_deg=med(pair_std), i2_rot_resid_deg=med(rot_res),
               i2_track_len_med=med(lens),
               i2_dominant_kept_frac=float(np.mean(kept)) if kept else nan)
    return out


def frames_features(frames: list[np.ndarray], c: dict, rng: np.random.Generator,
                    return_frames: bool = False):
    per_frame = []
    for g in frames:
        seg = detect_segments(g, c.get("min_len_frac", 0.04), c.get("max_segments", 300))
        per_frame.append(frame_vps(seg, g.shape[:2], rng, c))
    feats = temporal_features(per_frame, c)
    return (feats, per_frame) if return_frames else feats


def sample_frames(frames: list, stride: int, max_frames: int) -> list[int]:
    """Equally spaced indices (every `stride` frames, at most max_frames, from the start)."""
    return list(range(0, len(frames), max(1, stride)))[:max_frames]


def video_features(path: Path, cfg: dict, return_frames: bool = False):
    """cfg: keys ``frames`` and ``i2`` of the pipeline config."""
    fr, c = cfg["frames"], cfg["i2"]
    vf = read_frames(path, skip_s=fr["skip_s"], crop_bottom=fr["crop_bottom"],
                     crop_top=fr["crop_top"], duration_s=fr.get("duration_s"))
    idx = sample_frames(vf.frames, c.get("stride", 2), c.get("max_frames", 48))
    rng = np.random.default_rng([int(c.get("seed", 0)), zlib.crc32(Path(path).stem.encode())])
    res = frames_features([vf.frames[i] for i in idx], c, rng, return_frames)
    if return_frames:
        return res[0], dict(per_frame=res[1], frames=[vf.frames[i] for i in idx],
                            times=[vf.times[i] for i in idx])
    return res


def image_features(path: Path, cfg: dict) -> dict:
    """Single-image I2 features (cfg: key ``i2``): the per-frame part only."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise IOError(f"cannot read image {path}")
    c = cfg["i2"]
    f = frame_vps(detect_segments(g, c.get("min_len_frac", 0.04), c.get("max_segments", 300)),
                  g.shape, np.random.default_rng(c.get("seed", 0)), c)
    return dict(i2_n_segments=f["n_segments"], i2_n_vps=len(f["dirs"]), i2_explained_frac=f["explained"],
                i2_resid_deg=f["resid"], i2_orth_err_deg=f["orth_err"])
