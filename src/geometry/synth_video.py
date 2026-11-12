# Cours : Session 4 (Stereo) : vidéos synthétiques à géométrie connue, pour valider les mesures vidéo
"""Synthetic videos with known geometry, to check what the video measures react to.

A rigid scene made of textured planes (ground, two walls, a far wall, a few panels) is
rendered from a camera that moves forward. Each pixel shows the nearest plane hit by its
ray (the texture is read through the plane-to-image homography), so every frame pair
satisfies one fundamental matrix exactly, up to resampling.

Two perturbations, applied to the rendered frames:
- ``blur_sigma``: Gaussian blur of every frame. The geometry is untouched; only the
  rendering changes (this imitates a generator that re-renders a real scene).
- ``drift_px``: a smooth displacement field whose amplitude grows linearly with time, up
  to ``drift_px`` pixels at the last frame. Frame pairs far apart in time no longer fit
  one rigid scene (this imitates a generated scene that slowly deforms).
"""
from __future__ import annotations

import cv2
import numpy as np


PX_PER_M = 64      # texture resolution of every plane


def _texture(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    """Random texture with detail at several scales, so that SIFT finds keypoints."""
    img = np.zeros((height, width), np.float32)
    for sigma, weight in [(1.0, 0.5), (3.0, 0.7), (9.0, 1.0), (27.0, 1.0)]:
        noise = rng.normal(size=(height, width)).astype(np.float32)
        layer = cv2.GaussianBlur(noise, (0, 0), sigma)
        img += weight * layer / (layer.std() + 1e-9)
    for _ in range(max(40, width * height // 6000)):      # sharp rectangles
        x, y = rng.integers(0, width - 40), rng.integers(0, height - 40)
        w, h = rng.integers(8, 40, size=2)
        img[y:y + h, x:x + w] += rng.normal() * 1.5
    img = (img - img.min()) / (img.max() - img.min())
    return (255 * img).astype(np.uint8)


def _planes(rng: np.random.Generator) -> list[dict]:
    """Each plane: origin P0, two edge vectors U and V (metres), and its own texture."""
    planes = [
        dict(P0=[-6, 1.5, -2], U=[12, 0, 0], V=[0, 0, 44]),     # ground
        dict(P0=[-6, -4, -2], U=[0, 0, 44], V=[0, 5.5, 0]),     # left wall
        dict(P0=[6, -4, -2], U=[0, 0, 44], V=[0, 5.5, 0]),      # right wall
        dict(P0=[-6, -4, 42], U=[12, 0, 0], V=[0, 5.5, 0]),     # far wall
    ]
    for z in (9.0, 14.0, 20.0, 27.0):                          # panels standing in the scene
        x = float(rng.uniform(-4.5, 2.5))
        planes.append(dict(P0=[x, -1.0, z], U=[2.0, 0, 0], V=[0, 2.5, 0]))
    for p in planes:
        p.update({k: np.asarray(p[k], float) for k in ("P0", "U", "V")})
        tw = int(round(np.linalg.norm(p["U"]) * PX_PER_M))
        th = int(round(np.linalg.norm(p["V"]) * PX_PER_M))
        p["tex"] = _texture(rng, tw, th)
    return planes


def _render_frame(planes, K, R, t, shape, pix) -> np.ndarray:
    """Ray casting with a depth buffer: each pixel shows the nearest plane it hits."""
    h, w = shape
    img = np.full(h * w, 200.0, np.float32)                    # sky
    zbuf = np.full(h * w, np.inf)
    for p in planes:
        th, tw = p["tex"].shape
        # texture pixel (a, b) -> 3D point P0 + (a / tw) U + (b / th) V -> image (homogeneous)
        Hm = K @ np.column_stack([R @ p["U"] / tw, R @ p["V"] / th, R @ p["P0"] + t])
        q = pix @ np.linalg.inv(Hm).T
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
            depth = Hm[2, 0] * a + Hm[2, 1] * b + Hm[2, 2]
        ok = np.isfinite(depth) & (a >= 0) & (a <= tw - 1) & (b >= 0) & (b <= th - 1) & (depth > 0.05) & (depth < zbuf)
        if not ok.any():
            continue
        a_map = np.where(ok, a, -1).astype(np.float32).reshape(h, w)
        b_map = np.where(ok, b, -1).astype(np.float32).reshape(h, w)
        val = cv2.remap(p["tex"], a_map, b_map, cv2.INTER_LINEAR).ravel()[ok]
        img[ok], zbuf[ok] = val, depth[ok]
    return img.reshape(h, w).astype(np.uint8)


def _warp_field(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """Smooth unit displacement field (h, w, 2): low-frequency, maximum norm 1."""
    h, w = shape
    f = rng.normal(size=(h // 40 + 2, w // 40 + 2, 2)).astype(np.float32)
    f = cv2.resize(f, (w, h), interpolation=cv2.INTER_CUBIC)
    return f / np.linalg.norm(f, axis=2).max()


def render_video(seed: int = 0, n_frames: int = 60, shape: tuple[int, int] = (360, 640),
                 focal: float = 520.0, step: float = 0.06, blur_sigma: float = 0.0,
                 drift_px: float = 0.0, noise_std: float = 1.0) -> dict:
    """Render the clip. Returns dict(frames, K, poses) with frames as (h, w) uint8 arrays.

    The scene and the camera path depend only on ``seed``: two calls with the same seed
    and different perturbations show the same scene and motion.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])
    planes = _planes(rng)
    field = _warp_field(rng, shape)
    sway = rng.uniform(0.5, 1.0)
    noise_rng = np.random.default_rng(seed + 1)
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pix = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.ones(h * w)]).astype(float)
    frames, poses = [], []
    for k in range(n_frames):
        # camera centre moves forward with a small sideways sway; small yaw and pitch
        c = np.array([0.25 * np.sin(sway * k * 0.08), 0.05 * np.sin(k * 0.11), k * step])
        yaw, pitch = 0.02 * np.sin(k * 0.05), 0.01 * np.sin(k * 0.07)
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        R = Rx @ Ry
        t = -R @ c
        img = _render_frame(planes, K, R, t, shape, pix)
        if drift_px > 0:
            a = drift_px * k / max(n_frames - 1, 1)
            img = cv2.remap(img, grid_x + a * field[..., 0], grid_y + a * field[..., 1],
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        if noise_std > 0:
            img = np.clip(img + noise_rng.normal(0, noise_std, img.shape), 0, 255).astype(np.uint8)
        frames.append(img)
        poses.append((R, t))
    return dict(frames=frames, K=K, poses=poses)


def write_video(frames: list[np.ndarray], path, fps: float = 24.0) -> None:
    """Write grayscale frames to an AVI (MJPG) file."""
    h, w = frames[0].shape
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        wr.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    wr.release()
