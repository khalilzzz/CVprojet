# Cours : Sessions 1-2 (filtrage, rééchantillonnage, Fourier) : I1, empreintes fréquentielles
"""I1: frequency fingerprints of the noise residual.

Upsampling layers (transposed convolutions, the VAE decoder of latent diffusion models)
leave periodic traces in the Fourier spectrum; a camera image has a smooth spectrum.

For each sampled frame (grayscale, after the same crop as V1):
1. noise residual r = I - median3(I)   (the median filter removes the scene, keeps the noise),
2. Hann window, 2D FFT, log10 power spectrum of the residual (centred),
3. the log spectra of the frames are averaged (periodic artifacts add up, noise averages out).

Features of the averaged spectrum (frequency f in cycles/pixel, 0.5 = Nyquist):
- ``i1_prof_k``: azimuthal average (radial profile) in ``n_bins`` bins of f,
- ``i1_res_slope``: slope of log power against log f of the residual (mid band),
- ``i1_img_slope``: same for the image itself (about -2 for natural images, 1/f amplitude),
- ``i1_peak_max`` / ``i1_peak_frac``: excess of the 2D spectrum over its radial average
  (log10 units): mean of the 8 strongest frequencies, and share of frequencies 10 times
  above it,
- ``i1_hf_ratio``: share of residual energy above f = 0.25,
- ``i1_noise_std``: standard deviation of the residual (grey levels).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.data.frames import read_frames

CODE_VERSION = "i1-2026-11-13a"
PEAK_TOP = 8          # i1_peak_max = mean excess of the 8 strongest frequencies


def noise_residual(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    g = np.ascontiguousarray(gray, dtype=np.uint8)
    return g.astype(np.float32) - cv2.medianBlur(g, ksize).astype(np.float32)


def _center_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(size, h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]


def log_spectrum(img: np.ndarray, size: int) -> np.ndarray:
    """Centred log10 power spectrum of the central size x size square (Hann window)."""
    a = _center_square(img.astype(np.float64), size)
    a = a - a.mean()
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(a * win))
    return np.log10(np.abs(F) ** 2 + 1e-8)


def radial_frequency(n: int) -> np.ndarray:
    """Frequency (cycles/pixel) of each cell of a centred n x n spectrum."""
    k = np.fft.fftshift(np.fft.fftfreq(n))
    return np.hypot(*np.meshgrid(k, k))


def radial_profile(logp: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of logp in n_bins rings of frequency between 0 and 0.5. Returns (centres, means)."""
    f = radial_frequency(logp.shape[0])
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    idx = np.digitize(f, edges) - 1
    ok = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[ok], weights=logp[ok], minlength=n_bins)
    cnt = np.bincount(idx[ok], minlength=n_bins)
    return 0.5 * (edges[1:] + edges[:-1]), sums / np.maximum(cnt, 1)


def _slope(logp: np.ndarray, lo: float = 0.05, hi: float = 0.4) -> float:
    """Slope of log10 power against log10 frequency, between lo and hi cycles/pixel."""
    f = radial_frequency(logp.shape[0])
    m = (f >= lo) & (f <= hi)
    return float(np.polyfit(np.log10(f[m]), logp[m], 1)[0])


def spectrum_features(res_logp: np.ndarray, img_logp: np.ndarray, n_bins: int,
                      noise_std: float) -> dict:
    centres, prof = radial_profile(res_logp, n_bins)
    f = radial_frequency(res_logp.shape[0])
    # excess of each frequency over the mean of its ring (periodic peaks stand out)
    ring = np.interp(f, centres, prof)
    excess = (res_logp - ring)[(f > 0.03) & (f <= 0.5)]
    power = 10.0 ** res_logp
    out = {f"i1_prof_{k:02d}": float(v) for k, v in enumerate(prof)}
    out.update(i1_res_slope=_slope(res_logp), i1_img_slope=_slope(img_logp),
               i1_peak_max=float(np.sort(excess)[-PEAK_TOP:].mean()),
               i1_peak_frac=float((excess > 1.0).mean()),
               i1_hf_ratio=float(power[f > 0.25].sum() / power[f > 0.0].sum()),
               i1_noise_std=float(noise_std))
    return out


def frames_features(frames: list[np.ndarray], cfg: dict) -> dict:
    """I1 features of a list of grayscale frames (log spectra averaged over the frames)."""
    res_sum, img_sum, noise = None, None, []
    for g in frames:
        r = noise_residual(g, cfg.get("median_ksize", 3))
        rl, il = log_spectrum(r, cfg["crop"]), log_spectrum(g, cfg["crop"])
        res_sum = rl if res_sum is None else res_sum + rl
        img_sum = il if img_sum is None else img_sum + il
        noise.append(float(r.std()))
    n = len(frames)
    return spectrum_features(res_sum / n, img_sum / n, cfg["n_bins"], float(np.median(noise)))


def video_features(path: Path, cfg: dict) -> dict:
    """cfg: keys ``frames`` and ``i1`` of the pipeline config."""
    fr, c = cfg["frames"], cfg["i1"]
    vf = read_frames(path, skip_s=fr["skip_s"], crop_bottom=fr["crop_bottom"],
                     crop_top=fr["crop_top"], duration_s=fr.get("duration_s"))
    idx = np.unique(np.linspace(0, len(vf.frames) - 1, c["n_frames"]).round().astype(int))
    return frames_features([vf.frames[i] for i in idx], c)


def image_features(path: Path, cfg: dict) -> dict:
    """I1 features of one still image (cfg: key ``i1``)."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise IOError(f"cannot read image {path}")
    return frames_features([g], cfg["i1"])
