# Cours : Session 5 (Deep Learning for CV) : baseline, linear probe sur des features CLIP (UnivFD)
"""Deep learning baseline: CLIP image embeddings of a few frames per video, averaged; the
linear probe (logistic regression) is trained in src/models/classify.py.
In the spirit of UnivFD (Ojha et al., CVPR 2023). Uses a GPU when one is available.

Needs ``open_clip_torch`` (requirements.txt); the weights (about 350 MB) are downloaded
on first use.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

CODE_VERSION = "clip-2026-11-13a"


def load_model(cfg: dict):
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(cfg["model"], pretrained=cfg["pretrained"])
    return model.eval().to(device), preprocess, device


def read_rgb_frames(path: Path, n: int, skip_s: float, crop_bottom: float, crop_top: float):
    """n frames spread over the clip (after skip_s), RGB, with the same crop as the other features."""
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    frames = frames[int(round(skip_s * fps)):] or frames
    if not frames:
        raise IOError(f"no frame in {path}")
    idx = np.unique(np.linspace(0, len(frames) - 1, n).round().astype(int))
    out = []
    for i in idx:
        f = frames[i]
        h = f.shape[0]
        f = f[int(round(crop_top * h)): h - int(round(crop_bottom * h))]
        out.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    return out


def embed_videos(items: list[tuple[str, Path, Path]], cfg: dict) -> None:
    """items: (video id, normalized video path, output .npy path). Writes one 1-D vector per
    video: mean of the L2-normalized frame embeddings, normalized again."""
    import torch

    if not items:
        return
    model, preprocess, device = load_model(cfg["clip"])
    fr = cfg["frames"]
    t0 = time.time()
    with torch.no_grad():
        for k, (vid, path, out) in enumerate(items, 1):
            try:
                imgs = read_rgb_frames(path, cfg["clip"]["n_frames"], fr["skip_s"],
                                       fr["crop_bottom"], fr["crop_top"])
                x = torch.stack([preprocess(im) for im in imgs]).to(device)
                e = model.encode_image(x).float()
                e = e / e.norm(dim=1, keepdim=True)
                v = e.mean(0)
                v = (v / v.norm()).cpu().numpy()
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, v.astype(np.float32))
                status = "done"
            except Exception as exc:                     # one bad video must not stop the run
                status = f"FAILED ({type(exc).__name__}: {exc})"
            if k % 50 == 0 or k == len(items) or status != "done":
                print(f"[clip] {k}/{len(items)} {status}: {vid} ({time.time() - t0:.0f} s, {device})",
                      flush=True)
