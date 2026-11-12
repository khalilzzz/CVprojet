# Cours : hors cours (gestion des données) : inventaire automatique de data/raw dans data/metadata.csv
"""Keep data/metadata.csv in sync with the files in data/raw/ (CLAUDE.md rule 7).

- Every video in data/raw/videos/{real,fake}/ and image in data/raw/images/{real,fake}/
  gets one row. The label comes from the folder name.
- A field already filled by hand is never overwritten: only empty fields are completed.
- New files named like ``vid7.mp4``, ``vid7fake.mp4``, ``vid7fake2.mp4`` share one
  ``scene_id`` (the scene of an existing ``vid7*`` row, or ``scene_vid7``), so that real
  and generated versions of the same scene are compared and never split across folds.

Extra columns (not in CLAUDE.md section 6): ``path`` (relative to the repo root),
``dataset`` (own, vifbench, ...), ``cond_type`` (image | video | text: what the generator
was given) and ``cond_s`` (seconds of the real video given to the generator).
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import pandas as pd

from src.utils.config import REPO_ROOT

METADATA = REPO_ROOT / "data" / "metadata.csv"
BASE_COLUMNS = ["id", "media_type", "label", "source", "generator", "model_version", "prompt",
                "negative_prompt", "seed", "input_image_id", "scene_id", "duration_s",
                "resolution", "fps", "date_generated", "notes"]
EXTRA_COLUMNS = ["path", "dataset", "cond_type", "cond_s"]
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


def read_metadata() -> pd.DataFrame:
    if METADATA.exists():
        df = pd.read_csv(METADATA, dtype=str, keep_default_na=False)
    else:
        df = pd.DataFrame(columns=BASE_COLUMNS)
    for col in BASE_COLUMNS + EXTRA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    ok, frame = cap.read()          # the decoded frame has the display orientation
    cap.release()
    if not ok:
        return {}
    h, w = frame.shape[:2]
    return dict(duration_s=f"{n / fps:.2f}" if fps else "", resolution=f"{w}x{h}",
                fps=f"{fps:.2f}".rstrip("0").rstrip("."))


def probe_image(path: Path) -> dict:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {}
    return dict(resolution=f"{img.shape[1]}x{img.shape[0]}")


def _scene_prefix(stem: str) -> str | None:
    m = re.match(r"(vid\d+)", stem, flags=re.IGNORECASE)
    return m.group(1).lower() if m else None


def scan_raw() -> list[dict]:
    """One dict per media file found in data/raw/{videos,images}/{real,fake}/."""
    found = []
    for media, exts, probe in [("videos", VIDEO_EXT, probe_video), ("images", IMAGE_EXT, probe_image)]:
        for label in ("real", "fake"):
            folder = REPO_ROOT / "data" / "raw" / media / label
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() in exts:
                    found.append(dict(id=path.stem, label=label,
                                      media_type="video" if media == "videos" else "image",
                                      path=path.relative_to(REPO_ROOT).as_posix(), probe=probe,
                                      abspath=path))
    return found


def ingest(write: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Add missing rows and fill empty fields. Returns the metadata table."""
    df = read_metadata()
    added, completed = [], 0
    for item in scan_raw():
        rows = df.index[df["id"] == item["id"]]
        if len(rows):                                   # existing row: only fill empty fields
            i = rows[0]
            for col, val in [("path", item["path"]), ("dataset", "own")]:
                if df.at[i, col] == "":
                    df.at[i, col] = val
                    completed += 1
            continue
        row = {c: "" for c in df.columns}
        row.update(id=item["id"], media_type=item["media_type"], label=item["label"],
                   path=item["path"], dataset="own", notes="row added automatically by src.data.ingest")
        row.update(item["probe"](item["abspath"]))
        prefix = _scene_prefix(item["id"])
        if prefix:
            same = df[df["id"].str.lower().str.startswith(prefix) & (df["scene_id"] != "")]
            row["scene_id"] = same["scene_id"].iloc[0] if len(same) else f"scene_{prefix}"
            if item["label"] == "fake":
                real = df[df["id"].str.lower().str.startswith(prefix) & (df["label"] == "real")]
                if len(real):
                    row["input_image_id"] = real["id"].iloc[0]
        else:
            row["scene_id"] = f"scene_{item['id']}"
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        added.append(item["id"])
    if write and (added or completed):
        df.to_csv(METADATA, index=False)
    if verbose:
        print(f"[ingest] {len(df)} rows in data/metadata.csv, {len(added)} added"
              + (f": {', '.join(added)}" if added else ""))
    return df


if __name__ == "__main__":
    ingest()
