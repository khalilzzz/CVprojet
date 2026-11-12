# Cours : hors cours (gestion des données) : inscription de ViF-Bench dans data/metadata.csv
"""Register the public benchmark ViF-Bench (Skyra, arXiv 2512.15693) in data/metadata.csv.

Download (0.6 GB) and unzip first:
    huggingface.co/datasets/JoeLeelyf/ViF-Bench, file source_videos.zip
    -> data/raw/public/vifbench/source_videos/{Real,Fake/<generator>}/<clip>.mp4

165 real clips; each generator has a clip with the same file name as its real counterpart,
so a real clip and its generated versions share one ``scene_id``. All clips are about 5 s,
256 px high, 16 fps (a few generators differ; configs/vifbench.yaml normalizes them).

    python -m src.data.vifbench            # adds the rows (existing rows are kept)
"""
from __future__ import annotations

import pandas as pd

from src.data.ingest import METADATA, read_metadata
from src.utils.config import REPO_ROOT

ROOT = REPO_ROOT / "data" / "raw" / "public" / "vifbench" / "source_videos"


def _cond_type(generator: str) -> str:
    """What the generator was given, read from the folder name (-I / I2V = image, -T / T2V = text)."""
    g = generator.upper()
    if "I2V" in g or g.endswith("-I"):
        return "image"
    if "T2V" in g or g.endswith("-T"):
        return "text"
    return ""


def register(verbose: bool = True) -> pd.DataFrame:
    if not ROOT.is_dir():
        raise SystemExit(f"{ROOT} not found: download and unzip source_videos.zip first")
    df = read_metadata()
    known = set(df["id"])
    rows = []
    for path in sorted(ROOT.rglob("*.mp4")):
        rel = path.relative_to(ROOT).parts
        real = rel[0] == "Real"
        generator = "" if real else rel[1]
        vid = f"vif__{'real' if real else generator}__{path.stem}"
        if vid in known:
            continue
        row = {c: "" for c in df.columns}
        row.update(id=vid, media_type="video", label="real" if real else "fake",
                   source="ViF-Bench", generator=generator, scene_id=f"vif__{path.stem}",
                   input_image_id="" if real else f"vif__real__{path.stem}",
                   path=path.relative_to(REPO_ROOT).as_posix(), dataset="vifbench",
                   cond_type=_cond_type(generator), cond_s="0" if not real else "",
                   notes="public benchmark, row added by src.data.vifbench")
        rows.append(row)
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(METADATA, index=False)
    if verbose:
        v = df[df["dataset"] == "vifbench"]
        print(f"[vifbench] {len(rows)} rows added; {len(v)} ViF-Bench videos in data/metadata.csv "
              f"({(v['label'] == 'real').sum()} real, {v['generator'].nunique() - 1} generators)")
    return df


if __name__ == "__main__":
    register()
