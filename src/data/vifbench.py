# Cours : hors cours (gestion des données) : inscription de ViF-Bench dans data/metadata.csv
"""Register the public benchmark ViF-Bench (Skyra, arXiv 2512.15693) in data/metadata.csv.

Download (0.6 GB) and unzip first:
    huggingface.co/datasets/JoeLeelyf/ViF-Bench, file source_videos.zip
    -> data/raw/public/vifbench/source_videos/{Real,Fake/<generator>}/<clip>.mp4

165 real clips; each generator has a clip with the same file name as its real counterpart,
so a real clip and its generated versions share one ``scene_id``. All clips are about 5 s,
256 px high, 16 fps (a few generators differ; configs/vifbench.yaml normalizes them).

    python -m src.data.vifbench --download   # download (0.6 GB), unzip, add the rows
    python -m src.data.vifbench              # only add the rows (files already there)
"""
from __future__ import annotations

import pandas as pd

from src.data.ingest import METADATA, read_metadata
from src.utils.config import REPO_ROOT

ROOT = REPO_ROOT / "data" / "raw" / "public" / "vifbench" / "source_videos"
REPO_ID, ZIP_NAME = "JoeLeelyf/ViF-Bench", "source_videos.zip"


def download() -> None:
    """Download source_videos.zip from Hugging Face (0.6 GB) and unzip it next to ROOT."""
    import zipfile

    from huggingface_hub import hf_hub_download

    dest = ROOT.parent
    dest.mkdir(parents=True, exist_ok=True)
    if ROOT.is_dir() and any(ROOT.rglob("*.mp4")):
        print(f"[vifbench] already there: {ROOT.relative_to(REPO_ROOT)}")
        return
    print(f"[vifbench] downloading {ZIP_NAME} (about 0.6 GB) from huggingface.co/datasets/{REPO_ID}")
    zip_path = hf_hub_download(REPO_ID, ZIP_NAME, repo_type="dataset", local_dir=str(dest))
    print("[vifbench] unzipping")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)


def _cond_type(generator: str) -> str:
    """What the generator was given, read from the folder name (-I / I2V = image, -T / T2V = text)."""
    g = generator.upper()
    if g.endswith("-T"):                 # suffix first: "Wan2.2-TI2V-5B-T" is the text variant
        return "text"
    if g.endswith("-I") or "I2V" in g:
        return "image"
    if "T2V" in g:
        return "text"
    return ""                            # not stated in the folder name (sora-2, kling-v1, ...)


def register(verbose: bool = True) -> pd.DataFrame:
    if not ROOT.is_dir():
        raise SystemExit(f"{ROOT} not found: download and unzip source_videos.zip first")
    df = read_metadata()
    # keep the conditioning type of rows already registered up to date with _cond_type
    rows_v = df["dataset"] == "vifbench"
    df.loc[rows_v, "cond_type"] = df.loc[rows_v, "generator"].map(_cond_type)
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
    import sys

    if "--download" in sys.argv[1:]:
        download()
    register()
