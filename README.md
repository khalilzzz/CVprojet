# Detecting AI-generated images and videos with physical and geometric cues

Course project for CSC_51073_EP (Image Analysis and Computer Vision, École polytechnique).
See `CLAUDE.md` for the full project description, methods and work plan.

## Install

```bash
python -m pip install -r requirements.txt
```

Python >= 3.10. Heavy dependencies (torch, pycolmap, diffusers, ...) are commented out in
`requirements.txt` and will be installed when the corresponding phase starts.

## Usage

Whole pipeline, one command (from the repo root):

```bash
python -m src.run                              # every video listed in data/metadata.csv
python -m src.run --videos vid4real vid4fake   # only these ids
python -m src.run --force                      # ignore the cache
```

Put new videos in `data/raw/videos/real/` or `data/raw/videos/fake/` first: they are added
to `data/metadata.csv`, normalized into `data/processed/videos/`, analysed (results are
cached per video in `results/cache/`), and compared. Each run writes its figures, tables
and a self-contained `report.html` to `results/runs/<date-time>/`, copied to
`results/latest/`. Parameters live in `configs/pipeline.yaml`.

Feature families (`--features`, default `v1,i1,v3`): V1 epipolar and three-view geometry,
I1 frequency fingerprint, V3 temporal coherence, and `clip` for the deep learning baseline
(GPU if available). When at least 10 scenes have a real and a generated clip, the report
adds a classification section: cross-validation grouped by scene, AUC per family, per
generator, and for generators left out of training.

Public benchmark ViF-Bench (165 real clips, 19 generators, paired by file name):

```bash
python -m src.data.vifbench --download     # 0.6 GB from Hugging Face, unzip, add to data/metadata.csv
python -m src.run --config configs/vifbench.yaml --dataset vifbench --features v1,i1,v3,clip --workers 16
# report in results/latest_vifbench/report.html, classifier outputs in results/latest_vifbench/classification/
python -m src.run --config configs/vifbench.yaml --dataset vifbench --max-scenes 20   # quick test
python -m src.run --config configs/vifbench.yaml --dataset vifbench --report-only     # report from the cache only
```

Run the tests:

```bash
python -m pytest -q
```

Generate the synthetic sanity figure for the geometry core:

```bash
python -m src.geometry.demo      # writes results/figures/geometry_synthetic.png
```

## Geometry core (`src/geometry/`)

Our own numpy implementations of the course algorithms (OpenCV is used only in the tests, as a check):

| Module | Content | Course |
|---|---|---|
| `homogeneous.py` | homogeneous coordinates, `[t]x`, line through 2 points, line intersection | EP4 slides 73-75 |
| `normalization.py` | course normalization (image to [-1, 1]^2) and Hartley normalization | EP4 slides 102-104 |
| `ransac.py` | generic RANSAC, adaptive number of rounds, final refit on inliers | EP3 slides 100-112 |
| `fundamental.py` | 8-point and normalized 8-point algorithms, rank 2, RANSAC estimate of F | EP4 slides 98-104 |
| `homography.py` | normalized DLT, RANSAC estimate of H | EP3 slides 93-94 |
| `errors.py` | algebraic error, Sampson distance, epipolar distance, transfer error | EP4; Hartley & Zisserman eq. 11.9 |
| `vanishing.py` | vanishing points by sequential RANSAC on line segments | EP4 slide 75 + RANSAC |
| `synthetic.py` | synthetic scenes with known ground truth (two views, homographies, Manhattan box) | EP4 slides 10-14 |

Parameters live in `configs/geometry.yaml`.

## Layout

```
configs/    parameters (YAML)
course/     slides (PDF) and lab code
data/       raw/ (never modified), processed/, metadata.csv
src/        data, geometry, features, localization, models, eval, utils
notebooks/  exploration and report figures
results/    extracted features, scores, figures (not versioned)
tests/      unit tests (synthetic data)
```
