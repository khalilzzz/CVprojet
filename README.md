# Do generated videos respect the geometry of a camera?

Course project for CSC_51073_EP (Image Analysis and Computer Vision, École polytechnique).

We measure whether AI-generated videos respect the physical constraints that a real camera
always satisfies. We use the methods of the course, implemented by us in numpy:
- epipolar geometry between frames (RANSAC, normalized 8-point algorithm, Sampson distance);
- a three-view test from the Structure-from-Motion lecture (triangulation, camera DLT,
  reprojection error);
- vanishing points over time (LSD segments + RANSAC);
- temporal coherence by optical flow;
- the Fourier spectrum of the noise residual.

The measures run on our own real/generated pairs and on the public ViF-Bench benchmark,
and are compared with a CLIP linear probe.

- Report: `report/main.pdf` (LaTeX source `report/main.tex`)
- Demo video: `results/demo/vid6_vs_vid6fake.mp4`, made by `demo.py`
- Code: `src/`, parameters: `configs/`, tests: `tests/`

## 1. Install

Python 3.10 or newer (tested with 3.13 on Windows 11 and Linux).

```bash
python -m pip install -r requirements.txt
```

- `ffmpeg` is not needed separately: `imageio-ffmpeg` brings its own binary.
- `torch`, `torchvision` and `open_clip_torch` are only used by the CLIP baseline
  (`--features ...,clip`); `transformers` only by V5 (`--features ...,v5`). A GPU is optional.

Check the install (89 tests on synthetic data, about 2 minutes, no video needed):

```bash
python -m pytest -q
```

## 2. Data

The videos are not in the git repository (too large).

- **Our clips** (6 real, 10 generated, 213 MB): download them from
  `<LINK TO ADD>` and put them in `data/raw/videos/real/` and `data/raw/videos/fake/`.
  Their description (generator, conditioning, scene) is in `data/metadata.csv`.
- **ViF-Bench** (165 real clips, 19 generators, 3160 videos, 0.6 GB, open access):
  ```bash
  python -m src.data.vifbench --download
  ```
  The archive is downloaded from Hugging Face, unzipped into `data/raw/public/vifbench/`,
  and the clips are added to `data/metadata.csv`.

`data/raw/` is never modified. Normalized copies are written to `data/processed/`:
24 fps, short side 720 px, H.264 CRF 18, the same for real and generated clips.

## 3. Demo: real and generated side by side

```bash
python demo.py data/raw/videos/real/vid6.mp4 data/raw/videos/fake/vid6fake.mp4
```

- What it does: every frame of both clips is measured by our code. It writes a
  1920x1080 video to `results/demo/<real>_vs_<fake>.mp4`, then plays it in a window.
- On the frames: the two clips side by side and one method at a time:
  - V1 epipolar lines and the points that fit F, in red;
  - the three-view test with its reprojection errors;
  - parallel lines grouped by vanishing point;
  - the flow residual;
  - the noise spectrum.

  Below the frames, a table of 11 values is updated at every frame.
- Verdict (last chapter, `src/demo/verdict.py`): the three-view error of the two clips is compared
  at equal image motion (log-linear fit on the frames, 95 % interval by block bootstrap). It says
  whether the generated clip agrees less with a real camera, the real one does, there is no
  measurable difference, or the motions of the two clips are too different to compare. It compares
  two clips of the same scene; it is not a detector. Settings in the `verdict` block of
  `configs/demo.yaml`.
- The medians and the verdict are also written to `results/demo/<real>_vs_<fake>.csv`.
- Time: about 3 min for two 6 s clips on 6 CPU processes the first time. The measures are
  then cached in `results/demo/cache/`, and the video is written in about 1 min.
- Keys in the window: `space` pause, `left`/`right` (or `a`/`d`) one frame, `1`-`7` chapter,
  `f` full screen, `q` or `Esc` quit.
- Options:
  - `--no-window`: only write the MP4;
  - `--speed 0.5`: fixed slow motion (the default, `auto`, stretches the clip over all
    chapters);
  - `--force`: recompute the measures;
  - `--workers N`: number of processes;
  - `--no-video`: no MP4 and no window, only the verdict in the terminal and the CSV (a few
    seconds once the measures are cached);
  - `--short-side 1080`: keep 1080p instead of 720p (only when both clips are 1080p). The clips
    go to `data/processed/videos_1080/` and the video to `..._1080p.mp4`, so the 720p files of
    the pipeline are never touched.
- Chapter list and durations: `configs/demo.yaml`.

Any pair of videos works, for example
`python demo.py data/raw/videos/real/vid5.mp4 data/raw/videos/fake/vid5fakeomni.mp4`.

## 4. Whole pipeline on our clips

```bash
python -m src.run                              # every clip of data/metadata.csv (dataset "own")
python -m src.run --videos vid6 vid6fake       # only these ids
python -m src.run --force                      # ignore the cache
```

New videos placed in `data/raw/videos/real/` or `data/raw/videos/fake/` are added to
`data/metadata.csv` automatically. Fill in their `generator`, `scene_id`, `cond_type` and
`cond_s` columns by hand; these fields are never overwritten.

Each video is then normalized and analysed (cache in `results/cache/`). The run writes its
figures, tables and a self-contained `report.html` to `results/runs/<date-time>/`, copied to
`results/latest/`.

Feature families (`--features`, default `v1,i1,v3,i2`):
- `v1`: V1 epipolar and three-view geometry;
- `i1`: I1 noise spectrum;
- `v3`: V3 temporal coherence;
- `i2`: I2 vanishing points over time;
- `v5`: V5 rigid flow with learned depth (outside the course, see below);
- `clip`: the CLIP baseline.

### V5: does the flow agree with the depth? (outside the course)

`src/features/video_rigidflow.py`. The instantaneous rigid-motion model (Longuet-Higgins and
Prazdny, 1980) says that, given the depth of every pixel, the optical flow of a camera moving in a
rigid scene is linear in 6 numbers (translation and rotation rates). The depth comes from a
monocular network, Depth Anything V2 (small, 25 M parameters, CPU is fine; downloaded from
Hugging Face on first use, about 100 MB). The network gives depth up to a scale and a shift, so
the fit has 9 linear unknowns, solved by robust least squares in numpy. The residual is the flow
that no camera motion can explain given that depth: moving objects, and a generated scene whose
motion does not follow its own depth.

```bash
python -m src.run --features v5                                   # features (cached)
python -m src.features.video_rigidflow data/processed/videos/vid6.mp4 data/processed/videos/vid6fake.mp4 --figure results/figures/v5_residual_vid6.png
```

The figure shows, per video: frame, learned depth, flow, and the per-pixel residual map.
Parameters in the `v5` block of the configs (frame gap, number of pairs, assumed focal length).

Parameters: `configs/pipeline.yaml`.

### One clip, or a real and a generated clip, in detail

```bash
python -m src.eval.inspect_video data/raw/videos/real/vid6.mp4 data/raw/videos/fake/vid6fake.mp4
python -m src.eval.inspect_video clip.mp4 --reference results/latest_vifbench/features.csv
```

The report is written to `results/inspect/<names>/report.html`. It gives every indicator,
what it means, and where the value falls among the real clips of the reference.
The reference must come from a run with the same config.

## 5. ViF-Bench benchmark (results of the report)

```bash
python -m src.data.vifbench --download
python -m src.run --config configs/vifbench.yaml --dataset vifbench --features v1,i1,v3,clip --workers 16
```

- About 26 min with 16 workers.
- For a quick test, add `--max-scenes 20`. A run restricted by `--max-scenes`, `--videos` or
  `--generators` is written to `results/latest_<dataset>_subset/`, so it never overwrites the
  full run.
- To rebuild the report from the cache only, add `--report-only`.
- Output in `results/latest_vifbench/`:
  - `report.html`;
  - `classification/classify_summary.csv`: AUC per feature family, CLIP baseline;
  - `classification/classify_per_generator.csv`;
  - `classification/classify_unseen_generator.csv`: generator left out of training.

Further analyses on that run:

```bash
python -m src.eval.by_conditioning results/latest_vifbench      # image- vs text-to-video generators
python -m src.eval.motion_vs_coherence results/latest_vifbench  # V1: camera motion or 3D coherence?
python -m src.eval.robustness --config configs/vifbench.yaml --dataset vifbench --max-scenes 40 --features v1,i1,v3,i2,clip --workers 16
```

The last command is experiment 3 (CRF 28 and 35, half resolution). Its output is in
`results/robustness_vifbench/`.

## 6. Other tools

```bash
python -m src.geometry.demo                    # synthetic check of F and vanishing points -> results/figures/geometry_synthetic.png
python extract.py data/raw/videos/real/vid5.mp4   # first frame of a real clip, for image-to-video generation
```

To compile the report (needs a LaTeX distribution such as MiKTeX or TeX Live):

```bash
cd report && latexmk -pdf main.tex
```

## 7. Code map

| Path | Content | Course |
|---|---|---|
| `src/geometry/homogeneous.py` | homogeneous coordinates, `[t]x`, line through 2 points, intersection | EP4 slides 73-75 |
| `src/geometry/normalization.py` | course and Hartley normalization | EP4 slides 102-104 |
| `src/geometry/ransac.py` | generic RANSAC, adaptive number of rounds, refit on inliers | EP3 slides 100-112 |
| `src/geometry/fundamental.py` | 8-point and normalized 8-point, rank 2, RANSAC estimate of F | EP4 slides 98-104 |
| `src/geometry/homography.py` | normalized DLT, RANSAC estimate of H | EP3 slides 93-94 |
| `src/geometry/errors.py` | algebraic error, Sampson distance, epipolar and transfer errors | EP4; H&Z eq. 11.9 |
| `src/geometry/multiview.py` | cameras from F, triangulation, camera DLT, three-view error | Session 7 |
| `src/geometry/vanishing.py` | vanishing points by sequential RANSAC | EP4 slide 75 + RANSAC |
| `src/geometry/synthetic.py`, `synth_video.py` | synthetic scenes and clips with known geometry | tests |
| `src/features/video_epipolar.py` | V1: pairs of frames (F, Sampson, H, parallax) and three-view test | Sessions 3, 4, 7 |
| `src/features/image_perspective.py` | I2: vanishing points per frame and over time | Sessions 1-4 |
| `src/features/video_temporal.py` | V3: flow warping residual, forward-backward check | Sessions 3, 6 |
| `src/features/image_frequency.py` | I1: noise residual spectrum, radial profile, periodic peaks | Sessions 1-2 |
| `src/features/video_rigidflow.py` | V5: rigid flow model with learned depth, robust fit, residual maps | outside the course (Szeliski ch. 9, 11, 12) |
| `src/models/classify.py` | classifiers per family, CV grouped by scene, unseen generator | Session 5 |
| `src/models/baseline_clip.py` | CLIP ViT-B/32 linear probe baseline | Session 5 |
| `src/eval/` | report, comparisons at equal motion, inspection, robustness | |
| `src/data/` | ingest into `metadata.csv`, normalization, frame reading, ViF-Bench | |
| `src/demo/`, `demo.py` | side-by-side demo (measures per frame, drawing) | Sessions 1-7 |
| `src/run.py` | the whole pipeline in one command | |

OpenCV is used for SIFT, LSD, Farnebäck flow and filtering, and in the tests to check our
implementations (`cv2.findFundamentalMat`, `cv2.findHomography`).

## 8. Layout

```
configs/    parameters (YAML): pipeline.yaml (our clips), vifbench.yaml, demo.yaml, geometry.yaml
data/       raw/ (never modified), processed/, metadata.csv
report/     LaTeX report and its figures
src/        data, geometry, features, models, eval, demo
results/    features, tables, figures, reports, demo videos (not versioned)
tests/      unit tests on synthetic data
demo.py     side-by-side demo
extract.py  first frame of a clip for image-to-video generation
```

Seeds are fixed (`seed` in the configs), and every parameter is in a YAML file.
