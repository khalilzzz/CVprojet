DO GENERATED VIDEOS RESPECT THE GEOMETRY OF A CAMERA?
CSC_51073_EP Image Analysis and Computer Vision, Ecole polytechnique

We measure, with the methods of the course implemented by us in numpy, whether
AI-generated videos respect the constraints a real camera always satisfies:
- epipolar geometry between frames (RANSAC, normalized 8-point, Sampson distance)
- three-view test of the SfM lecture (triangulation, camera DLT, reprojection error)
- vanishing points over time (LSD segments + RANSAC)
- temporal coherence by optical flow
- Fourier spectrum of the noise residual

The full documentation is in README.md. The report is report/main.pdf.


1. INSTALL (Python 3.10 or newer)

    python -m pip install -r requirements.txt
    python -m pytest -q          # 79 tests on synthetic data, about 2 min, no video needed

ffmpeg is brought by the imageio-ffmpeg package. torch and open_clip_torch are only
needed for the CLIP baseline. A GPU is optional.


2. DATA (not in the git repository)

Our clips (6 real, 10 generated, 213 MB):
- download them from <LINK TO ADD>
- put them in data/raw/videos/real/ and data/raw/videos/fake/
- their description is in data/metadata.csv

Public benchmark ViF-Bench (3160 videos, 0.6 GB, open access):

    python -m src.data.vifbench --download

data/raw/ is never modified. Normalized copies go to data/processed/.


3. DEMO: A REAL AND A GENERATED CLIP SIDE BY SIDE

    python demo.py data/raw/videos/real/vid6.mp4 data/raw/videos/fake/vid6fake.mp4

What it does:
- Measures every frame of both clips with our code. The first time this takes about
  3 min on 6 processes; the result is cached in results/demo/cache/.
- Writes results/demo/vid6_vs_vid6fake.mp4 (1920x1080).
- Plays it in a window.

Keys in the window:
- space: pause
- left / right arrow: one frame
- 1 to 7: chapter
- f: full screen
- q: quit

Options:
- --no-window: only write the MP4
- --speed 0.5: fixed slow motion
- --force: recompute

Chapters and their durations are in configs/demo.yaml.


4. WHOLE PIPELINE ON OUR CLIPS

    python -m src.run

The run writes results/latest/report.html, figures and tables. Parameters are in
configs/pipeline.yaml.

To analyse one pair in detail:

    python -m src.eval.inspect_video data/raw/videos/real/vid6.mp4 data/raw/videos/fake/vid6fake.mp4

The report is written to results/inspect/<names>/report.html.


5. VIF-BENCH (THE NUMBERS OF THE REPORT)

    python -m src.run --config configs/vifbench.yaml --dataset vifbench --features v1,i1,v3,clip --workers 16

This takes about 26 min with 16 workers. Add --max-scenes 20 for a quick test.

Output in results/latest_vifbench/:
- report.html
- classification/classify_summary.csv: AUC per family and the CLIP baseline
- classification/classify_per_generator.csv
- classification/classify_unseen_generator.csv

Further analyses:

    python -m src.eval.by_conditioning results/latest_vifbench
    python -m src.eval.motion_vs_coherence results/latest_vifbench


6. REPORT

    cd report
    latexmk -pdf main.tex


7. WHERE THINGS ARE

src/geometry/   our implementations: RANSAC, 8-point, DLT homography, Sampson,
                triangulation, camera DLT, vanishing points
src/features/   V1 (video_epipolar.py), I2 (image_perspective.py),
                V3 (video_temporal.py), I1 (image_frequency.py),
                V5 (video_rigidflow.py): rigid flow with learned depth, outside the
                course; python -m src.run --features v5 (needs transformers)
src/models/     classifiers (cross-validation grouped by scene) and the CLIP baseline
src/eval/       reports, inspection, robustness
src/demo/       demo.py measures and drawing
configs/        every parameter (YAML). Seeds are fixed.
tests/          unit tests on synthetic data
