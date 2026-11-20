# Graph Report - projet  (2026-11-15)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 610 nodes · 1681 edges · 23 communities (18 shown, 5 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 35 edges (avg confidence: 0.85)
- Token cost: 45,089 input · 332 output

## Graph Freshness
- Built from commit: `eb9df5af`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Synthetic Geometry Demo
- Fundamental Matrix Estimation
- V1 Comparison Report
- Video Frame Epipolar Analysis
- Media Probing and Extraction
- Two-View Residual Errors
- Feature Classifier Training
- Vanishing Point Perspective
- Frequency Feature Sheets
- Epipolar Plotting Figures
- Conditioning Window Analysis
- Pipeline Run Orchestration
- Metadata and Dataset Ingest
- Readable Indicators and Tests
- Config and CLI Selection
- Robustness Degradation Experiment
- Video Normalization and Inspection
- CLIP Baseline Embeddings
- Project Root

## God Nodes (most connected - your core abstractions)
1. `build_report_v1()` - 26 edges
2. `two_view_scene()` - 23 edges
3. `to_h()` - 22 edges
4. `Report` - 20 edges
5. `read_frames()` - 19 edges
6. `estimate_F_ransac()` - 17 edges
7. `main()` - 17 edges
8. `from_h()` - 16 edges
9. `sampson_distance()` - 16 edges
10. `load_config()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `test_normalization_reduces_error_under_noise()` --calls--> `symmetric_epipolar_distance()`  [EXTRACTED]
  tests/test_fundamental.py → src/geometry/errors.py
- `test_exact_from_four_points()` --calls--> `homography_scene()`  [EXTRACTED]
  tests/test_homography.py → src/geometry/synthetic.py
- `test_matches_opencv()` --calls--> `homography_scene()`  [EXTRACTED]
  tests/test_homography.py → src/geometry/synthetic.py
- `test_ransac_with_outliers()` --calls--> `homography_scene()`  [EXTRACTED]
  tests/test_homography.py → src/geometry/synthetic.py
- `test_pair_stats_planar_scene_is_flagged_degenerate()` --calls--> `homography_scene()`  [EXTRACTED]
  tests/test_pipeline_v1.py → src/geometry/synthetic.py

## Import Cycles
- None detected.

## Communities (23 total, 5 thin omitted)

### Community 0 - "Synthetic Geometry Demo"
Cohesion: 0.06
Nodes (37): _draw_line(), main(), symmetric_epipolar_distance(), _as_h(), from_h(), intersect(), line_through(), normalize_lines() (+29 more)

### Community 1 - "Fundamental Matrix Estimation"
Cohesion: 0.07
Nodes (37): sampson_distance(), eight_point(), enforce_rank2(), estimate_F_ransac(), fit(), residuals(), normalization_transforms(), normalized_eight_point() (+29 more)

### Community 2 - "V1 Comparison Report"
Cohesion: 0.07
Nodes (20): attach_meta(), benchmark_table(), bin_by_motion(), matched_summary(), paired_scenes(), _ratio(), split_windows(), three_view_table() (+12 more)

### Community 3 - "Video Frame Epipolar Analysis"
Cohesion: 0.07
Nodes (25): _process(), read_frames(), VideoFrames, analyse_frames(), analyse_video(), main(), match_descriptors(), pair_stats() (+17 more)

### Community 4 - "Media Probing and Extraction"
Cohesion: 0.07
Nodes (21): check(), frame_at(), main(), probe_image(), probe_video(), scan_raw(), _flow(), frames_features() (+13 more)

### Community 5 - "Two-View Residual Errors"
Cohesion: 0.08
Nodes (18): algebraic_error(), symmetric_transfer_error(), transfer_error(), apply_H(), _dlt(), dlt_homography(), estimate_H_ransac(), fit() (+10 more)

### Community 6 - "Feature Classifier Training"
Cohesion: 0.08
Nodes (19): analyse(), figure(), _impute(), metric_of(), oof_corrected(), residualize(), v1_groups(), _align() (+11 more)

### Community 7 - "Vanishing Point Perspective"
Cohesion: 0.11
Nodes (21): angle_deg(), assumed_K(), build_tracks(), detect_segments(), frame_vps(), frames_features(), image_features(), rotation_between() (+13 more)

### Community 8 - "Frequency Feature Sheets"
Cohesion: 0.11
Nodes (17): all_features(), sheet_i1_v3_i2(), _center_square(), frames_features(), image_features(), log_spectrum(), noise_residual(), radial_frequency() (+9 more)

### Community 9 - "Epipolar Plotting Figures"
Cohesion: 0.20
Nodes (13): apply_style(), classification_figures(), _draw_epilines(), equal_motion(), _plain_log(), ratio_by_generator(), _save(), scene_delta_grid() (+5 more)

### Community 10 - "Conditioning Window Analysis"
Cohesion: 0.15
Nodes (8): _cond_type(), _auc(), bootstrap(), figure(), main(), _physics(), _shown(), main()

### Community 11 - "Pipeline Run Orchestration"
Cohesion: 0.13
Nodes (8): compute_families(), _free_gb(), _key(), _ram_ok(), run_clip(), run_family(), _run_pool(), simple_job()

### Community 12 - "Metadata and Dataset Ingest"
Cohesion: 0.22
Nodes (5): ingest(), read_metadata(), _scene_prefix(), download(), register()

### Community 13 - "Readable Indicators and Tests"
Cohesion: 0.20
Nodes (6): _gaps(), resolve(), _table(), test_evaluate_auc_drops_with_degradation(), test_indicators_resolve_middle_gap(), test_motion_groups_and_correction()

### Community 14 - "Config and CLI Selection"
Cohesion: 0.21
Nodes (6): main(), add_selection_args(), main(), normalize_all(), select_videos(), load_config()

### Community 15 - "Robustness Degradation Experiment"
Cohesion: 0.23
Nodes (3): degrade(), figure_auc(), figure_indicators()

### Community 16 - "Video Normalization and Inspection"
Cohesion: 0.31
Nodes (3): _ffmpeg(), normalize_video(), _stamp()

### Community 17 - "CLIP Baseline Embeddings"
Cohesion: 0.31
Nodes (3): embed_videos(), load_model(), read_rgb_frames()

## Knowledge Gaps
- **1 isolated node(s):** `ai-media-detection`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 225 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_report_v1()` connect `V1 Comparison Report` to `Epipolar Plotting Figures`, `Pipeline Run Orchestration`?**
  _High betweenness centrality (0.023) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `build_report_v1()` (e.g. with `.details()` and `.figure()`) actually correct?**
  _`build_report_v1()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ai-media-detection` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Synthetic Geometry Demo` be split into smaller, more focused modules?**
  _Cohesion score 0.06164383561643835 - nodes in this community are weakly interconnected._
- **Why does `Report` connect `V1 Comparison Report` to `Video Normalization and Inspection`, `Pipeline Run Orchestration`, `Config and CLI Selection`, `Robustness Degradation Experiment`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Should `Fundamental Matrix Estimation` be split into smaller, more focused modules?**
  _Cohesion score 0.07433489827856025 - nodes in this community are weakly interconnected._
- **Why does `read_frames()` connect `Video Frame Epipolar Analysis` to `Fundamental Matrix Estimation`, `Media Probing and Extraction`, `Vanishing Point Perspective`, `Frequency Feature Sheets`, `Video Normalization and Inspection`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._