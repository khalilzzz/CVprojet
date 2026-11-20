# Graph Report - projet  (2026-11-16)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 683 nodes · 1928 edges · 32 communities (27 shown, 5 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.85)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `eb9df5af`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- run.py
- test_multiview.py
- classify.py
- video_epipolar.py
- test_fundamental.py
- image_perspective.py
- measure.py
- render.py
- plots.py
- vanishing.py
- synthetic.py
- image_frequency.py
- compare.py
- inspect_video.py
- numpy
- test_demo.py
- by_conditioning.py
- test_robustness_motion.py
- to_h
- cv2
- synth_video.py
- homography.py
- pandas
- pathlib
- test_features_i1_v3_classify.py
- baseline_clip.py
- ingest.py
- ai-media-detection

## God Nodes (most connected - your core abstractions)
1. `build_report_v1()` - 26 edges
2. `two_view_scene()` - 23 edges
3. `to_h()` - 22 edges
4. `Report` - 20 edges
5. `load_config()` - 20 edges
6. `read_frames()` - 19 edges
7. `estimate_F_ransac()` - 19 edges
8. `main()` - 17 edges
9. `ransac()` - 16 edges
10. `from_h()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `_measures()` --calls--> `VideoFrames`  [EXTRACTED]
  tests/test_synth_measures.py → src/data/frames.py
- `main()` --calls--> `Renderer`  [EXTRACTED]
  demo.py → src/demo/render.py
- `test_measure_and_render_on_synthetic_clip()` --calls--> `Renderer`  [EXTRACTED]
  tests/test_demo.py → src/demo/render.py
- `main()` --calls--> `load_config()`  [EXTRACTED]
  demo.py → src/utils/config.py
- `_cfg()` --calls--> `load_config()`  [EXTRACTED]
  tests/test_demo.py → src/utils/config.py

## Import Cycles
- None detected.

## Communities (32 total, 5 thin omitted)

### Community 0 - "run.py"
Cohesion: 0.06
Nodes (26): _ffmpeg(), normalize_video(), _stamp(), main(), _img(), Report, _table(), degrade() (+18 more)

### Community 1 - "test_multiview.py"
Cohesion: 0.08
Nodes (22): three_view_points(), camera_dlt(), cameras_from_F(), estimate_camera_ransac(), fit(), residuals(), project(), reprojection_error() (+14 more)

### Community 2 - "classify.py"
Cohesion: 0.07
Nodes (22): classifier_scores(), analyse(), figure(), _impute(), main(), metric_of(), oof_corrected(), residualize() (+14 more)

### Community 3 - "video_epipolar.py"
Cohesion: 0.08
Nodes (23): analyse_frames(), analyse_video(), main(), match_descriptors(), pair_stats(), select_on_grid(), sift_features(), triplet_stats() (+15 more)

### Community 4 - "test_fundamental.py"
Cohesion: 0.11
Nodes (21): sampson_distance(), eight_point(), enforce_rank2(), estimate_F_ransac(), fit(), residuals(), normalization_transforms(), normalized_eight_point() (+13 more)

### Community 5 - "image_perspective.py"
Cohesion: 0.11
Nodes (21): angle_deg(), assumed_K(), build_tracks(), detect_segments(), frame_vps(), frames_features(), image_features(), rotation_between() (+13 more)

### Community 6 - "measure.py"
Cohesion: 0.10
Nodes (13): display_name(), main(), play(), timeline(), write_mp4(), _frame_rng(), _get(), i2_series() (+5 more)

### Community 7 - "render.py"
Cohesion: 0.12
Nodes (16): _dashed(), draw_i1(), draw_i2(), draw_three_view(), draw_v1(), draw_v3(), fit_frame(), fmt() (+8 more)

### Community 8 - "plots.py"
Cohesion: 0.17
Nodes (15): apply_style(), classification_figures(), _draw_epilines(), equal_motion(), _plain_log(), ratio_by_generator(), _save(), scene_delta_grid() (+7 more)

### Community 9 - "vanishing.py"
Cohesion: 0.13
Nodes (14): T_hartley(), vp_direction_error_deg(), estimate_vps(), fit(), residuals(), refine_vp(), segments_to_lines(), vp_residual_angle() (+6 more)

### Community 10 - "synthetic.py"
Cohesion: 0.15
Nodes (14): transfer_error(), from_h(), apply_H(), _add_outliers(), homography_scene(), _inside(), make_K(), manhattan_segments() (+6 more)

### Community 11 - "image_frequency.py"
Cohesion: 0.18
Nodes (11): i1_series(), _center_square(), frames_features(), image_features(), log_spectrum(), noise_residual(), radial_frequency(), radial_profile() (+3 more)

### Community 12 - "compare.py"
Cohesion: 0.14
Nodes (9): attach_meta(), benchmark_table(), bin_by_motion(), matched_summary(), paired_scenes(), _ratio(), split_windows(), three_view_table() (+1 more)

### Community 13 - "inspect_video.py"
Cohesion: 0.16
Nodes (8): _process(), read_frames(), VideoFrames, all_features(), read_reference(), reading(), sheet_i1_v3_i2(), video_features()

### Community 14 - "numpy"
Cohesion: 0.26
Nodes (10): _as_h(), intersect(), line_through(), normalize_lines(), skew(), test_line_through_and_intersection(), test_normalized_line_gives_point_distance(), test_parallel_lines_meet_at_infinity() (+2 more)

### Community 15 - "test_demo.py"
Cohesion: 0.18
Nodes (5): render_video(), _cfg(), test_measure_and_render_on_synthetic_clip(), test_three_view_points_equals_pipeline_triplet(), _measures()

### Community 16 - "by_conditioning.py"
Cohesion: 0.19
Nodes (7): _cond_type(), _auc(), bootstrap(), figure(), main(), _physics(), _shown()

### Community 17 - "test_robustness_motion.py"
Cohesion: 0.17
Nodes (6): _scene_prefix(), _gaps(), resolve(), _table(), test_evaluate_auc_drops_with_degradation(), test_indicators_resolve_middle_gap()

### Community 18 - "to_h"
Cohesion: 0.22
Nodes (6): _draw_line(), main(), algebraic_error(), symmetric_epipolar_distance(), to_h(), set_seed()

### Community 19 - "cv2"
Cohesion: 0.35
Nodes (6): measure_chunk(), _flow(), frames_features(), _highpass(), pair_stats(), residual_map()

### Community 20 - "synth_video.py"
Cohesion: 0.21
Nodes (5): _planes(), _render_frame(), _texture(), _warp_field(), write_video()

### Community 21 - "homography.py"
Cohesion: 0.26
Nodes (6): symmetric_transfer_error(), _dlt(), dlt_homography(), estimate_H_ransac(), fit(), residuals()

### Community 22 - "pandas"
Cohesion: 0.27
Nodes (4): ingest(), read_metadata(), download(), register()

### Community 23 - "pathlib"
Cohesion: 0.27
Nodes (3): check(), frame_at(), main()

### Community 24 - "test_features_i1_v3_classify.py"
Cohesion: 0.27
Nodes (6): _natural(), _shifted_clip(), test_classifier_groups_scenes_and_finds_signal(), test_i1_detects_periodic_upsampling_trace(), test_i1_image_slope_of_natural_image(), test_v3_boiling_texture_raises_warp_residual()

### Community 25 - "baseline_clip.py"
Cohesion: 0.36
Nodes (3): embed_videos(), load_model(), read_rgb_frames()

### Community 26 - "ingest.py"
Cohesion: 0.43
Nodes (3): probe_image(), probe_video(), scan_raw()

## Knowledge Gaps
- **1 isolated node(s):** `ai-media-detection`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 245 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_report_v1()` connect `run.py` to `plots.py`, `compare.py`?**
  _High betweenness centrality (0.020) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `build_report_v1()` (e.g. with `.details()` and `.figure()`) actually correct?**
  _`build_report_v1()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ai-media-detection` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run.py` be split into smaller, more focused modules?**
  _Cohesion score 0.05789473684210526 - nodes in this community are weakly interconnected._
- **Why does `Report` connect `run.py` to `inspect_video.py`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._
- **Should `test_multiview.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0792156862745098 - nodes in this community are weakly interconnected._
- **Why does `load_config()` connect `run.py` to `video_epipolar.py`, `measure.py`, `inspect_video.py`, `test_demo.py`, `to_h`?**
  _High betweenness centrality (0.018) - this node is a cross-community bridge._