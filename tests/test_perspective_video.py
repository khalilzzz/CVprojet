"""I2 over time: vanishing-point directions of a rigid scene are stable, those of a drifting
scene are not. Synthetic Manhattan scenes seen by a slowly rotating camera."""
import numpy as np
from scipy.spatial.transform import Rotation

from src.features.image_perspective import (angle_deg, assumed_K, build_tracks, detect_segments,
                                            frame_vps, temporal_features)
from src.geometry.synthetic import manhattan_segments

C = dict(min_segments=12, n_vps=3, angle_deg=2.0, min_inliers=6, max_iters=1000,
         focal_factor=1000.0 / 1280.0, max_jump_deg=6.0, min_track_len=5)


def _sequence(seed: int, n_frames: int = 24, wobble_deg: float = 0.0, shake_deg: float = 0.0):
    """Camera turning 0.4 degree per frame around the vertical axis. wobble_deg: each scene
    direction is tilted at random in every frame (a scene that is not rigid over time).
    shake_deg: the whole camera turns at random in every frame (handheld shake, rigid scene)."""
    rng = np.random.default_rng(seed)
    R0 = Rotation.from_euler("yx", [35.0, 15.0], degrees=True).as_matrix()
    per_frame = []
    for j in range(n_frames):
        R = Rotation.from_euler("y", 0.4 * j, degrees=True).as_matrix() @ R0
        if shake_deg:
            R = Rotation.from_rotvec(np.radians(shake_deg) * rng.normal(size=3) / np.sqrt(3)).as_matrix() @ R
        if wobble_deg:
            cols = [Rotation.from_rotvec(np.radians(wobble_deg) * rng.normal(size=3) / np.sqrt(3)).as_matrix()
                    @ R[:, i] for i in range(3)]
            R = np.stack(cols, axis=1)
        sc = manhattan_segments(rng, R=R, n_per_axis=30, n_outliers=20, noise_px=0.4)
        per_frame.append(frame_vps(sc["segments"], sc["shape"], rng, C))
    return per_frame


def test_assumed_K_matches_synthetic_camera():
    K = assumed_K((720, 1280), C["focal_factor"])
    assert np.isclose(K[0, 0], 1000.0) and np.isclose(K[0, 2], 640.0) and np.isclose(K[1, 2], 360.0)


def test_frame_vps_finds_three_orthogonal_directions():
    pf = _sequence(0, n_frames=1)[0]
    assert len(pf["dirs"]) == 3
    assert pf["orth_err"] < 2.0                      # the true camera K is used here
    assert pf["explained"] > 0.7


def test_tracks_follow_a_rotating_camera():
    pf = _sequence(1)
    tracks, assign = build_tracks(pf, C["max_jump_deg"])
    long = [t for t in tracks if len(t) >= 20]
    assert len(long) == 3                             # each axis is followed over the clip
    assert all(len(a) == len(f["dirs"]) for a, f in zip(assign, pf))


def test_rigid_scene_is_more_stable_than_wobbling_scene():
    rigid = temporal_features(_sequence(2), C)
    wobble = temporal_features(_sequence(2, wobble_deg=2.0), C)
    assert rigid["i2_jitter_med_deg"] < 0.5 * wobble["i2_jitter_med_deg"]
    assert rigid["i2_pair_angle_std_deg"] < 0.5 * wobble["i2_pair_angle_std_deg"]
    assert rigid["i2_dominant_kept_frac"] > 0.9


def test_rotation_residual_ignores_camera_shake_but_sees_wobble():
    shake = temporal_features(_sequence(3, shake_deg=2.0), C)
    wobble = temporal_features(_sequence(3, wobble_deg=2.0), C)
    rigid = temporal_features(_sequence(3), C)
    assert shake["i2_jitter_med_deg"] > 3 * rigid["i2_jitter_med_deg"]    # jitter sees the shake
    assert shake["i2_rot_resid_deg"] < 2 * rigid["i2_rot_resid_deg"]      # one rotation explains it
    assert wobble["i2_rot_resid_deg"] > 3 * shake["i2_rot_resid_deg"]


def test_rotation_between_recovers_a_rotation():
    rng = np.random.default_rng(4)
    R = Rotation.random(random_state=5).as_matrix()
    A = rng.normal(size=(3, 3))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    from src.features.image_perspective import rotation_between
    assert np.allclose(rotation_between(A, A @ R.T), R, atol=1e-9)


def test_detect_segments_on_a_drawn_box():
    img = np.full((240, 320), 40, np.uint8)
    img[60:180, 80:240] = 200                         # a bright rectangle: 4 long edges
    seg = detect_segments(img, min_len_frac=0.1)
    assert len(seg) >= 4
    horizontal = [s for s in seg if abs(s[3] - s[1]) < 2]
    assert len(horizontal) >= 2


def test_angle_is_sign_free():
    assert angle_deg(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0])) == 0.0
    assert np.isclose(angle_deg(np.array([1.0, 0, 0]), np.array([0, 1.0, 0])), 90.0)
